import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def load_and_prepare_data(csv_path: str, metadata_path: str = None) -> Tuple[np.ndarray, dict]:
    """
    Load stationary returns from preprocessed data.
    
    The data is already transformed by download_sp500.py:
    - Either log_return (if stationary) OR
    - differenced_log_return (if log returns were non-stationary)
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date")
    
    # Load metadata to determine which column to use
    if metadata_path and os.path.exists(metadata_path):
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        use_column = metadata.get('use_column', 'log_return')
        transformation = metadata.get('final_transformation', 'log_returns')
    else:
        # Fallback: check if differenced column exists
        if 'differenced_log_return' in df.columns:
            use_column = 'differenced_log_return'
            transformation = 'differenced_log_returns'
        elif 'log_return' in df.columns:
            use_column = 'log_return'
            transformation = 'log_returns'
        else:
            raise ValueError("No transformed column found. Run download_sp500.py first.")
    
    returns = df[use_column].values
    
    stats = {
        'n_observations': len(df),
        'n_returns': len(returns),
        'transformation': transformation,
        'column_used': use_column,
        'mean': float(returns.mean()),
        'std': float(returns.std()),
    }
    
    return returns, stats


def make_sequences(values: np.ndarray, seq_len: int, forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for prediction.
    
    Args:
        values: Input time series
        seq_len: Length of input sequence
        forecast_horizon: Number of steps ahead to predict (default: 1 for next-day)
    """
    X, y = [], []
    for i in range(len(values) - seq_len - forecast_horizon + 1):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len + forecast_horizon - 1])
    return np.array(X), np.array(y)


def compute_temporal_weights(n_samples: int, decay_rate: float = 0.001) -> np.ndarray:
    """Compute exponentially increasing weights for time series samples.
    
    More recent samples get higher weight. decay_rate controls how much emphasis
    is placed on recent vs historical data.
    
    Args:
        n_samples: Number of samples
        decay_rate: Higher = more emphasis on recent data (default: 0.001)
    
    Returns:
        Array of weights, normalized to sum to n_samples (for loss averaging)
    """
    # Exponentially increasing weights: older samples get exp(-decay * age)
    ages = np.arange(n_samples, 0, -1)  # [n, n-1, ..., 2, 1]
    weights = np.exp(-decay_rate * ages)
    # Normalize so mean weight = 1.0 (keeps loss magnitude comparable)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        return self.fc(out)


def main():
    # Resolve paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_csv = os.path.join(root_dir, "data", "sp500_close.csv")
    metadata_json = os.path.join(root_dir, "data", "transformation_metadata.json")
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Hyperparameters
    seq_len = 60
    forecast_horizon = 5  # Predict 5 days ahead (weekly)
    batch_size = 64
    epochs = 20  # Increased epochs for better convergence
    lr = 1e-3
    weight_decay_rate = 0.0001  # Reduced temporal weighting (was 0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Configuration:")
    print(f"  Sequence length: {seq_len} days")
    print(f"  Forecast horizon: {forecast_horizon} days")
    print(f"  Temporal decay rate: {weight_decay_rate}")
    print(f"  Device: {device}")
    print()

    # Load and prepare data (already transformed and tested for stationarity)
    returns, data_stats = load_and_prepare_data(data_csv, metadata_json)
    print(f"Data: {data_stats['n_returns']} observations")
    print(f"  Transformation: {data_stats['transformation']}")
    print(f"  Using column: {data_stats['column_used']}")
    print(f"  Mean: {data_stats['mean']:.6f}, Std: {data_stats['std']:.6f}")
    
    # Standardize returns (z-score normalization)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(returns.reshape(-1, 1)).astype(np.float32).flatten()

    # Sequences
    X, y = make_sequences(scaled, seq_len, forecast_horizon)
    n = len(X)
    print(f"Created {n} sequences (predicting {forecast_horizon} days ahead)")
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    
    # Compute temporal weights for training samples (higher weight for recent data)
    train_weights = compute_temporal_weights(n_train, decay_rate=weight_decay_rate)
    train_weights_tensor = torch.from_numpy(train_weights)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    model = LSTMRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    def evaluate(loader):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                xb = xb.unsqueeze(-1)  # (B, T, 1)
                pred = model(xb).squeeze(-1)  # (B, 1) -> (B)
                loss = loss_fn(pred, yb)
                total += loss.item() * len(xb)
                count += len(xb)
        return total / max(count, 1)

    for epoch in range(1, epochs + 1):
        model.train()
        running_weighted = 0.0
        running_unweighted = 0.0
        batch_start_idx = 0
        
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.unsqueeze(-1)  # (B, T, 1)
            
            # Get weights for this batch
            batch_size_actual = len(xb)
            batch_weights = train_weights_tensor[batch_start_idx : batch_start_idx + batch_size_actual].to(device)
            batch_start_idx += batch_size_actual
            
            opt.zero_grad()
            pred = model(xb).squeeze(-1)  # (B, 1) -> (B)
            
            # Compute per-sample losses
            losses = (pred - yb) ** 2
            # Apply temporal weights
            weighted_loss = (losses.squeeze() * batch_weights).mean()
            
            weighted_loss.backward()
            opt.step()
            
            # Track both weighted and unweighted for reporting
            running_weighted += weighted_loss.item() * batch_size_actual
            running_unweighted += losses.mean().item() * batch_size_actual
        
        train_loss_weighted = running_weighted / max(len(train_ds), 1)
        train_loss_unweighted = running_unweighted / max(len(train_ds), 1)
        val_loss = evaluate(val_dl)
        print(f"Epoch {epoch:02d} | train MSE (weighted): {train_loss_weighted:.6f} | "
              f"train MSE (unweighted): {train_loss_unweighted:.6f} | val MSE: {val_loss:.6f}")

    test_loss = evaluate(test_dl)
    print(f"Test MSE: {test_loss:.6f}")

    # Save model with metadata
    model_path = os.path.join(models_dir, "sp500_lstm.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "seq_len": seq_len,
        "forecast_horizon": forecast_horizon,
        "scaler_mean": scaler.mean_[0],
        "scaler_scale": scaler.scale_[0],
    }, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

"""
Improved LSTM training with key fixes for better predictions:

1. Scale target variable (log returns) to [-1, 1] range
2. Use Huber Loss (robust to outliers)
3. Filter to modern data only (2000+)
4. Inverse transform predictions for evaluation
5. Use distance-from-MA features (already in prepare_features.py)
6. Multi-step prediction (5 days ahead at once)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List


class MultiStepLSTM(nn.Module):
    """
    LSTM that predicts multiple future timesteps at once.
    
    Key improvement: Predicts 5-day vector instead of iterative 1-day predictions.
    This prevents error accumulation from feeding predictions back as inputs.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_steps: int = 5
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_steps = output_steps
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output head - predicts multiple steps
        lstm_output_size = hidden_size * 2  # Bidirectional
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_steps)  # Output: 5 future days
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            (batch_size, output_steps) - predictions for next 5 days
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        x_proj = x.reshape(-1, self.input_size)
        x_proj = self.input_proj(x_proj)
        x_proj = x_proj.reshape(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x_proj)  # (B, T, hidden*2)
        
        # Attention
        attn_weights = self.attention(lstm_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, hidden*2)
        
        # Multi-step output
        out = self.fc(context)  # (B, output_steps)
        return out


def load_feature_data(csv_path: str, start_year: int = 2000) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load features and filter to modern era only.
    
    Fix #3: Drop pre-2000 data to avoid regime mixing.
    """
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Filter to modern era
    df = df[df['Date'].dt.year >= start_year].reset_index(drop=True)
    
    # Feature columns (exclude Date, Close, log_return)
    exclude = ['Date', 'Close', 'log_return']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    return df, feature_cols


def create_sequences_multistep(
    features: np.ndarray,
    target: np.ndarray,
    seq_len: int,
    output_steps: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for multi-step prediction.
    
    Fix #6: Predict 5-day vector at once instead of iterative 1-day.
    
    Args:
        features: (n_samples, n_features)
        target: (n_samples,) - scaled log returns
        seq_len: Input sequence length
        output_steps: Number of future steps to predict
    
    Returns:
        X: (n_sequences, seq_len, n_features)
        y: (n_sequences, output_steps) - next 5 days
    """
    X, y = [], []
    n_samples = len(features)
    
    for i in range(n_samples - seq_len - output_steps + 1):
        X.append(features[i:i + seq_len])
        # Target: next output_steps values
        y.append(target[i + seq_len:i + seq_len + output_steps])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for multi-step predictions.
    
    Args:
        y_true: (n_samples, n_steps)
        y_pred: (n_samples, n_steps)
    """
    # Average across all timesteps
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Directional accuracy (sign match)
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    # Per-step metrics
    step_mse = np.mean((y_true - y_pred) ** 2, axis=0)
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'directional_accuracy': float(directional_accuracy),
        'step_mse': step_mse.tolist()
    }


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0


def main():
    # Configuration
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    feature_csv = os.path.join(root_dir, "data", "sp500_features.csv")
    models_dir = os.path.join(root_dir, "models")
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Hyperparameters
    seq_len = 60
    output_steps = 5  # Predict 5 days at once
    batch_size = 64
    epochs = 100
    lr = 1e-3
    hidden_size = 128
    num_layers = 2
    dropout = 0.3
    patience = 20
    start_year = 2000  # Modern data only
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("IMPROVED S&P 500 FORECASTING MODEL")
    print("=" * 70)
    print("Key improvements:")
    print("  1. Target scaling (log returns normalized)")
    print("  2. Huber Loss (robust to outliers)")
    print(f"  3. Modern data only (>= {start_year})")
    print("  4. Inverse transform for evaluation")
    print("  5. Distance-from-MA features")
    print(f"  6. Multi-step prediction ({output_steps} days at once)")
    print()
    print(f"Configuration:")
    print(f"  Sequence length: {seq_len} days")
    print(f"  Output steps: {output_steps} days")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Device: {device}")
    print()
    
    # Load data (filtered to modern era)
    print("Loading features...")
    if not os.path.exists(feature_csv):
        print(f"Error: {feature_csv} not found")
        print("Run: python src/data/prepare_features.py")
        return
    
    df, feature_cols = load_feature_data(feature_csv, start_year=start_year)
    print(f"  {len(df)} samples from {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  {len(feature_cols)} features")
    print()
    
    # Prepare features and target
    features = df[feature_cols].values.astype(np.float32)
    target = df['log_return'].values.astype(np.float32)
    
    # FIX #1: Scale target variable
    print("Scaling target variable (log returns)...")
    target_scaler = StandardScaler()
    target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
    print(f"  Original: mean={target.mean():.6f}, std={target.std():.6f}")
    print(f"  Scaled:   mean={target_scaled.mean():.6f}, std={target_scaled.std():.6f}")
    print()
    
    # Scale features (fit on train only)
    n_samples = len(features)
    train_end = int(n_samples * 0.8)
    
    feature_scaler = StandardScaler()
    features_train = features[:train_end]
    feature_scaler.fit(features_train)
    features_scaled = feature_scaler.transform(features)
    
    # Create sequences with multi-step targets
    print("Creating multi-step sequences...")
    X, y = create_sequences_multistep(features_scaled, target_scaled, seq_len, output_steps)
    print(f"  Total sequences: {len(X)}")
    print(f"  Input shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print()
    
    # Train/Val/Test split
    n_seq = len(X)
    train_seq_end = int(n_seq * 0.8)
    val_seq_end = int(n_seq * 0.9)
    
    X_train, y_train = X[:train_seq_end], y[:train_seq_end]
    X_val, y_val = X[train_seq_end:val_seq_end], y[train_seq_end:val_seq_end]
    X_test, y_test = X[val_seq_end:], y[val_seq_end:]
    
    print(f"Data splits:")
    print(f"  Train: {len(X_train)} sequences")
    print(f"  Val:   {len(X_val)} sequences")
    print(f"  Test:  {len(X_test)} sequences")
    print()
    
    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    
    # Model
    model = MultiStepLSTM(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_steps=output_steps
    ).to(device)
    
    print(f"Model architecture:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # FIX #2: Use Huber Loss (robust to outliers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    loss_fn = nn.HuberLoss(delta=1.0)  # Huber instead of MSE
    print("Using Huber Loss (robust to outliers)")
    print()
    
    early_stopping = EarlyStopping(patience=patience)
    
    # Training loop
    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                
                total_loss += loss.item() * len(xb)
                predictions.append(pred.cpu().numpy())
                targets.append(yb.cpu().numpy())
        
        avg_loss = total_loss / len(loader.dataset)
        predictions = np.vstack(predictions)
        targets = np.vstack(targets)
        
        return avg_loss, predictions, targets
    
    print("Training...")
    print()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * len(xb)
        
        train_loss = running_loss / len(train_ds)
        val_loss, _, _ = evaluate(val_dl)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        early_stopping(val_loss, model)
        if early_stopping.should_stop:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best validation loss: {early_stopping.best_loss:.6f}")
            model.load_state_dict(early_stopping.best_model)
            break
    
    print()
    print("=" * 70)
    print("EVALUATION (with inverse transform)")
    print("=" * 70)
    
    # Evaluate on all sets
    train_loss, train_pred_scaled, train_true_scaled = evaluate(train_dl)
    val_loss, val_pred_scaled, val_true_scaled = evaluate(val_dl)
    test_loss, test_pred_scaled, test_true_scaled = evaluate(test_dl)
    
    # FIX #4: Inverse transform predictions back to original scale
    print("\nInverse transforming predictions to original scale...")
    
    # Reshape for inverse transform
    train_pred = target_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).reshape(train_pred_scaled.shape)
    train_true = target_scaler.inverse_transform(train_true_scaled.reshape(-1, 1)).reshape(train_true_scaled.shape)
    
    val_pred = target_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).reshape(val_pred_scaled.shape)
    val_true = target_scaler.inverse_transform(val_true_scaled.reshape(-1, 1)).reshape(val_true_scaled.shape)
    
    test_pred = target_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).reshape(test_pred_scaled.shape)
    test_true = target_scaler.inverse_transform(test_true_scaled.reshape(-1, 1)).reshape(test_true_scaled.shape)
    
    # Compute metrics on original scale
    train_metrics = compute_metrics(train_true, train_pred)
    val_metrics = compute_metrics(val_true, val_pred)
    test_metrics = compute_metrics(test_true, test_pred)
    
    print(f"\nTrain Metrics (original scale):")
    for k, v in train_metrics.items():
        if k != 'step_mse':
            print(f"  {k:25s}: {v:.6f}")
    
    print(f"\nValidation Metrics (original scale):")
    for k, v in val_metrics.items():
        if k != 'step_mse':
            print(f"  {k:25s}: {v:.6f}")
    
    print(f"\nTest Metrics (original scale):")
    for k, v in test_metrics.items():
        if k != 'step_mse':
            print(f"  {k:25s}: {v:.6f}")
    
    # Per-step analysis
    print(f"\nPer-step Test MSE:")
    for i, mse in enumerate(test_metrics['step_mse'], 1):
        print(f"  Day {i}: {mse:.6f}")
    
    # Baseline comparison
    baseline_mse = np.mean(test_true ** 2)
    print(f"\nBaseline (predict zero) MSE: {baseline_mse:.6f}")
    print(f"Model improvement: {(1 - test_metrics['mse'] / baseline_mse) * 100:.2f}%")
    
    # Save model
    model_path = os.path.join(models_dir, "sp500_lstm_improved.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': len(feature_cols),
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'output_steps': output_steps,
        },
        'feature_columns': feature_cols,
        'feature_scaler_mean': feature_scaler.mean_.tolist(),
        'feature_scaler_scale': feature_scaler.scale_.tolist(),
        'target_scaler_mean': float(target_scaler.mean_[0]),
        'target_scaler_scale': float(target_scaler.scale_[0]),
        'seq_len': seq_len,
        'start_year': start_year,
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        }
    }, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save predictions
    predictions_path = os.path.join(results_dir, "predictions_improved.npz")
    np.savez(
        predictions_path,
        test_pred=test_pred,
        test_true=test_true,
    )
    print(f"✓ Predictions saved to {predictions_path}")
    
    print()
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

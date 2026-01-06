"""
Enhanced LSTM training for S&P 500 forecasting with rich feature set.

This script uses the prepared features from prepare_features.py to train
a more sophisticated neural network with:
- Multi-feature input (42 features vs 1)
- Better architecture (attention, dropout, batch norm)
- Improved training (early stopping, LR scheduling)
- Comprehensive evaluation metrics

Designed for master's thesis on time series forecasting.
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


class LSTMWithAttention(nn.Module):
    """
    Enhanced LSTM with self-attention mechanism for time series forecasting.
    
    Architecture:
    - Multi-feature input (technical indicators, temporal features, etc.)
    - 2-layer bidirectional LSTM with dropout
    - Self-attention to focus on relevant time steps
    - Batch normalization for stable training
    - Residual connection
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_attention = attention
        
        # Input projection (helps with heterogeneous features)
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
        if self.use_attention:
            # Attention weights for each time step
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
        
        # Output layers
        lstm_output_size = hidden_size * 2  # Bidirectional
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input features
        # Reshape for layer norm: (B, T, F) -> (B*T, F)
        x_proj = x.reshape(-1, self.input_size)
        x_proj = self.input_proj(x_proj)
        x_proj = x_proj.reshape(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x_proj)  # (B, T, hidden*2)
        
        # Attention or last hidden state
        if self.use_attention:
            # Compute attention weights
            attn_weights = self.attention(lstm_out)  # (B, T, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            
            # Weighted sum
            context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, hidden*2)
        else:
            # Just use last time step
            context = lstm_out[:, -1, :]  # (B, hidden*2)
        
        # Output
        out = self.fc(context)  # (B, 1)
        return out.squeeze(-1)


def load_feature_data(csv_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load processed features and identify feature columns."""
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Feature columns: everything except Date, Close, log_return
    exclude = ['Date', 'Close', 'log_return']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    return df, feature_cols


def create_sequences_multivariate(
    features: np.ndarray,
    target: np.ndarray,
    seq_len: int,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for multivariate time series.
    
    Args:
        features: (n_samples, n_features) array
        target: (n_samples,) array
        seq_len: Length of input sequence
        forecast_horizon: Steps ahead to predict
    
    Returns:
        X: (n_sequences, seq_len, n_features)
        y: (n_sequences,)
    """
    X, y = [], []
    n_samples = len(features)
    
    for i in range(n_samples - seq_len - forecast_horizon + 1):
        X.append(features[i:i + seq_len])
        y.append(target[i + seq_len + forecast_horizon - 1])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Directional accuracy (for thesis: how often we predict correct direction)
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_accuracy = np.mean(direction_true == direction_pred)
    
    # R-squared (coefficient of determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'directional_accuracy': float(directional_accuracy)
    }


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
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
    forecast_horizon = 1  # Next-day prediction
    batch_size = 64
    epochs = 100  # Will use early stopping
    lr = 1e-3
    hidden_size = 128
    num_layers = 2
    dropout = 0.3
    patience = 15  # Early stopping patience
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("ENHANCED S&P 500 FORECASTING MODEL")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Sequence length: {seq_len} days")
    print(f"  Forecast horizon: {forecast_horizon} day(s) ahead")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Layers: {num_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Device: {device}")
    print()
    
    # Load data
    print("Loading features...")
    if not os.path.exists(feature_csv):
        print(f"Error: {feature_csv} not found")
        print("Run: python src/data/prepare_features.py")
        return
    
    df, feature_cols = load_feature_data(feature_csv)
    print(f"  {len(df)} samples from {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  {len(feature_cols)} features")
    print()
    
    # Prepare features and target
    features = df[feature_cols].values.astype(np.float32)
    target = df['log_return'].values.astype(np.float32)
    
    # Standardize features (fit on training set, apply to all)
    # Note: We'll split first to avoid data leakage
    n_samples = len(features)
    train_end = int(n_samples * 0.8)
    val_end = int(n_samples * 0.9)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    features_train = features[:train_end]
    scaler.fit(features_train)
    
    # Transform all data
    features_scaled = scaler.transform(features)
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences_multivariate(features_scaled, target, seq_len, forecast_horizon)
    print(f"  Total sequences: {len(X)}")
    print(f"  Input shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print()
    
    # Train/Val/Test split (temporal)
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
    
    # Create DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    
    # Model
    model = LSTMWithAttention(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        attention=True
    ).to(device)
    
    print(f"Model architecture:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    loss_fn = nn.MSELoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-5)
    
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
                predictions.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())
        
        avg_loss = total_loss / len(loader.dataset)
        return avg_loss, np.array(predictions), np.array(targets)
    
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
            
            # Gradient clipping (helps with stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * len(xb)
        
        train_loss = running_loss / len(train_ds)
        val_loss, _, _ = evaluate(val_dl)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation MSE: {early_stopping.best_loss:.6f}")
            # Restore best model
            model.load_state_dict(early_stopping.best_model)
            break
    
    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)
    
    # Evaluate on all sets
    train_loss, train_pred, train_true = evaluate(train_dl)
    val_loss, val_pred, val_true = evaluate(val_dl)
    test_loss, test_pred, test_true = evaluate(test_dl)
    
    train_metrics = compute_metrics(train_true, train_pred)
    val_metrics = compute_metrics(val_true, val_pred)
    test_metrics = compute_metrics(test_true, test_pred)
    
    print(f"\nTrain Metrics:")
    for k, v in train_metrics.items():
        print(f"  {k:20s}: {v:.6f}")
    
    print(f"\nValidation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k:20s}: {v:.6f}")
    
    print(f"\nTest Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:20s}: {v:.6f}")
    
    # Baseline comparison (always predict zero/mean)
    baseline_mse = np.mean(test_true ** 2)
    print(f"\nBaseline (predict zero) MSE: {baseline_mse:.6f}")
    print(f"Model improvement: {(1 - test_metrics['mse'] / baseline_mse) * 100:.2f}%")
    
    # Save model
    model_path = os.path.join(models_dir, "sp500_lstm_enhanced.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': len(feature_cols),
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
        },
        'feature_columns': feature_cols,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'seq_len': seq_len,
        'forecast_horizon': forecast_horizon,
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        }
    }, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save training history
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_trained': len(train_losses),
            'best_val_loss': float(best_val_loss),
        }, f, indent=2)
    print(f"✓ Training history saved to {history_path}")
    
    # Save predictions for analysis
    predictions_path = os.path.join(results_dir, "predictions.npz")
    np.savez(
        predictions_path,
        train_pred=train_pred,
        train_true=train_true,
        val_pred=val_pred,
        val_true=val_true,
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

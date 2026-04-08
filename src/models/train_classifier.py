"""
Bidirectional LSTM Classifier for S&P 500 Directional Forecasting.
Thesis Topic: Neural networks and their applications for time series forecasting.

This script:
1. Loads the processed stationary dataset.
2. Creates chronological sequences (lookback window).
3. Builds a Bidirectional LSTM with an Attention mechanism.
4. Trains using BCEWithLogitsLoss (Binary Classification) with Class Weights.
5. Auto-detects and rigorously tests hardware, with seamless CPU fallback.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# --- 1. HARDWARE AUTO-DETECTION ---
def get_device():
    """Safely auto-detects hardware by actually testing execution."""
    if torch.cuda.is_available():
        try:
            # Layer 1 Defense: Test basic kernel execution
            _test = torch.tensor([1.0]).to("cuda") * 2.0
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
        except RuntimeError as e:
            print(f"\n[!] CUDA detected but basic execution failed. Falling back to CPU.\n    Reason: {e}")
            device = torch.device("cpu")
            device_name = "Standard CPU (CUDA Fallback)"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon (MPS)"
    else:
        device = torch.device("cpu")
        device_name = "Standard CPU"
        
    print(f"Hardware Auto-Detection: Using {device_name} ({device})")
    return device

# --- 2. MODEL ARCHITECTURE ---
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, lstm_outputs):
        attn_weights = self.attention(lstm_outputs) 
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_outputs, dim=1) 
        return context, attn_weights

class SP500Classifier(nn.Module):
    # Reduced hidden_dim to 32 to prevent overfitting
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.3):
        super(SP500Classifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = Attention(hidden_dim * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        out = self.fc(context)
        return out.squeeze()

# --- 3. DATA PREPARATION ---
def create_sequences(features, targets, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X), np.array(y)

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_csv = os.path.join(root_dir, "data", "processed", "sp500_processed.csv")
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("="*60)
    print("TRAINING BINARY LSTM CLASSIFIER")
    print("="*60)

    # Auto-detect device
    device = get_device()

    # 1. Load Data
    if not os.path.exists(processed_csv):
        raise FileNotFoundError("Processed data not found. Run make_dataset.py first.")
    
    df = pd.read_csv(processed_csv, parse_dates=['Date'])
    
    exclude_cols = ['Date', 'Close', 'log_return', 'target']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    features = df[feature_cols].values
    targets = df['target'].values
    
    print(f"Loaded {len(df)} days of data. Using {len(feature_cols)} features.")

    # 2. Chronological Split (80% / 10% / 10%)
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    # 3. Fit Scaler ONLY on Training Data
    scaler = StandardScaler()
    features[:train_end] = scaler.fit_transform(features[:train_end])
    features[train_end:val_end] = scaler.transform(features[train_end:val_end])
    features[val_end:] = scaler.transform(features[val_end:])

    # 4. Create Sequences
    seq_len = 10 
    X, y = create_sequences(features, targets, seq_len)
    
    X_train, y_train = X[:train_end-seq_len], y[:train_end-seq_len]
    X_val, y_val = X[train_end-seq_len:val_end-seq_len], y[train_end-seq_len:val_end-seq_len]

    # Convert to PyTorch Tensors
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 5. Initialize Model (Layer 2 Defense: Try/Catch the LSTM initialization)
    try:
        model = SP500Classifier(input_dim=len(feature_cols)).to(device)
        # Force a dummy pass to catch CuDNN-specific kernel errors
        _dummy = model(torch.zeros(1, seq_len, len(feature_cols)).to(device))
    except RuntimeError as e:
        if device.type == 'cuda':
            print(f"\n[!] CUDA execution failed during model initialization (Likely unsupported GPU architecture).")
            print(f"    Falling back to CPU...\n")
            device = torch.device("cpu")
            model = SP500Classifier(input_dim=len(feature_cols)).to(device)
        else:
            raise e

    # --- THE FIX: Class Weights to stop 100% Recall on "Up" days ---
    y_train_tensor = torch.FloatTensor(y_train)
    num_pos = y_train_tensor.sum()
    num_neg = len(y_train_tensor) - num_pos
    
    # Calculate the weight for the positive class
    pos_weight = (num_neg / num_pos).to(device)
    print(f"\nApplying Class Weight: {pos_weight:.4f} to penalize 'always guessing Up'")

    # Pass the weight into the loss function
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
    
    # Optimizer (Removed weight_decay to preserve LSTM memory)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Training Loop with Early Stopping
    epochs = 100
    patience = 15
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            predictions = model(xb)
            loss = loss_fn(predictions, yb)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                predictions = model(xb)
                loss = loss_fn(predictions, yb)
                val_loss += loss.item()
                
                probs = torch.sigmoid(predictions)
                preds = (probs > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_,
                'feature_cols': feature_cols,
                'seq_len': seq_len
            }, os.path.join(models_dir, "sp500_lstm_classifier.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

    print("\n✓ Training Complete. Best model saved to models/sp500_lstm_classifier.pt")
    print("="*60)

if __name__ == "__main__":
    main()
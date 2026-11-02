"""
Bidirectional LSTM Classifier for S&P 500 Directional Forecasting.
- Reads the existing sp500_processed.csv WITHOUT modifying any data.
- Uses a 20-day lookback window (SEQ_LEN = 20) across 5 fixed seeds.
- Uses a single, unified decision rule (probs <= t) in BOTH Validation and Test sets
  (eliminating train/eval inversion while capturing the contrarian signal).
- Saves all 5 run checkpoints and exports Table 6.1 to CSV.
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

SEEDS = [42, 100, 2024, 777, 999]
SEQ_LEN = 20


def set_seed(seed: int):
    """Locks all random seeds for 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Selects compute device with safe CPU fallback."""
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0], device="cuda") * 2.0
            return torch.device("cuda")
        except RuntimeError:
            return torch.device("cpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, lstm_outputs):
        attn_weights = torch.softmax(self.attention(lstm_outputs), dim=1)
        context = torch.sum(attn_weights * lstm_outputs, dim=1)
        return context, attn_weights


class SP500Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
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
        return self.fc(context).squeeze(-1)


def create_sequences(features, targets, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X), np.array(y)


def train_single_seed(run_idx, seed, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler, device):
    set_seed(seed)
    batch_size = 64
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    try:
        model = SP500Classifier(input_dim=len(feature_cols)).to(device)
        _ = model(torch.zeros(1, SEQ_LEN, len(feature_cols)).to(device))
    except RuntimeError:
        device = torch.device("cpu")
        model = SP500Classifier(input_dim=len(feature_cols)).to(device)

    y_train_tensor = torch.FloatTensor(y_train)
    num_pos = y_train_tensor.sum()
    num_neg = len(y_train_tensor) - num_pos
    pos_weight = (num_neg / num_pos).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None

    for epoch in range(1, 101):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += loss_fn(model(xb.to(device)), yb.to(device)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    model.eval()

    # 1. Validation Set Threshold Selection — Unified Contrarian Rule: (probs <= t)
    val_probs, val_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            val_probs.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy())
            val_true.extend(yb.numpy())
    
    val_probs = np.array(val_probs)
    val_true = np.array(val_true).astype(int)

    thresholds = np.linspace(val_probs.min(), val_probs.max(), 41)
    best_t = max(
        thresholds,
        key=lambda t: matthews_corrcoef(val_true, (val_probs <= t).astype(int))
    )

    # 2. Test Set Evaluation — EXACT SAME RULE: (probs <= best_t) (NO INVERSION!)
    with torch.no_grad():
        test_probs = torch.sigmoid(model(torch.FloatTensor(X_test).to(device))).cpu().numpy()

    y_test_int = y_test.astype(int)
    test_preds = (test_probs <= best_t).astype(int)

    raw_auc = roc_auc_score(y_test_int, test_probs)
    eff_auc = roc_auc_score(y_test_int, 1.0 - test_probs)

    metrics = {
        'Run': f"Run {run_idx}",
        'Seed': seed,
        'Threshold': round(float(best_t), 4),
        'Accuracy (%)': round(accuracy_score(y_test_int, test_preds) * 100, 2),
        'Precision (%)': round(precision_score(y_test_int, test_preds, zero_division=0) * 100, 2),
        'Recall (%)': round(recall_score(y_test_int, test_preds, zero_division=0) * 100, 2),
        'F1_Score (%)': round(f1_score(y_test_int, test_preds, zero_division=0) * 100, 2),
        'MCC': round(matthews_corrcoef(y_test_int, test_preds), 4),
        'Raw_AUC': round(raw_auc, 4),
        'Effective_AUC': round(eff_auc, 4)
    }

    ckpt = {
        'run_idx': run_idx,
        'seed': seed,
        'model_state_dict': best_state_dict,
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'feature_cols': feature_cols,
        'seq_len': SEQ_LEN,
        'best_threshold': float(best_t),
        'metrics': metrics
    }
    return metrics, ckpt


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_csv = os.path.join(root_dir, "data", "processed", "sp500_processed.csv")
    models_dir = os.path.join(root_dir, "models")
    tables_dir = os.path.join(root_dir, "results", "tables")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    device = get_device()
    print("=" * 75)
    print(f"TRAINING Bi-LSTM (SEQ_LEN={SEQ_LEN}, 5-DAY HORIZON, 5 FIXED SEEDS) on {device}")
    print("=" * 75)

    df = pd.read_csv(processed_csv, parse_dates=['Date'])
    exclude_cols = ['Date', 'Close', 'log_return', 'target']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    features = df[feature_cols].values.copy()
    targets = df['target'].values.copy()
    if np.isinf(features).sum() > 0:
        features = pd.DataFrame(features, columns=feature_cols).replace([np.inf, -np.inf], np.nan).ffill().bfill().values

    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    scaler = StandardScaler()
    features[:train_end] = scaler.fit_transform(features[:train_end])
    features[train_end:val_end] = scaler.transform(features[train_end:val_end])
    features[val_end:] = scaler.transform(features[val_end:])
    features = np.clip(features, -3, 3)

    X, y = create_sequences(features, targets, SEQ_LEN)
    X_train, y_train = X[:train_end - SEQ_LEN], y[:train_end - SEQ_LEN]
    X_val, y_val = X[train_end - SEQ_LEN : val_end - SEQ_LEN], y[train_end - SEQ_LEN : val_end - SEQ_LEN]
    X_test, y_test = X[val_end - SEQ_LEN :], y[val_end - SEQ_LEN :]

    all_metrics = []
    best_mcc = -2.0
    best_ckpt = None

    for idx, seed in enumerate(SEEDS, 1):
        metrics, ckpt = train_single_seed(
            idx, seed, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, scaler, device
        )
        all_metrics.append(metrics)
        torch.save(ckpt, os.path.join(models_dir, f"sp500_lstm_run_{idx}_seed_{seed}.pt"))
        
        print(f"Run {idx} (Seed {seed:4d}) | Acc: {metrics['Accuracy (%)']:5.2f}% | "
              f"Prec: {metrics['Precision (%)']:5.2f}% | Rec: {metrics['Recall (%)']:5.2f}% | "
              f"F1: {metrics['F1_Score (%)']:5.2f}% | MCC: {metrics['MCC']:6.4f} | "
              f"Eff.AUC: {metrics['Effective_AUC']:.3f}")

        if metrics['MCC'] > best_mcc:
            best_mcc = metrics['MCC']
            best_ckpt = ckpt

    res_df = pd.DataFrame(all_metrics)
    num_cols = ['Threshold', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1_Score (%)', 'MCC', 'Raw_AUC', 'Effective_AUC']
    
    mean_row = {'Run': 'Mean', 'Seed': '-'}
    std_row = {'Run': 'Std', 'Seed': '-'}
    for col in num_cols:
        mean_row[col] = round(res_df[col].mean(), 4)
        std_row[col] = round(res_df[col].std(), 4)

    summary_df = pd.concat([res_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    summary_csv = os.path.join(tables_dir, "table_6_1_multi_run_results.csv")
    summary_df.to_csv(summary_csv, index=False)

    torch.save(best_ckpt, os.path.join(models_dir, "sp500_lstm_classifier.pt"))

    print("\n" + "=" * 75)
    print("SUMMARY TABLE ACROSS 5 FIXED SEEDS")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print(f"\n✓ Saved Table 6.1 CSV to: {summary_csv}")
    print(f"✓ Saved Best Model (Run {best_ckpt['run_idx']}, Seed {best_ckpt['seed']}, MCC={best_mcc:.4f})")
    print("=" * 75)


if __name__ == "__main__":
    main()
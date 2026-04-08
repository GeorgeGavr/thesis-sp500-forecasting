"""
Evaluation Script for S&P 500 Directional Forecasting
Generates thesis-ready metrics and an academic suite of plots.
Auto-detects hardware with seamless CPU fallback.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score)
import warnings
warnings.filterwarnings('ignore')

# --- 1. HARDWARE AUTO-DETECTION ---
def get_device():
    if torch.cuda.is_available():
        try:
            _test = torch.tensor([1.0]).to("cuda") * 2.0
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
        except RuntimeError:
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
        attn_weights = torch.softmax(self.attention(lstm_outputs), dim=1)
        context = torch.sum(attn_weights * lstm_outputs, dim=1)
        return context, attn_weights

class SP500Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, dropout=0.3):
        super(SP500Classifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
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
        return self.fc(context).squeeze()

def create_sequences(features, targets, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X), np.array(y)

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_csv = os.path.join(root_dir, "data", "processed", "sp500_processed.csv")
    model_path = os.path.join(root_dir, "models", "sp500_lstm_classifier.pt")
    plots_dir = os.path.join(root_dir, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("="*60)
    print("EVALUATING MODEL ON UNSEEN TEST DATA")
    print("="*60)

    # 1. Auto-detect Hardware
    device = get_device()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model checkpoint not found. Run train_classifier.py first.")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    feature_cols = checkpoint['feature_cols']
    seq_len = 10 
    
    # 2. Initialize Model
    try:
        model = SP500Classifier(input_dim=len(feature_cols)).to(device)
        _dummy = model(torch.zeros(1, seq_len, len(feature_cols)).to(device))
    except RuntimeError as e:
        if device.type == 'cuda':
            print(f"\n[!] CUDA execution failed during model load. Falling back to CPU...\n")
            device = torch.device("cpu")
            model = SP500Classifier(input_dim=len(feature_cols)).to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        else:
            raise e

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Load and Split Data 
    df = pd.read_csv(processed_csv, parse_dates=['Date'])
    features = df[feature_cols].values
    targets = df['target'].values
    
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    test_features = features[val_end - seq_len:] 
    test_targets = targets[val_end - seq_len:]
    test_features = (test_features - checkpoint['scaler_mean']) / checkpoint['scaler_scale']
    
    X_test, y_test = create_sequences(test_features, test_targets, seq_len)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    # 4. Generate Predictions & Dynamic Thresholding
    print("Generating predictions...")
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    y_test = y_test.astype(int)

    # --- THE MEDIAN THRESHOLD ---
    # Force the threshold to the exact middle of the model's predictions
    # This prevents the model from defaulting to the majority class.
    best_threshold = float(np.median(probs))
    print(f"\nForcing Threshold to Median Probability: {best_threshold:.4f}")
    
    # Apply the threshold
    preds = (probs > best_threshold).astype(int)
    
    # 5. Calculate Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    
    print("\n--- TEST SET METRICS ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    
    # ==========================================
    # 6. THESIS PLOT GENERATION
    # ==========================================
    print("\nGenerating academic plots...")
    sns.set_theme(style="whitegrid")

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Down (0)', 'Predicted Up (1)'],
                yticklabels=['Actual Down (0)', 'Actual Up (1)'])
    plt.title(f'Test Set Confusion Matrix (Threshold: {best_threshold:.2f})')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "1_confusion_matrix.png"), dpi=300)
    plt.close()

    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "2_roc_curve.png"), dpi=300)
    plt.close()

    # Plot 3: Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
    ap_score = average_precision_score(y_test, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='green', lw=2, label=f'PR curve (AP = {ap_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "3_precision_recall_curve.png"), dpi=300)
    plt.close()

    # Plot 4: Probability Distribution Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(probs[y_test == 0], color='red', alpha=0.5, label='Actual Down (0)', stat='density', bins=30)
    sns.histplot(probs[y_test == 1], color='green', alpha=0.5, label='Actual Up (1)', stat='density', bins=30)
    plt.axvline(best_threshold, color='black', linestyle='dashed', linewidth=2, label=f'Threshold ({best_threshold:.2f})')
    plt.title('Model Prediction Probability Distribution')
    plt.xlabel('Predicted Probability of "Up"')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "4_probability_distribution.png"), dpi=300)
    plt.close()

    print(f"✓ 4 High-Resolution plots saved to: {plots_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
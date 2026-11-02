"""
Comprehensive Evaluation & EDA Script for S&P 500 Directional Forecasting.
Reads existing sp500_processed.csv WITHOUT modifying data, and generates:
1. Feature Correlation Matrix CSV & Heatmap (0_correlation_heatmap.png)
2. Autocorrelation (ACF) & Partial Autocorrelation (PACF) Plots + Ljung-Box CSV (0_acf_pacf_analysis.png)
3. All 4 evaluation plots (Confusion Matrix, ROC, PR, Prob. Distribution) for ALL 5 runs + Best Model.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score,
                             matthews_corrcoef)
import warnings
warnings.filterwarnings('ignore')

SEEDS = [42, 100, 2024, 777, 999]


def get_device():
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
        return self.fc(context).squeeze(-1)


def create_sequences(features, targets, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X), np.array(y)


def generate_eda_plots_and_tables(df: pd.DataFrame, plots_dir: str, tables_dir: str):
    """Generates Correlation Heatmap + CSVs, and ACF/PACF Plots + Ljung-Box CSV."""
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Save Dataset Sample (Table 5.2)
    df.head(10).round(4).to_csv(os.path.join(tables_dir, "table_5_2_dataset_sample.csv"), index=False)

    # 1. Correlation Matrix & Heatmap
    exclude_cols = ['Date', 'Close', 'Volume']
    corr_cols = [c for c in df.columns if c not in exclude_cols]
    corr_matrix = df[corr_cols].corr(method='pearson')
    
    corr_matrix.round(4).to_csv(os.path.join(tables_dir, "table_5_3_feature_correlation_matrix.csv"))
    target_corr = corr_matrix[['target']].sort_values(by='target', ascending=False).round(4)
    target_corr.to_csv(os.path.join(tables_dir, "table_5_4_target_correlations.csv"))

    plt.figure(figsize=(14, 11))
    sns.set_theme(style="white")
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='coolwarm',
        vmin=-1.0, vmax=1.0, center=0.0,
        annot=True, fmt='.2f',
        annot_kws={'size': 7},
        linewidths=0.5,
        cbar_kws={'label': 'Pearson Correlation Coefficient (r)'}
    )
    plt.title('Feature & Target Correlation Heatmap (S&P 500)', fontsize=13, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    heatmap_path = os.path.join(plots_dir, "0_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"✓ Saved Correlation Heatmap: {heatmap_path}")

    # 2. Autocorrelation (ACF) & Partial Autocorrelation (PACF) + Ljung-Box Test
    returns = df['log_return'].dropna()
    sq_returns = returns ** 2

    lags_to_test = [1, 5, 10, 15, 20]
    lb_ret = acorr_ljungbox(returns, lags=lags_to_test, return_df=True)
    lb_sq = acorr_ljungbox(sq_returns, lags=lags_to_test, return_df=True)

    acf_table = pd.DataFrame({
        'Lag': lags_to_test,
        'Log_Return_LB_Q': lb_ret['lb_stat'].round(4).values,
        'Log_Return_p_val': lb_ret['lb_pvalue'].round(4).values,
        'Squared_Return_LB_Q': lb_sq['lb_stat'].round(4).values,
        'Squared_Return_p_val': lb_sq['lb_pvalue'].round(6).values
    })
    acf_csv_path = os.path.join(tables_dir, "table_5_5_autocorrelation_ljung_box.csv")
    acf_table.to_csv(acf_csv_path, index=False)

    print("\n--- AUTOCORRELATION LJUNG-BOX TABLE (SAVED TO CSV) ---")
    print(acf_table.to_string(index=False))

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_acf(returns, lags=20, ax=axes[0, 0], zero=False, title='ACF of Log Returns (Lags 1–20)')
    plot_pacf(returns, lags=20, ax=axes[0, 1], zero=False, method='ywm', title='PACF of Log Returns (Lags 1–20)')
    plot_acf(sq_returns, lags=20, ax=axes[1, 0], zero=False, color='darkred', title='ACF of Squared Returns (Volatility Clustering)')
    plot_pacf(sq_returns, lags=20, ax=axes[1, 1], zero=False, method='ywm', color='darkred', title='PACF of Squared Returns (Lags 1–20)')
    
    for ax in axes.flat:
        ax.set_xlabel('Lag (Trading Days)')
        ax.set_ylabel('Correlation')

    plt.tight_layout()
    acf_plot_path = os.path.join(plots_dir, "0_acf_pacf_analysis.png")
    plt.savefig(acf_plot_path, dpi=300)
    plt.close()
    print(f"✓ Saved ACF/PACF Plot:       {acf_plot_path}")


def evaluate_run(ckpt_path: str, df: pd.DataFrame, out_dir: str, device):
    """Evaluates a checkpoint using the unified rule (probs <= best_threshold) and saves all 4 plots."""
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    feature_cols = checkpoint['feature_cols']
    seq_len = checkpoint['seq_len']
    seed = checkpoint.get('seed', 'Best')
    run_idx = checkpoint.get('run_idx', 'Best')

    try:
        model = SP500Classifier(input_dim=len(feature_cols)).to(device)
        _ = model(torch.zeros(1, seq_len, len(feature_cols)).to(device))
    except RuntimeError:
        device = torch.device("cpu")
        model = SP500Classifier(input_dim=len(feature_cols)).to(device)

    model.load_state_dict({k: v.to(device) for k, v in checkpoint['model_state_dict'].items()})
    model.eval()

    features = df[feature_cols].values.copy()
    targets = df['target'].values.copy()
    if np.isinf(features).sum() > 0:
        features = pd.DataFrame(features, columns=feature_cols).replace([np.inf, -np.inf], np.nan).ffill().bfill().values

    val_end = int(len(df) * 0.9)
    test_features = features[val_end - seq_len:]
    test_targets = targets[val_end - seq_len:]
    test_features = (test_features - checkpoint['scaler_mean']) / checkpoint['scaler_scale']
    test_features = np.clip(test_features, -3, 3)

    X_test, y_test = create_sequences(test_features, test_targets, seq_len)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_tensor)).cpu().numpy()

    y_test = y_test.astype(int)
    best_threshold = checkpoint.get('best_threshold', float(np.median(probs)))

    # Unified Contrarian Decision Rule (probs <= best_threshold) — matches train_classifier.py!
    preds = (probs <= best_threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, 1.0 - probs)
    ap_score = average_precision_score(y_test, 1.0 - probs)

    sns.set_theme(style="whitegrid")

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Down (0)', 'Predicted Up (1)'],
                yticklabels=['Actual Down (0)', 'Actual Up (1)'])
    plt.title(f'Test Set Confusion Matrix — Run {run_idx} (Seed {seed}, Threshold: {best_threshold:.2f})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_confusion_matrix.png"), dpi=300)
    plt.close()

    # Plot 2: ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f} | Eff. AUC = {1-roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) — Run {run_idx} (Seed {seed})')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_roc_curve.png"), dpi=300)
    plt.close()

    # Plot 3: Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='green', lw=2, label=f'PR curve (AP = {ap_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve — Run {run_idx} (Seed {seed})')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_precision_recall_curve.png"), dpi=300)
    plt.close()

    # Plot 4: Probability Distribution Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(probs[y_test == 0], color='red', alpha=0.5, label='Actual Down (0)', stat='density', bins=30)
    sns.histplot(probs[y_test == 1], color='green', alpha=0.5, label='Actual Up (1)', stat='density', bins=30)
    plt.axvline(best_threshold, color='black', linestyle='dashed', linewidth=2, label=f'Threshold ({best_threshold:.2f})')
    plt.title(f'Model Prediction Probability Distribution — Run {run_idx} (Seed {seed})')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "4_probability_distribution.png"), dpi=300)
    plt.close()


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_csv = os.path.join(root_dir, "data", "processed", "sp500_processed.csv")
    models_dir = os.path.join(root_dir, "models")
    plots_dir = os.path.join(root_dir, "results", "plots")
    tables_dir = os.path.join(root_dir, "results", "tables")

    print("=" * 75)
    print("RUNNING FULL EVALUATION: HEATMAP, ACF/PACF, TABLES & ALL 5 RUN PLOTS")
    print("=" * 75)

    device = get_device()
    df = pd.read_csv(processed_csv, parse_dates=['Date'])

    # 1. Generate Correlation Heatmap, ACF/PACF Plots, and CSV Tables
    generate_eda_plots_and_tables(df, plots_dir, tables_dir)

    # 2. Generate all 4 plots for each of the 5 runs
    print("\nGenerating 4 evaluation plots for each run...")
    for idx, seed in enumerate(SEEDS, 1):
        ckpt_file = os.path.join(models_dir, f"sp500_lstm_run_{idx}_seed_{seed}.pt")
        if os.path.exists(ckpt_file):
            run_dir = os.path.join(plots_dir, f"run_{idx}_seed_{seed}")
            evaluate_run(ckpt_file, df, run_dir, device)
            print(f"  ✓ Saved 4 plots for Run {idx} (Seed {seed}) -> {run_dir}")

    # 3. Generate all 4 plots for the Best Model in the root plots directory
    best_ckpt = os.path.join(models_dir, "sp500_lstm_classifier.pt")
    if os.path.exists(best_ckpt):
        evaluate_run(best_ckpt, df, plots_dir, device)
        print(f"  ✓ Saved Best Model plots -> {plots_dir}")

    print("\n✓ Everything completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
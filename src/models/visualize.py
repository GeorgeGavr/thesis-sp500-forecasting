import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from train import LSTMRegressor, load_series, compute_log_returns, make_sequences


def main():
    # Resolve paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_csv = os.path.join(root_dir, "data", "sp500_close.csv")
    model_path = os.path.join(root_dir, "models", "sp500_lstm.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    
    # Load model metadata
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    seq_len = checkpoint.get("seq_len", 60)
    forecast_horizon = checkpoint.get("forecast_horizon", 1)
    
    print(f"Model Configuration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Forecast horizon: {forecast_horizon} days")
    print()
    
    # Load data
    prices = load_series(data_csv).values
    log_returns = compute_log_returns(prices)
    
    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(log_returns.reshape(-1, 1)).astype(np.float32).flatten()
    
    # Create sequences
    X, y = make_sequences(scaled, seq_len, forecast_horizon)
    n = len(X)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    # Load model
    model = LSTMRegressor().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # Make predictions on test set
    X_test_tensor = torch.from_numpy(X_test).unsqueeze(-1).to(device)  # (N, T, 1)
    with torch.no_grad():
        predictions = model(X_test_tensor).squeeze(-1).cpu().numpy()
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Predictions vs Actual (normalized log returns)
    ax1 = axes[0]
    test_indices = np.arange(len(y_test))
    ax1.plot(test_indices, y_test, label="Actual", alpha=0.7, linewidth=1)
    ax1.plot(test_indices, predictions, label="Predicted", alpha=0.7, linewidth=1)
    ax1.set_title(f"Test Set: Predicted vs Actual Log Returns ({forecast_horizon}-day ahead, Normalized)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Test Sample Index")
    ax1.set_ylabel("Normalized Log Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction errors
    ax2 = axes[1]
    errors = predictions - y_test
    ax2.plot(test_indices, errors, color='red', alpha=0.6, linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.fill_between(test_indices, 0, errors, alpha=0.3, color='red')
    ax2.set_title("Prediction Errors (Predicted - Actual)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Test Sample Index")
    ax2.set_ylabel("Error")
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    mse = np.mean(errors ** 2)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(mse)
    
    stats_text = f"MSE: {mse:.6f}\nMAE: {mae:.6f}\nRMSE: {rmse:.6f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(root_dir, "models", "training_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("TEST SET PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Number of test samples: {len(y_test)}")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Correlation: {np.corrcoef(y_test, predictions)[0,1]:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()

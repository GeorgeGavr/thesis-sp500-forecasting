"""
Visualize results from the improved LSTM model.

Creates plots for:
1. Predictions vs actual (all 5 days)
2. Per-step performance
3. Directional accuracy analysis
4. Error distribution
5. Time series comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_predictions():
    """Load saved predictions."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pred_path = os.path.join(root_dir, "results", "predictions_improved.npz")
    
    data = np.load(pred_path)
    return data['test_pred'], data['test_true']


def plot_predictions_vs_actual(test_pred, test_true, save_dir):
    """Plot predictions vs actual for all 5 days."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('5-Day Ahead Predictions vs Actual Log Returns', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for day in range(5):
        ax = axes[day]
        
        # Scatter plot
        ax.scatter(test_true[:, day], test_pred[:, day], alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(test_true[:, day].min(), test_pred[:, day].min())
        max_val = max(test_true[:, day].max(), test_pred[:, day].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        
        # Labels
        ax.set_xlabel('Actual Log Return', fontsize=10)
        ax.set_ylabel('Predicted Log Return', fontsize=10)
        ax.set_title(f'Day {day + 1} Ahead', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Compute R²
        ss_res = np.sum((test_true[:, day] - test_pred[:, day]) ** 2)
        ss_tot = np.sum((test_true[:, day] - test_true[:, day].mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Add R² annotation
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 1_predictions_vs_actual.png")
    plt.close()


def plot_per_step_metrics(test_pred, test_true, save_dir):
    """Plot MSE and MAE per prediction step."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = np.arange(1, 6)
    
    # Compute metrics per step
    mse_per_step = np.mean((test_true - test_pred) ** 2, axis=0)
    mae_per_step = np.mean(np.abs(test_true - test_pred), axis=0)
    
    # MSE plot
    axes[0].plot(steps, mse_per_step, marker='o', linewidth=2, markersize=8, label='MSE')
    axes[0].set_xlabel('Prediction Day', fontsize=12)
    axes[0].set_ylabel('Mean Squared Error', fontsize=12)
    axes[0].set_title('MSE by Prediction Horizon', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(steps)
    
    # Add values on points
    for i, mse in enumerate(mse_per_step):
        axes[0].text(i+1, mse, f'{mse:.6f}', ha='center', va='bottom')
    
    # MAE plot
    axes[1].plot(steps, mae_per_step, marker='s', linewidth=2, markersize=8, color='orange', label='MAE')
    axes[1].set_xlabel('Prediction Day', fontsize=12)
    axes[1].set_ylabel('Mean Absolute Error', fontsize=12)
    axes[1].set_title('MAE by Prediction Horizon', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(steps)
    
    # Add values on points
    for i, mae in enumerate(mae_per_step):
        axes[1].text(i+1, mae, f'{mae:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '2_per_step_metrics.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 2_per_step_metrics.png")
    plt.close()


def plot_directional_accuracy(test_pred, test_true, save_dir):
    """Plot directional accuracy analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Per-step directional accuracy
    steps = np.arange(1, 6)
    dir_acc_per_step = []
    
    for day in range(5):
        dir_true = np.sign(test_true[:, day])
        dir_pred = np.sign(test_pred[:, day])
        accuracy = np.mean(dir_true == dir_pred)
        dir_acc_per_step.append(accuracy)
    
    # Bar plot
    bars = axes[0].bar(steps, dir_acc_per_step, color='steelblue', alpha=0.7)
    axes[0].axhline(y=0.5, color='red', linestyle='--', label='Random (50%)')
    axes[0].set_xlabel('Prediction Day', fontsize=12)
    axes[0].set_ylabel('Directional Accuracy', fontsize=12)
    axes[0].set_title('Directional Accuracy by Horizon', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticks(steps)
    
    # Add percentage labels on bars
    for bar, acc in zip(bars, dir_acc_per_step):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc*100:.1f}%', ha='center', va='bottom')
    
    # Confusion matrix for Day 1
    dir_true_day1 = np.sign(test_true[:, 0])
    dir_pred_day1 = np.sign(test_pred[:, 0])
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(dir_true_day1, dir_pred_day1, labels=[-1, 1])
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'], 
                yticklabels=['Down', 'Up'],
                ax=axes[1], cbar_kws={'label': 'Count'})
    axes[1].set_xlabel('Predicted Direction', fontsize=12)
    axes[1].set_ylabel('Actual Direction', fontsize=12)
    axes[1].set_title('Confusion Matrix (Day 1 Predictions)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '3_directional_accuracy.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 3_directional_accuracy.png")
    plt.close()


def plot_error_distribution(test_pred, test_true, save_dir):
    """Plot error distribution."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Prediction Error Distribution by Horizon', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for day in range(5):
        ax = axes[day]
        
        errors = test_pred[:, day] - test_true[:, day]
        
        # Histogram
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax.axvline(x=errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.6f}')
        
        ax.set_xlabel('Prediction Error (Pred - Actual)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Day {day + 1} Ahead', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add stats
        stats_text = f'Std: {errors.std():.6f}\nSkew: {pd.Series(errors).skew():.3f}'
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '4_error_distribution.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 4_error_distribution.png")
    plt.close()


def plot_time_series_sample(test_pred, test_true, save_dir, n_samples=100):
    """Plot time series of predictions vs actual (first 100 samples)."""
    fig, axes = plt.subplots(5, 1, figsize=(16, 12))
    fig.suptitle('Time Series: Predictions vs Actual (First 100 Samples)', fontsize=16, fontweight='bold')
    
    x = np.arange(n_samples)
    
    for day in range(5):
        ax = axes[day]
        
        # Plot actual and predicted
        ax.plot(x, test_true[:n_samples, day], label='Actual', linewidth=1.5, alpha=0.7)
        ax.plot(x, test_pred[:n_samples, day], label='Predicted', linewidth=1.5, alpha=0.7)
        ax.fill_between(x, test_true[:n_samples, day], test_pred[:n_samples, day], 
                        alpha=0.2, label='Error region')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_ylabel('Log Return', fontsize=10)
        ax.set_title(f'Day {day + 1} Ahead Predictions', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if day == 4:
            ax.set_xlabel('Sample Index', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '5_time_series_sample.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 5_time_series_sample.png")
    plt.close()


def plot_cumulative_returns(test_pred, test_true, save_dir):
    """Plot cumulative returns comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use day 1 predictions
    actual_returns = test_true[:, 0]
    pred_returns = test_pred[:, 0]
    
    # Cumulative returns
    cum_actual = np.cumsum(actual_returns)
    cum_pred = np.cumsum(pred_returns)
    
    x = np.arange(len(cum_actual))
    
    ax.plot(x, cum_actual, label='Actual Cumulative Returns', linewidth=2)
    ax.plot(x, cum_pred, label='Predicted Cumulative Returns', linewidth=2)
    ax.fill_between(x, cum_actual, cum_pred, alpha=0.2)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Time (samples)', fontsize=12)
    ax.set_ylabel('Cumulative Log Return', fontsize=12)
    ax.set_title('Cumulative Returns: Actual vs Predicted (Day 1 Ahead)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add final values
    final_text = f'Final Actual: {cum_actual[-1]:.4f}\nFinal Predicted: {cum_pred[-1]:.4f}'
    ax.text(0.02, 0.98, final_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '6_cumulative_returns.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 6_cumulative_returns.png")
    plt.close()


def create_summary_plot(test_pred, test_true, save_dir):
    """Create a comprehensive summary plot."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Improved LSTM Model - Performance Summary', fontsize=18, fontweight='bold')
    
    # 1. Scatter plot (Day 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(test_true[:, 0], test_pred[:, 0], alpha=0.5, s=15)
    min_val = min(test_true[:, 0].min(), test_pred[:, 0].min())
    max_val = max(test_true[:, 0].max(), test_pred[:, 0].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title('Day 1: Pred vs Actual')
    ax1.grid(True, alpha=0.3)
    
    # 2. MSE per step
    ax2 = fig.add_subplot(gs[0, 1])
    mse_per_step = np.mean((test_true - test_pred) ** 2, axis=0)
    ax2.plot(range(1, 6), mse_per_step, marker='o', linewidth=2, markersize=8)
    ax2.set_xlabel('Day')
    ax2.set_ylabel('MSE')
    ax2.set_title('MSE by Horizon')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 6))
    
    # 3. Directional accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    dir_acc = []
    for day in range(5):
        acc = np.mean(np.sign(test_true[:, day]) == np.sign(test_pred[:, day]))
        dir_acc.append(acc)
    ax3.bar(range(1, 6), dir_acc, color='steelblue', alpha=0.7)
    ax3.axhline(y=0.5, color='red', linestyle='--', label='Random')
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Directional Accuracy')
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(range(1, 6))
    
    # 4. Error distribution (Day 1)
    ax4 = fig.add_subplot(gs[1, 0])
    errors = test_pred[:, 0] - test_true[:, 0]
    ax4.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Error')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution (Day 1)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Time series (sample)
    ax5 = fig.add_subplot(gs[1, 1:])
    n = 50
    x = np.arange(n)
    ax5.plot(x, test_true[:n, 0], label='Actual', linewidth=2)
    ax5.plot(x, test_pred[:n, 0], label='Predicted', linewidth=2, alpha=0.8)
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax5.set_xlabel('Sample')
    ax5.set_ylabel('Log Return')
    ax5.set_title('Time Series Sample (Day 1, First 50 samples)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative returns
    ax6 = fig.add_subplot(gs[2, :])
    cum_actual = np.cumsum(test_true[:, 0])
    cum_pred = np.cumsum(test_pred[:, 0])
    x = np.arange(len(cum_actual))
    ax6.plot(x, cum_actual, label='Actual', linewidth=2)
    ax6.plot(x, cum_pred, label='Predicted', linewidth=2)
    ax6.fill_between(x, cum_actual, cum_pred, alpha=0.2)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax6.set_xlabel('Sample')
    ax6.set_ylabel('Cumulative Log Return')
    ax6.set_title('Cumulative Returns Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, '0_summary_dashboard.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: 0_summary_dashboard.png")
    plt.close()


def main():
    print("=" * 70)
    print("VISUALIZING IMPROVED MODEL RESULTS")
    print("=" * 70)
    print()
    
    # Create results directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    save_dir = os.path.join(root_dir, "results", "plots")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading predictions...")
    test_pred, test_true = load_predictions()
    print(f"  Loaded {len(test_pred)} test samples")
    print(f"  Shape: {test_pred.shape} (samples, days)")
    print()
    
    print("Creating visualizations...")
    print()
    
    # Create all plots
    create_summary_plot(test_pred, test_true, save_dir)
    plot_predictions_vs_actual(test_pred, test_true, save_dir)
    plot_per_step_metrics(test_pred, test_true, save_dir)
    plot_directional_accuracy(test_pred, test_true, save_dir)
    plot_error_distribution(test_pred, test_true, save_dir)
    plot_time_series_sample(test_pred, test_true, save_dir)
    plot_cumulative_returns(test_pred, test_true, save_dir)
    
    print()
    print("=" * 70)
    print(f"All plots saved to: {save_dir}")
    print("=" * 70)
    print()
    print("Files created:")
    print("  0_summary_dashboard.png      - Comprehensive overview")
    print("  1_predictions_vs_actual.png  - Scatter plots for all 5 days")
    print("  2_per_step_metrics.png       - MSE and MAE by horizon")
    print("  3_directional_accuracy.png   - Direction prediction analysis")
    print("  4_error_distribution.png     - Error histograms")
    print("  5_time_series_sample.png     - Time series comparison")
    print("  6_cumulative_returns.png     - Cumulative return tracking")


if __name__ == "__main__":
    main()

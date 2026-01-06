"""
Enhanced feature preparation for S&P 500 time series forecasting.

This script transforms raw closing prices into a rich feature set suitable
for neural network training, including:
- Log returns (target variable)
- Technical indicators (momentum, volatility, trend)
- Temporal/calendar features
- Rolling statistics

Designed for academic time series forecasting (not trading strategies).
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators that capture market dynamics.
    
    All indicators are normalized/scaled to work well with neural networks.
    """
    prices = df['Close'].values
    
    # === MOMENTUM INDICATORS ===
    
    # RSI (Relative Strength Index) - bounded [0, 100]
    # Measures momentum: >70 = overbought, <30 = oversold
    window = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'] / 100.0  # Normalize to [0, 1]
    
    # Rate of Change (ROC) - percentage change over N days
    for period in [5, 10, 20]:
        df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = (ema_12 - ema_26) / df['Close']  # Normalized by price
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # === TREND INDICATORS ===
    
    # Moving averages and distance from them
    for window in [10, 20, 50, 200]:
        sma = df['Close'].rolling(window=window).mean()
        df[f'sma_{window}'] = sma
        df[f'distance_sma_{window}'] = (df['Close'] - sma) / sma  # Normalized distance
    
    # Trend strength: slope of 20-day MA
    df['ma_slope_20'] = df['sma_20'].diff(5) / df['sma_20'].shift(5)
    
    # === VOLATILITY INDICATORS ===
    
    # Rolling standard deviation of returns
    returns = df['log_return']
    for window in [5, 10, 20, 60]:
        df[f'volatility_{window}'] = returns.rolling(window=window).std()
    
    # Bollinger Bands position: where is price relative to bands?
    bb_window = 20
    bb_std = 2
    bb_ma = df['Close'].rolling(window=bb_window).mean()
    bb_std_dev = df['Close'].rolling(window=bb_window).std()
    df['bb_upper'] = bb_ma + (bb_std_dev * bb_std)
    df['bb_lower'] = bb_ma - (bb_std_dev * bb_std)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    # bb_position: 0 = at lower band, 0.5 = at middle, 1 = at upper band
    
    # Volatility of volatility (regime changes)
    df['vol_of_vol'] = df['volatility_20'].rolling(window=20).std()
    
    # === STATISTICAL FEATURES ===
    
    # Z-score: how many std deviations from recent mean?
    for window in [20, 60]:
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()
        df[f'zscore_{window}'] = (df['Close'] - rolling_mean) / (rolling_std + 1e-8)
    
    # Skewness and kurtosis of recent returns (distribution shape)
    df['return_skew_20'] = returns.rolling(window=20).skew()
    df['return_kurt_20'] = returns.rolling(window=20).kurt()
    
    # === CALENDAR/TEMPORAL FEATURES ===
    
    # Day of week (Monday effect, Friday effect)
    df['day_of_week'] = df['Date'].dt.dayofweek / 6.0  # Normalize to [0, 1]
    
    # Month (January effect, seasonal patterns)
    df['month'] = df['Date'].dt.month / 12.0  # Normalize to [0, 1]
    
    # Quarter
    df['quarter'] = df['Date'].dt.quarter / 4.0  # Normalize to [0, 1]
    
    # Day of month
    df['day_of_month'] = df['Date'].dt.day / 31.0  # Normalize to [0, 1]
    
    # Days since start (captures long-term trend)
    df['days_since_start'] = (df['Date'] - df['Date'].min()).dt.days
    df['days_since_start_normalized'] = df['days_since_start'] / df['days_since_start'].max()
    
    # === LAGGED FEATURES ===
    
    # Recent return history (autoregressive features)
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = returns.shift(lag)
    
    # Recent volatility
    for lag in [1, 5]:
        df[f'volatility_20_lag_{lag}'] = df['volatility_20'].shift(lag)
    
    return df


def prepare_dataset(
    csv_path: str,
    output_path: str,
    start_date: str = None,
    end_date: str = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Load raw price data and prepare enhanced feature set.
    
    Args:
        csv_path: Path to raw sp500_close.csv
        output_path: Path to save processed features
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    
    Returns:
        Tuple of (processed DataFrame, metadata dict)
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Raw data: {len(df)} days from {df['Date'].min()} to {df['Date'].max()}")
    
    # Filter date range if specified
    if start_date:
        df = df[df['Date'] >= start_date]
    if end_date:
        df = df[df['Date'] <= end_date]
    
    # Compute log returns (target variable)
    df['log_return'] = compute_log_returns(df['Close'])
    
    print("Computing technical indicators...")
    df = add_technical_indicators(df)
    
    # Drop rows with NaN (from rolling windows and indicators)
    # Keep track of how many rows we lose
    original_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = original_len - len(df)
    
    print(f"Dropped {dropped} rows due to indicator warmup period")
    print(f"Final dataset: {len(df)} days from {df['Date'].min()} to {df['Date'].max()}")
    print(f"Feature count: {len(df.columns) - 2} features (excluding Date, Close)")
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Saved processed features to {output_path}")
    
    # Metadata for reproducibility
    metadata = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 2,  # Exclude Date and Close
        'date_range': (str(df['Date'].min()), str(df['Date'].max())),
        'feature_columns': [c for c in df.columns if c not in ['Date', 'Close']],
        'target_column': 'log_return',
    }
    
    return df, metadata


def get_feature_groups() -> dict:
    """
    Return organized feature groups for analysis and ablation studies.
    Useful for thesis: analyze which feature groups contribute most.
    """
    return {
        'target': ['log_return'],
        'momentum': ['rsi', 'roc_5', 'roc_10', 'roc_20', 'macd', 'macd_signal', 'macd_diff'],
        'trend': ['sma_10', 'sma_20', 'sma_50', 'sma_200', 
                  'distance_sma_10', 'distance_sma_20', 'distance_sma_50', 'distance_sma_200',
                  'ma_slope_20'],
        'volatility': ['volatility_5', 'volatility_10', 'volatility_20', 'volatility_60',
                       'bb_position', 'vol_of_vol'],
        'statistical': ['zscore_20', 'zscore_60', 'return_skew_20', 'return_kurt_20'],
        'temporal': ['day_of_week', 'month', 'quarter', 'day_of_month', 'days_since_start_normalized'],
        'lagged': ['return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_10',
                   'volatility_20_lag_1', 'volatility_20_lag_5'],
    }


def main():
    # Resolve paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_csv = os.path.join(root_dir, "data", "sp500_close.csv")
    output_csv = os.path.join(root_dir, "data", "sp500_features.csv")
    
    # Check if input exists
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        print("Please run: python src/data/download_sp500.py")
        return
    
    # Prepare features
    df, metadata = prepare_dataset(input_csv, output_csv)
    
    # Print feature summary
    print("\n=== Feature Groups ===")
    feature_groups = get_feature_groups()
    for group_name, features in feature_groups.items():
        available = [f for f in features if f in df.columns]
        print(f"{group_name:12s}: {len(available)} features")
    
    print(f"\n=== Sample Statistics ===")
    print(f"Target (log_return) mean: {df['log_return'].mean():.6f}")
    print(f"Target (log_return) std:  {df['log_return'].std():.6f}")
    print(f"Target (log_return) min:  {df['log_return'].min():.6f}")
    print(f"Target (log_return) max:  {df['log_return'].max():.6f}")
    
    # Check for any remaining NaNs
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"\nWarning: NaN values found in columns: {nan_cols}")
    else:
        print(f"\n✓ No NaN values in processed dataset")
    
    print(f"\n✓ Feature preparation complete!")
    print(f"  Next step: python src/models/train_enhanced.py")


if __name__ == "__main__":
    main()

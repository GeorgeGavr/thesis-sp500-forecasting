"""
Unified Data Pipeline for S&P 500 Directional Forecasting
Thesis Topic: Neural networks and their applications for time series forecasting in the stock market.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

def test_stationarity(series: pd.Series, name: str):
    """Runs ADF and KPSS tests to mathematically prove stationarity for the thesis."""
    print(f"\n--- Stationarity Tests for {name} ---")
    
    # ADF Test (H0: Non-stationary)
    adf_result = adfuller(series.dropna(), autolag='AIC')
    print(f"ADF p-value: {adf_result[1]:.6f} (Needs to be < 0.05 to be stationary)")
    
    # KPSS Test (H0: Stationary)
    kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
    print(f"KPSS p-value: {kpss_result[1]:.6f} (Needs to be > 0.05 to be stationary)")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates strictly bounded and stationary technical indicators."""
    print("\nEngineering bounded technical features...")
    
    # 1. Target & Baseline Returns
    returns = df['log_return']
    
    # 2. MOMENTUM INDICATORS
    # RSI (Bounded 0 to 1)
    window = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['rsi_14'] = 1.0 - (1.0 / (1.0 + rs))
    
    # MACD (Normalized by price)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd_norm'] = (ema_12 - ema_26) / df['Close']
    df['macd_signal'] = df['macd_norm'].ewm(span=9, adjust=False).mean()

    # 3. TREND & MEAN REVERSION INDICATORS
    # Distance from Moving Averages (Normalized)
    for window in [10, 20, 50, 200]:
        sma = df['Close'].rolling(window=window).mean()
        df[f'dist_sma_{window}'] = (df['Close'] - sma) / sma

    # Bollinger Band Position (Bounded around 0 to 1)
    bb_window = 20
    bb_ma = df['Close'].rolling(window=bb_window).mean()
    bb_std = df['Close'].rolling(window=bb_window).std()
    bb_upper = bb_ma + (bb_std * 2)
    bb_lower = bb_ma - (bb_std * 2)
    df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

    # 4. VOLATILITY INDICATORS
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = returns.rolling(window=window).std()
    
    # Volatility of volatility (Detects regime changes)
    df['vol_of_vol'] = df['volatility_20'].rolling(window=20).std()

    # 5. CLASSIFICATION-SPECIFIC FEATURES (The Level-Up)
    # Consecutive Up/Down Days (Streak)
    direction = np.sign(returns).fillna(0)
    streak = direction.groupby((direction != direction.shift()).cumsum()).cumsum()
    # Cap streaks at +/- 5 to prevent extreme outliers
    df['return_streak'] = streak.clip(-5, 5) / 5.0  

    # 6. TEMPORAL FEATURES
    df['day_of_week'] = df['Date'].dt.dayofweek / 4.0  # 0 to 1 (Mon-Fri)
    df['month'] = (df['Date'].dt.month - 1) / 11.0     # 0 to 1 (Jan-Dec)

    # 7. AUTOREGRESSIVE LAGGED FEATURES
    for lag in [1, 2, 3, 5]:
        df[f'return_lag_{lag}'] = returns.shift(lag)

    return df

def main():
    # 1. Setup Directories
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(root_dir, "data", "raw")
    processed_dir = os.path.join(root_dir, "data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    print("="*60)
    print("S&P 500 DATA PIPELINE: BINARY CLASSIFICATION")
    print("="*60)

    # 2. Download Data (Modern Era Only)
    print("\nDownloading S&P 500 data (2015-Present)...")
    # This automatically pulls up to today's live close.
    df = yf.download("^GSPC", start="2015-01-01", interval="1d", auto_adjust=True, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[["Close", "Volume"]].dropna()
    df.reset_index(inplace=True)
    
    # Save raw data just in case
    df.to_csv(os.path.join(raw_dir, "sp500_raw.csv"), index=False)

    # 3. Transform to Stationarity
    print("\nCalculating Log Returns...")
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Thesis requirement: Prove stationarity
    test_stationarity(df['log_return'], "Log Returns")

    # 4. Feature Engineering
    df = engineer_features(df)

    # 5. Create Binary Target (Predicting Tomorrow)
    print("\nCreating Shifted Binary Target Variable...")
    # If tomorrow's log return is > 0, today's target is 1. Else 0.
    df['target'] = (df['log_return'].shift(-1) > 0).astype(float)

    # 6. Clean and Save
    original_len = len(df)
    df_clean = df.dropna().reset_index(drop=True)
    
    print(f"\nDropped {original_len - len(df_clean)} rows due to indicator warmup & shifting.")
    print(f"Final usable dataset size: {len(df_clean)} days.")
    
    up_days = df_clean['target'].mean() * 100
    down_days = 100 - up_days
    print(f"\nTarget (Classification) Balance:")
    print(f"  Up Days (1):   {up_days:.2f}%")
    print(f"  Down Days (0): {down_days:.2f}%")

    out_path = os.path.join(processed_dir, "sp500_processed.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"\n✓ Processed data saved successfully to: {out_path}")
    print("="*60)

if __name__ == "__main__":
    main()
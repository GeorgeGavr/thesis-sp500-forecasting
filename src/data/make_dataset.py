"""
Unified Data Pipeline for S&P 500 Directional Forecasting
Thesis Topic: Neural networks and their applications for time series forecasting in the stock market.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    returns = df['log_return']
    
    # 1. MOMENTUM INDICATORS
    window = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['rsi_14'] = 1.0 - (1.0 / (1.0 + rs))
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd_norm'] = (ema_12 - ema_26) / df['Close']
    df['macd_signal'] = df['macd_norm'].ewm(span=9, adjust=False).mean()

    # 2. TREND & MEAN REVERSION INDICATORS
    for window in [10, 20, 50, 200]:
        sma = df['Close'].rolling(window=window).mean()
        df[f'dist_sma_{window}'] = (df['Close'] - sma) / sma

    bb_window = 20
    bb_ma = df['Close'].rolling(window=bb_window).mean()
    bb_std = df['Close'].rolling(window=bb_window).std()
    bb_upper = bb_ma + (bb_std * 2)
    bb_lower = bb_ma - (bb_std * 2)
    df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

    # 3. VOLATILITY INDICATORS
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = returns.rolling(window=window).std()
    
    df['vol_of_vol'] = df['volatility_20'].rolling(window=20).std()

    # 4. VOLUME INDICATOR
    df['volume_change'] = np.log(df['Volume'] / df['Volume'].shift(1))

    # 5. CLASSIFICATION-SPECIFIC FEATURES (Streak)
    direction = np.sign(returns).fillna(0)
    streak = direction.groupby((direction != direction.shift()).cumsum()).cumsum()
    df['return_streak'] = streak.clip(-5, 5) / 5.0  

    # 6. TEMPORAL FEATURES
    df['day_of_week'] = df['Date'].dt.dayofweek / 4.0
    df['month'] = (df['Date'].dt.month - 1) / 11.0

    # 7. AUTOREGRESSIVE LAGGED FEATURES
    for lag in [1, 2, 3, 5]:
        df[f'return_lag_{lag}'] = returns.shift(lag)

    return df


def plot_correlation_heatmap(df: pd.DataFrame, plots_dir: str):
    """Generates Pearson Correlation Matrix & Heatmap for the engineered features."""
    os.makedirs(plots_dir, exist_ok=True)
    exclude_cols = ['Date', 'Close', 'Volume']
    corr_cols = [c for c in df.columns if c not in exclude_cols]
    
    corr_matrix = df[corr_cols].corr(method='pearson')
    
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
    print(f"✓ Correlation heatmap saved to: {heatmap_path}")


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_dir = os.path.join(root_dir, "data", "raw")
    processed_dir = os.path.join(root_dir, "data", "processed")
    plots_dir = os.path.join(root_dir, "results", "plots")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print("=" * 60)
    print("S&P 500 DATA PIPELINE: BINARY CLASSIFICATION (5-DAY HORIZON)")
    print("=" * 60)

    raw_csv_path = os.path.join(raw_dir, "sp500_raw.csv")
    if os.path.exists(raw_csv_path):
        print(f"\nLoading existing raw dataset from: {raw_csv_path}")
        df = pd.read_csv(raw_csv_path, parse_dates=['Date'])
    else:
        print("\nDownloading S&P 500 data (2015-Present)...")
        df = yf.download("^GSPC", start="2015-01-01", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close", "Volume"]].dropna()
        df.reset_index(inplace=True)
        df.to_csv(raw_csv_path, index=False)

    print("\nCalculating Log Returns...")
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    test_stationarity(df['log_return'], "Log Returns")

    df = engineer_features(df)

    print("\nCreating 5-Day Binary Target Variable...")
    df['target'] = (df['log_return'].rolling(5).sum().shift(-5) > 0).astype(float)

    original_len = len(df)
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    
    print(f"\nDropped {original_len - len(df_clean)} rows due to indicator warmup & shifting.")
    print(f"Final usable dataset size: {len(df_clean)} days.")

    plot_correlation_heatmap(df_clean, plots_dir)

    out_path = os.path.join(processed_dir, "sp500_processed.csv")
    if not os.path.exists(out_path):
        df_clean.to_csv(out_path, index=False)
        print(f"\n✓ Processed data saved successfully to: {out_path}")
    else:
        print(f"\n✓ Existing processed data preserved at: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
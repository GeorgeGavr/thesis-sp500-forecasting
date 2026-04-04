import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')


def adf_test(series: pd.Series, significance_level: float = 0.05) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    H0: Series has a unit root (non-stationary)
    H1: Series is stationary
    """
    result = adfuller(series.dropna(), autolag='AIC')
    p_value = result[1]
    is_stationary = p_value < significance_level
    
    return {
        'test': 'ADF',
        'statistic': float(result[0]),
        'p_value': float(p_value),
        'is_stationary': bool(is_stationary),
        'interpretation': f"{'Stationary' if is_stationary else 'Non-stationary'} (p={p_value:.6f})"
    }


def kpss_test(series: pd.Series, significance_level: float = 0.05) -> dict:
    """
    KPSS test for stationarity.
    H0: Series is stationary
    H1: Series has a unit root (non-stationary)
    """
    result = kpss(series.dropna(), regression='c', nlags='auto')
    p_value = result[1]
    is_stationary = p_value >= significance_level
    
    return {
        'test': 'KPSS',
        'statistic': float(result[0]),
        'p_value': float(p_value),
        'is_stationary': bool(is_stationary),
        'interpretation': f"{'Stationary' if is_stationary else 'Non-stationary'} (p={p_value:.6f})"
    }


def test_stationarity(series: pd.Series, name: str = "Series") -> dict:
    """
    Test stationarity using both ADF and KPSS tests.
    Returns combined decision: stationary only if both tests agree.
    """
    print(f"\n{'='*70}")
    print(f"Testing stationarity: {name}")
    print(f"{'='*70}")
    
    adf_result = adf_test(series)
    kpss_result = kpss_test(series)
    
    # Both tests must agree for stationarity
    if adf_result['is_stationary'] and kpss_result['is_stationary']:
        final_decision = 'stationary'
    else:
        final_decision = 'non-stationary'
    
    print(f"ADF Test:  {adf_result['interpretation']}")
    print(f"KPSS Test: {kpss_result['interpretation']}")
    print(f"\nFinal Decision: {final_decision.upper()}")
    
    return {
        'series_name': name,
        'adf': adf_result,
        'kpss': kpss_result,
        'final_decision': final_decision
    }


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns: r_t = ln(P_t / P_{t-1})"""
    return np.log(prices / prices.shift(1))


def compute_first_difference(series: pd.Series) -> pd.Series:
    """Compute first difference: Δy_t = y_t - y_{t-1}"""
    return series.diff()


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_csv = os.path.join(data_dir, "sp500_close.csv")
    metadata_json = os.path.join(data_dir, "transformation_metadata.json")

    print("="*70)
    print("S&P 500 DATA DOWNLOAD AND TRANSFORMATION")
    print("="*70)
    print("\nStep 1: Downloading historical data from Yahoo Finance...")
    df = yf.download("^GSPC", period="max", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("No data returned from yfinance for ^GSPC")

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[["Close"]].dropna()
    df.reset_index(inplace=True)
    print(f"Downloaded {len(df)} observations from {df['Date'].min()} to {df['Date'].max()}")
    
    # Initialize metadata
    metadata = {
        'data_source': 'Yahoo Finance (^GSPC)',
        'download_date': str(pd.Timestamp.now()),
        'n_observations': len(df),
        'date_range': [str(df['Date'].min()), str(df['Date'].max())],
        'transformation_steps': [],
        'stationarity_tests': {}
    }
    
    # Step 2: Compute log returns
    print("\nStep 2: Computing log returns...")
    df['log_return'] = compute_log_returns(df['Close'])
    df = df.dropna()  # Remove first NaN from log returns
    metadata['transformation_steps'].append('log_returns')
    print(f"Computed log returns (removed 1 NaN row)")
    
    # Step 3: Test stationarity of log returns
    print("\nStep 3: Testing stationarity of log returns...")
    log_returns_test = test_stationarity(df['log_return'], name="Log Returns")
    metadata['stationarity_tests']['log_returns'] = log_returns_test
    
    # Step 4: If non-stationary, apply first difference
    if log_returns_test['final_decision'] != 'stationary':
        print("\n⚠️  Log returns are NON-STATIONARY")
        print("\nStep 4: Applying first difference to log returns...")
        df['differenced_log_return'] = compute_first_difference(df['log_return'])
        df = df.dropna()  # Remove first NaN from differencing
        metadata['transformation_steps'].append('first_difference')
        print(f"Applied first difference (removed 1 NaN row)")
        
        # Step 5: Test stationarity of differenced series
        print("\nStep 5: Testing stationarity of differenced log returns...")
        diff_test = test_stationarity(df['differenced_log_return'], name="Differenced Log Returns")
        metadata['stationarity_tests']['differenced_log_returns'] = diff_test
        
        if diff_test['final_decision'] == 'stationary':
            print("\n✓ Differenced log returns are STATIONARY")
            metadata['final_transformation'] = 'differenced_log_returns'
            metadata['use_column'] = 'differenced_log_return'
        else:
            print("\n⚠️  Warning: Even after differencing, series may not be fully stationary")
            print("    Proceeding with differenced data anyway")
            metadata['final_transformation'] = 'differenced_log_returns'
            metadata['use_column'] = 'differenced_log_return'
    else:
        print("\n✓ Log returns are STATIONARY")
        metadata['final_transformation'] = 'log_returns'
        metadata['use_column'] = 'log_return'
    
    # Save data
    print(f"\n{'='*70}")
    print("SAVING DATA")
    print(f"{'='*70}")
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Saved transformed data to {out_csv}")
    print(f"  Final shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Use column: {metadata['use_column']}")
    
    # Save metadata
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved transformation metadata to {metadata_json}")
    
    print(f"\n{'='*70}")
    print("DOWNLOAD AND TRANSFORMATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nFinal transformation: {metadata['final_transformation']}")
    print(f"Column to use for training: {metadata['use_column']}")
    print(f"\nNext steps:")
    print("  1. Review {metadata_json} for stationarity test results")
    print("  2. Use the '{metadata['use_column']}' column for model training")


if __name__ == "__main__":
    main()

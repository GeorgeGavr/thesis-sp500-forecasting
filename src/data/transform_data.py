"""
Data transformation with stationarity testing for S&P 500 time series.

This script implements proper time series preprocessing:
1. Test for stationarity (ADF and KPSS tests)
2. Apply differencing if non-stationary
3. Calculate and remove noise components if stationary
4. Save transformed data with metadata

Academic rigor: Follows Box-Jenkins methodology for time series analysis.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')


def adf_test(series: pd.Series, significance_level: float = 0.05) -> Dict:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    H0 (null hypothesis): Series has a unit root (non-stationary)
    H1 (alternative): Series is stationary
    
    Args:
        series: Time series to test
        significance_level: Significance level (default: 0.05)
    
    Returns:
        Dictionary with test results
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    adf_stat = result[0]
    p_value = result[1]
    used_lag = result[2]
    n_obs = result[3]
    critical_values = result[4]
    
    # Decision: reject H0 if p-value < significance level
    is_stationary = p_value < significance_level
    
    return {
        'test': 'ADF',
        'statistic': float(adf_stat),
        'p_value': float(p_value),
        'used_lag': int(used_lag),
        'n_obs': int(n_obs),
        'critical_values': {k: float(v) for k, v in critical_values.items()},
        'is_stationary': bool(is_stationary),
        'significance_level': float(significance_level),
        'interpretation': f"{'Stationary' if is_stationary else 'Non-stationary'} (p={p_value:.6f})"
    }


def kpss_test(series: pd.Series, significance_level: float = 0.05) -> Dict:
    """
    Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
    
    H0 (null hypothesis): Series is stationary
    H1 (alternative): Series has a unit root (non-stationary)
    
    Note: KPSS is the "opposite" of ADF - null is stationarity.
    
    Args:
        series: Time series to test
        significance_level: Significance level (default: 0.05)
    
    Returns:
        Dictionary with test results
    """
    result = kpss(series.dropna(), regression='c', nlags='auto')
    
    kpss_stat = result[0]
    p_value = result[1]
    used_lag = result[2]
    critical_values = result[3]
    
    # Decision: reject H0 if p-value < significance level
    # (reject stationarity → series is non-stationary)
    is_stationary = p_value >= significance_level
    
    return {
        'test': 'KPSS',
        'statistic': float(kpss_stat),
        'p_value': float(p_value),
        'used_lag': int(used_lag),
        'critical_values': {k: float(v) for k, v in critical_values.items()},
        'is_stationary': bool(is_stationary),
        'significance_level': float(significance_level),
        'interpretation': f"{'Stationary' if is_stationary else 'Non-stationary'} (p={p_value:.6f})"
    }


def test_stationarity(series: pd.Series, name: str = "Series") -> Dict:
    """
    Comprehensive stationarity testing with ADF and KPSS.
    
    Best practice: Use both tests together:
    - ADF stationary + KPSS stationary → Stationary
    - ADF non-stationary + KPSS non-stationary → Non-stationary
    - Conflicting results → Difference series (safer approach)
    
    Args:
        series: Time series to test
        name: Name of the series (for reporting)
    
    Returns:
        Dictionary with both test results and final decision
    """
    print(f"\n{'='*70}")
    print(f"STATIONARITY TESTING: {name}")
    print(f"{'='*70}")
    
    # Run both tests
    adf_result = adf_test(series)
    kpss_result = kpss_test(series)
    
    # Combined decision
    if adf_result['is_stationary'] and kpss_result['is_stationary']:
        final_decision = 'stationary'
        recommendation = 'Series is stationary - can use as is'
    elif not adf_result['is_stationary'] and not kpss_result['is_stationary']:
        final_decision = 'non-stationary'
        recommendation = 'Series is non-stationary - apply differencing'
    else:
        final_decision = 'inconclusive'
        recommendation = 'Tests disagree - apply differencing (conservative approach)'
    
    print(f"\n{name} Statistics:")
    print(f"  Mean: {series.mean():.6f}")
    print(f"  Std:  {series.std():.6f}")
    print(f"  Min:  {series.min():.6f}")
    print(f"  Max:  {series.max():.6f}")
    print(f"  N:    {len(series)}")
    
    print(f"\nADF Test (H0: non-stationary):")
    print(f"  Statistic: {adf_result['statistic']:.6f}")
    print(f"  p-value:   {adf_result['p_value']:.6f}")
    print(f"  Result:    {adf_result['interpretation']}")
    
    print(f"\nKPSS Test (H0: stationary):")
    print(f"  Statistic: {kpss_result['statistic']:.6f}")
    print(f"  p-value:   {kpss_result['p_value']:.6f}")
    print(f"  Result:    {kpss_result['interpretation']}")
    
    print(f"\n{'='*70}")
    print(f"FINAL DECISION: {final_decision.upper()}")
    print(f"RECOMMENDATION: {recommendation}")
    print(f"{'='*70}")
    
    return {
        'series_name': name,
        'adf': adf_result,
        'kpss': kpss_result,
        'final_decision': final_decision,
        'recommendation': recommendation,
        'statistics': {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'n': int(len(series))
        }
    }


def compute_first_difference(series: pd.Series) -> pd.Series:
    """
    Compute first difference: Δy_t = y_t - y_{t-1}
    
    First differencing removes linear trends and makes series stationary.
    """
    return series.diff()


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
    
    Log returns are approximately equal to percentage returns for small changes,
    and they are additive over time (useful for time series analysis).
    """
    return np.log(prices / prices.shift(1))


def estimate_noise_variance(stationary_series: pd.Series, method: str = 'mad') -> float:
    """
    Estimate noise variance from a stationary time series.
    
    Methods:
    - 'mad': Median Absolute Deviation (robust to outliers)
    - 'std': Standard deviation (classical approach)
    - 'diff': Based on first differences (for smooth signals)
    
    Args:
        stationary_series: Stationary time series
        method: Estimation method
    
    Returns:
        Estimated noise standard deviation
    """
    series = stationary_series.dropna()
    
    if method == 'mad':
        # MAD estimator: σ ≈ 1.4826 * median(|x - median(x)|)
        median = series.median()
        mad = np.median(np.abs(series - median))
        noise_std = 1.4826 * mad
    
    elif method == 'std':
        # Classical standard deviation
        noise_std = series.std()
    
    elif method == 'diff':
        # Based on first differences (assumes smooth underlying signal)
        # σ ≈ median(|Δx|) / sqrt(2)
        diffs = np.abs(series.diff().dropna())
        noise_std = np.median(diffs) / np.sqrt(2)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return float(noise_std)


def denoise_series(series: pd.Series, noise_std: float, threshold: float = 2.0) -> pd.Series:
    """
    Remove noise from stationary series using threshold.
    
    Simple denoising: set values within ±threshold*noise_std to zero.
    This is a basic approach - more sophisticated methods exist (wavelets, filters).
    
    Args:
        series: Stationary time series
        noise_std: Estimated noise standard deviation
        threshold: Threshold multiplier (default: 2 sigma)
    
    Returns:
        Denoised series
    """
    denoised = series.copy()
    noise_threshold = threshold * noise_std
    
    # Set small values to zero
    mask = np.abs(denoised) < noise_threshold
    denoised[mask] = 0
    
    return denoised


def transform_price_series(
    prices: pd.Series,
    test_stationarity_flag: bool = True,
    denoise: bool = False
) -> Tuple[pd.Series, Dict]:
    """
    Complete transformation pipeline for price series.
    
    Pipeline:
    1. Test raw prices for stationarity
    2. If non-stationary: compute log returns (first difference of log prices)
    3. If stationary: use as is
    4. Test transformed series for stationarity
    5. Optionally: estimate and remove noise
    
    Args:
        prices: Raw price series
        test_stationarity_flag: Whether to perform stationarity tests
        denoise: Whether to apply denoising (only if stationary)
    
    Returns:
        Tuple of (transformed_series, metadata_dict)
    """
    metadata = {
        'transformation_steps': [],
        'stationarity_tests': {}
    }
    
    # Step 1: Test raw prices
    if test_stationarity_flag:
        price_test = test_stationarity(prices, name="Raw Prices")
        metadata['stationarity_tests']['raw_prices'] = price_test
        is_stationary = (price_test['final_decision'] == 'stationary')
    else:
        # Assume non-stationary (prices are typically non-stationary)
        is_stationary = False
        print("\nSkipping stationarity test - assuming prices are non-stationary")
    
    # Step 2: Transform if needed
    if not is_stationary:
        print("\n→ Applying log returns transformation (first difference of log prices)")
        transformed = compute_log_returns(prices)
        metadata['transformation_steps'].append('log_returns')
        metadata['transformation'] = 'log_returns'
        
        # Test log returns for stationarity
        if test_stationarity_flag:
            returns_test = test_stationarity(transformed.dropna(), name="Log Returns")
            metadata['stationarity_tests']['log_returns'] = returns_test
            
            # If still non-stationary, difference again
            if returns_test['final_decision'] != 'stationary':
                print("\n→ Log returns still non-stationary - applying second difference")
                transformed = transformed.diff()
                metadata['transformation_steps'].append('second_difference')
                metadata['transformation'] = 'log_returns_differenced'
    else:
        print("\n→ Prices are stationary - using raw prices")
        transformed = prices.copy()
        metadata['transformation'] = 'none'
    
    # Step 3: Denoise if requested and series is stationary
    if denoise and test_stationarity_flag:
        final_test = test_stationarity(transformed.dropna(), name="Transformed Series")
        metadata['stationarity_tests']['final'] = final_test
        
        if final_test['final_decision'] == 'stationary':
            print("\n→ Estimating and removing noise...")
            noise_std = estimate_noise_variance(transformed, method='mad')
            print(f"  Estimated noise std: {noise_std:.6f}")
            
            denoised = denoise_series(transformed, noise_std, threshold=2.0)
            n_removed = (denoised == 0).sum()
            pct_removed = 100 * n_removed / len(denoised)
            
            print(f"  Removed {n_removed} values ({pct_removed:.2f}%) below noise threshold")
            
            transformed = denoised
            metadata['transformation_steps'].append('denoising')
            metadata['noise_std'] = float(noise_std)
        else:
            print("\n→ Series not stationary - skipping denoising")
    
    return transformed, metadata


def main():
    # Setup paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_csv = os.path.join(root_dir, "data", "sp500_close.csv")
    output_csv = os.path.join(root_dir, "data", "sp500_transformed.csv")
    metadata_json = os.path.join(root_dir, "data", "transformation_metadata.json")
    
    # Check input
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        print("Please run: python src/data/download_sp500.py")
        return
    
    print("="*70)
    print("S&P 500 DATA TRANSFORMATION WITH STATIONARITY TESTING")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from {input_csv}...")
    df = pd.read_csv(input_csv, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"  Loaded {len(df)} observations from {df['Date'].min()} to {df['Date'].max()}")
    
    # Transform
    transformed_series, metadata = transform_price_series(
        df['Close'],
        test_stationarity_flag=True,
        denoise=False  # Set to True if you want denoising
    )
    
    # Create output dataframe
    df_out = pd.DataFrame({
        'Date': df['Date'],
        'Close': df['Close'],
        'Transformed': transformed_series
    })
    
    # Drop NaN from transformation
    df_out = df_out.dropna().reset_index(drop=True)
    
    print(f"\n{'='*70}")
    print("FINAL TRANSFORMED SERIES")
    print(f"{'='*70}")
    print(f"  Shape: {df_out.shape}")
    print(f"  Date range: {df_out['Date'].min()} to {df_out['Date'].max()}")
    print(f"  Transformation: {metadata['transformation']}")
    print(f"\nTransformed series statistics:")
    print(f"  Mean: {df_out['Transformed'].mean():.6f}")
    print(f"  Std:  {df_out['Transformed'].std():.6f}")
    print(f"  Min:  {df_out['Transformed'].min():.6f}")
    print(f"  Max:  {df_out['Transformed'].max():.6f}")
    
    # Save
    df_out.to_csv(output_csv, index=False)
    print(f"\n✓ Saved transformed data to {output_csv}")
    
    # Save metadata
    metadata['output_shape'] = df_out.shape
    metadata['date_range'] = [str(df_out['Date'].min()), str(df_out['Date'].max())]
    
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to {metadata_json}")
    
    print(f"\n{'='*70}")
    print("TRANSFORMATION COMPLETE")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("  1. Review transformation_metadata.json for test results")
    print("  2. Use sp500_transformed.csv for model training")
    print("  3. Update train.py to load transformed data")


if __name__ == "__main__":
    main()

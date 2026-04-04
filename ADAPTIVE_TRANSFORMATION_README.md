# Adaptive Data Transformation System

## Overview

This document explains the updated data transformation pipeline that automatically ensures stationarity through sequential testing.

## How It Works

### Step 1: Download and Transform (`download_sp500.py`)

When you run `python src/data/download_sp500.py`, the script now:

1. **Downloads** S&P 500 historical data from Yahoo Finance
2. **Computes log returns**: `log_return = ln(P_t / P_{t-1})`
3. **Tests stationarity** of log returns using:
   - **ADF test** (Augmented Dickey-Fuller): H0 = non-stationary
   - **KPSS test**: H0 = stationary
   - **Decision**: Both tests must agree for stationarity

4. **If log returns are stationary**:
   - Save data with `log_return` column
   - Set `use_column = 'log_return'` in metadata
   - Done! ✓

5. **If log returns are NON-stationary**:
   - Compute first difference: `differenced_log_return = Δ(log_return)`
   - Test stationarity of differenced series
   - Save data with both `log_return` AND `differenced_log_return` columns
   - Set `use_column = 'differenced_log_return'` in metadata

### Step 2: Feature Engineering (`prepare_features.py`)

When you run `python src/data/prepare_features.py`:

- Loads data from `sp500_close.csv`
- Uses `log_return` column (always present) for feature engineering
- Computes 41 technical indicators based on prices and log returns
- Saves to `sp500_features.csv`

**Note**: Feature engineering always uses `log_return`, not `differenced_log_return`. The differencing is only applied to the target variable for training.

### Step 3: Model Training (all `train*.py` scripts)

When you run any training script:

1. Loads `sp500_close.csv` or `sp500_features.csv`
2. Reads `transformation_metadata.json` to determine which column to use
3. Uses either:
   - `log_return` if log returns were stationary, OR
   - `differenced_log_return` if log returns needed differencing

**Backward compatible**: If metadata is missing, scripts check for column existence and fall back appropriately.

## File Structure

```
data/
├── sp500_close.csv                   # Transformed data
│   ├── Date
│   ├── Close
│   ├── log_return                    # Always present
│   └── differenced_log_return        # Only if needed
│
├── transformation_metadata.json      # Transformation info
│   ├── use_column                    # Which column to use for training
│   ├── final_transformation          # 'log_returns' or 'differenced_log_returns'
│   └── stationarity_tests            # ADF and KPSS test results
│
└── sp500_features.csv                # Enhanced features
    ├── Date
    ├── Close
    ├── log_return                    # Target variable
    ├── [41 technical indicators]
    └── (differenced_log_return)      # Only if sp500_close.csv has it
```

## Expected Outcomes

### Scenario A: Log Returns Are Stationary (Most Likely)

Based on financial theory and empirical evidence, log returns of stock prices are typically stationary. This is why they're used in time series analysis.

**Output**:
```
Testing stationarity: Log Returns
ADF Test:  Stationary (p=0.000000)
KPSS Test: Stationary (p=0.100000)
Final Decision: STATIONARY

✓ Log returns are STATIONARY
```

**Metadata**:
```json
{
  "final_transformation": "log_returns",
  "use_column": "log_return"
}
```

**Training**: Models will use `log_return` column

### Scenario B: Log Returns Are Non-Stationary (Unlikely but Possible)

In rare cases (e.g., structural breaks, regime changes), log returns might not be stationary.

**Output**:
```
Testing stationarity: Log Returns
ADF Test:  Non-stationary (p=0.123456)
KPSS Test: Non-stationary (p=0.012345)
Final Decision: NON-STATIONARY

⚠️  Log returns are NON-STATIONARY

Applying first difference to log returns...
Testing stationarity: Differenced Log Returns
ADF Test:  Stationary (p=0.000000)
KPSS Test: Stationary (p=0.100000)
Final Decision: STATIONARY

✓ Differenced log returns are STATIONARY
```

**Metadata**:
```json
{
  "final_transformation": "differenced_log_returns",
  "use_column": "differenced_log_return"
}
```

**Training**: Models will use `differenced_log_return` column

## Interpretation

### What are "differenced log returns"?

If log returns need differencing, we compute:

```
differenced_log_return_t = log_return_t - log_return_{t-1}
```

This is essentially the **second difference** of log prices:

```
differenced_log_return_t = [ln(P_t) - ln(P_{t-1})] - [ln(P_{t-1}) - ln(P_{t-2})]
                         = ln(P_t) - 2*ln(P_{t-1}) + ln(P_{t-2})
```

**Interpretation**: This measures the *acceleration* or *change in returns* rather than returns themselves.

### Academic Justification

This approach follows the **Box-Jenkins methodology** for time series analysis:

1. **Check stationarity** (ADF + KPSS)
2. **If non-stationary**: Apply differencing
3. **Recheck stationarity**
4. **If still non-stationary**: Apply second differencing (our differenced log returns)

Using both ADF and KPSS tests reduces the risk of:
- **Type I errors**: Falsely rejecting stationarity (ADF alone)
- **Type II errors**: Falsely accepting stationarity (KPSS alone)

## Benefits

1. **Automatic**: No manual intervention needed
2. **Rigorous**: Uses proper statistical tests
3. **Documented**: Metadata tracks all transformations
4. **Reproducible**: Same data preparation every time
5. **Consistent**: All models use the same stationary series

## Workflow

### Complete pipeline from scratch:

```bash
# Step 1: Download and transform data
python src/data/download_sp500.py
# → Creates: data/sp500_close.csv, data/transformation_metadata.json

# Step 2: (Optional) Prepare enhanced features
python src/data/prepare_features.py
# → Creates: data/sp500_features.csv

# Step 3: Train models
python src/models/train.py              # Baseline
python src/models/train_enhanced.py     # Enhanced
python src/models/train_improved.py     # Improved
```

### Checking transformation results:

```bash
# View metadata
cat data/transformation_metadata.json

# Check which column is being used
python -c "import json; m = json.load(open('data/transformation_metadata.json')); print(f\"Using: {m['use_column']} ({m['final_transformation']})\")"
```

## Troubleshooting

### Problem: "No transformed column found"

**Cause**: Data hasn't been processed yet  
**Solution**: Run `python src/data/download_sp500.py`

### Problem: Old data without new columns

**Cause**: Data was downloaded before this update  
**Solution**: Delete `data/sp500_close.csv` and `data/transformation_metadata.json`, then run `python src/data/download_sp500.py`

### Problem: Models using wrong column

**Cause**: Metadata is out of sync with data  
**Solution**: Delete both files and re-download as above

## Testing

To verify the system works correctly:

```bash
# 1. Clean start
rm data/sp500_close.csv data/transformation_metadata.json

# 2. Download and transform
python src/data/download_sp500.py

# 3. Check the output
# Should see stationarity test results and final decision

# 4. Verify metadata exists
cat data/transformation_metadata.json

# 5. Train a model (should work without errors)
python src/models/train.py
```

## References

- **Augmented Dickey-Fuller test**: Dickey, D. A., & Fuller, W. A. (1979)
- **KPSS test**: Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992)
- **Box-Jenkins methodology**: Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015)

---

**Last Updated**: 2026-01-27  
**Author**: George  
**Status**: Production Ready ✓

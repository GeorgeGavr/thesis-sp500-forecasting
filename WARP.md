# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Commands

Environment setup (from README):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Fetch data:

```bash
python src/data/download_sp500.py
```

Train the model (uses CPU or GPU automatically):

```bash
python src/models/train.py
```

Artifacts and inputs:
- Input CSV: `data/sp500_close.csv` (columns: `Date, Close`)
- Saved model: `models/sp500_lstm.pt`

Linting:
- No linter/formatter is configured in this repo.

Tests:
- No test suite is present. There is currently no command to run a single test.

## High-level architecture

This repo is a small, script-first Python project for time-series forecasting of S&P 500 closes using PyTorch.

- Data acquisition (`src/data/download_sp500.py`)
  - Downloads ^GSPC daily data via yfinance.
  - Writes a portable CSV with columns `Date, Close` to `data/sp500_close.csv`.

- Modeling and training (`src/models/train.py`)
  - Loads and sorts the CSV by `Date`, extracts the `Close` series.
  - Converts prices to **log returns**: `ln(P_t / P_t-1)` for scale-invariant representation.
  - Normalizes log returns using **StandardScaler** (z-score: mean=0, std=1).
  - Windowing: builds (X, y) pairs of length `seq_len=60` for next-day return prediction.
  - Splits dataset 80%/10%/10% into train/val/test.
  - **Temporal weighting**: training samples weighted exponentially (recent data emphasized via `weight_decay_rate=0.001`).
  - Model: `LSTMRegressor(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)` → final `Linear` head.
  - Training: Adam (`lr=1e-3`), weighted MSE loss, batched `DataLoader`s (`batch_size=64`).
  - Evaluation: per-epoch validation MSE (unweighted), final test MSE.
  - Device: selects CUDA if available (`torch.cuda.is_available()`), otherwise CPU.
  - Persistence: saves weights to `models/sp500_lstm.pt`.

## Notes for agents

- Run the data download script before training; `train.py` expects `data/sp500_close.csv` to exist.
- Network access is required for yfinance on first run.
- Adjust hyperparameters by editing constants near the top of `main()` in `src/models/train.py` (e.g., `seq_len`, `epochs`, `lr`, `weight_decay_rate`).
- The model predicts **log returns**, not absolute prices. To convert predictions back to price levels, apply inverse transformations.
- `weight_decay_rate` controls temporal weighting: higher values (e.g., 0.005) emphasize recent years more; lower values (e.g., 0.0001) treat all eras more equally.

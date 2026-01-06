# WARP.md

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download data
python src/data/download_sp500.py

# Train baseline model
python src/models/train.py

# Or train enhanced model with features
python src/data/prepare_features.py
python src/models/train_enhanced.py
```

## Architecture

**Data Pipeline:**
1. Download S&P 500 prices → `data/sp500_close.csv`
2. Transform to log returns (stationary): `r_t = ln(P_t / P_{t-1})`
3. Normalize with StandardScaler (z-score)
4. Create sequences: 60-day windows → predict next day

**Models:**
- `train.py`: Simple LSTM (1 feature: log returns)
- `train_enhanced.py`: LSTM + Attention (41 features: technical indicators, momentum, volatility, temporal)

**Key Files:**
- `data/sp500_close.csv`: Raw prices
- `data/sp500_features.csv`: Enhanced features (41 columns)
- `models/sp500_lstm.pt`: Baseline model
- `models/sp500_lstm_enhanced.pt`: Enhanced model

## Data Transformation

Prices are **non-stationary** (random walk). Log returns are **stationary**:
- Tested with ADF and KPSS tests
- Log returns have constant mean/variance
- Suitable for LSTM training

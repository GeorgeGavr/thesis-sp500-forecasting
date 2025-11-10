# Thesis: Neural Networks for S&P 500 Time Series Forecasting

This project forecasts the S&P 500 index closing prices using neural networks. Data are sourced from Yahoo Finance and saved to a CSV file.

## Project structure
- `src/data/download_sp500.py`: downloads S&P 500 close prices to `data/sp500_close.csv`
- `src/models/train.py`: trains a simple LSTM model on the CSV time series
- `data/`: CSV data folder
- `models/`: saved model artifacts
- `notebooks/`: exploratory notebooks (optional)
- `scripts/`: utility scripts (optional)

## Setup
1) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2) Download data to CSV:

```bash
python src/data/download_sp500.py
```

3) Train the model:

```bash
python src/models/train.py
```

## Notes
- Dataset is stored at `data/sp500_close.csv` (Date, Close).
- The baseline model is a small LSTM for next-day close prediction. You can evolve this to more advanced architectures.

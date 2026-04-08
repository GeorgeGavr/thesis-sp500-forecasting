# S&P 500 Directional Forecasting: A Deep Learning Approach

This repository contains the codebase for a Master's thesis investigating the application of Bidirectional Long Short-Term Memory (Bi-LSTM) neural networks with Attention mechanisms for financial time series forecasting. 

Rather than attempting naive continuous price prediction (regression), which typically flatlines due to the Efficient Market Hypothesis (EMH), this project treats market forecasting as a **binary classification problem**, optimizing for high-precision momentum detection.

## Project Architecture & Methodology
This pipeline was built to address common pitfalls in quantitative machine learning:
* **Stationarity:** Conversion of raw prices to log returns to ensure mathematical stability.
* **Lookback Window Optimization:** Utilizing a restricted 10-day short-term memory window to prevent stale noise from burying immediate momentum signals.
* **Class Imbalance & EMH:** Implementing BCE Loss class weights and Dynamic Median Thresholding to prevent the model from defaulting to majority-class ("always guess Up") predictions.

## Repository Structure

```text
thesis-sp500-forecasting/
│
├── data/
│   └── processed/          # Stores the stationary, engineered feature dataset
│
├── models/                 # Saved PyTorch checkpoints and scalers (.pt files)
│
├── results/
│   └── plots/              # Thesis-grade evaluation visuals (ROC, PR, Confusion Matrix)
│
├── src/
│   ├── data/
│   │   └── make_dataset.py         # Downloads yfinance data, engineers features, tests stationarity
│   ├── models/
│   │   ├── train_classifier.py     # Bi-LSTM + Attention architecture & training loop
│   │   └── evaluate.py             # Evaluation script with dynamic thresholding and plot generation
│
├── requirements.txt        # Standard environment dependencies
└── README.md               # Project documentation
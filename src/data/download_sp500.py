import os
import pandas as pd
import yfinance as yf


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    out_csv = os.path.join(data_dir, "sp500_close.csv")

    print("Downloading S&P 500 historical data (^GSPC) from Yahoo Finance...")
    df = yf.download("^GSPC", period="max", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("No data returned from yfinance for ^GSPC")

    # Handle MultiIndex columns (yfinance returns columns like ('Close', '^GSPC'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    close = df[["Close"]].dropna()
    # Save with Date as a column for portability
    close.reset_index(inplace=True)
    close.to_csv(out_csv, index=False)
    print(f"Saved close prices to {out_csv} (rows={len(close)})")


if __name__ == "__main__":
    main()

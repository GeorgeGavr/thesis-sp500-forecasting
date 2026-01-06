# S&P 500 Forecasting with LSTM

Master's thesis project on time series forecasting using neural networks.

## Setup (First Time Only)

### Mac/Linux (Terminal)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (Command Prompt)
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### VS Code (Any OS)
1. Open folder in VS Code
2. Open Terminal (View → Terminal or `` Ctrl+` ``)
3. Run setup commands above for your OS
4. Select Python interpreter: `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows) → "Python: Select Interpreter" → Choose `.venv`

---

## How to Run Scripts

### Step 1: Download Data (Required First)

**Mac/Linux:**
```bash
source .venv/bin/activate
python src/data/download_sp500.py
```

**Windows:**
```cmd
.venv\Scripts\activate
python src/data/download_sp500.py
```

**VS Code:**
- Right-click `src/data/download_sp500.py` → "Run Python File in Terminal"
- Or press `F5` (if debugger configured)
- Or use terminal as above

**What it does:** Downloads S&P 500 price data → saves to `data/sp500_close.csv`

---

### Step 2a: Train Baseline Model (Simple)

**Mac/Linux:**
```bash
source .venv/bin/activate
python src/models/train.py
```

**Windows:**
```cmd
.venv\Scripts\activate
python src/models/train.py
```

**VS Code:**
- Right-click `src/models/train.py` → "Run Python File in Terminal"

**What it does:**
- Uses 1 feature (log returns)
- Trains for 20 epochs
- Saves model to `models/sp500_lstm.pt`
- Takes ~2-5 minutes

---

### Step 2b: Train Enhanced Model (Better Performance)

**First, prepare features:**

**Mac/Linux:**
```bash
source .venv/bin/activate
python src/data/prepare_features.py
```

**Windows:**
```cmd
.venv\Scripts\activate
python src/data/prepare_features.py
```

**VS Code:**
- Right-click `src/data/prepare_features.py` → "Run Python File in Terminal"

**What it does:** Creates 41 technical indicators → saves to `data/sp500_features.csv`

---

**Then, train the model:**

**Mac/Linux:**
```bash
python src/models/train_enhanced.py
```

**Windows:**
```cmd
python src/models/train_enhanced.py
```

**VS Code:**
- Right-click `src/models/train_enhanced.py` → "Run Python File in Terminal"

**What it does:**
- Uses 41 features
- Trains with early stopping (~20-50 epochs)
- Saves model to `models/sp500_lstm_enhanced.pt`
- Takes ~5-15 minutes

---

### Optional: Run Stationarity Tests (For Thesis)

**Mac/Linux:**
```bash
source .venv/bin/activate
python src/data/transform_data.py
```

**Windows:**
```cmd
.venv\Scripts\activate
python src/data/transform_data.py
```

**VS Code:**
- Right-click `src/data/transform_data.py` → "Run Python File in Terminal"

**What it does:**
- Runs ADF and KPSS stationarity tests
- Confirms log returns are stationary
- Saves results to `data/transformation_metadata.json`

---

## Troubleshooting

### "python not found" or "python3 not found"
- **Mac**: Use `python3` instead of `python`
- **Windows**: Make sure Python is installed and in PATH

### "No module named 'torch'" or similar
- Activate virtual environment first:
  - **Mac/Linux**: `source .venv/bin/activate`
  - **Windows**: `.venv\Scripts\activate`
- Then run: `pip install -r requirements.txt`

### "sp500_close.csv not found"
- Run `python src/data/download_sp500.py` first

### "sp500_features.csv not found" (when training enhanced model)
- Run `python src/data/prepare_features.py` first

### VS Code doesn't recognize virtual environment
- Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows)
- Type "Python: Select Interpreter"
- Choose the one with `.venv` in the path

---

### Step 2c: Train Improved Model (6 Key Fixes)

**Mac/Linux:**
```bash
python src/models/train_improved.py
```

**Windows:**
```cmd
python src/models/train_improved.py
```

**VS Code:**
- Right-click `src/models/train_improved.py` → "Run Python File in Terminal"

**What it does:**
- Applies 6 key improvements:
  1. **Target scaling** - normalizes log returns to [-1, 1]
  2. **Huber Loss** - robust to outliers (vs MSE)
  3. **Modern data** - uses only 2000+ data (avoids regime mixing)
  4. **Inverse transform** - converts predictions back to original scale
  5. **Distance-from-MA** - already in features (relative metrics)
  6. **Multi-step prediction** - predicts 5 days at once (prevents error accumulation)
- Trains with early stopping
- Saves model to `models/sp500_lstm_improved.pt`
- Takes ~5-15 minutes

---

### Step 3: Visualize Results

**Mac/Linux:**
```bash
python src/models/visualize_improved.py
```

**Windows:**
```cmd
python src/models/visualize_improved.py
```

**VS Code:**
- Right-click `src/models/visualize_improved.py` → "Run Python File in Terminal"

**What it does:**
- Creates 7 comprehensive plots
- Saved to `results/plots/`
- Includes:
  - Summary dashboard
  - Predictions vs actual (all 5 days)
  - Per-step metrics (MSE/MAE)
  - Directional accuracy analysis
  - Error distributions
  - Time series comparisons
  - Cumulative returns

---

## Data Transformation

Stock prices are **non-stationary** (random walk) → transform to **log returns** (stationary)

```
Log Return = ln(P_t / P_{t-1})
```

Verified with ADF and KPSS statistical tests.

## Models

### Baseline LSTM
- 1 feature (log returns)
- 60-day sequence → predict next day
- **Test MSE: 0.893**

### Enhanced LSTM + Attention
- 41 features (technical indicators, momentum, volatility, temporal)
- Bidirectional LSTM with attention
- **Test MSE: 0.132** (85% improvement)
- **Directional Accuracy: 54.8%**

### Improved LSTM (6 Key Fixes)
- 41 features + target scaling + Huber loss
- Modern data only (2000+)
- Multi-step prediction (5 days at once)
- **Test MSE: 0.000098** (on original scale)
- **Directional Accuracy: 55.2%**
- **Training Directional Accuracy: 64.2%** ← Model is learning!

## Results

| Model | Features | Data | Prediction | Test MSE | Dir. Acc. |
|-------|----------|------|------------|----------|------------|
| Baseline | 1 | All (1928+) | 1-day | 0.893 | ~50% |
| Enhanced | 41 | All (1928+) | 1-day | 0.132 | 54.8% |
| **Improved** | **41** | **Modern (2000+)** | **5-day** | **0.000098** | **55.2%** |

**Note**: Low R² (~0) is expected for stock returns (Efficient Market Hypothesis). Directional accuracy > 50% is meaningful.

## Project Structure

```
├── data/                    # CSV files
├── src/
│   ├── data/               # Download & feature engineering
│   └── models/             # Training scripts
├── models/                 # Saved checkpoints
└── results/                # Training history
```

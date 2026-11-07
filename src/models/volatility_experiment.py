"""
Walk-forward πείραμα πρόβλεψης ΜΕΤΑΒΛΗΤΟΤΗΤΑΣ (όχι κατεύθυνσης) για τον S&P 500.

Στόχος:  y = log( Σ r²  των επόμενων 5 ημερών )   (λογάριθμος ρεαλιζόμενης διακύμανσης 5 ημερών)
         Το παράθυρο που τελειώνει στη γραμμή j προβλέπει τις ημέρες j+1..j+5, όπως στο
         walkforward_experiment.py. Ίδιες περίοδοι (folds), ίδιο embargo, ίδιοι σπόροι.

Μοντέλα και βάσεις:
    Mean         σταθερή πρόβλεψη = μέσος όρος του train
    Persistence  «η επόμενη εβδομάδα = η τελευταία εβδομάδα» (log Σ r² των 5 τελευταίων ημερών)
    HAR          γραμμική παλινδρόμηση σε log διακύμανση 1/5/22 ημερών (Corsi, 2009), κλασική βάση
    Ridge        Ridge παλινδρόμηση στα χαρακτηριστικά της ημέρας j (alpha=10)
    LSTM, BiLSTM, BiLSTM+Attn   ίδια αρχιτεκτονική με το προηγούμενο πείραμα, με έξοδο παλινδρόμησης

Σύνολα χαρακτηριστικών (το κύριο, «regime», δηλώνεται ΠΡΙΝ από την εκτέλεση):
    regime       (ΚΥΡΙΟ) τα default + log διακύμανση 1/5/22/60/120/252 ημερών, percentile
                 διακύμανσης, μεταβλητότητα της μεταβλητότητας, drawdown, απόσταση από SMA200,
                 τάση 60 ημερών. Όλα υπολογίζονται μόνο από παρελθοντικά δεδομένα.
    base         μόνο τα default χαρακτηριστικά του προηγούμενου πειράματος (αφαιρείται το Volume)
    har_only     μόνο οι τρεις είσοδοι του HAR (log διακύμανση 1/5/22 ημερών)
    regime_vix   regime + VIX (απαιτεί --vix αρχείο, βλ. παρακάτω)

VIX (προαιρετικό). Κατέβασέ το τοπικά, π.χ.:
    python -c "import yfinance as yf; yf.download('^VIX', start='2014-01-01', auto_adjust=True)['Close'].to_csv('data/raw/vix.csv')"
και τρέξε:   python src/models/volatility_experiment.py --features regime_vix --vix data/raw/vix.csv

Μετρικές (εκτός δείγματος, όλες οι περίοδοι δοκιμής μαζί):
    RMSE, MAE στο log-διακύμανσης, QLIKE (στην κλίμακα της διακύμανσης, με διόρθωση Jensen
    από την παραμένουσα διασπορά του validation), R² έναντι Mean, R² έναντι HAR,
    AUC για «διακύμανση > διάμεσο του train» (χρησιμοποιώντας την πρόβλεψη ως σκορ),
    και 95% block-bootstrap CI για τη μέση διαφορά τετραγωνικού σφάλματος έναντι του HAR.
    Ένα μοντέλο «ξεπερνά» το HAR μόνο αν το άνω όριο του CI της διαφοράς είναι < 0.

Εκτέλεση:
    python src/models/volatility_experiment.py
    python src/models/volatility_experiment.py --features base
    python src/models/volatility_experiment.py --shuffle-labels
    python src/models/volatility_experiment.py --quick
"""
import argparse
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    TORCH = True
except ImportError:
    TORCH = False

# --------------------------------------------------------------------------------------
# Ρυθμίσεις (δηλώνονται ΠΡΙΝ από την εκτέλεση και δεν αλλάζουν βάσει του test)
# --------------------------------------------------------------------------------------
SEQ_LEN = 20
HORIZON = 5
N_FOLDS = 4
TEST_DAYS = 252
VAL_DAYS = 252
SEEDS = [42, 100, 2024, 777, 999]
N_BOOT = 1000
BLOCK = 10
CLIP = 5.0               # μεγαλύτερο από 3: οι ακραίες τιμές διακύμανσης (κρίσεις) είναι η πληροφορία
FLOOR = 1e-6             # κάτω όριο για Σ r² πριν τον λογάριθμο
RIDGE_ALPHA = 10.0

NN_CFG = dict(hidden=32, layers=1, dropout=0.4, lr=3e-4, weight_decay=1e-4,
              batch=64, max_epochs=60, patience=10)

HAR_COLS = ["log_rv_1", "log_rv_5", "log_rv_22"]
REGIME_COLS = ["log_rv_1", "log_rv_5", "log_rv_22", "log_rv_60", "log_rv_120", "log_rv_252",
               "vol_pct", "vol_of_vol", "drawdown", "dist_sma200", "trend_60"]
VIX_COLS = ["log_vix", "vix_chg_5", "vix_rel"]


# --------------------------------------------------------------------------------------
# Δεδομένα
# --------------------------------------------------------------------------------------
def find_root():
    root = os.path.abspath(os.path.dirname(__file__))
    while not os.path.exists(os.path.join(root, "data", "processed", "sp500_processed.csv")):
        parent = os.path.dirname(root)
        if parent == root:
            raise FileNotFoundError("Δεν βρέθηκε data/processed/sp500_processed.csv")
        root = parent
    return root


def log_mean_rv(r2, w):
    """log της μέσης τετραγωνισμένης απόδοσης των τελευταίων w ημερών (γνωστή στο κλείσιμο της ημέρας j)."""
    return np.log(np.maximum(r2.rolling(w).mean(), FLOOR / 5))


def add_vol_features(df):
    """Υπολογίζει στόχο και χαρακτηριστικά regime. Όλα χρησιμοποιούν μόνο δεδομένα έως τη γραμμή j."""
    r = df["log_return"]
    r2 = r ** 2
    # Στόχος: Σ r² των ημερών j+1..j+5
    fwd = r2.rolling(HORIZON).sum().shift(-HORIZON)
    df["y_vol"] = np.log(np.maximum(fwd, FLOOR))
    # Persistence: Σ r² των 5 τελευταίων ημερών (ίδια κλίμακα με τον στόχο)
    df["persist"] = np.log(np.maximum(r2.rolling(HORIZON).sum(), FLOOR))
    for w in (1, 5, 22, 60, 120, 252):
        df[f"log_rv_{w}"] = log_mean_rv(r2, w)
    vol20 = r.rolling(20).std()
    df["vol_pct"] = vol20.rolling(252).apply(lambda a: (a[:-1] < a[-1]).mean(), raw=True)
    df["vol_of_vol"] = vol20.rolling(60).std() / vol20.rolling(60).mean()
    df["drawdown"] = df["Close"] / df["Close"].rolling(252).max() - 1
    df["dist_sma200"] = df["Close"] / df["Close"].rolling(200).mean() - 1
    df["trend_60"] = np.log(df["Close"] / df["Close"].shift(60))
    return df


def add_vix(df, path):
    v = pd.read_csv(path)
    n_before = len(v)
    v = pd.DataFrame({"Date": pd.to_datetime(v.iloc[:, 0], errors="coerce"),
                      "vix": pd.to_numeric(v.iloc[:, 1], errors="coerce")}).dropna()
    df = df.merge(v, on="Date", how="left")
    cover = df["vix"].notna().mean()
    if cover < 0.95:
        raise SystemExit(f"Το VIX ταιριάζει μόνο στο {cover:.0%} των ημερομηνιών ({n_before} γραμμές στο αρχείο). Έλεγξε το {path}.")
    df["vix"] = df["vix"].ffill().bfill()
    lv = np.log(df["vix"])
    df["log_vix"] = lv
    df["vix_chg_5"] = lv.diff(5)
    df["vix_rel"] = lv - lv.rolling(252).mean()
    return df


def make_windows(F, y, end_rows):
    X = np.stack([F[j - SEQ_LEN + 1: j + 1] for j in end_rows]).astype(np.float32)
    return X, y[end_rows].astype(np.float32)


def build_folds(n_rows, first_valid_row, n_folds=N_FOLDS):
    folds = []
    last = n_rows - 1 - HORIZON
    for k in range(n_folds):
        test_end = last - (n_folds - 1 - k) * TEST_DAYS
        test_start = test_end - TEST_DAYS + 1
        val_end = test_start - 1 - HORIZON
        val_start = val_end - VAL_DAYS + 1
        train_end = val_start - 1 - HORIZON
        if train_end - first_valid_row < 500:
            raise ValueError("Πολύ λίγα δεδομένα εκπαίδευσης για τόσες περιόδους δοκιμής.")
        folds.append(dict(train=np.arange(first_valid_row, train_end + 1),
                          val=np.arange(val_start, val_end + 1),
                          test=np.arange(test_start, test_end + 1)))
    return folds


# --------------------------------------------------------------------------------------
# Μετρικές
# --------------------------------------------------------------------------------------
def qlike(y_log, pred_log, s2):
    """QLIKE στην κλίμακα της διακύμανσης. s2: διόρθωση Jensen ώστε E[exp(·)] να μην υποεκτιμάται."""
    v = np.exp(y_log)
    f = np.exp(pred_log + 0.5 * s2)
    return v / f - np.log(v / f) - 1


def block_boot_mean(d, rng):
    n = len(d)
    nb = int(np.ceil(n / BLOCK))
    means = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        means.append(d[idx].mean())
    return np.percentile(means, [2.5, 97.5])


# --------------------------------------------------------------------------------------
# Νευρωνικά μοντέλα (παλινδρόμηση)
# --------------------------------------------------------------------------------------
if TORCH:
    class Attention(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.score = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1, bias=False))

        def forward(self, h):
            w = torch.softmax(self.score(h), dim=1)
            return (w * h).sum(dim=1)

    class SeqRegressor(nn.Module):
        def __init__(self, n_in, hidden, layers, dropout, bidirectional, attention):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
            d = hidden * (2 if bidirectional else 1)
            self.attention = Attention(d) if attention else None
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(d, 1)

        def forward(self, x):
            h, _ = self.lstm(x)
            z = self.attention(h) if self.attention is not None else h[:, -1, :]
            return self.head(self.drop(z)).squeeze(-1)

    def set_seed(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_device():
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda") * 2
                return torch.device("cuda")
            except RuntimeError:
                pass
        return torch.device("cpu")

    @torch.no_grad()
    def predict(model, X, device, bs=512):
        model.eval()
        out = [model(torch.from_numpy(X[i:i + bs]).to(device)).cpu().numpy() for i in range(0, len(X), bs)]
        return np.concatenate(out)

    def train_nn(kind, Xtr, ytr, Xva, yva, seed, device, cfg, history, tag):
        """ytr, yva: ήδη τυποποιημένοι στόχοι (μέσος/τυπ. απόκλιση του train)."""
        set_seed(seed)
        bi = kind in ("BiLSTM", "BiLSTM+Attn")
        model = SeqRegressor(Xtr.shape[2], cfg["hidden"], cfg["layers"], cfg["dropout"],
                             bidirectional=bi, attention=(kind == "BiLSTM+Attn")).to(device)
        loss_fn = nn.MSELoss()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        g = torch.Generator().manual_seed(seed)
        Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)

        best, best_state, best_epoch, wait = np.inf, None, 0, 0
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), cfg["batch"]):
                b = perm[i:i + cfg["batch"]]
                xb, yb = Xt[b].to(device), yt[b].to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tr = float(np.mean((predict(model, Xtr, device) - ytr) ** 2))
            va = float(np.mean((predict(model, Xva, device) - yva) ** 2))
            history.append(dict(**tag, model=kind, seed=seed, epoch=epoch, train_mse=tr, val_mse=va))
            if va < best:
                best, best_epoch, wait = va, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg["patience"]:
                    break
        model.load_state_dict(best_state)
        return model, best_epoch

    def train_fixed(kind, X, y, n_epochs, seed, device, cfg):
        """Επανεκπαίδευση σε train+val για σταθερό αριθμό εποχών (χωρίς early stopping)."""
        set_seed(seed)
        bi = kind in ("BiLSTM", "BiLSTM+Attn")
        model = SeqRegressor(X.shape[2], cfg["hidden"], cfg["layers"], cfg["dropout"],
                             bidirectional=bi, attention=(kind == "BiLSTM+Attn")).to(device)
        loss_fn = nn.MSELoss()
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        g = torch.Generator().manual_seed(seed)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        for _ in range(n_epochs):
            model.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), cfg["batch"]):
                b = perm[i:i + cfg["batch"]]
                xb, yb = Xt[b].to(device), yt[b].to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        return model


# --------------------------------------------------------------------------------------
# Κύριο πείραμα
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="regime", choices=["regime", "base", "har_only", "regime_vix"])
    ap.add_argument("--vix", default=None, help="CSV με στήλες Date, Close του ^VIX (για regime_vix)")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="έλεγχος ορθότητας: ανακατεύει τους στόχους του train (αναμένεται καμία βελτίωση)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default=os.path.join("data", "processed", "sp500_processed.csv"),
                    help="αρχείο δεδομένων (π.χ. data/processed/sp500_processed_long.csv)")
    ap.add_argument("--folds", type=int, default=N_FOLDS, help="αριθμός ετήσιων περιόδων δοκιμής")
    ap.add_argument("--shuffle-seed", type=int, default=0, help="σπόρος για το --shuffle-labels")
    ap.add_argument("--models", default="Mean,Persistence,HAR,Ridge,LSTM,BiLSTM,BiLSTM+Attn")
    args = ap.parse_args()

    seeds = SEEDS[:1] if args.quick else SEEDS
    cfg = dict(NN_CFG, max_epochs=5, patience=3) if args.quick else NN_CFG
    models = [m.strip() for m in args.models.split(",")]
    if not TORCH:
        dropped = [m for m in models if m in ("LSTM", "BiLSTM", "BiLSTM+Attn")]
        if dropped:
            print(f"[!] Δεν βρέθηκε PyTorch: παραλείπονται τα {dropped}")
        models = [m for m in models if m not in dropped]

    root = find_root()
    data_path = args.data if os.path.isabs(args.data) else os.path.join(root, args.data)
    df = pd.read_csv(data_path, parse_dates=["Date"])
    base_cols = [c for c in df.columns if c not in ("Date", "Close", "log_return", "target", "Volume")]
    df = add_vol_features(df)
    if args.features == "regime_vix":
        vix_path = args.vix or os.path.join("data", "raw", "vix.csv")
        if not os.path.isabs(vix_path):
            vix_path = os.path.join(root, vix_path)
        if not os.path.exists(vix_path):
            print(f"[i] Δεν βρέθηκε {vix_path}: κατεβάζω ^VIX με yfinance...")
            import yfinance as yf
            os.makedirs(os.path.dirname(vix_path), exist_ok=True)
            start = (df["Date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            vx = yf.download("^VIX", start=start, auto_adjust=True, progress=False)["Close"]
            vx.to_csv(vix_path)
        df = add_vix(df, vix_path)
        print(f"[i] VIX: {vix_path}")

    feats = {"regime": base_cols + REGIME_COLS,
             "base": base_cols,
             "har_only": HAR_COLS,
             "regime_vix": base_cols + REGIME_COLS + VIX_COLS}[args.features]
    tag = ("vol_" + args.features + (f"_f{args.folds}" if args.folds != N_FOLDS else "")
           + ("_long" if "long" in os.path.basename(args.data) else "")
           + (f"_shuffled{args.shuffle_seed}" if args.shuffle_labels else ""))

    # Κοινή πρώτη έγκυρη γραμμή (ίδια για όλα τα σύνολα ώστε τα folds να είναι τα ίδια)
    need = REGIME_COLS + ["persist"]
    first_ok = int(df[need].dropna().index[0])
    first_valid = max(first_ok, 0) + SEQ_LEN - 1

    F_raw = df[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill().values.astype(np.float64)
    y_all = df["y_vol"].values.astype(np.float64)
    persist_all = df["persist"].values.astype(np.float64)
    har_all = df[HAR_COLS].values.astype(np.float64)
    folds = build_folds(len(df), first_valid, args.folds)
    device = get_device() if TORCH else None

    print("=" * 78)
    print(f"ΜΕΤΑΒΛΗΤΟΤΗΤΑ 5 ΗΜΕΡΩΝ | features={args.features} ({len(feats)}) | seeds={len(seeds)} | device={device}")
    print(f"Στόχος: log Σ r² των επόμενων {HORIZON} ημερών | πρώτη έγκυρη γραμμή {first_valid} ({df['Date'].iloc[first_valid].date()})")
    for k, f in enumerate(folds, 1):
        d = lambda r: df["Date"].iloc[r].date()
        print(f"  Fold {k}: train {d(f['train'][0])}→{d(f['train'][-1])} | "
              f"val {d(f['val'][0])}→{d(f['val'][-1])} | test {d(f['test'][0])}→{d(f['test'][-1])}")
    print("=" * 78)

    rng = np.random.default_rng(args.shuffle_seed)
    fold_rows, history = [], []
    P = {m: [] for m in models}                 # προβλέψεις ανά μοντέλο (test, συνενωμένες)
    Y, MEAN_F, MED, S2 = [], [], [], {m: [] for m in models}

    # Πρωτόκολλο v2 (συμμετρικό): το validation χρησιμοποιείται ΜΟΝΟ για επιλογή υπερπαραμέτρων
    # (αριθμός εποχών). Μετά ΟΛΑ τα μοντέλα (Mean, HAR, Ridge, νευρωνικά) επανεκπαιδεύονται
    # σε train+val και προβλέπουν το test. Έτσι κανένα μοντέλο δεν έχει πιο πρόσφατη πληροφορία
    # από τα άλλα. Στο --shuffle-labels ανακατεύονται οι στόχοι ΚΑΙ του train ΚΑΙ του val.
    for k, f in enumerate(folds, 1):
        fit_rows = np.concatenate([f["train"], f["val"]])
        y_lab = y_all.copy()
        if args.shuffle_labels:
            y_lab[fit_rows] = rng.permutation(y_lab[fit_rows])
        yva_lab = y_lab[f["val"]]                     # στόχοι val όπως τους «βλέπει» η επιλογή εποχών
        yte_true = y_all[f["test"]]
        mu, sd = y_lab[f["train"]].mean(), y_lab[f["train"]].std()
        mu2, sd2 = y_lab[fit_rows].mean(), y_lab[fit_rows].std()
        Y.append(yte_true); MEAN_F.append(np.full(len(yte_true), mu2))
        MED.append(np.full(len(yte_true), np.median(y_lab[fit_rows])))

        # Στάδιο 1: κλίμακα από το train, επιλογή εποχών στο val
        sc1 = StandardScaler().fit(F_raw[f["train"]])
        F1 = np.clip(sc1.transform(F_raw), -CLIP, CLIP).astype(np.float32)
        Xtr, ytr_s = make_windows(F1, ((y_lab - mu) / sd).astype(np.float32), f["train"])
        Xva, yva_s = make_windows(F1, ((y_lab - mu) / sd).astype(np.float32), f["val"])
        # Στάδιο 2: κλίμακα από train+val, τελική εκπαίδευση, πρόβλεψη test
        sc2 = StandardScaler().fit(F_raw[fit_rows])
        F2 = np.clip(sc2.transform(F_raw), -CLIP, CLIP).astype(np.float32)
        Xfit, yfit_s = make_windows(F2, ((y_lab - mu2) / sd2).astype(np.float32), fit_rows)
        Xte, _ = make_windows(F2, np.zeros(len(y_all), dtype=np.float32), f["test"])

        for m in models:
            t0 = time.time(); info = ""
            if m == "Mean":
                p_va = np.full(len(yva_lab), mu); p_te = np.full(len(yte_true), mu2)
            elif m == "Persistence":
                p_va = persist_all[f["val"]]; p_te = persist_all[f["test"]]
            elif m == "HAR":
                p_va = LinearRegression().fit(har_all[f["train"]], y_lab[f["train"]]).predict(har_all[f["val"]])
                p_te = LinearRegression().fit(har_all[fit_rows], y_lab[fit_rows]).predict(har_all[f["test"]])
            elif m == "Ridge":
                p_va = Ridge(alpha=RIDGE_ALPHA).fit(Xtr[:, -1, :], y_lab[f["train"]]).predict(Xva[:, -1, :])
                p_te = Ridge(alpha=RIDGE_ALPHA).fit(Xfit[:, -1, :], y_lab[fit_rows]).predict(Xte[:, -1, :])
            else:
                pv, pt, epochs = [], [], []
                for s in seeds:
                    model, ep = train_nn(m, Xtr, ytr_s, Xva, yva_s, s, device, cfg, history, dict(fold=k))
                    pv.append(predict(model, Xva, device) * sd + mu)
                    final = train_fixed(m, Xfit, yfit_s, ep, s, device, cfg)
                    pt.append(predict(final, Xte, device) * sd2 + mu2)
                    epochs.append(ep)
                p_va, p_te = np.mean(pv, axis=0), np.mean(pt, axis=0)
                info = f"| εποχές {epochs}"
            yva_true = yva_lab
            P[m].append(p_te)
            S2[m].append(np.full(len(yte_true), np.var(yva_true - p_va)))
            rm = np.sqrt(np.mean((yte_true - p_te) ** 2))
            print(f"Fold {k} | {m:12s} | val RMSE {np.sqrt(np.mean((yva_true - p_va) ** 2)):.3f} "
                  f"| test RMSE {rm:.3f} | {time.time() - t0:5.1f}s {info}")
            fold_rows.append(dict(fold=k, model=m, val_RMSE=np.sqrt(np.mean((yva_true - p_va) ** 2)), test_RMSE=rm,
                                  test_MAE=np.mean(np.abs(yte_true - p_te))))

    y = np.concatenate(Y)
    mean_f = np.concatenate(MEAN_F)
    med = np.concatenate(MED)
    pooled = {m: np.concatenate(P[m]) for m in models}
    s2 = {m: np.concatenate(S2[m]) for m in models}
    sse_mean = np.sum((y - mean_f) ** 2)
    har_se = (y - pooled["HAR"]) ** 2 if "HAR" in pooled else None
    fold_idx = np.concatenate([np.full(len(a), i) for i, a in enumerate(Y)])
    thr_high = np.concatenate([np.full(len(a), np.median(y_all[np.concatenate([f["train"], f["val"]])]))
                               for a, f in zip(Y, folds)])
    y_bin = (y > thr_high).astype(int)

    summary = []
    for m in models:
        p = pooled[m]
        se = (y - p) ** 2
        row = dict(model=m, n_test=len(y), RMSE=np.sqrt(se.mean()), MAE=np.mean(np.abs(y - p)),
                   QLIKE=qlike(y, p, s2[m]).mean(), R2_vs_mean=1 - se.sum() / sse_mean,
                   AUC_high_vol=(0.5 if m == "Mean" else roc_auc_score(y_bin, p)))   # Mean: σταθερή πρόβλεψη -> 0,5
        if har_se is not None:
            row["R2_vs_HAR"] = 1 - se.sum() / har_se.sum()
            if m != "HAR":
                d = se - har_se
                # Έλεγχος σε επίπεδο fold: κάθε περίοδος δοκιμής = μία ανεξάρτητη παρατήρηση.
                # Συντηρητικότερος από το block bootstrap, γιατί λαμβάνει υπόψη ότι ένα τυχαίο
                # «σήκωμα» του επιπέδου πρόβλεψης επηρεάζει ολόκληρη την περίοδο.
                fd = np.array([d[fold_idx == i].mean() for i in range(len(folds))])
                fold_t_p = float(stats.ttest_1samp(fd, 0, alternative="less").pvalue) if len(fd) > 2 else np.nan
                fold_sign_p = float(stats.binomtest(int((fd < 0).sum()), len(fd), 0.5, alternative="greater").pvalue)
                lo, hi = block_boot_mean(d, rng)
                row.update(dMSE_vs_HAR=d.mean(), dMSE_lo=lo, dMSE_hi=hi,
                           folds_better_than_HAR=f"{sum(fd < 0)}/{len(fd)}",
                           fold_t_p=fold_t_p, fold_sign_p=fold_sign_p,
                           verdict=("καλύτερο από HAR" if (hi < 0 and fold_t_p < 0.05) else "ένδειξη μόνο (όχι σε επίπεδο fold)" if hi < 0 else "χειρότερο από HAR" if lo > 0 else "δεν διακρίνεται από HAR"))
        if "Ridge" in pooled and m not in ("Ridge", "Mean", "Persistence", "HAR"):
            d2 = se - (y - pooled["Ridge"]) ** 2
            lo2, hi2 = block_boot_mean(d2, rng)
            row.update(dMSE_vs_Ridge=d2.mean(), dR_lo=lo2, dR_hi=hi2,
                       verdict_vs_Ridge=("καλύτερο από Ridge" if hi2 < 0 else "χειρότερο από Ridge" if lo2 > 0
                                         else "δεν διακρίνεται από Ridge"))
        summary.append(row)

    np.savez(os.path.join(root, "results", f"wf_{tag}_predictions.npz"), y=y, fold=fold_idx,
             **{m.replace("+", "_"): pooled[m] for m in models})
    out = os.path.join(root, "results", "tables"); os.makedirs(out, exist_ok=True)
    pd.DataFrame(fold_rows).round(4).to_csv(os.path.join(out, f"wf_{tag}_folds.csv"), index=False)
    S = pd.DataFrame(summary).round(4)
    S.to_csv(os.path.join(out, f"wf_{tag}_summary.csv"), index=False)
    if history:
        pd.DataFrame(history).round(4).to_csv(os.path.join(out, f"wf_{tag}_history.csv"), index=False)

    print("\n" + "=" * 78)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΑ (όλες οι περίοδοι δοκιμής, εκτός δείγματος)")
    print("=" * 78)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(S.to_string(index=False))
    print("\nΕρμηνεία: «καλύτερο από HAR» απαιτεί ΚΑΙ άνω όριο bootstrap CI < 0 ΚΑΙ fold_t_p < 0,05")
    print("(t-test στις διαφορές MSE ανά περίοδο δοκιμής). Με 4 folds ο έλεγχος είναι αδύναμος· χρειάζονται ~15+.")
    print("Αν R2_vs_mean είναι ≤ 0 για όλα, δεν υπάρχει προβλεψιμότητα· αν το Persistence/HAR είναι")
    print("ήδη καλά και τα νευρωνικά δεν τα ξεπερνούν, το βαθύ μοντέλο δεν προσθέτει τίποτα.")
    print(f"Αρχεία: {out}")


if __name__ == "__main__":
    main()
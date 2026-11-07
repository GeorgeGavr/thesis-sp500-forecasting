"""
Walk-forward πείραμα: Bi-LSTM που εκτιμά την ΚΑΤΑΝΟΜΗ της μεταβολής της τιμής του S&P 500
στις επόμενες 5 ημέρες (μέση τιμή μ και τυπική απόκλιση σ), με είσοδο τις 20 τελευταίες ημέρες.

Στόχος:  r5 = Σ log_return των ημερών j+1..j+5   (το παράθυρο που τελειώνει στη γραμμή j)
Έξοδος:  N(μ, σ²). Εκπαίδευση με Gaussian negative log-likelihood (NLL).

Μοντέλα:
    Const           μ, σ = μέσος όρος και τυπ. απόκλιση των r5 στο train (σταθερή κατανομή)
    HARsigma        μ = σταθερό, σ από το HAR μοντέλο διακύμανσης (log Σr² 1/5/22 ημερών)
    Ridge+HARsigma  μ από Ridge στα χαρακτηριστικά της ημέρας j, σ από το HAR
    LSTM, BiLSTM, BiLSTM+Attn   δύο έξοδοι (μ, log σ²), ίδια αρχιτεκτονική με το volatility_experiment.py

Πρωτόκολλο: ίδιο με το volatility_experiment.py (πρωτόκολλο v2): ίδια folds, embargo, σπόροι,
validation μόνο για επιλογή εποχών, μετά επανεκπαίδευση σε train+val για ΟΛΑ τα μοντέλα.

Μετρικές (εκτός δείγματος, όλες οι περίοδοι δοκιμής μαζί):
    ΜΕΤΑΒΟΛΗ ΤΙΜΗΣ (μ)   R²_mu_vs_const (Campbell–Thompson), έλεγχος σε επίπεδο fold (t-test),
                         ποσοστό σωστού προσήμου, AUC του μ για «r5 > 0» με block-bootstrap CI
    ΚΑΤΑΝΟΜΗ (μ και σ)   NLL, CRPS, κάλυψη διαστημάτων 80% και 95%, std των τυποποιημένων σφαλμάτων
                         (πρέπει να είναι ≈ 1), διαφορά NLL έναντι Const και έναντι HARsigma
Το «ΝΙΚΗ» για τη μεταβλητότητα κρίνεται από το dNLL έναντι HARsigma: αν το δίκτυο προσθέτει
πληροφορία πέρα από το HAR, το dNLL είναι < 0 (CI < 0 ΚΑΙ fold_t_p < 0,05).

Έκδοση v2 (προστέθηκε μετά το πρώτο τρέξιμο, δηλώνεται ΠΡΙΝ από την εκτέλεσή της):
    --sigma-target rv  Η σ των δικτύων εκπαιδεύεται στη ρεαλιζόμενη διακύμανση log Σr² των 5 ημερών
                       (όπως ήδη το HARsigma), όχι στη μία τιμή r5. Διορθώνει μια ασυμμετρία του v1:
                       το HAR έβλεπε 5 ημερήσιες αποδόσεις ανά στόχο, το δίκτυο μόνο το άθροισμά τους.
    --dist t           Αξιολόγηση με Student-t (ίδια μέση τιμή/διασπορά) αντί Gaussian. Το ν
                       εκτιμάται ανά fold μόνο από train+val και είναι ΙΔΙΟ για όλα τα μοντέλα.
    Κύρια σύγκριση v2: BiLSTM έναντι HARsigma στο NLL· «καλύτερο» αν CI < 0 ΚΑΙ fold_t_p < 0,05.

Εκτέλεση:
    python src/models/return_distribution_experiment.py --data data/processed/sp500_processed_long.csv --folds 20 --sigma-target rv --dist t
    python src/models/return_distribution_experiment.py --data data/processed/sp500_processed_long.csv --folds 20 --features regime
    python src/models/return_distribution_experiment.py --data data/processed/sp500_processed_long.csv --folds 20 --features regime --shuffle-labels
    python src/models/return_distribution_experiment.py --data data/processed/sp500_processed_long.csv --features regime_vix --folds 20
    python src/models/return_distribution_experiment.py --legacy-split     # ένα test στο τελευταίο 10%, όπως στο πρώτο Bi-LSTM
    python src/models/return_distribution_experiment.py --quick
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import volatility_experiment as ve  # noqa: E402

warnings.filterwarnings("ignore")

HORIZON, SEQ_LEN, SEEDS = ve.HORIZON, ve.SEQ_LEN, ve.SEEDS
CLIP, NN_CFG = ve.CLIP, ve.NN_CFG
RIDGE_ALPHA_RET = 100.0      # ισχυρότερο shrinkage από τη μεταβλητότητα: τα σήματα στην απόδοση είναι ασθενή
LOGVAR_CLIP = 6.0
NN_KINDS = ("LSTM", "BiLSTM", "BiLSTM+Attn")
NO_MU = ("Const", "HARsigma")   # μ σταθερό ανά fold: το AUC/ποσοστό προσήμου δεν έχει νόημα

if ve.TORCH:
    import torch
    import torch.nn as nn

    class DistNet(nn.Module):
        def __init__(self, n_in, hidden, layers, dropout, bidirectional, attention):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
            d = hidden * (2 if bidirectional else 1)
            self.attention = ve.Attention(d) if attention else None
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(d, 2)

        def forward(self, x):
            h, _ = self.lstm(x)
            z = self.attention(h) if self.attention is not None else h[:, -1, :]
            o = self.head(self.drop(z))
            return o[:, 0], o[:, 1].clamp(-LOGVAR_CLIP, LOGVAR_CLIP)

    def nll_loss(mu, lv, y, mode):
        """y: (N,2) = [τυποποιημένη r5, τυποποιημένη log RV].
        nll: Gaussian NLL μόνο στην r5 (η 2η έξοδος = log σ²).
        rv : MSE(μ, r5) + MSE(2η έξοδος, log RV) — η σ μαθαίνεται από τη ρεαλιζόμενη διακύμανση."""
        if mode == "rv":
            return ((mu - y[:, 0]) ** 2).mean() + ((lv - y[:, 1]) ** 2).mean()
        return (0.5 * (lv + (y[:, 0] - mu) ** 2 * torch.exp(-lv))).mean()

    def make_model(kind, n_in, cfg, device):
        return DistNet(n_in, cfg["hidden"], cfg["layers"], cfg["dropout"],
                       bidirectional=kind in ("BiLSTM", "BiLSTM+Attn"),
                       attention=(kind == "BiLSTM+Attn")).to(device)

    @torch.no_grad()
    def predict2(model, X, device, bs=512):
        model.eval()
        mus, lvs = [], []
        for i in range(0, len(X), bs):
            m, l = model(torch.from_numpy(X[i:i + bs]).to(device))
            mus.append(m.cpu().numpy()); lvs.append(l.cpu().numpy())
        return np.concatenate(mus), np.concatenate(lvs)

    def _epoch(model, Xt, yt, opt, g, cfg, device):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            xb, yb = Xt[b].to(device), yt[b].to(device)
            opt.zero_grad()
            m, l = model(xb)
            nll_loss(m, l, yb, cfg["sigma_target"]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    def np_nll(mu, lv, y, mode):
        if mode == "rv":
            return float(np.mean((mu - y[:, 0]) ** 2) + np.mean((lv - y[:, 1]) ** 2))
        return float(np.mean(0.5 * (lv + (y[:, 0] - mu) ** 2 * np.exp(-lv))))

    def train_dist(kind, Xtr, ytr, Xva, yva, seed, device, cfg, history, tag):
        ve.set_seed(seed)
        model = make_model(kind, Xtr.shape[2], cfg, device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        g = torch.Generator().manual_seed(seed)
        Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
        best, best_state, best_epoch, wait = np.inf, None, 0, 0
        for epoch in range(1, cfg["max_epochs"] + 1):
            _epoch(model, Xt, yt, opt, g, cfg, device)
            tr = np_nll(*predict2(model, Xtr, device), ytr, cfg["sigma_target"])
            va = np_nll(*predict2(model, Xva, device), yva, cfg["sigma_target"])
            history.append(dict(**tag, model=kind, seed=seed, epoch=epoch, train_nll=tr, val_nll=va))
            if va < best:
                best, best_epoch, wait = va, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg["patience"]:
                    break
        model.load_state_dict(best_state)
        return model, best_epoch

    def train_dist_fixed(kind, X, y, n_epochs, seed, device, cfg):
        ve.set_seed(seed)
        model = make_model(kind, X.shape[2], cfg, device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        g = torch.Generator().manual_seed(seed)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        for _ in range(n_epochs):
            _epoch(model, Xt, yt, opt, g, cfg, device)
        return model


# --------------------------------------------------------------------------------------
# Βοηθητικά
# --------------------------------------------------------------------------------------
def gauss_nll(y, mu, sig):
    return 0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * ((y - mu) / sig) ** 2


def std_t_logpdf(z, nu):
    """log-πυκνότητα Student-t με ν βαθμούς ελευθερίας και ΜΟΝΑΔΙΑΙΑ διασπορά."""
    s = np.sqrt((nu - 2) / nu)
    return stats.t.logpdf(z / s, nu) - np.log(s)


def fit_nu(z):
    """ML εκτίμηση του ν από τυποποιημένα σφάλματα (μόνο δεδομένα train+val του fold)."""
    res = minimize_scalar(lambda nu: -std_t_logpdf(z, nu).sum(), bounds=(2.1, 60.0), method="bounded")
    return float(res.x)


def score_nll(y, mu, sig, nu=None):
    """NLL ανά παρατήρηση: Gaussian (nu=None) ή Student-t με ίδια μέση τιμή και διασπορά."""
    if nu is None:
        return gauss_nll(y, mu, sig)
    return -(std_t_logpdf((y - mu) / sig, nu) - np.log(sig))


def z_quantile(p, nu=None):
    """Ποσοστημόριο της τυποποιημένης κατανομής (μονάδες σ) για κεντρικό διάστημα πιθανότητας p."""
    q = 0.5 + p / 2
    if nu is None:
        return stats.norm.ppf(q)
    return stats.t.ppf(q, nu) * np.sqrt((nu - 2) / nu)


def gauss_crps(y, mu, sig):
    z = (y - mu) / sig
    return sig * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))


def mix_ensemble(mus, lvs, sd, mu0):
    """Συνδυασμός σπόρων ως μείξη Gaussian: διασπορά = E[σ²+μ²] − (E μ)². Επιστρέφει (μ, σ) στην κλίμακα απόδοσης."""
    mus, var = np.asarray(mus), np.exp(np.asarray(lvs))
    m = mus.mean(0)
    v = (var + mus ** 2).mean(0) - m ** 2
    return m * sd + mu0, np.sqrt(np.maximum(v, 1e-12)) * sd


def block_boot_auc(y_bin, score, rng):
    n = len(y_bin)
    nb = int(np.ceil(n / ve.BLOCK))
    out = []
    for _ in range(ve.N_BOOT):
        starts = rng.integers(0, n - ve.BLOCK + 1, nb)
        idx = np.concatenate([np.arange(s, s + ve.BLOCK) for s in starts])[:n]
        yb = y_bin[idx]
        if yb.min() != yb.max():
            out.append(roc_auc_score(yb, score[idx]))
    return np.percentile(out, [2.5, 97.5])


def compare(d, fold_idx, n_folds, rng):
    """d: διαφορά σφάλματος/NLL ανά παρατήρηση (αρνητικό = καλύτερο το μοντέλο)."""
    fd = np.array([d[fold_idx == i].mean() for i in range(n_folds)])
    t_p = float(stats.ttest_1samp(fd, 0, alternative="less").pvalue) if n_folds > 2 else np.nan
    sign_p = float(stats.binomtest(int((fd < 0).sum()), n_folds, 0.5, alternative="greater").pvalue)
    lo, hi = ve.block_boot_mean(d, rng)
    return dict(mean=d.mean(), lo=lo, hi=hi, better=f"{int((fd < 0).sum())}/{n_folds}", t_p=t_p, sign_p=sign_p)


def verdict(c, name):
    ok_t = np.isnan(c["t_p"]) or c["t_p"] < 0.05
    if c["hi"] < 0 and ok_t:
        return f"καλύτερο από {name}"
    if c["hi"] < 0:
        return "ένδειξη μόνο (όχι σε επίπεδο fold)"
    if c["lo"] > 0:
        return f"χειρότερο από {name}"
    return f"δεν διακρίνεται από {name}"


def legacy_fold(n_rows, first_valid):
    """Ένα test στο τελευταίο 10% των γραμμών (όπως το πρώτο Bi-LSTM), με embargo."""
    test_start, val_start = int(n_rows * 0.9), int(n_rows * 0.8)
    test_end = n_rows - 1 - HORIZON
    val_end = test_start - 1 - HORIZON
    train_end = val_start - 1 - HORIZON
    return [dict(train=np.arange(first_valid, train_end + 1),
                 val=np.arange(val_start, val_end + 1),
                 test=np.arange(test_start, test_end + 1))]


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="regime", choices=["regime", "base", "regime_vix"])
    ap.add_argument("--vix", default=None)
    ap.add_argument("--shuffle-labels", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--legacy-split", action="store_true", help="ένα test στο τελευταίο 10% (80/10/10), όπως το πρώτο Bi-LSTM")
    ap.add_argument("--data", default=os.path.join("data", "processed", "sp500_processed.csv"))
    ap.add_argument("--folds", type=int, default=ve.N_FOLDS)
    ap.add_argument("--models", default="Const,HARsigma,Ridge+HARsigma,LSTM,BiLSTM,BiLSTM+Attn")
    ap.add_argument("--sigma-target", default="nll", choices=["nll", "rv"],
                    help="nll: σ από Gaussian NLL στην r5 (v1) · rv: σ εκπαιδευμένη στη ρεαλιζόμενη "
                         "διακύμανση log Σr², όπως ήδη το HARsigma (v2)")
    ap.add_argument("--dist", default="gauss", choices=["gauss", "t"],
                    help="κατανομή αξιολόγησης· t = Student-t με ν εκτιμημένο ανά fold (ίδιο για όλα τα μοντέλα)")
    args = ap.parse_args()

    seeds = SEEDS[:1] if args.quick else SEEDS
    cfg = dict(NN_CFG, max_epochs=5, patience=3) if args.quick else dict(NN_CFG)
    cfg["sigma_target"] = args.sigma_target
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not ve.TORCH:
        dropped = [m for m in models if m in NN_KINDS]
        if dropped:
            print(f"[!] Δεν βρέθηκε PyTorch: παραλείπονται τα {dropped}")
        models = [m for m in models if m not in dropped]
    for need in ("Const", "HARsigma"):
        if need not in models:
            models.insert(0, need)          # οι βάσεις αναφοράς είναι απαραίτητες για τα τεστ

    root = ve.find_root()
    data_path = args.data if os.path.isabs(args.data) else os.path.join(root, args.data)
    df = pd.read_csv(data_path, parse_dates=["Date"])
    base_cols = [c for c in df.columns if c not in ("Date", "Close", "log_return", "target", "Volume")]
    df = ve.add_vol_features(df)
    if args.features == "regime_vix":
        default_vix = "vix_long.csv" if "long" in os.path.basename(args.data) else "vix.csv"
        vix_path = args.vix or os.path.join("data", "raw", default_vix)
        if not os.path.isabs(vix_path):
            vix_path = os.path.join(root, vix_path)
        if not os.path.exists(vix_path):
            print(f"[i] Δεν βρέθηκε {vix_path}: κατεβάζω ^VIX με yfinance...")
            import yfinance as yf
            os.makedirs(os.path.dirname(vix_path), exist_ok=True)
            start = (df["Date"].min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            yf.download("^VIX", start=start, auto_adjust=True, progress=False)["Close"].to_csv(vix_path)
        df = ve.add_vix(df, vix_path)

    # Στόχος: συνολική log απόδοση των ημερών j+1..j+5
    df["y_ret"] = df["log_return"].rolling(HORIZON).sum().shift(-HORIZON)

    feats = {"regime": base_cols + ve.REGIME_COLS, "base": base_cols,
             "regime_vix": base_cols + ve.REGIME_COLS + ve.VIX_COLS}[args.features]
    tag = ("ret_" + args.features + ("_legacy" if args.legacy_split else f"_f{args.folds}")
           + ("_long" if "long" in os.path.basename(args.data) else "")
           + ("_rv" if args.sigma_target == "rv" else "") + ("_t" if args.dist == "t" else "")
           + (f"_shuffled{args.shuffle_seed}" if args.shuffle_labels else ""))

    first_ok = int(df[ve.REGIME_COLS + ["persist"]].dropna().index[0])
    first_valid = first_ok + SEQ_LEN - 1
    F_raw = df[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill().values.astype(np.float64)
    ret_all = df["y_ret"].values.astype(np.float64)
    vol_all = df["y_vol"].values.astype(np.float64)
    har_all = df[ve.HAR_COLS].values.astype(np.float64)
    folds = legacy_fold(len(df), first_valid) if args.legacy_split else ve.build_folds(len(df), first_valid, args.folds)
    n_folds = len(folds)
    device = ve.get_device() if ve.TORCH else None

    print("=" * 80)
    print(f"ΚΑΤΑΝΟΜΗ ΑΠΟΔΟΣΗΣ 5 ΗΜΕΡΩΝ | features={args.features} ({len(feats)}) | παράθυρο={SEQ_LEN} ημέρες | "
          f"seeds={len(seeds)} | device={device}")
    print(f"σ-στόχος δικτύων: {args.sigma_target} | κατανομή αξιολόγησης: {args.dist}")
    for k, f in enumerate(folds, 1):
        d = lambda r: df["Date"].iloc[r].date()
        print(f"  Fold {k}: train {d(f['train'][0])}→{d(f['train'][-1])} | val {d(f['val'][0])}→{d(f['val'][-1])} "
              f"| test {d(f['test'][0])}→{d(f['test'][-1])}")
    print("=" * 80)

    rng = np.random.default_rng(args.shuffle_seed)
    fold_rows, history = [], []
    Y, MEAN_C, NU = [], [], []
    MU = {m: [] for m in models}
    SIG = {m: [] for m in models}

    for k, f in enumerate(folds, 1):
        fit_rows = np.concatenate([f["train"], f["val"]])
        yr, yv = ret_all.copy(), vol_all.copy()
        if args.shuffle_labels:      # ίδια μετάθεση σε απόδοση και διακύμανση: καταστρέφει κάθε σχέση με τα features
            perm = rng.permutation(len(fit_rows))
            yr[fit_rows] = yr[fit_rows][perm]
            yv[fit_rows] = yv[fit_rows][perm]
        yte = ret_all[f["test"]]
        mu1, sd1 = yr[f["train"]].mean(), yr[f["train"]].std()
        mu2, sd2 = yr[fit_rows].mean(), yr[fit_rows].std()
        mv1, sv1 = yv[f["train"]].mean(), yv[f["train"]].std()
        mv2, sv2 = yv[fit_rows].mean(), yv[fit_rows].std()
        Y.append(yte); MEAN_C.append(np.full(len(yte), mu2))

        def targets2(mr, sr, mvv, svv):
            """(N,2): τυποποιημένη r5 και τυποποιημένη log RV (NaN στις τελευταίες γραμμές δεν χρησιμοποιούνται)."""
            return np.stack([(yr - mr) / sr, (yv - mvv) / svv], axis=1).astype(np.float32)

        sc1 = StandardScaler().fit(F_raw[f["train"]])
        F1 = np.clip(sc1.transform(F_raw), -CLIP, CLIP).astype(np.float32)
        T1 = targets2(mu1, sd1, mv1, sv1)
        Xtr, ytr_s = ve.make_windows(F1, T1, f["train"])
        Xva, yva_s = ve.make_windows(F1, T1, f["val"])
        sc2 = StandardScaler().fit(F_raw[fit_rows])
        F2 = np.clip(sc2.transform(F_raw), -CLIP, CLIP).astype(np.float32)
        Xfit, yfit_s = ve.make_windows(F2, targets2(mu2, sd2, mv2, sv2), fit_rows)
        Xte, _ = ve.make_windows(F2, np.zeros(len(ret_all), dtype=np.float32), f["test"])
        yva = yr[f["val"]]

        def har_sigma(train_rows, pred_rows):
            lr = LinearRegression().fit(har_all[train_rows], yv[train_rows])
            res = yv[train_rows] - lr.predict(har_all[train_rows])
            # E[Σr²] = exp(E[log Σr²] + ½ Var): διόρθωση Jensen από τα υπόλοιπα του fit
            return np.sqrt(np.exp(lr.predict(har_all[pred_rows]) + 0.5 * res.var()))

        # ν της Student-t: εκτιμάται ΜΟΝΟ από train+val, από τα τυποποιημένα σφάλματα του HARsigma,
        # και χρησιμοποιείται ίδιο για ΟΛΑ τα μοντέλα του fold (δίκαιη σύγκριση).
        if args.dist == "t":
            nu_k = fit_nu((yr[fit_rows] - mu2) / har_sigma(fit_rows, fit_rows))
            nu_val = fit_nu((yr[f["train"]] - mu1) / har_sigma(f["train"], f["train"]))
        else:
            nu_k = nu_val = None
        NU.append(np.full(len(yte), np.nan if nu_k is None else nu_k))

        for m in models:
            t0 = time.time(); info = ""
            if m == "Const":
                mv, sv = np.full(len(yva), mu1), np.full(len(yva), sd1)
                mt, st = np.full(len(yte), mu2), np.full(len(yte), sd2)
            elif m == "HARsigma":
                mv, sv = np.full(len(yva), mu1), har_sigma(f["train"], f["val"])
                mt, st = np.full(len(yte), mu2), har_sigma(fit_rows, f["test"])
            elif m == "Ridge+HARsigma":
                mv = Ridge(alpha=RIDGE_ALPHA_RET).fit(Xtr[:, -1, :], yr[f["train"]]).predict(Xva[:, -1, :])
                mt = Ridge(alpha=RIDGE_ALPHA_RET).fit(Xfit[:, -1, :], yr[fit_rows]).predict(Xte[:, -1, :])
                sv, st = har_sigma(f["train"], f["val"]), har_sigma(fit_rows, f["test"])
            else:
                pv, pt, epochs = [], [], []
                for s in seeds:
                    model, ep = train_dist(m, Xtr, ytr_s, Xva, yva_s, s, device, cfg, history, dict(fold=k))
                    pv.append(predict2(model, Xva, device))
                    final = train_dist_fixed(m, Xfit, yfit_s, ep, s, device, cfg)
                    pt.append(predict2(final, Xte, device))
                    epochs.append(ep)
                if args.sigma_target == "rv":
                    # μ: μέσος όρος σπόρων· σ: από την πρόβλεψη log RV με διόρθωση Jensen, όπου η
                    # διασπορά υπολοίπων εκτιμάται στο validation (όπως στο volatility_experiment.py)
                    mv = np.mean([a for a, _ in pv], axis=0) * sd1 + mu1
                    mt = np.mean([a for a, _ in pt], axis=0) * sd2 + mu2
                    lrv_v = np.mean([b for _, b in pv], axis=0) * sv1 + mv1
                    lrv_t = np.mean([b for _, b in pt], axis=0) * sv2 + mv2
                    s2v = np.var(yv[f["val"]] - lrv_v)
                    sv, st = np.sqrt(np.exp(lrv_v + 0.5 * s2v)), np.sqrt(np.exp(lrv_t + 0.5 * s2v))
                else:
                    mv, sv = mix_ensemble([a for a, _ in pv], [b for _, b in pv], sd1, mu1)
                    mt, st = mix_ensemble([a for a, _ in pt], [b for _, b in pt], sd2, mu2)
                info = f"| εποχές {epochs}"
            MU[m].append(mt); SIG[m].append(st)
            nv, nt = score_nll(yva, mv, sv, nu_val).mean(), score_nll(yte, mt, st, nu_k).mean()
            if m == models[0] and nu_k is not None:
                print(f"Fold {k:2d} | ν (Student-t, από train+val) = {nu_k:.2f}")
            print(f"Fold {k:2d} | {m:15s} | val NLL {nv:7.3f} | test NLL {nt:7.3f} | "
                  f"test RMSE(μ) {np.sqrt(np.mean((yte - mt) ** 2)):.4f} | {time.time() - t0:5.1f}s {info}")
            fold_rows.append(dict(fold=k, model=m, val_NLL=nv, test_NLL=nt,
                                  test_RMSE_mu=np.sqrt(np.mean((yte - mt) ** 2))))

    # ---------------- Συγκεντρωτικά ----------------
    y = np.concatenate(Y)
    mean_c = np.concatenate(MEAN_C)
    fold_idx = np.concatenate([np.full(len(a), i) for i, a in enumerate(Y)])
    mu = {m: np.concatenate(MU[m]) for m in models}
    sig = {m: np.concatenate(SIG[m]) for m in models}
    nu_all = np.concatenate(NU) if args.dist == "t" else None
    nll = {m: score_nll(y, mu[m], sig[m], nu_all) for m in models}
    y_up = (y > 0).astype(int)
    se_c = (y - mean_c) ** 2
    q80, q95 = z_quantile(0.80, nu_all), z_quantile(0.95, nu_all)

    summary = []
    for m in models:
        z = (y - mu[m]) / sig[m]
        row = dict(model=m, n_test=len(y),
                   R2_mu_vs_const=1 - ((y - mu[m]) ** 2).sum() / se_c.sum(),
                   NLL=nll[m].mean(), CRPS_gauss=gauss_crps(y, mu[m], sig[m]).mean(),
                   cov80=np.mean(np.abs(z) < q80), cov95=np.mean(np.abs(z) < q95), z_std=z.std())
        if m in NO_MU:
            row.update(hit_rate=np.nan, base_up=y_up.mean(), AUC_up=0.5, AUC_lo=np.nan, AUC_hi=np.nan)
        else:
            lo, hi = block_boot_auc(y_up, mu[m], rng)
            row.update(hit_rate=np.mean((mu[m] > 0) == (y > 0)), base_up=y_up.mean(),
                       AUC_up=roc_auc_score(y_up, mu[m]), AUC_lo=lo, AUC_hi=hi)
            c = compare((y - mu[m]) ** 2 - se_c, fold_idx, n_folds, rng)
            row.update(muSE_folds_better=c["better"], mu_t_p=c["t_p"], mu_verdict=verdict(c, "σταθερό μ"))
        if m != "Const":
            c = compare(nll[m] - nll["Const"], fold_idx, n_folds, rng)
            row.update(dNLL_vs_Const=c["mean"], dNLL_C_lo=c["lo"], dNLL_C_hi=c["hi"])
        if m not in ("Const", "HARsigma"):
            c = compare(nll[m] - nll["HARsigma"], fold_idx, n_folds, rng)
            row.update(dNLL_vs_HARsig=c["mean"], dNLL_H_lo=c["lo"], dNLL_H_hi=c["hi"],
                       folds_better_than_HARsig=c["better"], fold_t_p=c["t_p"], fold_sign_p=c["sign_p"],
                       verdict_vs_HARsig=verdict(c, "HARsigma"))
        if "Ridge+HARsigma" in models and m in NN_KINDS:
            c = compare(nll[m] - nll["Ridge+HARsigma"], fold_idx, n_folds, rng)
            row.update(dNLL_vs_Ridge=c["mean"], dNLL_R_lo=c["lo"], dNLL_R_hi=c["hi"],
                       verdict_vs_Ridge=verdict(c, "Ridge+HARsigma"))
        summary.append(row)

    out = os.path.join(root, "results", "tables"); os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(root, "results", f"wf_{tag}_predictions.npz"), y=y, fold=fold_idx,
             nu=(np.full(len(y), np.nan) if nu_all is None else nu_all),
             **{f"{m.replace('+', '_')}_mu": mu[m] for m in models},
             **{f"{m.replace('+', '_')}_sig": sig[m] for m in models})
    pd.DataFrame(fold_rows).round(4).to_csv(os.path.join(out, f"wf_{tag}_folds.csv"), index=False)
    S = pd.DataFrame(summary).round(4)
    S.to_csv(os.path.join(out, f"wf_{tag}_summary.csv"), index=False)
    if history:
        pd.DataFrame(history).round(4).to_csv(os.path.join(out, f"wf_{tag}_history.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("\n" + "=" * 80)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΑ (εκτός δείγματος)")
    print("=" * 80)
    print("\n-- Κατανομή (NLL/CRPS χαμηλότερα = καλύτερα· κάλυψη ιδανικά 0,80 / 0,95· z_std ιδανικά 1) --")
    if nu_all is not None:
        print(f"Student-t: ν ανά fold από {np.nanmin(nu_all):.2f} έως {np.nanmax(nu_all):.2f}")
    cols1 = [c for c in ("model", "NLL", "CRPS_gauss", "cov80", "cov95", "z_std", "dNLL_vs_Const", "dNLL_vs_HARsig",
                         "dNLL_H_lo", "dNLL_H_hi", "folds_better_than_HARsig", "fold_t_p", "fold_sign_p",
                         "verdict_vs_HARsig",
                         "verdict_vs_Ridge") if c in S.columns]
    print(S[cols1].to_string(index=False))
    print("\n-- Μεταβολή τιμής (μ): R² > 0 σημαίνει καλύτερο από σταθερό μέσο όρο; AUC 0,5 = καμία πληροφορία --")
    cols2 = [c for c in ("model", "R2_mu_vs_const", "hit_rate", "base_up", "AUC_up", "AUC_lo", "AUC_hi",
                         "muSE_folds_better", "mu_t_p", "mu_verdict") if c in S.columns]
    print(S[cols2].to_string(index=False))
    print("\nΕρμηνεία: «καλύτερο από HARsigma» στο NLL = το δίκτυο προσθέτει πληροφορία για την κίνηση πέρα από το HAR.")
    print("Για το μ, δες το R²/AUC μαζί με το CI: αν περιέχει 0 / 0,5, δεν υπάρχει αποδείξιμο σήμα στο πρόσημο.")
    if args.legacy_split:
        print("[!] --legacy-split: ένα μόνο test, χωρίς έλεγχο ανά fold· ενδεικτικό για τη σύγκριση με το πρώτο Bi-LSTM.")
    print(f"Αρχεία: {out}")


if __name__ == "__main__":
    main()
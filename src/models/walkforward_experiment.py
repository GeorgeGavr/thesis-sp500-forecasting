"""
Walk-forward πειραματικό πρωτόκολλο για την κατευθυντική πρόβλεψη 5 ημερών του S&P 500.

Τι διορθώνει σε σχέση με το train_classifier.py:
  1. Πολλαπλές περίοδοι δοκιμής (walk-forward, expanding window) αντί για ένα μόνο έτος.
  2. Embargo ίσο με τον ορίζοντα (5 ημέρες) μεταξύ train / val / test, ώστε οι
     επικαλυπτόμενες ετικέτες να μη διαρρέουν από το ένα σύνολο στο άλλο.
  3. Σταθερός κανόνας απόφασης (probs > t). Καμία αντιστροφή. Το AUC αναφέρεται όπως βγαίνει.
  4. Early stopping και επιλογή εποχής με βάση το AUC του validation (όχι το BCE loss).
  5. Μικρότερα μοντέλα, ισχυρότερη κανονικοποίηση, μικρότερος ρυθμός μάθησης.
  6. Baselines: πλειοψηφία, λογιστική παλινδρόμηση, LSTM, Bi-LSTM, Bi-LSTM + Attention.
  7. Ensemble των σπόρων (μέσος όρος πιθανοτήτων). Καμία επιλογή «καλύτερου run» στο test.
  8. Block-bootstrap 95% διαστήματα εμπιστοσύνης για AUC και MCC.
  9. Προαιρετικός έλεγχος με ανακατεμένες ετικέτες (--shuffle-labels).

Εκτέλεση από οπουδήποτε μέσα στο repository:
    python src/models/walkforward_experiment.py
    python src/models/walkforward_experiment.py --features no_rsi
    python src/models/walkforward_experiment.py --shuffle-labels
    python src/models/walkforward_experiment.py --quick          (γρήγορη δοκιμή)

Έξοδοι στο results/tables/:
    wf_<tag>_folds.csv     μετρικές ανά fold και μοντέλο
    wf_<tag>_summary.csv   συγκεντρωτικά (όλες οι περίοδοι δοκιμής μαζί) με 95% CI
    wf_<tag>_history.csv   train/val AUC ανά εποχή (για να φαίνεται αν μαθαίνει)
"""
import argparse
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    TORCH = True
except ImportError:  # επιτρέπει να τρέξουν τα baselines χωρίς PyTorch
    TORCH = False

# --------------------------------------------------------------------------------------
# Ρυθμίσεις (δηλώνονται ΠΡΙΝ από την εκτέλεση και δεν αλλάζουν βάσει του test)
# --------------------------------------------------------------------------------------
SEQ_LEN = 20
HORIZON = 5              # ορίζοντας στόχου -> μέγεθος embargo
N_FOLDS = 4              # αριθμός περιόδων δοκιμής
TEST_DAYS = 252          # ~1 χρηματιστηριακό έτος ανά περίοδο δοκιμής
VAL_DAYS = 252
SEEDS = [42, 100, 2024, 777, 999]
N_BOOT = 1000
BLOCK = 10               # μήκος μπλοκ για το block bootstrap (> ορίζοντας)

NN_CFG = dict(hidden=32, layers=1, dropout=0.4, lr=3e-4, weight_decay=1e-4,
              batch=64, max_epochs=60, patience=10)

FEATURE_SETS = {
    # Το ακατέργαστο Volume αφαιρείται: είναι μη στάσιμο (μετακίνηση +1,2 τυπ. απόκλ. στο test).
    "default": lambda cols: [c for c in cols if c != "Volume"],
    "no_rsi": lambda cols: [c for c in cols if c not in ("Volume", "rsi_14")],
    "no_bb": lambda cols: [c for c in cols if c not in ("Volume", "bb_position")],
    "no_calendar": lambda cols: [c for c in cols if c not in ("Volume", "day_of_week", "month")],
    "all": lambda cols: list(cols),
}


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


def describe_target(df):
    """Ελέγχει τι σημαίνει η κλάση 1, συγκρίνοντας με τη μελλοντική απόδοση 5 ημερών."""
    fwd = df["log_return"].rolling(HORIZON).sum().shift(-HORIZON)
    m1 = fwd[df["target"] == 1].mean()
    m0 = fwd[df["target"] == 0].mean()
    meaning = "ΠΤΩΣΗ (fwd ≤ 0)" if m1 < m0 else "ΑΝΟΔΟΣ (fwd > 0)"
    agree = ((fwd <= 0) == (df["target"] == 1)).mean() if m1 < m0 else ((fwd > 0) == (df["target"] == 1)).mean()
    return meaning, agree


def make_windows(F, y, end_rows):
    """Παράθυρο που τελειώνει στη γραμμή j -> στόχος y[j] (απόδοση των ημερών j+1..j+5)."""
    X = np.stack([F[j - SEQ_LEN + 1: j + 1] for j in end_rows]).astype(np.float32)
    return X, y[end_rows].astype(np.float32)


def build_folds(n_rows, first_valid_row):
    """Expanding-window walk-forward με embargo HORIZON γραμμών πριν από κάθε val και test."""
    folds = []
    last = n_rows - 1 - HORIZON          # οι τελευταίες γραμμές δεν έχουν πλήρη ετικέτα
    for k in range(N_FOLDS):
        test_end = last - (N_FOLDS - 1 - k) * TEST_DAYS
        test_start = test_end - TEST_DAYS + 1
        val_end = test_start - 1 - HORIZON
        val_start = val_end - VAL_DAYS + 1
        train_end = val_start - 1 - HORIZON
        train_start = first_valid_row
        if train_end - train_start < 500:
            raise ValueError("Πολύ λίγα δεδομένα εκπαίδευσης για τόσες περιόδους δοκιμής.")
        folds.append(dict(train=np.arange(train_start, train_end + 1),
                          val=np.arange(val_start, val_end + 1),
                          test=np.arange(test_start, test_end + 1)))
    return folds


# --------------------------------------------------------------------------------------
# Μετρικές
# --------------------------------------------------------------------------------------
def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def pick_threshold(y_val, p_val):
    """Κατώφλι μέσω MCC στο validation με ΣΤΑΘΕΡΟ κανόνα (p > t)."""
    qs = np.quantile(p_val, np.linspace(0.05, 0.95, 37))
    scores = [matthews_corrcoef(y_val, (p_val > t).astype(int)) for t in qs]
    return float(qs[int(np.argmax(scores))])


def metrics(y, p, t):
    pred = (p > t).astype(int)
    return dict(AUC=safe_auc(y, p), MCC=matthews_corrcoef(y, pred),
                Acc=accuracy_score(y, pred), BalAcc=balanced_accuracy_score(y, pred),
                MajorityAcc=max(y.mean(), 1 - y.mean()), PredPos=pred.mean())


def block_bootstrap_ci(y, p, pred, rng):
    """Moving-block bootstrap πάνω στη χρονική σειρά των out-of-sample προβλέψεων."""
    n = len(y)
    nb = int(np.ceil(n / BLOCK))
    aucs, mccs = [], []
    for _ in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK + 1, nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        mccs.append(matthews_corrcoef(y[idx], pred[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5]), np.percentile(mccs, [2.5, 97.5])
    return lo, hi


# --------------------------------------------------------------------------------------
# Νευρωνικά μοντέλα
# --------------------------------------------------------------------------------------
if TORCH:
    class Attention(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.score = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1, bias=False))

        def forward(self, h):
            w = torch.softmax(self.score(h), dim=1)
            return (w * h).sum(dim=1)

    class SeqClassifier(nn.Module):
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
        out = [torch.sigmoid(model(torch.from_numpy(X[i:i + bs]).to(device))).cpu().numpy()
               for i in range(0, len(X), bs)]
        return np.concatenate(out)

    def train_nn(kind, Xtr, ytr, Xva, yva, seed, device, cfg, history, tag):
        set_seed(seed)
        bi = kind in ("BiLSTM", "BiLSTM+Attn")
        model = SeqClassifier(Xtr.shape[2], cfg["hidden"], cfg["layers"], cfg["dropout"],
                              bidirectional=bi, attention=(kind == "BiLSTM+Attn")).to(device)
        pos_w = torch.tensor((1 - ytr.mean()) / ytr.mean(), device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        g = torch.Generator().manual_seed(seed)
        Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)

        best_auc, best_state, best_epoch, wait = -np.inf, None, 0, 0
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
            tr_auc = safe_auc(ytr, predict(model, Xtr, device))
            va_auc = safe_auc(yva, predict(model, Xva, device))
            history.append(dict(**tag, model=kind, seed=seed, epoch=epoch, train_auc=tr_auc, val_auc=va_auc))
            if va_auc > best_auc:
                best_auc, best_epoch, wait = va_auc, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg["patience"]:
                    break
        model.load_state_dict(best_state)
        return model, best_epoch


# --------------------------------------------------------------------------------------
# Κύριο πείραμα
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="default", choices=FEATURE_SETS.keys())
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="έλεγχος ορθότητας: ανακατεύει τις ετικέτες του train (αναμένεται AUC ≈ 0,5)")
    ap.add_argument("--quick", action="store_true", help="1 σπόρος, λίγες εποχές (μόνο για δοκιμή)")
    ap.add_argument("--models", default="Majority,Logistic,LSTM,BiLSTM,BiLSTM+Attn")
    args = ap.parse_args()

    seeds = SEEDS[:1] if args.quick else SEEDS
    cfg = dict(NN_CFG, max_epochs=5, patience=3) if args.quick else NN_CFG
    models = [m.strip() for m in args.models.split(",")]
    if not TORCH:
        dropped = [m for m in models if m not in ("Majority", "Logistic")]
        if dropped:
            print(f"[!] Δεν βρέθηκε PyTorch: παραλείπονται τα {dropped}")
        models = [m for m in models if m in ("Majority", "Logistic")]

    root = find_root()
    df = pd.read_csv(os.path.join(root, "data", "processed", "sp500_processed.csv"), parse_dates=["Date"])
    base_cols = [c for c in df.columns if c not in ("Date", "Close", "log_return", "target")]
    feats = FEATURE_SETS[args.features](base_cols)
    tag = args.features + ("_shuffled" if args.shuffle_labels else "")

    meaning, agree = describe_target(df)
    F_raw = df[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill().values.astype(np.float64)
    y_all = df["target"].values.astype(np.float32)
    folds = build_folds(len(df), first_valid_row=SEQ_LEN - 1)
    device = get_device() if TORCH else None

    print("=" * 78)
    print(f"WALK-FORWARD | features={args.features} ({len(feats)}) | seeds={len(seeds)} | device={device}")
    print(f"Κλάση 1 = {meaning} (συμφωνία με τη μελλοντική απόδοση: {agree*100:.1f}%)")
    for k, f in enumerate(folds, 1):
        d = lambda r: df["Date"].iloc[r].date()
        print(f"  Fold {k}: train {d(f['train'][0])}→{d(f['train'][-1])} | "
              f"val {d(f['val'][0])}→{d(f['val'][-1])} | test {d(f['test'][0])}→{d(f['test'][-1])} "
              f"| κλάση 1 στο test {y_all[f['test']].mean()*100:.1f}%")
    print("=" * 78)

    rng = np.random.default_rng(0)
    fold_rows, history = [], []
    pooled = {m: dict(y=[], p=[], pred=[]) for m in models}

    for k, f in enumerate(folds, 1):
        y = y_all.copy()
        if args.shuffle_labels:
            y[f["train"]] = rng.permutation(y[f["train"]])

        scaler = StandardScaler().fit(F_raw[f["train"]])           # μόνο στο train
        F = np.clip(scaler.transform(F_raw), -3, 3).astype(np.float32)
        Xtr, ytr = make_windows(F, y, f["train"])
        Xva, yva = make_windows(F, y, f["val"])
        Xte, yte = make_windows(F, y, f["test"])

        for m in models:
            t0 = time.time()
            if m == "Majority":
                # σταθερή «πιθανότητα» 0,5 (AUC = 0,5) και πρόβλεψη πάντα της πλειοψηφίας του train
                p_va = np.full(len(yva), 0.5); p_te = np.full(len(yte), 0.5)
                thr = 0.5 if ytr.mean() < 0.5 else -np.inf
                info = ""
            elif m == "Logistic":
                clf = LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000)
                clf.fit(Xtr[:, -1, :], ytr)
                p_va = clf.predict_proba(Xva[:, -1, :])[:, 1]
                p_te = clf.predict_proba(Xte[:, -1, :])[:, 1]
                thr = pick_threshold(yva, p_va); info = ""
            else:
                pv, pt, epochs = [], [], []
                for s in seeds:
                    model, ep = train_nn(m, Xtr, ytr, Xva, yva, s, device, cfg, history, dict(fold=k))
                    pv.append(predict(model, Xva, device)); pt.append(predict(model, Xte, device))
                    epochs.append(ep)
                p_va, p_te = np.mean(pv, axis=0), np.mean(pt, axis=0)   # ensemble σπόρων
                thr = pick_threshold(yva, p_va)
                seed_aucs = [safe_auc(yte, q) for q in pt]
                info = f"| βέλτιστες εποχές {epochs} | test AUC ανά σπόρο {np.round(seed_aucs, 3).tolist()}"

            r_va = metrics(yva, p_va, thr) if m != "Majority" else dict(AUC=np.nan)
            r_te = metrics(yte, p_te, thr)
            fold_rows.append(dict(fold=k, model=m, val_AUC=r_va["AUC"], **{f"test_{a}": b for a, b in r_te.items()},
                                  threshold=thr))
            pooled[m]["y"].append(yte); pooled[m]["p"].append(p_te); pooled[m]["pred"].append((p_te > thr).astype(int))
            print(f"Fold {k} | {m:12s} | val AUC {r_va['AUC']:.3f} | test AUC {r_te['AUC']:.3f} "
                  f"| MCC {r_te['MCC']:+.3f} | Acc {r_te['Acc']*100:5.1f}% (πλειοψ. {r_te['MajorityAcc']*100:5.1f}%) "
                  f"| {time.time()-t0:5.1f}s {info}")

    # Συγκεντρωτικά πάνω σε όλες τις περιόδους δοκιμής
    summary = []
    for m in models:
        y = np.concatenate(pooled[m]["y"]).astype(int)
        p = np.concatenate(pooled[m]["p"])
        pred = np.concatenate(pooled[m]["pred"])
        auc = safe_auc(y, p)
        mcc = matthews_corrcoef(y, pred)
        if m == "Majority":
            auc_ci, mcc_ci = (np.nan, np.nan), (np.nan, np.nan)
        else:
            auc_ci, mcc_ci = block_bootstrap_ci(y, p, pred, rng)
        fold_aucs = [r["test_AUC"] for r in fold_rows if r["model"] == m]
        summary.append(dict(model=m, n_test=len(y), AUC=auc, AUC_lo=auc_ci[0], AUC_hi=auc_ci[1],
                            MCC=mcc, MCC_lo=mcc_ci[0], MCC_hi=mcc_ci[1],
                            Acc=accuracy_score(y, pred), MajorityAcc=max(y.mean(), 1 - y.mean()),
                            BalAcc=balanced_accuracy_score(y, pred),
                            folds_AUC_above_05=f"{sum(a > 0.5 for a in fold_aucs)}/{len(fold_aucs)}"))

    out = os.path.join(root, "results", "tables"); os.makedirs(out, exist_ok=True)
    pd.DataFrame(fold_rows).round(4).to_csv(os.path.join(out, f"wf_{tag}_folds.csv"), index=False)
    S = pd.DataFrame(summary).round(4)
    S.to_csv(os.path.join(out, f"wf_{tag}_summary.csv"), index=False)
    if history:
        pd.DataFrame(history).round(4).to_csv(os.path.join(out, f"wf_{tag}_history.csv"), index=False)

    print("\n" + "=" * 78)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΑ (όλες οι περίοδοι δοκιμής, 95% block-bootstrap CI)")
    print("=" * 78)
    pd.set_option("display.width", 160)
    print(S.to_string(index=False))
    print("\nΕρμηνεία: ένα μοντέλο έχει σήμα μόνο αν το κάτω όριο του CI για το AUC είναι > 0,5,")
    print("το AUC είναι > 0,5 στις περισσότερες περιόδους, και ξεπερνά τη λογιστική παλινδρόμηση.")
    print(f"Αρχεία: {out}")


if __name__ == "__main__":
    main()
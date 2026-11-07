"""
Διαγνωστικά για το πρόβλημα Raw AUC < 0.5 στο test.
Τρέξιμο από τη ρίζα του repository:   python src/diagnostics.py
Δεν χρησιμοποιεί PyTorch. Διαβάζει data/processed/sp500_processed.csv.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Βρίσκει τη ρίζα του repository ανεξάρτητα από το πού βρίσκεται το script
root = os.path.abspath(os.path.dirname(__file__))
while not os.path.exists(os.path.join(root, "data", "processed", "sp500_processed.csv")):
    parent = os.path.dirname(root)
    if parent == root:
        raise FileNotFoundError("Δεν βρέθηκε data/processed/sp500_processed.csv")
    root = parent
csv = os.path.join(root, "data", "processed", "sp500_processed.csv")
df = pd.read_csv(csv, parse_dates=["Date"])

exclude = ["Date", "Close", "log_return", "target"]
feats = [c for c in df.columns if c not in exclude]
n = len(df); tr_end = int(n * 0.8); va_end = int(n * 0.9)
splits = {"train": (0, tr_end), "val": (tr_end, va_end), "test": (va_end, n)}

print("=" * 70)
print(f"Γραμμές: {n} | Χαρακτηριστικά ({len(feats)}): {feats}")
print("=" * 70)

# 1. Επικράτηση κλάσης 1 και περίοδοι
print("\n[1] ΕΠΙΚΡΑΤΗΣΗ ΚΛΑΣΗΣ 1 ΑΝΑ ΣΥΝΟΛΟ")
for name, (a, b) in splits.items():
    t = df["target"].iloc[a:b]
    print(f"  {name:5s} {df['Date'].iloc[a].date()} -> {df['Date'].iloc[b-1].date()} | "
          f"n={b-a:4d} | ποσοστό 1 = {t.mean()*100:5.1f}%")

# Στο pipeline το παράθυρο που τελειώνει στη γραμμή j χρησιμοποιεί target[j+1]
# (create_sequences: y = targets[i+seq_len], X = features[i:i+seq_len]).
def auc_safe(y, x):
    y = np.asarray(y); x = np.asarray(x)
    if len(np.unique(y)) < 2 or np.nanstd(x) == 0: return np.nan
    return roc_auc_score(y, x)

def per_feature_auc(shift):
    rows = []
    for f in feats:
        r = {"feature": f}
        for name, (a, b) in splits.items():
            x = df[f].values[a:b - shift] if shift else df[f].values[a:b]
            y = df["target"].values[a + shift:b]
            r[name] = auc_safe(y, x)
        r["flip"] = "ΝΑΙ" if (r["train"] - 0.5) * (r["test"] - 0.5) < 0 and abs(r["test"] - 0.5) > 0.05 else ""
        rows.append(r)
    return pd.DataFrame(rows).set_index("feature").round(3)

pd.set_option("display.width", 140)
print("\n[2] AUC ΚΑΘΕ ΧΑΡΑΚΤΗΡΙΣΤΙΚΟΥ ΜΕ ΤΟΝ ΣΤΟΧΟ (τιμή > 0.5: υψηλή τιμή ~ κλάση 1)")
print("    Ευθυγράμμιση όπως στο pipeline (χαρακτηριστικό ημέρας j ↔ target της j+1)")
print(per_feature_auc(1).to_string())
print("\n    Ευθυγράμμιση 'σωστή' (χαρακτηριστικό ημέρας j ↔ target της j)")
print(per_feature_auc(0).to_string())

# 3. Λογιστική παλινδρόμηση baseline (τελευταία γραμμή παραθύρου, pipeline-ευθυγράμμιση)
print("\n[3] BASELINE: ΛΟΓΙΣΤΙΚΗ ΠΑΛΙΝΔΡΟΜΗΣΗ (μόνο η τελευταία ημέρα του παραθύρου)")
def run_lr(cols, label):
    X = df[cols].values[:-1]; y = df["target"].values[1:]
    # ευθυγράμμιση: X[j] ↔ y[j+1]
    idx = {k: (a, min(b, n - 1)) for k, (a, b) in splits.items()}
    a0, b0 = idx["train"]
    sc = StandardScaler().fit(X[a0:b0])
    Xs = np.clip(sc.transform(X), -3, 3)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xs[a0:b0], y[a0:b0])
    out = []
    for k in ["train", "val", "test"]:
        a, b = idx[k]
        out.append(auc_safe(y[a:b], clf.predict_proba(Xs[a:b])[:, 1]))
    print(f"  {label:32s} AUC train={out[0]:.3f} val={out[1]:.3f} test={out[2]:.3f}")
run_lr(feats, "όλα τα χαρακτηριστικά")
if "Volume" in feats:
    run_lr([c for c in feats if c != "Volume"], "χωρίς Volume")
run_lr([c for c in feats if c in ("rsi_14", "dist_sma_20")], "μόνο RSI + dist_sma_20")

# 4. Naive baselines στο test
t_test = df["target"].values[va_end:]
print("\n[4] NAIVE ΒΑΣΕΙΣ ΣΤΟ TEST")
print(f"  πάντα '0' (πλειοψηφία): accuracy = {(1 - t_test.mean())*100:.1f}%")
print(f"  πάντα '1'            : accuracy = {t_test.mean()*100:.1f}%")

# 5. Μη-στασιμότητα: μέση τιμή / τυπική απόκλιση ανά σύνολο
print("\n[5] ΜΕΣΗ ΤΙΜΗ ΚΑΘΕ ΧΑΡΑΚΤΗΡΙΣΤΙΚΟΥ ΑΝΑ ΣΥΝΟΛΟ, σε μονάδες τυπικής απόκλισης του train")
rows = []
for f in feats:
    mu = df[f].iloc[:tr_end].mean(); sd = df[f].iloc[:tr_end].std() or 1
    rows.append({"feature": f, **{k: (df[f].iloc[a:b].mean() - mu) / sd for k, (a, b) in splits.items()}})
print(pd.DataFrame(rows).set_index("feature").round(2).to_string())
print("\n(σε μονάδες >1 στο test το χαρακτηριστικό έχει αλλάξει επίπεδο έναντι του train)")
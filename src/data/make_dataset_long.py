"""
Μακρύ ιστορικό S&P 500 (από 1990) με ΕΛΕΓΧΟ ΣΤΑΣΙΜΟΤΗΤΑΣ και αυτόματο μετασχηματισμό.

Ροή:
    1. Κατέβασμα/φόρτωση ^GSPC (Close, Volume) από το 1990.
    2. Log returns και τα ίδια χαρακτηριστικά με το make_dataset.py (engineer_features).
    3. Έλεγχος στασιμότητας ΚΑΘΕ μεταβλητής με ADF (H0: μοναδιαία ρίζα) και KPSS (H0: στάσιμη).
    4. Μετασχηματισμός όσων δεν είναι στάσιμες:
           θετικές μεταβλητές επιπέδου (τιμή, όγκος)  ->  log returns  (Δ log x)
           οποιαδήποτε άλλη                           ->  πρώτες διαφορές (Δ x), στήλη <όνομα>_d1
       και ΕΠΑΝΕΛΕΓΧΟΣ μετά τον μετασχηματισμό.
    5. Αποθήκευση δεδομένων και πλήρους αναφοράς στασιμότητας (πριν/μετά).

Κανόνας απόφασης (ανά μεταβλητή, α = 0,05):
    ADF απορρίπτει  & KPSS δεν απορρίπτει  -> ΣΤΑΣΙΜΗ                    (κρατιέται)
    ADF δεν απορρίπτει                     -> ΜΟΝΑΔΙΑΙΑ ΡΙΖΑ / ΑΣΑΦΗΣ     (μετασχηματισμός)
    ADF απορρίπτει  & KPSS απορρίπτει      -> ΣΥΓΚΡΟΥΟΜΕΝΑ ΑΠΟΤΕΛΕΣΜΑΤΑ  (κρατιέται, εκτός αν --strict)
Τα συγκρουόμενα αποτελέσματα είναι τυπικά για επίμονες αλλά mean-reverting σειρές (π.χ. μεταβλητότητα,
απόσταση από κινητό μέσο): το KPSS τείνει να απορρίπτει σε μεγάλα δείγματα. Η διαφορά τέτοιας σειράς
θα έσβηνε ακριβώς την πληροφορία του επιπέδου της, γι' αυτό η προεπιλογή είναι να ΜΗΝ διαφοριστεί.
Με --strict διαφορίζονται και αυτές.

Αρχεία:
    data/raw/sp500_raw_long.csv
    data/processed/sp500_processed_long.csv           (νέο, μετά τους μετασχηματισμούς)
    data/processed/sp500_processed_long_prev.csv      (αντίγραφο ασφαλείας του προηγούμενου, μία φορά)
    results/tables/stationarity_report_long.csv       (πλήρης αναφορά)

ΠΡΟΣΟΧΗ: αν αλλάξουν στήλες, αλλάζει το σύνολο χαρακτηριστικών του volatility_experiment.py και τα
αποτελέσματα που ήδη έχεις ΔΕΝ είναι πια συγκρίσιμα. Ξανατρέξε το πείραμα μετά από κάθε αλλαγή.

Τοποθέτηση: src/data/make_dataset_long.py (δίπλα στο make_dataset.py)
Εκτέλεση:
    python src/data/make_dataset_long.py
    python src/data/make_dataset_long.py --strict
    python src/data/make_dataset_long.py --test-until 2005-09-29   (έλεγχοι μόνο στο πρώτο train παράθυρο)
"""
import argparse
import os
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_dataset import engineer_features  # noqa: E402

START = "1990-01-01"
ALPHA = 0.05
# Δεν ελέγχονται ως χαρακτηριστικά: ημερομηνία, δυαδικός στόχος, αποδεδειγμένα περιοδικές/φραγμένες εποχικές στήλες
NOT_TESTED = {"Date", "target", "day_of_week", "month"}
LEVEL_VARS = {"Close": "log_return (Δ log Close)", "Volume": "volume_change (Δ log Volume)"}


def run_tests(x):
    """ADF (autolag=AIC, σταθερός όρος) και KPSS (σταθερός όρος, nlags='auto'), όπως στο make_dataset.py."""
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    adf = adfuller(x, autolag="AIC", regression="c")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")           # το KPSS προειδοποιεί όταν το p βγαίνει εκτός πίνακα (0,01–0,10)
        k = kpss(x, regression="c", nlags="auto")
    return dict(ADF_stat=adf[0], ADF_p=adf[1], KPSS_stat=k[0], KPSS_p=k[1], n=len(x))


def verdict(t):
    adf_rejects = t["ADF_p"] < ALPHA                  # απορρίπτει μοναδιαία ρίζα (ένδειξη στασιμότητας)
    kpss_ok = t["KPSS_p"] > ALPHA                     # δεν απορρίπτει στασιμότητα
    if adf_rejects and kpss_ok:
        return "στάσιμη"
    if not adf_rejects:
        return "μοναδιαία ρίζα / ασαφής" if not kpss_ok else "ασαφής (ADF δεν απορρίπτει)"
    return "συγκρουόμενα (ADF στάσιμη, KPSS όχι)"


def needs_transform(v, strict):
    if v == "στάσιμη":
        return False
    if v.startswith("συγκρουόμενα"):
        return strict
    return True


def transform(series, name):
    """Θετικές σειρές επιπέδου -> log returns, αλλιώς πρώτες διαφορές."""
    s = series.replace([np.inf, -np.inf], np.nan)
    if (s.dropna() > 0).all() and name in LEVEL_VARS:
        return np.log(s).diff(), f"{name}_logret"
    return s.diff(), f"{name}_d1"


def stationarity_pass(df, strict, test_until):
    """Επιστρέφει (νέο df, αναφορά). Δεν αγγίζει Close/Volume/log_return/target (χρειάζονται στη συνέχεια)."""
    mask = pd.Series(True, index=df.index) if test_until is None else (df["Date"] <= pd.Timestamp(test_until))
    protected = {"Date", "Close", "Volume", "log_return", "target"}
    rows, drop_cols, new_cols = [], [], {}
    for col in df.columns:
        if col in NOT_TESTED:
            continue
        t = run_tests(df.loc[mask, col])
        v = verdict(t)
        row = dict(variable=col, role="τιμή/όγκος (επίπεδο)" if col in LEVEL_VARS else ("απόδοση" if col == "log_return" else "χαρακτηριστικό"),
                   ADF_p=t["ADF_p"], KPSS_p=t["KPSS_p"], verdict=v, action="", new_variable="",
                   ADF_p_after=np.nan, KPSS_p_after=np.nan, verdict_after="")
        if col in LEVEL_VARS:
            row["action"] = f"όχι στάσιμη όπως αναμένεται· χρησιμοποιείται {LEVEL_VARS[col]}" if v != "στάσιμη" else "στάσιμη"
        elif col == "log_return":
            row["action"] = "κρατιέται" if v == "στάσιμη" else "ΠΡΟΣΟΧΗ: οι αποδόσεις δεν βγήκαν στάσιμες"
        elif needs_transform(v, strict):
            new, new_name = transform(df[col], col)
            t2 = run_tests(new[mask])
            v2 = verdict(t2)
            new_cols[new_name] = new
            drop_cols.append(col)
            row.update(action="μετασχηματίστηκε", new_variable=new_name, ADF_p_after=t2["ADF_p"],
                       KPSS_p_after=t2["KPSS_p"], verdict_after=v2)
            if v2 != "στάσιμη":
                row["action"] += " (ΠΡΟΣΟΧΗ: δεν έγινε πλήρως στάσιμη ούτε μετά την πρώτη διαφορά)"
        else:
            row["action"] = "κρατιέται" if v == "στάσιμη" else "κρατιέται (συγκρουόμενα αποτελέσματα, βλ. --strict)"
        rows.append(row)
    out = df.drop(columns=[c for c in drop_cols if c not in protected])
    for k, v in new_cols.items():
        out[k] = v
    return out, pd.DataFrame(rows)


def regime_report(df, test_until):
    """Μόνο έλεγχος (όχι μετασχηματισμός) των χαρακτηριστικών regime που υπολογίζει το volatility_experiment.py."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
    try:
        import volatility_experiment as ve
    except Exception as e:
        print(f"[i] Παραλείπεται ο έλεγχος των regime features ({type(e).__name__}).")
        return pd.DataFrame()
    d = ve.add_vol_features(df[["Date", "Close", "log_return"]].copy())
    mask = pd.Series(True, index=d.index) if test_until is None else (d["Date"] <= pd.Timestamp(test_until))
    rows = []
    for c in ve.REGIME_COLS:
        t = run_tests(d.loc[mask, c])
        v = verdict(t)
        rows.append(dict(variable=c, role="regime (μόνο έλεγχος)", ADF_p=t["ADF_p"], KPSS_p=t["KPSS_p"], verdict=v,
                         action="υπολογίζεται μέσα στο πείραμα· δεν αλλάζει εδώ" + ("" if v == "στάσιμη" else " (βλ. σχόλιο)"),
                         new_variable="", ADF_p_after=np.nan, KPSS_p_after=np.nan, verdict_after=""))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="διαφόριση και των «συγκρουόμενων» περιπτώσεων")
    ap.add_argument("--test-until", default=None, help="περιορισμός των ελέγχων σε δεδομένα έως αυτή την ημερομηνία")
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_path = os.path.join(root, "data", "raw", "sp500_raw_long.csv")
    out_path = os.path.join(root, "data", "processed", "sp500_processed_long.csv")
    prev_path = os.path.join(root, "data", "processed", "sp500_processed_long_prev.csv")
    rep_path = os.path.join(root, "results", "tables", "stationarity_report_long.csv")
    for p in (raw_path, out_path, rep_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path, parse_dates=["Date"])
    else:
        import yfinance as yf
        print(f"Κατέβασμα ^GSPC από {START}...")
        df = yf.download("^GSPC", start=START, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close", "Volume"]].dropna().reset_index()
        df.to_csv(raw_path, index=False)

    df["Volume"] = df["Volume"].replace(0, np.nan)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df = engineer_features(df)
    df = df.replace([np.inf, -np.inf], np.nan)

    print("\n" + "=" * 78)
    print("ΕΛΕΓΧΟΣ ΣΤΑΣΙΜΟΤΗΤΑΣ (ADF: H0 μοναδιαία ρίζα · KPSS: H0 στάσιμη · α = 0,05)")
    if args.test_until:
        print(f"Οι έλεγχοι γίνονται σε δεδομένα έως {args.test_until}")
    print("=" * 78)
    df2, rep = stationarity_pass(df, args.strict, args.test_until)
    rep = pd.concat([rep, regime_report(df, args.test_until)], ignore_index=True)

    df2["target"] = (df2["log_return"].rolling(5).sum().shift(-5) > 0).astype(float)
    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200); pd.set_option("display.max_colwidth", 46)
    show = rep.copy()
    for c in ("ADF_p", "KPSS_p", "ADF_p_after", "KPSS_p_after"):
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    print(show[["variable", "ADF_p", "KPSS_p", "verdict", "action", "new_variable", "verdict_after"]].to_string(index=False))
    rep.round(6).to_csv(rep_path, index=False, encoding="utf-8-sig")

    n_tr = int((rep["action"].str.startswith("μετασχηματίστηκε")).sum())
    n_conf = int(rep["verdict"].str.startswith("συγκρουόμενα").sum())
    print(f"\nΣύνοψη: {len(rep)} μεταβλητές ελέγχθηκαν · {n_tr} μετασχηματίστηκαν · {n_conf} με συγκρουόμενα αποτελέσματα"
          f"{' (δεν διαφορίστηκαν)' if not args.strict else ''}.")
    print(f"✓ Αναφορά: {rep_path}")

    if os.path.exists(out_path) and not os.path.exists(prev_path):
        shutil.copy(out_path, prev_path)
        print(f"✓ Αντίγραφο ασφαλείας του προηγούμενου αρχείου: {prev_path}")
    df2.to_csv(out_path, index=False)
    years = (df2["Date"].iloc[-1] - df2["Date"].iloc[0]).days / 365.25
    print(f"✓ {len(df2)} ημέρες ({df2['Date'].iloc[0].date()} → {df2['Date'].iloc[-1].date()}, ~{years:.0f} έτη)")
    print(f"✓ Αποθηκεύτηκε: {out_path}")
    if n_tr:
        print("\n⚠ Άλλαξαν στήλες: ξανατρέξε το volatility_experiment.py. Τα προηγούμενα αποτελέσματα δεν είναι συγκρίσιμα.")


if __name__ == "__main__":
    main()
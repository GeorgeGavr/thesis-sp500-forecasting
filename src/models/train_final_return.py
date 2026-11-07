"""
Τελικό Bi-LSTM κατανομής απόδοσης 5 ημερών και πρόβλεψη για την επόμενη εβδομάδα.

Ίδιο μοντέλο και ίδιες ρυθμίσεις με το κύριο πείραμα (return_distribution_experiment.py, v2):
    --sigma-target rv (μ από MSE στην r5, σ από τη log ρεαλιζόμενη διακύμανση) και Student-t.
Ίδιο πρωτόκολλο εκπαίδευσης με κάθε fold του πειράματος, αλλά σε ΟΛΑ τα διαθέσιμα δεδομένα:
    1. train = όλες οι γραμμές με γνωστό στόχο εκτός των τελευταίων 252 (+ embargo 5 ημερών),
       validation = οι τελευταίες 252 → επιλογή εποχών με early stopping (ανά σπόρο).
    2. επανεκπαίδευση σε train+validation με τις επιλεγμένες εποχές.
    3. πρόβλεψη από το παράθυρο των 20 τελευταίων ημερών (έως την τελευταία ημερομηνία του αρχείου)
       για την log απόδοση των επόμενων 5 ημερών διαπραγμάτευσης.

Έξοδοι:
    models/return_dist/<tag>_<μοντέλο>_seed<s>.pt
    results/return_forecast_<tag>.json
    (εκτύπωση: μ, σ, ετήσια μεταβλητότητα, διαστήματα 80%/95% σε απόδοση και σε μονάδες δείκτη,
     πιθανότητα ανόδου, σύγκριση με HAR-σ και σταθερή κατανομή)

Εκτέλεση (από τη ρίζα του project):
    python src/models/train_final_return.py
    python src/models/train_final_return.py --predict-only        # μόνο πρόβλεψη από τα αποθηκευμένα μοντέλα
    python src/models/train_final_return.py --models BiLSTM,LSTM
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import volatility_experiment as ve            # noqa: E402
import return_distribution_experiment as rde  # noqa: E402

warnings.filterwarnings("ignore")
HORIZON, SEQ_LEN, SEEDS, CLIP = ve.HORIZON, ve.SEQ_LEN, ve.SEEDS, ve.CLIP
VAL_DAYS = 252
ANN = np.sqrt(252 / HORIZON)


def load_frame(root, data, features, vix):
    path = data if os.path.isabs(data) else os.path.join(root, data)
    df = pd.read_csv(path, parse_dates=["Date"])
    base_cols = [c for c in df.columns if c not in ("Date", "Close", "log_return", "target", "Volume")]
    df = ve.add_vol_features(df)
    if features == "regime_vix":
        default_vix = "vix_long.csv" if "long" in os.path.basename(data) else "vix.csv"
        vix_path = vix or os.path.join(root, "data", "raw", default_vix)
        df = ve.add_vix(df, vix_path)
    df["y_ret"] = df["log_return"].rolling(HORIZON).sum().shift(-HORIZON)
    feats = {"regime": base_cols + ve.REGIME_COLS, "base": base_cols,
             "regime_vix": base_cols + ve.REGIME_COLS + ve.VIX_COLS}[features]
    return df, feats


def har_fit(har, yv, rows):
    lr = LinearRegression().fit(har[rows], yv[rows])
    res = yv[rows] - lr.predict(har[rows])
    return lr, res.var()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "processed", "sp500_processed_long.csv"))
    ap.add_argument("--features", default="regime", choices=["regime", "base", "regime_vix"])
    ap.add_argument("--vix", default=None)
    ap.add_argument("--models", default="BiLSTM", help="π.χ. BiLSTM ή BiLSTM,LSTM,BiLSTM+Attn ή κενό για μόνο βάσεις")
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    root = ve.find_root()
    tag = f"ret_{args.features}_rv_t"
    model_dir = os.path.join(root, "models", "return_dist")
    os.makedirs(model_dir, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if models and not ve.TORCH:
        raise SystemExit("Χρειάζεται PyTorch για τα δίκτυα (ή --models \"\" για μόνο βάσεις).")
    seeds = SEEDS[:1] if args.quick else SEEDS
    cfg = dict(ve.NN_CFG, max_epochs=5, patience=3) if args.quick else dict(ve.NN_CFG)
    cfg["sigma_target"] = "rv"

    df, feats = load_frame(root, args.data, args.features, args.vix)
    n = len(df)
    first_ok = int(df[ve.REGIME_COLS + ["persist"]].dropna().index[0])
    first_valid = first_ok + SEQ_LEN - 1
    last_lab = n - 1 - HORIZON                      # τελευταία γραμμή με γνωστό στόχο 5 ημερών
    val = np.arange(last_lab - VAL_DAYS + 1, last_lab + 1)
    train = np.arange(first_valid, val[0] - HORIZON)
    fit = np.concatenate([train, val])
    origin = n - 1                                   # η πρόβλεψη γίνεται στην τελευταία ημέρα του αρχείου
    d = lambda r: df["Date"].iloc[r].date()

    F_raw = df[feats].replace([np.inf, -np.inf], np.nan).ffill().bfill().values.astype(np.float64)
    yr, yv = df["y_ret"].values.astype(np.float64), df["y_vol"].values.astype(np.float64)
    har = df[ve.HAR_COLS].values.astype(np.float64)
    close = float(df["Close"].iloc[origin])
    next_days = pd.bdate_range(df["Date"].iloc[origin] + pd.Timedelta(days=1), periods=HORIZON)

    print("=" * 78)
    print(f"ΤΕΛΙΚΟ ΜΟΝΤΕΛΟ ΚΑΤΑΝΟΜΗΣ ΑΠΟΔΟΣΗΣ 5 ΗΜΕΡΩΝ | features={args.features} ({len(feats)}) | {tag}")
    print(f"train {d(train[0])}→{d(train[-1])} | val {d(val[0])}→{d(val[-1])} | "
          f"πρόβλεψη από {d(origin)} (κλείσιμο {close:,.2f}) για {next_days[0].date()}→{next_days[-1].date()}*")
    print("=" * 78)

    # ---------------- βάσεις σύγκρισης (εκπαίδευση σε όλα τα δεδομένα με γνωστό στόχο) ----------------
    mu_c, sd_c = yr[fit].mean(), yr[fit].std()
    lr, s2h = har_fit(har, yv, fit)
    sig_h = float(np.sqrt(np.exp(lr.predict(har[[origin]])[0] + 0.5 * s2h)))
    sig_fit = np.sqrt(np.exp(lr.predict(har[fit]) + 0.5 * s2h))
    nu = rde.fit_nu((yr[fit] - mu_c) / sig_fit)
    out = {"Const": dict(mu=mu_c, sigma=sd_c), "HARsigma": dict(mu=mu_c, sigma=sig_h)}

    # ---------------- δίκτυα ----------------
    if models:
        import torch
        device = ve.get_device()
        mv1, sv1 = yv[train].mean(), yv[train].std()
        mu1, sd1 = yr[train].mean(), yr[train].std()
        mv2, sv2 = yv[fit].mean(), yv[fit].std()
        mu2, sd2 = yr[fit].mean(), yr[fit].std()
        sc2 = StandardScaler().fit(F_raw[fit])
        F2 = np.clip(sc2.transform(F_raw), -CLIP, CLIP).astype(np.float32)
        X_origin, _ = ve.make_windows(F2, np.zeros(n, dtype=np.float32), [origin])
        for m in models:
            ckpts = [os.path.join(model_dir, f"{tag}_{m.replace('+', '_')}_seed{s}.pt") for s in seeds]
            mus, lrvs = [], []
            if args.predict_only:
                missing = [c for c in ckpts if not os.path.exists(c)]
                if missing:
                    raise SystemExit(f"Λείπουν μοντέλα: {missing[0]} ... Τρέξε χωρίς --predict-only.")
                for c in ckpts:
                    ck = torch.load(c, map_location=device, weights_only=False)
                    if ck["feats"] != feats:
                        raise SystemExit(f"Τα features του {c} δεν ταιριάζουν με τα δεδομένα.")
                    net = rde.make_model(m, len(feats), ck["cfg"], device)
                    net.load_state_dict(ck["state_dict"])
                    F = np.clip((F_raw - ck["scaler_mean"]) / ck["scaler_scale"], -CLIP, CLIP).astype(np.float32)
                    Xo, _ = ve.make_windows(F, np.zeros(n, dtype=np.float32), [origin])
                    a, b = rde.predict2(net, Xo, device)
                    mus.append(a[0] * ck["sd2"] + ck["mu2"]); lrvs.append(b[0] * ck["sv2"] + ck["mv2"])
                    s2v = ck["s2v"]
                    print(f"  {m} seed {ck['seed']}: φορτώθηκε (εκπαιδευμένο έως {ck['trained_through']})")
            else:
                sc1 = StandardScaler().fit(F_raw[train])
                F1 = np.clip(sc1.transform(F_raw), -CLIP, CLIP).astype(np.float32)
                T1 = np.stack([(yr - mu1) / sd1, (yv - mv1) / sv1], axis=1).astype(np.float32)
                T2 = np.stack([(yr - mu2) / sd2, (yv - mv2) / sv2], axis=1).astype(np.float32)
                Xtr, ytr = ve.make_windows(F1, T1, train)
                Xva, yva = ve.make_windows(F1, T1, val)
                Xfit, yfit = ve.make_windows(F2, T2, fit)
                lrv_val = []
                for s, c in zip(seeds, ckpts):
                    net1, ep = rde.train_dist(m, Xtr, ytr, Xva, yva, s, device, cfg, [], dict(fold=0))
                    lrv_val.append(rde.predict2(net1, Xva, device)[1] * sv1 + mv1)
                    net = rde.train_dist_fixed(m, Xfit, yfit, ep, s, device, cfg)
                    a, b = rde.predict2(net, X_origin, device)
                    mus.append(a[0] * sd2 + mu2); lrvs.append(b[0] * sv2 + mv2)
                    torch.save(dict(state_dict={k: v.cpu() for k, v in net.state_dict().items()}, cfg=cfg,
                                    model=m, seed=s, epochs=ep, feats=feats, features_set=args.features,
                                    scaler_mean=sc2.mean_, scaler_scale=sc2.scale_, clip=CLIP, seq_len=SEQ_LEN,
                                    mu2=mu2, sd2=sd2, mv2=mv2, sv2=sv2, s2v=None, nu=nu,
                                    trained_through=str(d(fit[-1])), sigma_target="rv"), c)
                    print(f"  {m} seed {s}: εποχές {ep} · αποθηκεύτηκε {os.path.relpath(c, root)}")
                # διασπορά υπολοίπων log RV στο validation (σύνολο σπόρων), όπως στο πείραμα
                s2v = float(np.var(yv[val] - np.mean(lrv_val, axis=0)))
                for c in ckpts:
                    ck = torch.load(c, map_location="cpu", weights_only=False)
                    ck["s2v"] = s2v
                    torch.save(ck, c)
            mu_m = float(np.mean(mus))
            sig_m = float(np.sqrt(np.exp(np.mean(lrvs) + 0.5 * s2v)))
            out[m] = dict(mu=mu_m, sigma=sig_m, seeds=len(mus))
            # Η εκδοχή που υποστηρίζεται από το πείραμα: σ από το δίκτυο, μ = ιστορικός μέσος όρος
            # (το μ του δικτύου δεν είχε αποδείξιμη πληροφορία εκτός δείγματος).
            out[f"{m}-σ"] = dict(mu=mu_c, sigma=sig_m, seeds=len(mus))

    # ---------------- αποτελέσματα ----------------
    rows = []
    for m, r in out.items():
        mu_, s_ = r["mu"], r["sigma"]
        q80, q95 = rde.z_quantile(0.80, nu), rde.z_quantile(0.95, nu)
        p_up = 1 - float(rde.stats.t.cdf((0 - mu_) / s_ / np.sqrt((nu - 2) / nu), nu))
        rows.append(dict(model=m, mu_5d=mu_, sigma_5d=s_, vol_annual=s_ * ANN, p_up=p_up,
                         r80_lo=mu_ - q80 * s_, r80_hi=mu_ + q80 * s_, r95_lo=mu_ - q95 * s_, r95_hi=mu_ + q95 * s_,
                         px80_lo=close * np.exp(mu_ - q80 * s_), px80_hi=close * np.exp(mu_ + q80 * s_),
                         px95_lo=close * np.exp(mu_ - q95 * s_), px95_hi=close * np.exp(mu_ + q95 * s_)))
    R = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    print(f"\nΠρόβλεψη για τις 5 ημέρες διαπραγμάτευσης μετά τις {d(origin)} · Student-t, ν = {nu:.2f}")
    show = R.copy()
    for c in ("mu_5d", "sigma_5d", "vol_annual", "p_up", "r80_lo", "r80_hi", "r95_lo", "r95_hi"):
        show[c] = show[c].map(lambda v: f"{v:+.2%}" if c.startswith(("mu", "r")) else f"{v:.2%}")
    for c in ("px80_lo", "px80_hi", "px95_lo", "px95_hi"):
        show[c] = show[c].map(lambda v: f"{v:,.0f}")
    print(show.to_string(index=False))
    print("\nΣτήλες: μ και σ της log απόδοσης 5 ημερών · ετήσια μεταβλητότητα · πιθανότητα ανόδου · "
          "διαστήματα 80%/95% σε απόδοση και σε μονάδες S&P 500.")
    print("Υπενθύμιση από το πείραμα: το μ ΔΕΝ έχει αποδείξιμη πληροφορία για την κατεύθυνση (AUC ≈ 0,5)· "
          "η αξία του μοντέλου είναι το σ και τα διαστήματα.")
    print("* οι 5 ημέρες υπολογίζονται ως εργάσιμες, χωρίς αργίες χρηματιστηρίου.")

    res = dict(tag=tag, origin_date=str(d(origin)), close=close, nu=nu,
               horizon_days=[str(x.date()) for x in next_days],
               trained_through=str(d(fit[-1])), forecasts=R.round(6).to_dict(orient="records"))
    jp = os.path.join(root, "results", f"return_forecast_{tag}.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print(f"\n✓ {os.path.relpath(jp, root)}")


if __name__ == "__main__":
    main()
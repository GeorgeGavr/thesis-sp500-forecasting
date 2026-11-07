"""
Γραφήματα για το πείραμα κατανομής απόδοσης 5 ημερών (return_distribution_experiment.py).

Διαβάζει (από results/ και results/tables/):
    wf_<tag>_predictions.npz, wf_<tag>_summary.csv           κύριο τρέξιμο (default: v2, rv + Student-t)
    wf_<tag>_shuffled*_summary.csv / _predictions.npz          έλεγχοι shuffle (όσοι υπάρχουν)
    wf_<v1-tag>_summary.csv                                   τρέξιμο v1 (προαιρετικό, για το fig7)

Γράφει στο results/plots/return_distribution/:
    fig1_nll_per_year.png         διαφορά NLL BiLSTM − HAR-σ ανά έτος δοκιμής (+ shuffle)
    fig2_real_vs_shuffle.png      διαφορά NLL έναντι HAR-σ με 95% CI: πραγματικοί vs ανακατεμένοι στόχοι
    fig3_calibration.png          κάλυψη διαστημάτων ανά ονομαστικό επίπεδο + ιστόγραμμα PIT
    fig4_timeseries.png           απόδοση 5 ημερών με ζώνη 95% και προβλεπόμενη μεταβλητότητα (όλη η περίοδος)
    fig5_timeseries_zoom.png      το ίδιο, μεγέθυνση (default: 2020)
    fig6_direction.png            AUC και ποσοστό σωστού προσήμου (κατεύθυνση) με CI
    fig7_v1_vs_v2.png             v1 (Gaussian NLL) vs v2 (σ από ρεαλιζόμενη διακύμανση, Student-t)
    fig8_convergence.png          σύγκλιση: διάμεση απώλεια train/validation ανά εποχή (κάθε αρχιτεκτονική)
    fig9_convergence_per_fold.png καμπύλες εκπαίδευσης του κύριου μοντέλου ανά έτος δοκιμής και σπόρο
    fig10_selected_epochs.png     κατανομή της εποχής που επέλεξε το early stopping
    captions.txt                  λεζάντες για την εργασία
και τον πίνακα results/tables/table_ret_main_<tag>.csv

Εκτέλεση (από τη ρίζα του project):
    python src/models/plot_return_distribution.py
    python src/models/plot_return_distribution.py --tag ret_regime_f20_long_rv_t --zoom-start 2008-01-01 --zoom-end 2009-06-30
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.dates as mdates        # noqa: E402
import matplotlib.ticker                 # noqa: E402,F401
import matplotlib.text                   # noqa: E402,F401
from scipy import stats                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import volatility_experiment as ve       # noqa: E402

# Χρώμα = οντότητα (ίδιο σε όλα τα γραφήματα), σταθερή σειρά
COL = {"BiLSTM": "#2a78d6", "HARsigma": "#eb6834", "LSTM": "#1baf7a", "BiLSTM+Attn": "#eda100",
       "Ridge+HARsigma": "#e87ba4", "Const": "#8a8984"}
NAME = {"BiLSTM": "BiLSTM", "HARsigma": "HAR-σ", "LSTM": "LSTM", "BiLSTM+Attn": "BiLSTM+Attn",
        "Ridge+HARsigma": "Ridge + HAR-σ", "Const": "Σταθερή κατανομή"}
INK, INK2, GRID, SHUF = "#0b0b0b", "#52514e", "#e6e5e0", "#52514e"
NN = ["LSTM", "BiLSTM", "BiLSTM+Attn"]
ANN = np.sqrt(252 / 5)          # σ 5 ημερών -> ετήσια

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


# --------------------------------------------------------------------------------------
# Φόρτωση
# --------------------------------------------------------------------------------------
def key(m):
    return m.replace("+", "_")


def load_run(root, tag):
    npz = os.path.join(root, "results", f"wf_{tag}_predictions.npz")
    summ = os.path.join(root, "results", "tables", f"wf_{tag}_summary.csv")
    if not os.path.exists(summ):
        return None
    run = dict(tag=tag, summary=pd.read_csv(summ))
    if os.path.exists(npz):
        z = np.load(npz)
        run["y"], run["fold"] = z["y"], z["fold"].astype(int)
        nu = z["nu"] if "nu" in z.files else np.full(len(z["y"]), np.nan)
        run["nu"] = None if np.all(np.isnan(nu)) else nu
        run["mu"] = {m: z[f"{key(m)}_mu"] for m in COL if f"{key(m)}_mu" in z.files}
        run["sig"] = {m: z[f"{key(m)}_sig"] for m in COL if f"{key(m)}_sig" in z.files}
    return run


def nll(y, mu, sig, nu):
    if nu is None:
        return 0.5 * np.log(2 * np.pi * sig ** 2) + 0.5 * ((y - mu) / sig) ** 2
    s = np.sqrt((nu - 2) / nu)
    return -(stats.t.logpdf((y - mu) / sig / s, nu) - np.log(s) - np.log(sig))


def std_cdf(z, nu):
    if nu is None:
        return stats.norm.cdf(z)
    return stats.t.cdf(z / np.sqrt((nu - 2) / nu), nu)


def z_q(p, nu):
    q = 0.5 + p / 2
    return stats.norm.ppf(q) if nu is None else stats.t.ppf(q, nu) * np.sqrt((nu - 2) / nu)


def test_calendar(root, data, n_folds, n_expected):
    """Ανακατασκευή των ημερομηνιών δοκιμής ακριβώς όπως στο πείραμα."""
    path = data if os.path.isabs(data) else os.path.join(root, data)
    df = ve.add_vol_features(pd.read_csv(path, parse_dates=["Date"]))
    first_ok = int(df[ve.REGIME_COLS + ["persist"]].dropna().index[0])
    folds = ve.build_folds(len(df), first_ok + ve.SEQ_LEN - 1, n_folds)
    rows = np.concatenate([f["test"] for f in folds])
    if len(rows) != n_expected:
        raise SystemExit(f"Οι ημερομηνίες ({len(rows)}) δεν ταιριάζουν με τις προβλέψεις ({n_expected}). "
                         f"Έλεγξε --data και --folds.")
    dates = pd.to_datetime(df["Date"].iloc[rows].values)
    realized = np.sqrt(np.exp(df["y_vol"].values[rows])) * ANN
    labels = []
    for f in folds:
        a, b = df["Date"].iloc[f["test"][0]], df["Date"].iloc[f["test"][-1]]
        labels.append(f"{a.year}–{str(b.year)[2:]}")
    return dates, realized, labels


_DEC = re.compile(r"(?<=\d)\.(?=\d)")


def gr(s):
    """Ελληνική υποδιαστολή: 0.034 -> 0,034 (μόνο τελεία ανάμεσα σε ψηφία)."""
    return _DEC.sub(",", s)


class _Comma(matplotlib.ticker.Formatter):
    """Τυλίγει τον υπάρχοντα formatter ενός άξονα και αλλάζει την υποδιαστολή σε κόμμα."""
    def __init__(self, f):
        self.f = f

    def set_axis(self, axis):
        super().set_axis(axis)
        self.f.set_axis(axis)

    def set_locs(self, locs):
        self.f.set_locs(locs)

    def __call__(self, x, pos=None):
        return gr(self.f(x, pos))

    def format_ticks(self, values):
        self.set_locs(values)
        return [self(v, i) for i, v in enumerate(values)]

    def get_offset(self):
        return gr(self.f.get_offset())


def greek_decimals(fig):
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            f = axis.get_major_formatter()
            if not isinstance(f, (mdates.DateFormatter, mdates.ConciseDateFormatter, mdates.AutoDateFormatter, _Comma)):
                axis.set_major_formatter(_Comma(f))
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        if s:
            t.set_text(gr(s))


def save(fig, out, name):
    greek_decimals(fig)
    fig.savefig(os.path.join(out, name), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


# --------------------------------------------------------------------------------------
# Γραφήματα
# --------------------------------------------------------------------------------------
def fig1(main, shuffles, labels, model, out):
    y, f = main["y"], main["fold"]
    d = nll(y, main["mu"][model], main["sig"][model], main["nu"]) - nll(y, main["mu"]["HARsigma"], main["sig"]["HARsigma"], main["nu"])
    fd = np.array([d[f == i].mean() for i in range(len(labels))])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x, fd, width=0.72, color=[COL[model] if v < 0 else COL["HARsigma"] for v in fd], zorder=2)
    for s in shuffles:
        if "mu" in s and model in s["mu"]:
            ds = nll(s["y"], s["mu"][model], s["sig"][model], s["nu"]) - nll(s["y"], s["mu"]["HARsigma"], s["sig"]["HARsigma"], s["nu"])
            fs = np.array([ds[s["fold"] == i].mean() for i in range(len(labels))])
            ax.scatter(x, fs, s=40, facecolor="white", edgecolor=SHUF, linewidth=1.5, zorder=3,
                       label="ανακατεμένοι στόχοι" if s is shuffles[0] else None)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.axhline(fd.mean(), color=INK, linewidth=1.2, linestyle="--", label=f"μέσος όρος {fd.mean():+.4f}")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Δ NLL ανά παρατήρηση")
    ax.set_title(f"{NAME[model]} έναντι HAR-σ ανά έτος δοκιμής "
                 f"(μπλε = καλύτερο το {NAME[model]}, {int((fd < 0).sum())}/{len(fd)} έτη)", loc="left")
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0, -0.22))
    save(fig, out, "fig1_nll_per_year.png")
    return fd


def _ci_plot(ax, runs, cols, ycol, lo, hi, models, ref_line):
    n = len(runs)
    offs = np.linspace(-0.25, 0.25, n) if n > 1 else [0.0]
    for j, (run, lab) in enumerate(runs):
        S = run["summary"].set_index("model")
        for i, m in enumerate(models):
            if m not in S.index or pd.isna(S.loc[m, ycol]):
                continue
            v, a, b = S.loc[m, ycol], S.loc[m, lo], S.loc[m, hi]
            filled = j == 0
            ax.errorbar(i + offs[j], v, yerr=[[v - a], [b - v]], fmt="o", ms=8, capsize=4, linewidth=2,
                        color=cols[m] if filled else SHUF, mfc=cols[m] if filled else "white",
                        mec=cols[m] if filled else SHUF, zorder=3)
    ax.axhline(ref_line, color=INK2, linewidth=1)
    ax.set_xticks(range(len(models)), [NAME[m] for m in models])


def fig2(main, shuffles, out):
    models = [m for m in ["Ridge+HARsigma"] + NN if m in set(main["summary"]["model"])]
    runs = [(main, "πραγματικοί στόχοι")] + [(s, s["tag"].split("_")[-1]) for s in shuffles]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _ci_plot(ax, runs, COL, "dNLL_vs_HARsig", "dNLL_H_lo", "dNLL_H_hi", models, 0)
    ax.set_ylabel("Δ NLL έναντι HAR-σ (95% CI)")
    ax.set_title("Κάτω από το 0 = καλύτερο από HAR-σ · γεμάτοι κύκλοι: πραγματικοί στόχοι, "
                 "κενοί: ανακατεμένοι", loc="left", fontsize=10)
    save(fig, out, "fig2_real_vs_shuffle.png")


def fig3(main, model, out):
    y, nu = main["y"], main["nu"]
    ps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    a1.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle="--", label="τέλεια βαθμονόμηση")
    for m in ["Const", "HARsigma", model]:
        if m not in main["mu"]:
            continue
        z = np.abs((y - main["mu"][m]) / main["sig"][m])
        cov = [np.mean(z < z_q(p, nu)) for p in ps]
        a1.plot(ps, cov, marker="o", ms=6, color=COL[m], label=NAME[m])
    a1.set_xlabel("Ονομαστική πιθανότητα διαστήματος"); a1.set_ylabel("Εμπειρική κάλυψη")
    a1.set_title("Κάλυψη κεντρικών διαστημάτων", loc="left"); a1.legend(loc="upper left")
    pit = std_cdf((y - main["mu"][model]) / main["sig"][model], nu)
    a2.hist(pit, bins=20, range=(0, 1), density=True, color=COL[model], edgecolor="white", linewidth=2)
    a2.axhline(1, color=INK, linewidth=1.2, linestyle="--", label="ομοιόμορφη (ιδανικό)")
    a2.set_xlabel("PIT = F(r5)"); a2.set_ylabel("Πυκνότητα")
    a2.set_title(f"Ιστόγραμμα PIT, {NAME[model]}", loc="left"); a2.legend(loc="upper right")
    save(fig, out, "fig3_calibration.png")


def fig_ts(main, dates, realized, model, out, name, start=None, end=None):
    y, nu = main["y"], main["nu"]
    q = z_q(0.95, nu)
    sel = np.ones(len(y), bool)
    if start:
        sel &= dates >= pd.Timestamp(start)
    if end:
        sel &= dates <= pd.Timestamp(end)
    if sel.sum() < 5:
        print(f"  [!] {name}: λίγες ημέρες στο διάστημα {start}–{end}, παραλείπεται")
        return
    d = dates[sel]
    mb, sb = main["mu"][model][sel], main["sig"][model][sel]
    mh, sh = main["mu"]["HARsigma"][sel], main["sig"]["HARsigma"][sel]
    qq = q[sel] if np.ndim(q) else q
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
    a1.fill_between(d, mb - qq * sb, mb + qq * sb, color=COL[model], alpha=0.22, linewidth=0,
                    label=f"ζώνη 95% {NAME[model]}")
    a1.plot(d, mh + qq * sh, color=COL["HARsigma"], linewidth=1, label="όρια 95% HAR-σ")
    a1.plot(d, mh - qq * sh, color=COL["HARsigma"], linewidth=1)
    a1.scatter(d, y[sel], s=4 if sel.sum() > 800 else 9, color=INK, zorder=3, label="πραγματική απόδοση 5 ημερών")
    out_b = np.mean(np.abs(y[sel] - mb) > qq * sb)
    a1.set_ylabel("log απόδοση 5 ημερών")
    a1.set_title(f"Απόδοση των επόμενων 5 ημερών και ζώνη 95% (εκτός ζώνης {NAME[model]}: {out_b:.1%})",
                 loc="left", pad=26)
    # υπόμνημα πάνω από το διάγραμμα, ώστε να μην καλύπτει ακραίες αποδόσεις
    a1.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9, borderaxespad=0.2)
    a2.plot(d, realized[sel], color="#b5b4ae", linewidth=1, label="ρεαλιζόμενη (επόμενες 5 ημέρες)")
    a2.plot(d, sh * ANN, color=COL["HARsigma"], linewidth=1.5, label="HAR-σ")
    a2.plot(d, sb * ANN, color=COL[model], linewidth=1.5, label=NAME[model])
    a2.set_ylabel("Ετήσια μεταβλητότητα")
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.legend(loc="upper left", ncol=3, fontsize=9)
    loc = mdates.AutoDateLocator()
    a2.xaxis.set_major_locator(loc); a2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.tight_layout()
    save(fig, out, name)


def fig6(main, shuffles, out):
    models = [m for m in ["Ridge+HARsigma"] + NN if m in set(main["summary"]["model"])]
    runs = [(main, "")] + [(s, "") for s in shuffles]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    _ci_plot(a1, runs, COL, "AUC_up", "AUC_lo", "AUC_hi", models, 0.5)
    a1.set_ylabel("AUC για «r5 > 0» (95% CI)")
    a1.set_title("Κατεύθυνση: AUC (0,5 = καμία πληροφορία)", loc="left")
    S = main["summary"].set_index("model")
    base = float(S["base_up"].dropna().iloc[0])
    x = np.arange(len(models))
    a2.bar(x, [S.loc[m, "hit_rate"] for m in models], width=0.6, color=[COL[m] for m in models], zorder=2)
    a2.axhline(base, color=INK, linestyle="--", linewidth=1.2, label=f"«πάντα πάνω» = {base:.1%}")
    a2.set_ylim(0.4, max(0.65, base + 0.03))
    a2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    a2.set_xticks(x, [NAME[m] for m in models])
    a2.set_title("Ποσοστό σωστού προσήμου", loc="left"); a2.legend(loc="upper right")
    for i, m in enumerate(models):
        a2.text(i, S.loc[m, "hit_rate"] + 0.005, f"{S.loc[m, 'hit_rate']:.1%}", ha="center", va="bottom",
                color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig6_direction.png")


def fig7(main, v1, out):
    models = [m for m in NN if m in set(main["summary"]["model"])]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for j, (run, filled, lab) in enumerate([(v1, False, "v1: Gaussian NLL"), (main, True, "v2: σ από Σr², Student-t")]):
        S = run["summary"].set_index("model")
        for i, m in enumerate(models):
            v, a, b = S.loc[m, "dNLL_vs_HARsig"], S.loc[m, "dNLL_H_lo"], S.loc[m, "dNLL_H_hi"]
            ax.errorbar(i + (-0.15 if j == 0 else 0.15), v, yerr=[[v - a], [b - v]], fmt="o", ms=8, capsize=4,
                        color=COL[m], mfc=COL[m] if filled else "white", mec=COL[m], linewidth=2,
                        label=lab if i == 0 else None)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(range(len(models)), [NAME[m] for m in models])
    ax.set_ylabel("Δ NLL έναντι HAR-σ (95% CI)")
    ax.set_title("Επίδραση του σχεδιασμού: κενοί = v1, γεμάτοι = v2", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2)
    save(fig, out, "fig7_v1_vs_v2.png")


# --------------------------------------------------------------------------------------
# Σύγκλιση εκπαίδευσης (στάδιο 1: train με early stopping στο validation)
# --------------------------------------------------------------------------------------
def load_history(root, tag):
    p = os.path.join(root, "results", "tables", f"wf_{tag}_history.csv")
    if not os.path.exists(p):
        return None
    H = pd.read_csv(p)
    # Κάθε καμπύλη (train ή validation, ανά fold/μοντέλο/σπόρο) ως μεταβολή από τη ΔΙΚΗ της 1η εποχή.
    # Τα επίπεδα της απώλειας train και validation δεν συγκρίνονται άμεσα: ο στόχος τυποποιείται με τα
    # στατιστικά του train, οπότε μια ήρεμη ή ταραγμένη περίοδος validation έχει άλλο επίπεδο απώλειας.
    g = H.groupby(["fold", "model", "seed"])
    H["train_rel"] = H["train_nll"] - g["train_nll"].transform("first")
    H["val_rel"] = H["val_nll"] - g["val_nll"].transform("first")
    return H


def fig8(H, out, labels):
    models = [m for m in NN if m in set(H["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), sharey=True)
    axes = np.atleast_1d(axes)
    n_runs = H.groupby("model").apply(lambda d: d[["fold", "seed"]].drop_duplicates().shape[0])
    for ax, m in zip(axes, models):
        d = H[H["model"] == m]
        st = d.groupby("epoch").agg(tr=("train_rel", "median"), va=("val_rel", "median"),
                                    lo=("val_rel", lambda s: s.quantile(0.25)),
                                    hi=("val_rel", lambda s: s.quantile(0.75)), n=("val_rel", "size"))
        st = st[st["n"] >= 0.25 * n_runs[m]]           # μόνο εποχές όπου τρέχει ≥ 25% των εκπαιδεύσεων
        ax.fill_between(st.index, st["lo"], st["hi"], color=COL[m], alpha=0.2, linewidth=0, label="validation, IQR")
        ax.plot(st.index, st["va"], color=COL[m], label="validation (διάμεσος)")
        ax.plot(st.index, st["tr"], color=INK2, linestyle="--", linewidth=1.5, label="train (διάμεσος)")
        best = d.loc[d.groupby(["fold", "seed"])["val_nll"].idxmin(), "epoch"]
        ax.axvline(best.median(), color=INK, linewidth=1, linestyle=":")
        ax.text(best.median(), max(st["va"].max(), st["tr"].max()), f" διάμεση επιλεγμένη\n εποχή = {best.median():.0f}",
                fontsize=8.5, color=INK, va="top")
        ax.axhline(0, color=GRID, linewidth=1)
        ax.set_title(f"{NAME[m]} ({n_runs[m]} εκπαιδεύσεις)", loc="left")
        ax.set_xlabel("Εποχή")
    axes[0].set_ylabel("Μεταβολή απώλειας από την 1η εποχή")
    axes[0].legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "fig8_convergence.png")


def fig9(H, out, labels, model):
    d = H[H["model"] == model]
    folds = sorted(d["fold"].unique())
    nc = 5
    nr = int(np.ceil(len(folds) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(13, 2.3 * nr), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, folds):
        dk = d[d["fold"] == k]
        for s, ds in dk.groupby("seed"):
            ax.plot(ds["epoch"], ds["train_rel"], color="#b5b4ae", linewidth=1)
            ax.plot(ds["epoch"], ds["val_rel"], color=COL[model], linewidth=1.2)
            b = ds.loc[ds["val_nll"].idxmin()]
            ax.scatter(b["epoch"], b["val_rel"], s=22, color=COL[model], edgecolor="white", linewidth=1, zorder=3)
        ax.axhline(0, color=INK2, linewidth=0.8)
        lab = labels[k - 1] if k - 1 < len(labels) else f"fold {k}"
        ax.set_title(lab, loc="left", fontsize=9.5)
        ax.tick_params(labelsize=8)
    for ax in axes[len(folds):]:
        ax.axis("off")
    for ax in axes[-nc:]:
        ax.set_xlabel("Εποχή", fontsize=9)
    for ax in axes[::nc]:
        ax.set_ylabel("Δ απώλειας", fontsize=9)
    fig.suptitle(f"Καμπύλες εκπαίδευσης {NAME[model]} ανά έτος δοκιμής (μεταβολή από την 1η εποχή): "
                 f"γκρι = train, χρώμα = validation, κουκκίδα = επιλεγμένη εποχή (5 σπόροι)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, out, "fig9_convergence_per_fold.png")


def fig10(H, out):
    models = [m for m in NN if m in set(H["model"])]
    best = H.loc[H.groupby(["fold", "model", "seed"])["val_nll"].idxmin(), ["model", "epoch"]]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        e = best.loc[best["model"] == m, "epoch"].values
        ax.boxplot(e, positions=[i], widths=0.45, showfliers=False, medianprops=dict(color=INK, linewidth=2),
                   boxprops=dict(color=INK2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(e)), e, s=12, color=COL[m], alpha=0.6, zorder=3)
        ax.text(i + 0.3, np.median(e), f"διάμεσος {np.median(e):.0f}", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(models)), [NAME[m] for m in models])
    ax.set_ylabel("Επιλεγμένη εποχή (early stopping)")
    ax.set_title("Εποχή ελάχιστης απώλειας validation ανά fold και σπόρο", loc="left")
    save(fig, out, "fig10_selected_epochs.png")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ret_regime_f20_long_rv_t")
    ap.add_argument("--v1-tag", default="ret_regime_f20_long")
    ap.add_argument("--data", default=os.path.join("data", "processed", "sp500_processed_long.csv"))
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--model", default="BiLSTM", help="κύριο μοντέλο για τα fig1/3/4/5")
    ap.add_argument("--zoom-start", default="2020-01-01")
    ap.add_argument("--zoom-end", default="2020-12-31")
    args = ap.parse_args()

    root = ve.find_root()
    main_run = load_run(root, args.tag)
    if main_run is None or "y" not in main_run:
        raise SystemExit(f"Δεν βρέθηκαν τα wf_{args.tag}_summary.csv / _predictions.npz. Τρέξε πρώτα το πείραμα.")
    if args.model not in main_run["mu"]:
        raise SystemExit(f"Το μοντέλο {args.model} δεν υπάρχει στο {args.tag}.")
    shuffles = []
    for p in sorted(glob.glob(os.path.join(root, "results", "tables", f"wf_{args.tag}_shuffled*_summary.csv"))):
        t = os.path.basename(p)[3:-len("_summary.csv")]
        r = load_run(root, t)
        if r is not None:
            shuffles.append(r)
    v1 = load_run(root, args.v1_tag) if args.v1_tag and args.v1_tag != args.tag else None

    out = os.path.join(root, "results", "plots", "return_distribution")
    os.makedirs(out, exist_ok=True)
    print(f"Κύριο: {args.tag} | shuffle: {[s['tag'] for s in shuffles] or 'κανένα'} | v1: {v1['tag'] if v1 else 'όχι'}")
    dates, realized, labels = test_calendar(root, args.data, args.folds, len(main_run["y"]))

    fd = fig1(main_run, shuffles, labels, args.model, out)
    fig2(main_run, shuffles, out)
    fig3(main_run, args.model, out)
    fig_ts(main_run, dates, realized, args.model, out, "fig4_timeseries.png")
    fig_ts(main_run, dates, realized, args.model, out, "fig5_timeseries_zoom.png", args.zoom_start, args.zoom_end)
    fig6(main_run, shuffles, out)
    if v1 is not None:
        fig7(main_run, v1, out)
    else:
        print(f"  [i] fig7: δεν βρέθηκε το wf_{args.v1_tag}_summary.csv, παραλείπεται")

    H = load_history(root, args.tag)
    if H is not None and len(H):
        fig8(H, out, labels)
        if args.model in set(H["model"]):
            fig9(H, out, labels, args.model)
        fig10(H, out)
    else:
        print(f"  [i] fig8–10: δεν βρέθηκε το wf_{args.tag}_history.csv, παραλείπονται")

    # Πίνακας για την εργασία
    keep = ["model", "NLL", "dNLL_vs_HARsig", "dNLL_H_lo", "dNLL_H_hi", "folds_better_than_HARsig", "fold_t_p",
            "fold_sign_p", "cov80", "cov95", "R2_mu_vs_const", "AUC_up", "AUC_lo", "AUC_hi", "hit_rate"]
    T = main_run["summary"][[c for c in keep if c in main_run["summary"].columns]]
    tpath = os.path.join(root, "results", "tables", f"table_ret_main_{args.tag}.csv")
    T.to_csv(tpath, index=False)
    print(f"  ✓ {os.path.basename(tpath)}")

    S = main_run["summary"].set_index("model")
    m = args.model
    period = f"{dates.min():%m/%Y}–{dates.max():%m/%Y}"
    caps = [
        f"Σχήμα 1. Διαφορά μέσης αρνητικής log-πιθανοφάνειας (NLL) μεταξύ {m} και HAR-σ ανά έτος δοκιμής "
        f"({len(labels)} ετήσιες περίοδοι, {period}). Αρνητικές τιμές: καλύτερη πρόβλεψη κατανομής από το {m}. "
        f"Το {m} είναι καλύτερο σε {int((fd < 0).sum())}/{len(fd)} έτη. Οι κενοί κύκλοι δείχνουν το ίδιο μέγεθος "
        f"όταν οι στόχοι εκπαίδευσης έχουν ανακατευτεί τυχαία.",
        f"Σχήμα 2. Διαφορά NLL έναντι HAR-σ με 95% block-bootstrap διάστημα εμπιστοσύνης. Γεμάτοι κύκλοι: "
        f"πραγματικοί στόχοι· κενοί: ανακατεμένοι στόχοι (έλεγχος ορθότητας). {m}: "
        f"{S.loc[m, 'dNLL_vs_HARsig']:+.4f} [{S.loc[m, 'dNLL_H_lo']:+.4f}, {S.loc[m, 'dNLL_H_hi']:+.4f}], "
        f"p (t-test ανά έτος) = {S.loc[m, 'fold_t_p']:.3f}.",
        f"Σχήμα 3. Αριστερά: εμπειρική κάλυψη των κεντρικών διαστημάτων πρόβλεψης ανά ονομαστικό επίπεδο. "
        f"Δεξιά: ιστόγραμμα PIT του {m}· ομοιόμορφο ιστόγραμμα σημαίνει σωστά βαθμονομημένη κατανομή. "
        f"Κάλυψη {m}: 80% → {S.loc[m, 'cov80']:.1%}, 95% → {S.loc[m, 'cov95']:.1%}.",
        f"Σχήμα 4. Πάνω: πραγματική log απόδοση των επόμενων 5 ημερών (σημεία) και ζώνη πρόβλεψης 95% του {m} "
        f"(σκιά) και του HAR-σ (γραμμές). Κάτω: προβλεπόμενη ετήσια μεταβλητότητα έναντι της ρεαλιζόμενης. "
        f"Η ημερομηνία είναι η ημέρα της πρόβλεψης· η απόδοση αφορά τις 5 επόμενες ημέρες διαπραγμάτευσης.",
        f"Σχήμα 5. Όπως το Σχήμα 4, για το διάστημα {args.zoom_start}–{args.zoom_end}.",
        f"Σχήμα 6. Κατεύθυνση της μεταβολής. Αριστερά: AUC της προβλεπόμενης μέσης τιμής μ για το γεγονός "
        f"«απόδοση 5 ημερών > 0» με 95% CI (0,5 = καμία πληροφορία). Δεξιά: ποσοστό σωστού προσήμου έναντι της "
        f"απλής πρόβλεψης «πάντα άνοδος» ({float(S['base_up'].dropna().iloc[0]):.1%}).",
    ]
    if v1 is not None:
        caps.append("Σχήμα 7. Διαφορά NLL έναντι HAR-σ στον αρχικό σχεδιασμό (v1: η σ μαθαίνεται από Gaussian NLL "
                    "στην απόδοση 5 ημερών) και στον τελικό (v2: η σ μαθαίνεται από τη ρεαλιζόμενη διακύμανση "
                    "Σr², αξιολόγηση με Student-t). Κάθε διάστημα είναι σχετικό με το HAR-σ της ίδιας έκδοσης.")
    if H is not None and len(H):
        loss_txt = ("άθροισμα MSE της μέσης τιμής και MSE της log ρεαλιζόμενης διακύμανσης, σε τυποποιημένες μονάδες"
                    if "_rv" in args.tag else "Gaussian NLL στην τυποποιημένη απόδοση")
        caps += [
            f"Σχήμα 8. Σύγκλιση της εκπαίδευσης. Διάμεση απώλεια train (διακεκομμένη) και validation (συνεχής) ανά "
            f"εποχή, σε όλα τα folds και τους σπόρους· κάθε καμπύλη ως μεταβολή από τη δική της τιμή στην πρώτη "
            f"εποχή· σκιά: ενδοτεταρτημοριακό εύρος του validation. Απώλεια: {loss_txt}. Εμφανίζονται μόνο εποχές "
            f"στις οποίες συνεχίζει τουλάχιστον το 25% των εκπαιδεύσεων (early stopping με υπομονή "
            f"{ve.NN_CFG['patience']} εποχών)· μετά τη διάμεση επιλεγμένη εποχή (κατακόρυφη γραμμή) οι καμπύλες "
            f"αντιπροσωπεύουν όλο και λιγότερες εκπαιδεύσεις, εκείνες που συνέχιζαν να βελτιώνονται.",
            f"Σχήμα 9. Καμπύλες εκπαίδευσης του {m} για κάθε έτος δοκιμής και κάθε σπόρο, ως μεταβολή από την "
            f"πρώτη εποχή (τα απόλυτα επίπεδα train και validation διαφέρουν, γιατί ο στόχος τυποποιείται με τα "
            f"στατιστικά του train). Γκρι: train· χρώμα: "
            f"validation· κουκκίδα: εποχή ελάχιστης απώλειας validation, που χρησιμοποιείται για την τελική "
            f"επανεκπαίδευση σε train+validation.",
            "Σχήμα 10. Κατανομή της επιλεγμένης εποχής (early stopping) ανά αρχιτεκτονική, σε όλα τα folds και τους "
            "σπόρους.",
        ]
    with open(os.path.join(out, "captions.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(gr(c) for c in caps) + "\n")
    print(f"  ✓ captions.txt\nΑρχεία: {out}")


if __name__ == "__main__":
    main()
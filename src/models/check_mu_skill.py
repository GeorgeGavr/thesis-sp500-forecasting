"""
Έλεγχος: όταν το BiLSTM προέβλεπε μεγάλο μ στο walk-forward, πήγε πράγματι καλύτερα η αγορά;
Διαβάζει results/wf_ret_regime_f20_long_rv_t_predictions.npz (κύριο πείραμα, εκτός δείγματος).

    python src/models/check_mu_skill.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import volatility_experiment as ve  # noqa: E402

root = ve.find_root()
z = np.load(os.path.join(root, "results", "wf_ret_regime_f20_long_rv_t_predictions.npz"))
y, mu, const = z["y"], z["BiLSTM_mu"], z["Const_mu"]
excess = mu - const                      # πόσο πάνω/κάτω από τον ιστορικό μέσο όρο ήταν το μ του δικτύου
bins = [-np.inf, -0.005, 0.0, 0.005, 0.01, np.inf]
labels = ["< −0,5%", "−0,5% έως 0", "0 έως +0,5%", "+0,5% έως +1%", "> +1%"]
df = pd.DataFrame(dict(y=y, excess=excess, cls=pd.cut(excess, bins, labels=labels)))
t = df.groupby("cls", observed=False).agg(εβδομάδες=("y", "size"),
                                          μέση_πραγματική_απόδοση=("y", "mean"),
                                          ποσοστό_ανόδου=("y", lambda s: (s > 0).mean()))
t["μέση_πραγματική_απόδοση"] = t["μέση_πραγματική_απόδοση"].map(lambda v: f"{v:+.2%}")
t["ποσοστό_ανόδου"] = t["ποσοστό_ανόδου"].map(lambda v: f"{v:.1%}")
print("Πραγματική απόδοση 5 ημερών ανάλογα με το πόσο αισιόδοξο ήταν το μ του BiLSTM (εκτός δείγματος, 20 έτη)")
print(t.to_string())
print(f"\nΣυνολικό ποσοστό ανόδου: {(y > 0).mean():.1%} · μέση απόδοση: {y.mean():+.2%}")
print(f"Συσχέτιση (Spearman) μ με πραγματική απόδοση: {pd.Series(excess).corr(pd.Series(y), method='spearman'):+.3f}")
# Νευρωνικά δίκτυα στις χρηματιστηριακές αγορές – S&P 500

Κώδικας της μεταπτυχιακής εργασίας «Χρήση Νευρωνικών Δικτύων στις Χρηματιστηριακές Αγορές».
Εξετάζεται αν Bi-LSTM/LSTM/Attention δίκτυα προσθέτουν πληροφορία στην πρόβλεψη της
**5ήμερης απόδοσης του S&P 500** (δεδομένα από το 1990) πέρα από απλά στατιστικά μοντέλα.

**Έκδοση:** v1.05

## Βασικά ευρήματα

| Ερώτημα | Αποτέλεσμα |
|---|---|
| Πρόβλεψη κατεύθυνσης (άνοδος/πτώση) | Καμία αποδεδειγμένη ικανότητα: AUC 0,506 [0,481–0,532], αρνητικό R² του μ σε σχέση με σταθερό μέσο όρο |
| Πρόβλεψη κατανομής (μεταβλητότητα) | Το BiLSTM βελτιώνει οριακά το NLL έναντι του HAR-σ: ΔNLL −0,0132 [−0,0253, −0,0011], 13/20 έτη, t-test ετών p = 0,034, sign test p = 0,13 |
| Έλεγχος shuffle | Με ανακατεμένες ετικέτες τα δίκτυα δεν νικούν το HAR-σ, άρα η βελτίωση δεν οφείλεται σε διαρροή πληροφορίας |

Το αποτέλεσμα για τη μεταβλητότητα είναι **οριακό** (μετά από διόρθωση για πολλαπλά μοντέλα p ≈ 0,10)
και η έκδοση v2 σχεδιάστηκε μετά την v1, γι' αυτό αναφέρονται και οι δύο. Η αρχική αναφορά
«Effective AUC 0,6716» του πρώτου Bi-LSTM δεν ισχύει και έχει αποσυρθεί.

## Πρωτόκολλο

- Walk-forward με επεκτεινόμενο παράθυρο, 20 ετήσια test folds (10/2006–10/2026)
- Embargo 5 ημερών, validation 252 ημερών (μόνο για επιλογή εποχής), επανεκπαίδευση σε train+val
- Παράθυρο εισόδου 20 ημέρες, ορίζοντας 5 ημέρες, 5 seeds (42, 100, 2024, 777, 999)
- Στόχος r5 = άθροισμα των log-αποδόσεων των επόμενων 5 ημερών
- Baselines: σταθερή κατανομή (Const), HAR-σ, Ridge + HAR-σ
- Μετρικές: NLL (Student-t), CRPS, κάλυψη 80/95%, PIT, block bootstrap, έλεγχοι σε επίπεδο έτους

## Δομή

```
data/            δεδομένα (τα CSV δεν ανεβαίνουν στο git)
models/          αποθηκευμένα μοντέλα (.pt)
results/         πίνακες, γραφήματα, προβλέψεις
src/data/        make_dataset.py (αρχικό), make_dataset_long.py (τελικό, με τεστ στασιμότητας)
src/models/
  return_distribution_experiment.py   κύριο πείραμα (walk-forward, v1/v2, shuffle)
  volatility_experiment.py            κοινές συναρτήσεις (χαρακτηριστικά, folds, bootstrap)
  plot_return_distribution.py         γραφήματα και λεζάντες
  train_final_return.py               τελικό μοντέλο και πρόβλεψη επόμενης εβδομάδας
  check_mu_skill.py                   διερευνητικός έλεγχος του μ
  walkforward_experiment.py, diagnostics.py   πειράματα κατεύθυνσης και διαγνωστικά
  train_classifier.py, evaluate.py    αρχικό Bi-LSTM (legacy, για σύγκριση)
```

## Εγκατάσταση

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch numpy pandas scipy scikit-learn statsmodels matplotlib yfinance
```

## Αναπαραγωγή (από τη ρίζα του repo)

```bash
# 1. Δεδομένα και έλεγχοι στασιμότητας (κατεβάζει τα ακατέργαστα δεδομένα μέσω yfinance)
python src/data/make_dataset_long.py

# 2. Κύριο πείραμα (v2: Student-t, σ από log RV)
python src/models/return_distribution_experiment.py --data data/processed/sp500_processed_long.csv --folds 20 --features regime --sigma-target rv --dist t

# 3. Έλεγχοι shuffle (δύο seeds)
python src/models/return_distribution_experiment.py --data data/processed/sp500_processed_long.csv --folds 20 --features regime --sigma-target rv --dist t --shuffle-labels --shuffle-seed 0
python src/models/return_distribution_experiment.py --data data/processed/sp500_processed_long.csv --folds 20 --features regime --sigma-target rv --dist t --shuffle-labels --shuffle-seed 1

# 4. Γραφήματα και λεζάντες -> results/plots/return_distribution/
python src/models/plot_return_distribution.py

# 5. Τελικό μοντέλο και πρόβλεψη επόμενης εβδομάδας -> models/return_dist/
python src/models/train_final_return.py
```

Χρήσιμα flags: `--quick` (γρήγορος έλεγχος), `--models BiLSTM,LSTM,BiLSTM+Attn`,
`--features {regime,base,regime_vix}`, και `--predict-only` στο `train_final_return.py`.

## Επιφυλάξεις

- Η πρόβλεψη μ (κατεύθυνση) δεν έχει αποδεδειγμένη αξία. Ως πρόβλεψη του μοντέλου θεωρείται η γραμμή `BiLSTM-σ`
  (σ από το δίκτυο, μ = ιστορικός μέσος όρος).
- Το δίκτυο αντιδρά αργά σε απότομα σοκ (2008, 2020) και η Student-t κατανομή είναι συμμετρική, ενώ το PIT δείχνει αρνητική ασυμμετρία.
- Τίποτα εδώ δεν αποτελεί επενδυτική συμβουλή.

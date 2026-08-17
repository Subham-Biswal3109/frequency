# Wire Watcher — ML Pipeline (README_ML)

Final Year Project — spectrum-availability prediction.

## ⚠️ Data disclaimer (read this first)

The training data (`ml/data/ml_training_samples.csv`) is **SYNTHETIC**. Every
row has `data_type = "synthetic"` and `label_source = "Synthetic dataset"`.
It is **not** a real-world Indian spectrum-occupancy measurement dataset.
Nothing in this pipeline, this README, or `model_results.txt` should be
quoted in your FYP as evidence the model works on real spectrum data. See
`README.txt` / `sources.txt` from the original data package for the
regulatory background (NFAP‑2025, WPC assignment requirements).

**Row-count note:** this pipeline was run against the CSV you pointed it to,
`ml_training_samples.csv`, which contains **10,000 rows** — not ~33,600. The
33,600-row file in your upload (`spectrum_occupancy_synthetic_33600.csv`) is
a different, raw observation file, not the exported ML training table. If
your FYP report claims ~33,600 training samples, that number needs
correcting, or you need to re-export a larger `ml_training_samples` table
from MySQL before rerunning this pipeline.

## Directory layout

```
ml/
├── data/
│   └── ml_training_samples.csv      # input data (10,000 rows, synthetic)
├── saved_models/
│   └── wire_watcher_model.pkl       # best pipeline (preprocessing + model) + metadata
├── results/
│   ├── confusion_matrix_*.png       # 4 confusion matrices (2 experiments x 2 models)
│   └── metrics.json                 # machine-readable metrics for all 4 runs
├── model_results.txt                # full text evaluation report (data quality + results + leakage discussion)
├── train_model.py                   # main training/experimentation script
├── evaluate_model.py                # reloads saved model and re-evaluates (no retraining)
├── requirements.txt
└── README_ML.md                     # this file
```

## How to run

```bash
cd ml
pip install -r requirements.txt
python train_model.py        # runs quality checks, trains + evaluates 4 models, saves everything
python evaluate_model.py     # reloads the saved model and re-evaluates on the held-out test set
```

Both scripts read from `ml/data/ml_training_samples.csv` and use
`random_state=42` throughout, so results are reproducible on re-run.

`evaluate_model.py` also accepts `--data path/to/new_export.csv` to score
the saved model against a fresh CSV export (same schema) without retraining.

## What the pipeline does

1. **Load** `ml_training_samples.csv` with pandas.
2. **Data-quality report**: row count, duplicates, missing values, invalid
   frequency ranges, invalid occupancy values, invalid target values, class
   distribution, numeric feature ranges, and a leakage check. Printed to
   console and saved into `model_results.txt`.
3. **Experiment A** (leakage-safe feature set): `start_frequency_mhz`,
   `end_frequency_mhz`, `bandwidth_mhz`, `hour_of_day`, `day_of_week`,
   `signal_power_dbm`, `noise_floor_dbm`, `snr_db`, plus one-hot encoded
   `region` / `state`. Does **not** use `occupancy_ratio` or
   `interference_score`.
4. **Experiment B**: Experiment A's features **plus** `occupancy_ratio` and
   `interference_score`.
5. Stratified 80/20 train/test split (`test_size=0.20, random_state=42`) —
   the split happens **after** feature selection and **before** any
   fitting, so there is no train/test leakage from that split.
6. Two models per experiment: **Logistic Regression** and **Random Forest**,
   both using `class_weight="balanced"` because the target is ~95%/5%
   imbalanced (accuracy alone would be misleading here).
7. Metrics: accuracy, precision, recall, F1, confusion matrix, ROC-AUC.
8. Comparison table across all 4 (experiment × model) runs.
9. Best model (by F1 on the minority class, restricted to Experiment A for
   reasons explained below) is saved with joblib as a single bundle
   containing the fitted `ColumnTransformer` (scaling + one-hot encoding)
   **and** the classifier, so any future prediction data goes through
   identical preprocessing.

## Actual results (from the run used to write this README)

Class distribution: **9,547 unavailable (0) / 453 available (1)** — a
~21:1 imbalance.

| Experiment | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| A (no occupancy) | Logistic Regression | 0.8345 | 0.1907 | 0.8132 | 0.3090 | 0.8971 |
| A (no occupancy) | Random Forest | 0.9540 | 0.4615 | 0.0659 | 0.1154 | 0.8928 |
| B (with occupancy) | Logistic Regression | 0.9815 | 0.7109 | 1.0000 | 0.8311 | 0.9999 |
| B (with occupancy) | Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Full per-model classification reports and confusion matrices are in
`model_results.txt` and `results/confusion_matrix_*.png`.

## Target-leakage discussion (important for your report)

`target_occupancy_ratio` is **identical** to `occupancy_ratio` in 100% of
rows, and the two `target_available` classes are almost perfectly separated
by `occupancy_ratio` alone (mean ≈0.43 for unavailable vs ≈0.21 for
available, with little overlap). This strongly indicates the synthetic
label-generation process derived `target_available` as a (near-)direct
function of `occupancy_ratio` / `interference_score`.

That is the most likely explanation for **Random Forest on Experiment B
scoring a perfect 1.0 on every metric** — this is a red flag for leakage,
not a genuinely "perfect" model. It means the model is recovering the
synthetic labelling rule rather than learning a generalizable relationship
that would hold for real, forward-looking availability prediction, where
`occupancy_ratio` for a not-yet-assigned candidate frequency wouldn't be
known the same way it's known here for already-labelled historical rows.

For this reason:
- **Experiment A is treated as the defensible, deployable setup** for this
  FYP, since its features aren't directly derived from the label-generation
  process.
- **`wire_watcher_model.pkl` saves the best Experiment A model**
  (Logistic Regression, F1=0.309, ROC-AUC=0.897), not the higher-scoring
  Experiment B models, precisely because Experiment B's high scores are
  suspected leakage artifacts rather than evidence of real predictive skill.
- Experiment A's own numbers are modest — recall of ~0.81 but precision of
  only ~0.19 for Logistic Regression, meaning it over-predicts
  "available" and would need further feature engineering, more real
  (non-synthetic) data, or a different modeling approach before being
  presented as a working prediction system.

## Known limitations to state explicitly in your FYP

- All training data is synthetic; no real Indian spectrum measurements were
  used for the classifier trained here.
- The class imbalance (~21:1) makes both models unreliable on the minority
  ("available") class in Experiment A; recall/precision trade-offs should
  be discussed rather than accuracy alone.
- `district` was 100% missing in this export and was dropped as a feature.
- Suspected target leakage in `occupancy_ratio` / `interference_score` vs.
  `target_available`, as discussed above.
- The MySQL connection (`mysqlconnect.py`) is currently broken; this
  pipeline was deliberately built to run entirely from the CSV export
  instead, per your instructions. It was **not** touched or debugged here.

# Wire Watcher — ML Pipeline (README_ML)

Final Year Project — spectrum-availability prediction.

## ⚠️ This is a rebuilt version — dataset correction

An earlier run of this pipeline accidentally used the wrong CSV
(`ml_training_samples.csv`, 10,000 rows). **That model was discarded and is
not used anywhere in this deliverable.** This version was rebuilt from
scratch on the correct, intended dataset:

**`spectrum_occupancy_synthetic_33600.csv` — 33,600 rows** (verified: 33,601
lines in the file including the header = 33,600 data rows).

## ⚠️ Data disclaimer (read this first)

The dataset is **entirely SYNTHETIC**. Every row has
`data_type = "SYNTHETIC_FOR_PIPELINE_TESTING"`, `assignment_status =
"synthetic_not_verified"`, and `occupancy_status = "synthetic_observation"`.
It is **not** a real-world Indian spectrum-occupancy measurement dataset.
This model is a **prototype** built to exercise the database / dashboard /
ML pipeline end-to-end. **It does not establish real-world spectrum
availability prediction accuracy**, and no such claim is made anywhere in
this pipeline, `model_results.txt`, or this README.

## Dataset verification (as requested, before any training)

| Check | Result |
|---|---|
| Exact row count | **33,600** data rows |
| Exact column names | `record_id, band_id, start_frequency_mhz, end_frequency_mhz, service_type, state, city, hour, day_of_week, bandwidth_mhz, signal_power_dbm, noise_floor_dbm, snr_db, occupancy_ratio, interference_score, assignment_status, occupancy_status, target_available, data_type` (19 columns) |
| Missing values | **None** in any column |
| Duplicate rows | **0** full-row duplicates, **0** duplicate `record_id` |
| `target_available` distribution | **32,326 (96.21%) unavailable (0) / 1,274 (3.79%) available (1)** — ~25.4:1 imbalance |
| Invalid frequency ranges | 0 |
| Invalid occupancy values (outside [0,1]) | 0 |
| Constant / no-signal columns | `assignment_status`, `occupancy_status`, `data_type` — all single-valued, dropped as features |

Note: this file has no `target_occupancy_ratio` column, no `district`
column, and uses `hour`/`city` instead of `hour_of_day`/`region` — the
pipeline renames `hour` → `hour_of_day` internally for consistency with the
feature spec and uses `city` directly.

## Directory layout

```
ml/
├── data/
│   └── spectrum_occupancy_synthetic_33600.csv   # input data (33,600 rows, synthetic)
├── saved_models/
│   └── wire_watcher_model.pkl                   # final pipeline (preprocessing + model), leakage-screened
├── results/
│   ├── confusion_matrix_*.png                   # 4 confusion matrices (2 experiments x 2 models)
│   └── metrics.json                             # machine-readable metrics for all 4 runs
├── model_results.txt                            # full text report: data quality + results + leakage investigation
├── train_model.py                               # main training/experimentation script
├── evaluate_model.py                            # reloads saved model and re-evaluates (no retraining)
├── requirements.txt
└── README_ML.md                                 # this file
```

## How to run

```bash
cd ml
pip install -r requirements.txt
python train_model.py        # data-quality checks + trains/evaluates 4 models + leakage investigation + saves everything
python evaluate_model.py     # reloads the saved model and re-evaluates on the held-out test set
```

Both scripts use `random_state=42` throughout (train/test split, both
classifiers), so results are reproducible on re-run — confirmed by running
`train_model.py` twice during development and getting identical metrics
both times.

`evaluate_model.py` also accepts `--data path/to/new_export.csv` to score
the saved model against a fresh CSV export with the same schema, without
retraining.

## Feature sets

**Experiment A — leakage-safe** (used for the final saved model):
`start_frequency_mhz, end_frequency_mhz, bandwidth_mhz, hour_of_day,
day_of_week, signal_power_dbm, noise_floor_dbm, snr_db` (numeric) +
`state, city, service_type` (one-hot encoded categorical).
Does **not** include `occupancy_ratio`, `interference_score`,
`target_available`, or any other target-derived column.

**Experiment B — measurement-assisted** (exploratory / leakage
investigation only): Experiment A's features **plus** `occupancy_ratio` and
`interference_score`. Explicitly labelled as measurement-assisted and never
considered as the deployable model — see leakage investigation below.

Both experiments use a stratified 80/20 train/test split
(`test_size=0.20, random_state=42`) and both Logistic Regression and Random
Forest use `class_weight="balanced"` to address the ~25:1 class imbalance.
Model selection is based on **F1-score on the minority class**, not
accuracy — accuracy alone is close to the majority-class baseline (96.2%)
and would be a misleading criterion here.

## Actual results (from the run used to write this README)

| Experiment | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| A — leakage-safe | Logistic Regression | 0.8257 | 0.1691 | 0.9176 | 0.2855 | 0.9405 |
| A — leakage-safe | Random Forest | 0.9625 | 0.5484 | 0.0667 | 0.1189 | 0.9354 |
| B — measurement-assisted | Logistic Regression | 0.9868 | 0.7413 | 1.0000 | 0.8514 | 1.0000 |
| B — measurement-assisted | Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Full per-model classification reports and confusion matrices are in
`model_results.txt` and `results/confusion_matrix_*.png`.

## Target-leakage investigation — Experiment B shows confirmed leakage risk

As instructed, Experiment B was investigated specifically to check whether
`occupancy_ratio` / `interference_score` are directly responsible for the
synthetic `target_available` label, rather than being independently useful
predictive signals.

**Findings:**
1. A trivial, model-free threshold rule — `occupancy_ratio < 0.25` — with
   **no classifier at all** reproduces `target_available` with **99.96%
   agreement** across all 33,600 rows.
2. `occupancy_ratio` for `target_available=1` rows is tightly clustered
   (mean 0.215, std 0.031) and almost entirely non-overlapping with
   `target_available=0` rows (mean 0.435).
3. Experiment B's best F1-score (1.0000, Random Forest) is dramatically
   higher than Experiment A's best F1-score (0.2855) on the exact same
   underlying rows — the only difference is adding `occupancy_ratio` /
   `interference_score` as features.

**Verdict: leakage risk confirmed.** This is strong evidence that the
synthetic label-generation process derived `target_available` as a
(near-)deterministic function of `occupancy_ratio`. Experiment B's
near-perfect / perfect scores should be read as "the model recovered the
label-generation rule", not as evidence of genuine predictive skill. In a
real forward-looking deployment, `occupancy_ratio` for a not-yet-assigned
candidate frequency would not be known the way it's known here for an
already-labelled historical row, so an occupancy-driven model is unlikely
to transfer to genuine prediction.

`target_occupancy_ratio` was **not** used anywhere (per instructions) —
note this specific dataset doesn't even contain that column, unlike the
earlier 10,000-row file.

## Final model selection

The model saved to `wire_watcher_model.pkl` was selected **only after**
this leakage investigation, and **only from Experiment A** (the leakage-safe
feature set):

> **Experiment_A_leakage_safe / Logistic Regression**
> F1 = 0.2855, ROC-AUC = 0.9405, Recall = 0.9176, Precision = 0.1691

Random Forest in Experiment A was *not* selected despite its higher raw
accuracy (0.9625 vs 0.8257) — with `class_weight="balanced"` it still
achieved only 0.067 recall on the minority class, meaning it essentially
predicts "unavailable" almost every time and would be useless for actually
surfacing available frequencies. This is a concrete example of why the
model was **not** selected on accuracy alone.

The saved bundle (`wire_watcher_model.pkl`) contains the fitted
`ColumnTransformer` (StandardScaler for numeric features + OneHotEncoder
for `state`/`city`/`service_type`) together with the classifier, so any
future prediction data is guaranteed to go through identical
preprocessing. It also stores a `leakage_note` field documenting why
Experiment B was excluded.

## Known limitations to state explicitly in your FYP

- **The 33,600-row dataset used is entirely synthetic** — this must be
  stated plainly; it is not real Indian spectrum measurement data.
- **Experiment B shows confirmed leakage risk** — its high scores reflect
  recovery of the synthetic label-generation rule, not genuine predictive
  performance, and it was excluded from the final saved model for that
  reason.
- **This model is a prototype and does not establish real-world spectrum
  availability prediction accuracy.** It demonstrates that the
  database → feature-engineering → training → evaluation → serialization
  pipeline works end-to-end, which is a valid FYP deliverable in its own
  right, but the specific accuracy numbers should not be presented as
  validated real-world performance.
- Even Experiment A's own precision (0.169) is low — it over-predicts
  "available" (high recall, low precision). This is a legitimate limitation
  to discuss (further feature engineering, more real/non-synthetic data, or
  a different imbalance-handling strategy would be needed before treating
  this as production-ready).
- Class imbalance is severe (~25:1); accuracy is not treated as the
  headline metric anywhere in this report.
- The MySQL connection (`mysqlconnect.py`) is currently broken; this
  pipeline was deliberately built to run entirely from the CSV export
  instead, per your instructions. It was **not** touched or debugged here.

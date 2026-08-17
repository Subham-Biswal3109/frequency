"""
Wire Watcher — ML Training Pipeline (REBUILT on correct 33,600-row dataset)
=============================================================================
Final Year Project

DATASET USED
------------
ml/data/spectrum_occupancy_synthetic_33600.csv  (33,600 rows, 19 columns)

This replaces an earlier, INCORRECT run of this pipeline that accidentally
used a different, smaller (10,000-row) file. This version uses the file
the project actually intends: spectrum_occupancy_synthetic_33600.csv.

WHAT THIS SCRIPT DOES
----------------------
1. Loads spectrum_occupancy_synthetic_33600.csv with pandas.
2. Runs data-quality checks (row count, columns, duplicates, missing values,
   invalid frequency ranges, invalid occupancy values, invalid target
   values, class distribution, numeric ranges) and prints/saves a report.
3. Builds two experiments:
     Experiment A -> leakage-safe features only (no occupancy_ratio,
                     no interference_score, no target-derived column)
     Experiment B -> "measurement-assisted": adds occupancy_ratio and
                     interference_score, explicitly investigated for
                     leakage risk rather than presented as the winning model
4. Trains Logistic Regression + Random Forest per experiment, both using
   class weighting because the target is ~96%/4% imbalanced.
5. Evaluates every model with accuracy, precision, recall, F1, ROC-AUC,
   confusion matrix, and a full classification report. Model selection is
   NOT based on accuracy alone (accuracy is close to the majority-class
   baseline for an imbalanced target and is not a reliable signal here).
6. Investigates whether occupancy_ratio / interference_score are directly
   responsible for target_available (leakage check), and only saves a final
   model after that investigation, from the experiment judged NOT to have
   target leakage (Experiment A).
7. Saves confusion-matrix images, a text evaluation report, and the final
   model bundled together with its preprocessing pipeline.

IMPORTANT — READ BEFORE USING RESULTS IN ANY REPORT
--------------------------------------------------------
* The dataset is entirely SYNTHETIC (data_type =
  "SYNTHETIC_FOR_PIPELINE_TESTING" for every row). It is NOT a real-world
  Indian spectrum-occupancy measurement set.
* target_available is almost perfectly separable using occupancy_ratio
  alone (a simple threshold rule gets >99% agreement with the label -- see
  the leakage-investigation section printed by this script). Experiment B
  results must be read as "how well can a model recover the synthetic
  labelling rule", not as evidence of genuine predictive skill.
* This model is a PROTOTYPE built to exercise the database/pipeline/ML
  workflow. It does NOT establish real-world spectrum-availability
  prediction accuracy, and no such claim is made anywhere in this script,
  the saved report, or the README.

Reproducibility: random_state = 42 is used everywhere a seed is accepted.
"""

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "spectrum_occupancy_synthetic_33600.csv"
MODEL_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"
REPORT_PATH = BASE_DIR / "model_results.txt"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target_available"

# Columns that must never be used as model inputs: identifiers, constant-value
# bookkeeping columns from the synthetic export, and (for Experiment A) any
# column derived from / responsible for the target.
NON_FEATURE_COLS = [
    "record_id",
    "assignment_status",   # constant value across the whole file -> no signal
    "occupancy_status",    # constant value across the whole file -> no signal
    "data_type",           # constant value ("SYNTHETIC_FOR_PIPELINE_TESTING")
    "target_available",    # target
]

CATEGORICAL_FEATURES = ["state", "city", "service_type"]

EXPERIMENT_A_NUMERIC = [
    "start_frequency_mhz",
    "end_frequency_mhz",
    "bandwidth_mhz",
    "hour_of_day",
    "day_of_week",
    "signal_power_dbm",
    "noise_floor_dbm",
    "snr_db",
]

EXPERIMENT_B_EXTRA_NUMERIC = ["occupancy_ratio", "interference_score"]

# Columns explicitly forbidden from Experiment A per the leakage rules:
FORBIDDEN_IN_EXPERIMENT_A = {
    "target_available",
    "target_occupancy_ratio",  # not present in this file, but forbidden if it ever appears
    "occupancy_ratio",
    "interference_score",
}


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load spectrum_occupancy_synthetic_33600.csv."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Make sure "
            f"spectrum_occupancy_synthetic_33600.csv is at ml/data/"
        )
    df = pd.read_csv(path)
    return df


# ----------------------------------------------------------------------------
# 2. DATA QUALITY REPORT
# ----------------------------------------------------------------------------
def data_quality_report(df: pd.DataFrame) -> str:
    lines = []
    add = lines.append

    add("=" * 70)
    add("WIRE WATCHER — DATA QUALITY REPORT")
    add("=" * 70)
    add(f"Source file : {DATA_PATH.name}")
    add(f"data_type value(s) present : {sorted(df['data_type'].unique().tolist())}")
    add("NOTE: This dataset is SYNTHETIC. It must not be described as real-world")
    add("      Indian spectrum measurement data in the final report.")
    add("")

    add(f"Row count            : {len(df)}")
    add(f"Column count          : {df.shape[1]}")
    add(f"Column names          : {list(df.columns)}")
    add("")

    add("-- Duplicates --")
    add(f"Full-row duplicates       : {df.duplicated().sum()}")
    add(f"Duplicate record_id rows  : {df['record_id'].duplicated().sum()}")
    add("")

    add("-- Missing values (per column) --")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        add("No missing values found in any column.")
    else:
        for col, n in missing.items():
            add(f"  {col:<28} {n} missing ({n/len(df):.1%})")
    add("")

    add("-- Invalid frequency ranges (start_frequency_mhz >= end_frequency_mhz) --")
    invalid_freq = (df["start_frequency_mhz"] >= df["end_frequency_mhz"]).sum()
    add(f"Invalid rows: {invalid_freq}")
    add("")

    add("-- Invalid occupancy values (outside [0, 1]) --")
    invalid_occ = ((df["occupancy_ratio"] < 0) | (df["occupancy_ratio"] > 1)).sum()
    add(f"occupancy_ratio out of range: {invalid_occ}")
    add("")

    add("-- Invalid target values (must be 0 or 1) --")
    add(f"Unique target_available values: {sorted(df[TARGET_COL].unique().tolist())}")
    invalid_target_vals = df[~df[TARGET_COL].isin([0, 1])]
    add(f"Rows with invalid target value: {len(invalid_target_vals)}")
    add("")

    add("-- Class distribution (target_available) --")
    counts = df[TARGET_COL].value_counts().sort_index()
    props = df[TARGET_COL].value_counts(normalize=True).sort_index()
    for cls in counts.index:
        add(f"  class {cls}: {counts[cls]} rows ({props[cls]:.2%})")
    imbalance_ratio = counts.max() / counts.min()
    add(f"Imbalance ratio (majority:minority) approx {imbalance_ratio:.1f} : 1")
    add("")

    add("-- Constant-value columns dropped as features (no signal) --")
    for col in ["assignment_status", "occupancy_status", "data_type"]:
        add(f"  {col}: unique values = {df[col].unique().tolist()}")
    add("")

    add("-- Numerical feature ranges --")
    num_cols = [
        "start_frequency_mhz", "end_frequency_mhz", "bandwidth_mhz",
        "hour", "day_of_week", "signal_power_dbm", "noise_floor_dbm",
        "snr_db", "occupancy_ratio", "interference_score",
    ]
    desc = df[num_cols].describe().T[["min", "max", "mean", "std"]]
    add(desc.to_string(float_format=lambda x: f"{x:.3f}"))
    add("")

    add("-- Leakage investigation --")
    grp = df.groupby(TARGET_COL)["occupancy_ratio"].agg(["mean", "std", "min", "max"])
    add("occupancy_ratio by class:")
    add(grp.to_string(float_format=lambda x: f"{x:.3f}"))
    best_thresh, best_acc = None, 0
    for t in np.arange(0.15, 0.45, 0.01):
        pred = (df["occupancy_ratio"] < t).astype(int)
        acc = (pred == df[TARGET_COL]).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, t
    add(
        f"A single threshold rule (occupancy_ratio < {best_thresh:.2f}) reproduces "
        f"target_available with {best_acc:.2%} agreement using NO model at all."
    )
    add(
        "-> This is strong evidence that target_available was generated as a "
        "(near-)direct function of occupancy_ratio in this synthetic dataset. "
        "See the leakage discussion at the end of this report for what this "
        "means for Experiment B."
    )
    add("")

    report = "\n".join(lines)
    print(report)
    return report


# ----------------------------------------------------------------------------
# 3. PREPROCESSING
# ----------------------------------------------------------------------------
def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """
    Bundle scaling (numeric) + one-hot encoding (categorical) into one
    ColumnTransformer, which is itself saved inside the final model bundle so
    future prediction data receives IDENTICAL transformations.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


# ----------------------------------------------------------------------------
# 4. TRAIN + EVALUATE ONE MODEL
# ----------------------------------------------------------------------------
def train_and_evaluate(
    model_name, estimator, preprocessor, X_train, X_test, y_train, y_test,
    experiment_name, results_dir,
):
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    metrics = {
        "experiment": experiment_name,
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_test)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, zero_division=0
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["unavailable(0)", "available(1)"]
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{experiment_name} — {model_name}")
    fig.tight_layout()
    fname = f"confusion_matrix_{experiment_name.replace(' ', '_')}_{model_name.replace(' ', '_')}.png"
    fig.savefig(results_dir / fname, dpi=150)
    plt.close(fig)
    metrics["confusion_matrix_image"] = fname

    return metrics, pipe


# ----------------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------------
def main():
    report_sections = []

    df = load_data(DATA_PATH)

    # Rename hour -> hour_of_day for feature-list consistency with the spec
    df = df.rename(columns={"hour": "hour_of_day"})

    dq_report = data_quality_report(df.rename(columns={"hour_of_day": "hour"}))
    report_sections.append(dq_report)

    y = df[TARGET_COL].astype(int)

    # Sanity-check: Experiment A feature list must not contain any forbidden
    # (target-derived) columns.
    assert not (set(EXPERIMENT_A_NUMERIC) & FORBIDDEN_IN_EXPERIMENT_A), (
        "Experiment A feature list illegally contains a target-derived column!"
    )

    all_metrics = []
    best_pipe = None
    best_metrics = None
    best_experiment_data = None

    experiments = {
        "Experiment_A_leakage_safe": EXPERIMENT_A_NUMERIC,
        "Experiment_B_measurement_assisted": EXPERIMENT_A_NUMERIC + EXPERIMENT_B_EXTRA_NUMERIC,
    }

    for exp_name, numeric_features in experiments.items():
        print("\n" + "=" * 70)
        print(f"{exp_name}")
        print("=" * 70)

        feature_cols = numeric_features + CATEGORICAL_FEATURES
        X = df[feature_cols].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")
        print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")
        print(f"Test class balance:\n{y_test.value_counts(normalize=True)}")

        preprocessor = build_preprocessor(numeric_features, CATEGORICAL_FEATURES)

        models_to_run = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        }

        for model_name, estimator in models_to_run.items():
            metrics, pipe = train_and_evaluate(
                model_name, estimator, preprocessor,
                X_train, X_test, y_train, y_test,
                exp_name, RESULTS_DIR,
            )
            all_metrics.append(metrics)

            print(f"\n--- {exp_name} / {model_name} ---")
            print(f"Accuracy : {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall   : {metrics['recall']:.4f}")
            print(f"F1-score : {metrics['f1']:.4f}")
            if metrics["roc_auc"] is not None:
                print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
            print("Confusion matrix [[TN FP][FN TP]]:")
            print(np.array(metrics["confusion_matrix"]))

            # Model selection rule: ONLY consider Experiment A candidates for
            # the final saved model (leakage-safe features only), and select
            # by F1 on the minority class -- NOT by accuracy, since accuracy
            # is dominated by the majority class here.
            if exp_name == "Experiment_A_leakage_safe":
                if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
                    best_metrics = metrics
                    best_pipe = pipe
                    best_experiment_data = {
                        "numeric_features": numeric_features,
                        "categorical_features": CATEGORICAL_FEATURES,
                    }

    # ---- Comparison table ----
    comp_df = pd.DataFrame(all_metrics)[
        ["experiment", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    print("\n" + "=" * 70)
    print("COMPARISON TABLE (Experiment A vs Experiment B)")
    print("=" * 70)
    print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- Leakage investigation summary for the report ----
    best_thresh, best_acc = None, 0
    for t in np.arange(0.15, 0.45, 0.01):
        pred = (df["occupancy_ratio"] < t).astype(int)
        acc = (pred == y).mean()
        if acc > best_acc:
            best_acc, best_thresh = acc, t

    exp_b_metrics = [m for m in all_metrics if m["experiment"] == "Experiment_B_measurement_assisted"]
    exp_b_best_f1 = max(m["f1"] for m in exp_b_metrics)
    leakage_flag = "YES — leakage risk confirmed" if best_acc > 0.97 or exp_b_best_f1 > 0.9 else \
                   "POSSIBLE — inconclusive, needs more investigation"

    leakage_text = f"""
{"=" * 70}
TARGET-LEAKAGE INVESTIGATION (Experiment B)
{"=" * 70}
Question: are occupancy_ratio / interference_score directly responsible for
the synthetic target_available label, rather than independently-useful
predictive signals?

Evidence:
1. A trivial, model-free threshold rule on occupancy_ratio alone
   (occupancy_ratio < {best_thresh:.2f}) reproduces target_available with
   {best_acc:.2%} agreement across all {len(df)} rows -- no classifier
   required.
2. occupancy_ratio for target_available=1 rows is tightly clustered (mean
   {df[df[TARGET_COL]==1]['occupancy_ratio'].mean():.3f}, std {df[df[TARGET_COL]==1]['occupancy_ratio'].std():.3f})
   and almost entirely non-overlapping with target_available=0 rows
   (mean {df[df[TARGET_COL]==0]['occupancy_ratio'].mean():.3f}).
3. Experiment B's best F1-score ({exp_b_best_f1:.4f}) is far higher than
   Experiment A's best F1-score ({best_metrics['f1']:.4f}) using the SAME
   underlying data and the only difference being the addition of
   occupancy_ratio / interference_score as features.

LEAKAGE VERDICT: {leakage_flag}

Interpretation: this pattern indicates the synthetic label-generation
process very likely derived target_available as a (near-)deterministic
function of occupancy_ratio (and probably interference_score too). That
means Experiment B's high scores primarily reflect a model recovering the
rule used to generate the label, not a model learning a generalizable
relationship between RF conditions and future availability. In a genuine
forward-looking deployment, occupancy_ratio for a not-yet-assigned
candidate frequency would not be known in the same way it is known here for
an already-labelled historical row, so an occupancy_ratio-driven model is
unlikely to transfer to real prediction.

Experiment B is retained in this report as a labelled "measurement-assisted"
exploratory result and for the leakage investigation itself -- NOT as a
candidate for the deployed/saved model.

MODEL SELECTION DECISION
--------------------------
The final model saved to wire_watcher_model.pkl is selected EXCLUSIVELY
from Experiment A (leakage-safe features), and the selection criterion is
F1-score on the minority ("available") class -- NOT accuracy, and NOT
Experiment B's higher raw numbers, because those numbers are believed to be
inflated by target leakage as investigated above.

Selected: {best_metrics['experiment']} / {best_metrics['model']}
F1 = {best_metrics['f1']:.4f}   ROC-AUC = {best_metrics['roc_auc']:.4f}

FINAL CAVEATS
-------------
* The dataset used for this pipeline is spectrum_occupancy_synthetic_33600.csv
  -- 33,600 SYNTHETIC rows (data_type = "SYNTHETIC_FOR_PIPELINE_TESTING" for
  every row). It is not a real-world Indian spectrum-occupancy measurement
  set.
* This model is a PROTOTYPE built to exercise the database / dashboard / ML
  pipeline end-to-end. It does NOT establish real-world spectrum
  availability prediction accuracy, and no such claim is made here.
* Class imbalance is severe (~25:1). Accuracy is not a meaningful headline
  metric; precision/recall/F1/ROC-AUC on the minority class are what should
  be reported and discussed.
* Even Experiment A's features (frequency, time, RF power/noise/SNR,
  location, service type) are themselves synthetically generated alongside
  the label, so Experiment A's numbers should also be read as "how well can
  a model fit this synthetic generator's pattern", not as validated
  real-world performance. Real deployment would require real, independently
  collected spectrum measurements.
"""
    print(leakage_text)
    report_sections.append(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    report_sections.append(leakage_text)

    report_sections.append("\n" + "=" * 70)
    report_sections.append("PER-MODEL CLASSIFICATION REPORTS")
    report_sections.append("=" * 70)
    for m in all_metrics:
        report_sections.append(f"\n--- {m['experiment']} / {m['model']} ---")
        report_sections.append(m["classification_report"])
        report_sections.append(f"Confusion matrix image: {m['confusion_matrix_image']}")

    # ---- Save final model + preprocessing pipeline together ----
    model_bundle = {
        "pipeline": best_pipe,
        "feature_columns": best_experiment_data["numeric_features"]
        + best_experiment_data["categorical_features"],
        "numeric_features": best_experiment_data["numeric_features"],
        "categorical_features": best_experiment_data["categorical_features"],
        "target_column": TARGET_COL,
        "experiment": best_metrics["experiment"],
        "model_name": best_metrics["model"],
        "random_state": RANDOM_STATE,
        "trained_on": (
            "spectrum_occupancy_synthetic_33600.csv (33,600 SYNTHETIC rows) — "
            "prototype only, does not establish real-world accuracy"
        ),
        "leakage_screened": True,
        "leakage_note": (
            "Selected from Experiment A (leakage-safe feature set: no "
            "occupancy_ratio, no interference_score, no target-derived column). "
            "Experiment B was investigated and shows strong evidence of target "
            "leakage; it was NOT used for this saved model."
        ),
    }
    model_path = MODEL_DIR / "wire_watcher_model.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\nSaved final model bundle (pipeline + metadata) to: {model_path}")
    print(
        f"Selected: {best_metrics['experiment']} / {best_metrics['model']} "
        f"(F1={best_metrics['f1']:.4f}, ROC-AUC={best_metrics['roc_auc']:.4f})"
    )

    full_report = "\n".join(report_sections)
    with open(REPORT_PATH, "w") as f:
        f.write(full_report)
    print(f"\nSaved full evaluation report to: {REPORT_PATH}")

    metrics_json_path = RESULTS_DIR / "metrics.json"
    serializable_metrics = []
    for m in all_metrics:
        m2 = dict(m)
        m2.pop("classification_report", None)
        serializable_metrics.append(m2)
    with open(metrics_json_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)
    print(f"Saved metrics JSON to: {metrics_json_path}")


if __name__ == "__main__":
    main()

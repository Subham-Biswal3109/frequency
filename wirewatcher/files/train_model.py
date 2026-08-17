"""
Wire Watcher — ML Training Pipeline
====================================
Final Year Project

WHAT THIS SCRIPT DOES
----------------------
1. Loads the ML training samples exported from MySQL (ml/data/ml_training_samples.csv).
2. Runs data-quality checks and prints a report.
3. Builds two experiments:
      Experiment A -> does NOT use occupancy_ratio / interference_score
      Experiment B -> DOES use occupancy_ratio / interference_score
4. Trains a Logistic Regression baseline and a Random Forest for each experiment,
   using class weighting to handle severe class imbalance.
5. Evaluates every model with precision / recall / F1 / ROC-AUC / confusion matrix
   (accuracy alone is not reported as the headline metric because the target is
   ~95% / 5% imbalanced and accuracy is misleading here).
6. Saves confusion-matrix images, a text evaluation report, and the best model
   (bundled together with its preprocessing pipeline) to disk.

IMPORTANT — READ BEFORE USING RESULTS IN YOUR REPORT
-------------------------------------------------------
* The dataset is SYNTHETIC (see README.txt / data_type column). It is NOT a
  real-world Indian spectrum-occupancy measurement set. Do not present these
  results as evidence that the system works on real spectrum data.
* target_occupancy_ratio is IDENTICAL to occupancy_ratio in every row of this
  export, and target_available is almost perfectly separable using
  occupancy_ratio alone. This means the synthetic label-generation process is
  very likely a (near-)direct function of occupancy_ratio / interference_score.
  Experiment B numbers should therefore be read as "how well can a model
  recover the synthetic labelling rule", not as a generalizable prediction
  result. See the leakage discussion printed at the end of this script and in
  ml/model_results.txt.
* No claims are made anywhere in this script about real-world deployment
  performance.

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
DATA_PATH = BASE_DIR / "data" / "ml_training_samples.csv"
MODEL_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"
REPORT_PATH = BASE_DIR / "model_results.txt"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target_available"

# Columns that must never be used as model inputs (identifiers, timestamps,
# leakage-labelled columns, or bookkeeping columns from the synthetic export).
NON_FEATURE_COLS = [
    "sample_id",
    "band_id",            # high-cardinality id-like field; region/state already
                           # carry the location signal in a lower-cardinality form
    "observation_time",
    "district",            # 100% missing in this export -> dropped entirely
    "target_available",    # target
    "target_occupancy_ratio",  # identical to occupancy_ratio -> pure leakage, never a feature
    "label_source",
    "data_type",
    "created_at",
]

CATEGORICAL_FEATURES = ["region", "state"]

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


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load the exported ML training samples CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Make sure ml_training_samples.csv is at "
            f"ml/data/ml_training_samples.csv"
        )
    df = pd.read_csv(path)
    return df


# ----------------------------------------------------------------------------
# 2. DATA QUALITY REPORT
# ----------------------------------------------------------------------------
def data_quality_report(df: pd.DataFrame) -> str:
    """Build a text data-quality report and return it as a string (also printed)."""
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
    add("")

    add("-- Duplicates --")
    add(f"Full-row duplicates       : {df.duplicated().sum()}")
    add(f"Duplicate sample_id rows  : {df['sample_id'].duplicated().sum()}")
    add("")

    add("-- Missing values (per column) --")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        add("No missing values found.")
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
    invalid_target_occ = (
        (df["target_occupancy_ratio"] < 0) | (df["target_occupancy_ratio"] > 1)
    ).sum()
    add(f"occupancy_ratio out of range        : {invalid_occ}")
    add(f"target_occupancy_ratio out of range  : {invalid_target_occ}")
    add("")

    add("-- Invalid target values (must be 0 or 1) --")
    valid_targets = set(df[TARGET_COL].unique().tolist())
    add(f"Unique target_available values: {sorted(valid_targets)}")
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

    add("-- Numerical feature ranges --")
    num_cols = [
        "start_frequency_mhz", "end_frequency_mhz", "bandwidth_mhz",
        "hour_of_day", "day_of_week", "signal_power_dbm", "noise_floor_dbm",
        "snr_db", "occupancy_ratio", "interference_score",
    ]
    desc = df[num_cols].describe().T[["min", "max", "mean", "std"]]
    add(desc.to_string(float_format=lambda x: f"{x:.3f}"))
    add("")

    add("-- Suspected leakage check --")
    exact_match = (df["target_occupancy_ratio"] == df["occupancy_ratio"]).mean()
    add(f"target_occupancy_ratio == occupancy_ratio for {exact_match:.1%} of rows.")
    grp = df.groupby(TARGET_COL)["occupancy_ratio"].agg(["mean", "std", "min", "max"])
    add("occupancy_ratio by class:")
    add(grp.to_string(float_format=lambda x: f"{x:.3f}"))
    add(
        "-> The two classes are almost perfectly separable using occupancy_ratio "
        "alone. This indicates the synthetic label-generation process derived "
        "target_available (near-)directly from occupancy_ratio / interference_score. "
        "See the leakage discussion at the end of this report."
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
    Build a ColumnTransformer that:
      - scales numeric features (StandardScaler)
      - one-hot encodes categorical features (region, state)
    This is bundled into the saved model so future prediction data goes
    through IDENTICAL transformations.
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
    """Fit a pipeline (preprocessor + estimator), evaluate it, save a confusion
    matrix image, and return a dict of metrics."""
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

    # Save confusion matrix image
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

    # ---- Load data ----
    df = load_data(DATA_PATH)

    # ---- Data quality report ----
    dq_report = data_quality_report(df)
    report_sections.append(dq_report)

    # ---- Basic cleanup ----
    # Drop the fully-empty district column and non-feature bookkeeping columns
    # up front; feature lists below explicitly select what each experiment uses.
    df = df.drop(columns=["district"], errors="ignore")

    y = df[TARGET_COL].astype(int)

    all_metrics = []
    best_pipe = None
    best_metrics = None
    best_experiment_data = None

    experiments = {
        "Experiment_A_no_occupancy": EXPERIMENT_A_NUMERIC,
        "Experiment_B_with_occupancy": EXPERIMENT_A_NUMERIC + EXPERIMENT_B_EXTRA_NUMERIC,
    }

    for exp_name, numeric_features in experiments.items():
        print("\n" + "=" * 70)
        print(f"{exp_name}")
        print("=" * 70)

        feature_cols = numeric_features + CATEGORICAL_FEATURES
        X = df[feature_cols].copy()

        # ---- Stratified train/test split ----
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

        exp_results = []
        for model_name, estimator in models_to_run.items():
            metrics, pipe = train_and_evaluate(
                model_name, estimator, preprocessor,
                X_train, X_test, y_train, y_test,
                exp_name, RESULTS_DIR,
            )
            exp_results.append(metrics)
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

            # Track best model by F1 on the positive (minority) class.
            # Experiment A models are preferred as the deployable candidate
            # because Experiment B is affected by the leakage issue described
            # below -- see best-model selection note further down.
            if exp_name == "Experiment_A_no_occupancy":
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

    # ---- Leakage discussion ----
    leakage_text = f"""
{"=" * 70}
TARGET LEAKAGE DISCUSSION
{"=" * 70}
target_occupancy_ratio is identical to occupancy_ratio in 100% of rows in
this export, and the two target_available classes are separated almost
perfectly by occupancy_ratio alone (see the data-quality report above).
This strongly suggests the synthetic data generator created
target_available as a (near-)deterministic function of occupancy_ratio /
interference_score, rather than these being independently-measured signals
that merely correlate with availability.

Consequence for Experiment B (which uses occupancy_ratio and
interference_score as features): the very high scores Experiment B produces
should NOT be interpreted as "the model learned to predict spectrum
availability well". They more likely reflect the model recovering the rule
that was used to generate the label in the first place -- i.e. target
leakage. In a real deployment, occupancy_ratio for a NOT-yet-assigned
candidate frequency would not be known in advance in the same way it is
known here for a labelled historical sample, so a model that leans heavily
on occupancy_ratio may not transfer to genuine forward-looking prediction.

Experiment A (excluding occupancy_ratio and interference_score) is the more
defensible predictive setup for this FYP, because its features
(frequency band, time, and RF measurements not derived from the label) do
not have this same direct relationship to the label-generation process.
For that reason, the BEST MODEL SAVED to disk (wire_watcher_model.pkl) is
selected from Experiment A, not Experiment B, even where Experiment B shows
higher raw metrics.

FINAL CAVEATS
-------------
* This dataset is entirely SYNTHETIC. All numbers above describe how well a
  model can recover patterns in synthetic, generator-created labels -- NOT
  how well the system would perform on real-world Indian spectrum
  allocation. No claim is made that this model "works" for real-world
  spectrum allocation.
* Class imbalance is severe (~95:5). Accuracy is not a meaningful headline
  metric here; precision/recall/F1/ROC-AUC on the minority ("available")
  class are what should be reported and discussed in the FYP write-up.
"""
    print(leakage_text)
    report_sections.append(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    report_sections.append(leakage_text)

    # Per-model classification reports
    report_sections.append("\n" + "=" * 70)
    report_sections.append("PER-MODEL CLASSIFICATION REPORTS")
    report_sections.append("=" * 70)
    for m in all_metrics:
        report_sections.append(f"\n--- {m['experiment']} / {m['model']} ---")
        report_sections.append(m["classification_report"])
        report_sections.append(f"Confusion matrix image: {m['confusion_matrix_image']}")

    # ---- Save best model + preprocessing pipeline together ----
    model_bundle = {
        "pipeline": best_pipe,  # includes preprocessor + classifier
        "feature_columns": best_experiment_data["numeric_features"]
        + best_experiment_data["categorical_features"],
        "numeric_features": best_experiment_data["numeric_features"],
        "categorical_features": best_experiment_data["categorical_features"],
        "target_column": TARGET_COL,
        "experiment": best_metrics["experiment"],
        "model_name": best_metrics["model"],
        "random_state": RANDOM_STATE,
        "trained_on": "SYNTHETIC data (data_type=synthetic) — see README.txt",
    }
    model_path = MODEL_DIR / "wire_watcher_model.pkl"
    joblib.dump(model_bundle, model_path)
    print(f"\nSaved best model bundle (pipeline + metadata) to: {model_path}")
    roc_str = f"{best_metrics['roc_auc']:.4f}" if best_metrics["roc_auc"] is not None else "N/A"
    print(
        f"Best model selected: {best_metrics['experiment']} / {best_metrics['model']} "
        f"(F1={best_metrics['f1']:.4f}, ROC-AUC={roc_str})"
    )

    # ---- Save text report ----
    full_report = "\n".join(report_sections)
    with open(REPORT_PATH, "w") as f:
        f.write(full_report)
    print(f"\nSaved full evaluation report to: {REPORT_PATH}")

    # ---- Save raw metrics as JSON too, for evaluate_model.py / reuse ----
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

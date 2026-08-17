"""
Wire Watcher — Standalone Model Evaluation Script (rebuilt)
================================================================
Final Year Project

Purpose
-------
Reload the saved model bundle (preprocessing pipeline + classifier +
metadata) from ml/saved_models/wire_watcher_model.pkl and re-evaluate it,
WITHOUT retraining. Useful for:
  - Verifying the saved model still works after a fresh checkout
  - Evaluating against a new CSV export with the same schema (e.g. a fresh
    pull from MySQL) once the DB connection is fixed
  - Producing evaluation numbers for your FYP report/demo without re-running
    the full training pipeline

Dataset schema expected
------------------------
Same columns as spectrum_occupancy_synthetic_33600.csv, i.e. at minimum:
record_id, band_id, start_frequency_mhz, end_frequency_mhz, service_type,
state, city, hour, day_of_week, bandwidth_mhz, signal_power_dbm,
noise_floor_dbm, snr_db, occupancy_ratio, interference_score,
target_available.

Reproducibility note
---------------------
When no --data argument is given, this script reloads
ml/data/spectrum_occupancy_synthetic_33600.csv and recreates the EXACT SAME
stratified train/test split used during training (same test_size=0.20 and
random_state=42), so the "test" portion here matches the held-out test set
train_model.py evaluated on. If you pass a different CSV via --data, the
model is evaluated on the ENTIRE file you provide instead (no split is
performed).

Usage
-----
    python evaluate_model.py
    python evaluate_model.py --data path/to/new_export.csv
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "spectrum_occupancy_synthetic_33600.csv"
MODEL_PATH = BASE_DIR / "saved_models" / "wire_watcher_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved Wire Watcher model.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to a CSV with the same schema as "
        "spectrum_occupancy_synthetic_33600.csv. If omitted, that file is "
        "used and the same train/test split as training is recreated.",
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"No saved model found at {MODEL_PATH}. Run train_model.py first."
        )

    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]
    target_column = bundle["target_column"]

    print("=" * 70)
    print("WIRE WATCHER — SAVED MODEL EVALUATION")
    print("=" * 70)
    print(f"Loaded model bundle from: {MODEL_PATH}")
    print(f"Model: {bundle['model_name']}   Experiment: {bundle['experiment']}")
    print(f"Features used: {feature_columns}")
    print(f"Trained on: {bundle['trained_on']}")
    print(f"Leakage screening note: {bundle['leakage_note']}")
    print()

    using_fresh_split = args.data is None
    data_path = Path(args.data) if args.data else DEFAULT_DATA_PATH
    df = pd.read_csv(data_path)

    # Match the hour -> hour_of_day rename used during training
    if "hour" in df.columns and "hour_of_day" not in df.columns:
        df = df.rename(columns={"hour": "hour_of_day"})

    missing_cols = [c for c in feature_columns + [target_column] if c not in df.columns]
    if missing_cols:
        raise SystemExit(
            f"The provided CSV is missing required columns: {missing_cols}"
        )

    X = df[feature_columns].copy()
    y = df[target_column].astype(int)

    if using_fresh_split:
        _, X_eval, _, y_eval = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print(
            f"No --data provided: recreated the original stratified test split "
            f"({len(X_eval)} rows) from spectrum_occupancy_synthetic_33600.csv."
        )
    else:
        X_eval, y_eval = X, y
        print(f"Evaluating on the full provided file: {data_path} ({len(X_eval)} rows).")

    y_pred = pipe.predict(X_eval)
    y_proba = pipe.predict_proba(X_eval)[:, 1] if hasattr(pipe, "predict_proba") else None

    print()
    print(f"Accuracy : {accuracy_score(y_eval, y_pred):.4f}")
    print(f"Precision: {precision_score(y_eval, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_eval, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_eval, y_pred, zero_division=0):.4f}")
    if y_proba is not None and len(np.unique(y_eval)) == 2:
        print(f"ROC-AUC  : {roc_auc_score(y_eval, y_proba):.4f}")

    print()
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(confusion_matrix(y_eval, y_pred))

    print()
    print("Classification report:")
    print(classification_report(y_eval, y_pred, zero_division=0))

    print(
        "\nReminder: this model was trained on a SYNTHETIC dataset "
        "(spectrum_occupancy_synthetic_33600.csv) and is a PROTOTYPE. These "
        "numbers do not establish real-world spectrum-allocation prediction "
        "accuracy."
    )


if __name__ == "__main__":
    main()

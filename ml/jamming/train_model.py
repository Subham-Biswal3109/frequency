"""
Trains the RF Interference/Jamming Detector on real experimental RF
spectral-scan measurements (release_artifacts.zip).

This is an ADDITIVE model. It does NOT retrain, modify, or replace the
existing spectrum-availability Random Forest
(ml/training/train_model.py -> ml/artifacts/wire_watcher_model.pkl).
This script writes to a separate artifact path:
    ml/artifacts/jamming_detector_model.pkl
    ml/artifacts/jamming_detector_metadata.json

Run:
    python ml/jamming/train_model.py
(Assumes ml/jamming/prepare_dataset.py has already been run — see its
docstring for how ml/data/jamming_release/jamming_features_labeled.csv.gz
was produced.)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
)
from sklearn.preprocessing import StandardScaler

from prepare_dataset import (
    FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS, split, build_controlled_eval_set,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "ml" / "data" / "jamming_release" / "jamming_features_labeled.csv.gz"
ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "jamming_detector_model.pkl"
METADATA_PATH = ARTIFACTS_DIR / "jamming_detector_metadata.json"
TEST_SAMPLES_PATH = ARTIFACTS_DIR / "jamming_detector_test_samples.json"

RANDOM_STATE = 42
N_DEMO_SAMPLES = 300  # small held-out subset persisted for the frontend demo picker


def encode_categoricals(df: pd.DataFrame, band_categories, scan_mode_categories) -> pd.DataFrame:
    """One-hot encode band/scan_mode with a fixed, documented category set
    (fit-time categories are saved to metadata so inference-time encoding
    is always consistent, even if a category is absent from a given batch)."""
    out = df[FEATURE_COLUMNS].copy()
    for cat in band_categories:
        out[f"band__{cat}"] = (df["band"] == cat).astype(int)
    for cat in scan_mode_categories:
        out[f"scan_mode__{cat}"] = (df["scan_mode"] == cat).astype(int)
    return out


def energy_baseline_predict(df: pd.DataFrame, threshold: float) -> np.ndarray:
    """
    Simple RF/energy-based baseline (Phase 8): flag a file as malicious if
    its mean max-spectral-magnitude exceeds a threshold fit only on the
    training set's malicious-vs-benign separation (not hand-picked to
    maximize test performance). Uses a single feature already in our
    approved, leakage-free feature set.
    """
    return (df["max_magnitude_mean"] > threshold).astype(int).values


def fit_energy_threshold(train_df: pd.DataFrame) -> float:
    """Midpoint between benign and malicious mean max_magnitude_mean on
    the training set — a simple, transparent, non-learned baseline."""
    benign_mean = train_df.loc[train_df["target_malicious"] == 0, "max_magnitude_mean"].mean()
    malicious_mean = train_df.loc[train_df["target_malicious"] == 1, "max_magnitude_mean"].mean()
    return (benign_mean + malicious_mean) / 2


def evaluate(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # [[TN, FP], [FN, TP]]
    }
    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba), 4)
        metrics["pr_auc"] = round(average_precision_score(y_true, y_proba), 4)
    return metrics


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    train_df, val_df, test_df = split(df)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    band_categories = sorted(df["band"].unique().tolist())
    scan_mode_categories = sorted(df["scan_mode"].unique().tolist())

    X_train = encode_categoricals(train_df, band_categories, scan_mode_categories)
    X_val = encode_categoricals(val_df, band_categories, scan_mode_categories)
    X_test = encode_categoricals(test_df, band_categories, scan_mode_categories)
    y_train, y_val, y_test = train_df["target_malicious"], val_df["target_malicious"], test_df["target_malicious"]

    feature_names = list(X_train.columns)
    print(f"Final feature count: {len(feature_names)}")

    # ---- Phase 8: baselines ----
    print("\n--- Baseline 1: Energy threshold (max_magnitude_mean) ---")
    energy_threshold = fit_energy_threshold(train_df)
    energy_pred_test = energy_baseline_predict(test_df, energy_threshold)
    energy_metrics = evaluate(y_test, energy_pred_test)
    print(json.dumps(energy_metrics, indent=2))

    print("\n--- Baseline 2: Logistic Regression (class_weight=balanced) ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    logreg.fit(X_train_scaled, y_train)
    logreg_pred = logreg.predict(X_test_scaled)
    logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_metrics = evaluate(y_test, logreg_pred, logreg_proba)
    print(json.dumps(logreg_metrics, indent=2))

    print("\n--- Baseline 3: Decision Tree (class_weight=balanced) ---")
    dtree = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE)
    dtree.fit(X_train, y_train)
    dtree_pred = dtree.predict(X_test)
    dtree_proba = dtree.predict_proba(X_test)[:, 1]
    dtree_metrics = evaluate(y_test, dtree_pred, dtree_proba)
    print(json.dumps(dtree_metrics, indent=2))

    # ---- Phase 7: Random Forest (primary model) ----
    print("\n--- Random Forest (class_weight=balanced) ---")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # ---- Controlled, environment-free evaluation set (the headline metric) ----
    # See prepare_dataset.build_controlled_eval_set(): whichever of val/test
    # actually contains BOTH classes within rf_chamber becomes the primary
    # reported evaluation, since it's the only genuinely same-environment
    # benign-vs-malicious comparison this dataset supports.
    controlled_df, controlled_source = build_controlled_eval_set(val_df, test_df)
    print(f"\nControlled (same-environment) evaluation set drawn from: {controlled_source} split, n={len(controlled_df)}")

    # Threshold tuning: since the controlled set equals one of val/test in
    # this dataset (val is 100% in-chamber here), reusing it to also pick
    # the decision threshold would be circular. Instead we tune the
    # threshold with group-aware cross-validation ON TRAIN ONLY, then
    # refit on the full training set — val/test are never touched for
    # threshold selection.
    print("Tuning threshold via 5-fold group-aware CV on the training set only...")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(
        RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                random_state=RANDOM_STATE, n_jobs=-1),
        X_train, y_train, cv=cv, groups=train_df["session_key"], method="predict_proba", n_jobs=-1,
    )[:, 1]
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1
    for t in thresholds:
        f1 = f1_score(y_train, (oof_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"Best threshold from train CV: {best_t:.2f} (CV F1={best_f1:.4f})")

    # ---- Headline metric: controlled (same-environment) evaluation ----
    X_controlled = encode_categoricals(controlled_df, band_categories, scan_mode_categories)
    controlled_proba = rf.predict_proba(X_controlled)[:, 1]
    controlled_pred = (controlled_proba >= best_t).astype(int)
    controlled_metrics = evaluate(controlled_df["target_malicious"], controlled_pred, controlled_proba)
    print(f"\n=== HEADLINE METRIC: controlled same-environment evaluation ({controlled_source} split, in-chamber only) ===")
    print(json.dumps(controlled_metrics, indent=2))

    # ---- Supplementary: raw test-split metrics (reported transparently, NOT the headline) ----
    X_test = encode_categoricals(test_df, band_categories, scan_mode_categories)
    test_proba = rf.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_t).astype(int)
    test_env_composition = test_df["collection_environment"].value_counts().to_dict()
    test_is_confounded = test_df.groupby("collection_environment")["target_malicious"].nunique().max() < 2 or \
        len(test_df["collection_environment"].unique()) < 2
    rf_metrics_raw_test = evaluate(test_df["target_malicious"], test_pred, test_proba)
    print(f"\n=== SUPPLEMENTARY: raw test-split metrics (test env composition: {test_env_composition}) ===")
    if test_is_confounded:
        print("WARNING: this test split does not contain a same-environment class comparison "
              "(see composition above) -- these numbers are NOT the headline result and should "
              "not be quoted as the model's real-world accuracy. Reported for transparency only.")
    print(json.dumps(rf_metrics_raw_test, indent=2))

    # ---- Group audit (Section 2 of the review): sizes, class/env distribution ----
    print("\n--- GROUP AUDIT: all 131 session groups (size, malicious rate, environment) ---")
    group_audit = df.groupby("session_key").agg(
        size=("target_malicious", "size"),
        malicious_rate=("target_malicious", "mean"),
        environment=("collection_environment", "first"),
    ).sort_values("size", ascending=False)
    print(group_audit.to_string())
    train_group_sizes = train_df.groupby("session_key").size()
    val_group_sizes = val_df.groupby("session_key").size()
    test_group_sizes = test_df.groupby("session_key").size()
    print(f"\nTrain groups: {len(train_group_sizes)}, sizes range {train_group_sizes.min()}-{train_group_sizes.max()}")
    print(f"Val groups: {len(val_group_sizes)}, sizes range {val_group_sizes.min()}-{val_group_sizes.max()}")
    print(f"Test groups: {len(test_group_sizes)}, sizes range {test_group_sizes.min()}-{test_group_sizes.max()}")

    # ---- Generalization test: unseen LOCATION (only axis this dataset actually supports) ----
    # Malicious samples exist only in the rf_chamber, so an unseen-location
    # test can only meaningfully be run within the benign class (comparing
    # real-world location1/2/3). We report this honestly as a benign-only
    # cross-location generalization check, NOT a jamming-detection
    # generalization test, since the dataset does not support the latter.
    print("\n--- GENERALIZATION TEST: unseen real-world location (benign-only; malicious is chamber-only, see limitations) ---")
    benign_real_world = df[(df["target_malicious"] == 0) & (df["collection_environment"] == "real_world_indoor")]
    locations = sorted(benign_real_world["location"].unique())
    print(f"Available real-world locations: {locations}")
    if len(locations) >= 2:
        held_out_location = locations[0]
        print(f"NOTE: this only tests whether ambient-RF FEATURE STATISTICS shift across physical "
              f"locations (a distribution-shift sanity check), since all rows here share the same "
              f"true label (benign). It is not a test of jamming-detection generalization.")
        loc_train = df[~((df["target_malicious"] == 0) & (df["collection_environment"] == "real_world_indoor") & (df["location"] == held_out_location))]
        loc_test = benign_real_world[benign_real_world["location"] == held_out_location]
        X_loc_train = encode_categoricals(loc_train, band_categories, scan_mode_categories)
        m_loc = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                        random_state=RANDOM_STATE, n_jobs=-1)
        m_loc.fit(X_loc_train, loc_train["target_malicious"])
        X_loc_test = encode_categoricals(loc_test, band_categories, scan_mode_categories)
        loc_proba = m_loc.predict_proba(X_loc_test)[:, 1]
        loc_pred = (loc_proba >= best_t).astype(int)
        print(f"Held out location: {held_out_location}, n={len(loc_test)} (all true-benign)")
        print(f"Predicted-benign rate on unseen location: {(loc_pred == 0).mean():.4f} "
              f"(1.0 would mean perfect agreement with the true label on unseen premises)")
    else:
        print("Insufficient distinct locations to run this check.")

    feature_importances = sorted(
        zip(feature_names, rf.feature_importances_.tolist()),
        key=lambda x: x[1], reverse=True,
    )
    print("\nTop 15 feature importances:")
    for name, imp in feature_importances[:15]:
        print(f"  {name}: {imp:.4f}")

    # ---- Save artifacts ----
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "feature_names": feature_names}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    metadata = {
        "model_name": "jamming_detector",
        "model_version": "1.1.0",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "task": "RF interference / jamming detection (benign vs malicious)",
        "NOT": "This is not a spectrum availability / occupancy model. See README.md.",
        "dataset_name": "release_artifacts (real experimental RF spectral-scan captures)",
        "dataset_type": "REAL_MEASURED",
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "num_session_groups": {"train": len(train_group_sizes), "val": len(val_group_sizes), "test": len(test_group_sizes)},
        "class_balance_overall": df["target_malicious"].value_counts(normalize=True).to_dict(),
        "feature_columns": feature_names,
        "categorical_encoding": {"band": band_categories, "scan_mode": scan_mode_categories},
        "algorithm": "RandomForestClassifier",
        "hyperparameters": rf.get_params(),
        "best_threshold": round(float(best_t), 2),
        "threshold_tuning_methodology": (
            "5-fold group-aware (StratifiedGroupKFold) cross-validation on the TRAINING set only, "
            "maximizing F1 on out-of-fold predictions. Neither val nor test was used to select the "
            "threshold, since the controlled evaluation set below is drawn from val/test directly."
        ),
        "split_methodology": (
            "Group-aware (StratifiedGroupKFold) split using whole, UNDIVIDED experimental sessions "
            "as groups (label, benign_subtype, location, waveform, power_dbm, channel_mhz, band, "
            "scan_mode) -- no continuous recording run is ever divided between splits. This dataset "
            "has only 131 such sessions and a few are very large, so stratifying by label alone can, "
            "by chance, produce a test split with skewed environment representation (this happened "
            "with this exact seed: this split's test set ended up 100% environment-confounded -- see "
            "'headline_vs_supplementary_metrics' below). We did NOT force artificial balance via "
            "chunking sessions into smaller pieces, since that would let temporally-adjacent (and "
            "therefore correlated) slices of the SAME recording run cross the train/test boundary -- "
            "a subtler violation of independence than the imbalance it would fix. True independence "
            "was prioritized over balanced classes, per the project's explicit instruction."
        ),
        "headline_vs_supplementary_metrics": {
            "controlled_same_environment_eval": {
                "description": (
                    f"Headline metric. Drawn entirely from the '{controlled_source}' split, "
                    "restricted to collection_environment=='rf_chamber' (the only environment "
                    "containing both classes), so environment cannot explain the score."
                ),
                "n": len(controlled_df),
                "class_distribution": controlled_df["target_malicious"].value_counts().to_dict(),
                "metrics": controlled_metrics,
            },
            "raw_test_split_eval": {
                "description": (
                    "Supplementary only. This test split happened to be 100% environment-confounded "
                    "(real_world_indoor is 100% benign, rf_chamber is 100% malicious in THIS split) -- "
                    "so a high score here partly reflects 'which environment' rather than 'is this "
                    "signal malicious'. Reported transparently, not used as the headline result."
                ),
                "n": len(test_df),
                "environment_composition": test_env_composition,
                "is_environment_confounded": bool(test_is_confounded),
                "metrics": rf_metrics_raw_test,
            },
        },
        "metrics": {
            "energy_baseline": energy_metrics,
            "logistic_regression_baseline": logreg_metrics,
            "decision_tree_baseline": dtree_metrics,
            "random_forest_controlled": controlled_metrics,
            "random_forest_raw_test_confounded": rf_metrics_raw_test,
        },
        "energy_baseline_threshold_max_magnitude_mean": round(float(energy_threshold), 4),
        "feature_importances": feature_importances,
        "limitations": [
            "Labels reflect a controlled benign-vs-jamming experimental protocol, not "
            "a general spectrum occupancy/availability ground truth.",
            "All malicious (jamming) samples were collected in an RF chamber; all "
            "real_world_indoor samples are benign -- this dataset CANNOT support a test of "
            "jamming-detection generalization to real-world environments, and no such claim is made.",
            "Only 131 distinct experimental sessions underlie 96,090 files, several of them very "
            "large (13,000+ rows); this limits how finely train/val/test proportions can be tuned "
            "without dividing a single continuous recording run across splits.",
            "The raw test split for this run happened to be 100% environment-confounded (see "
            "headline_vs_supplementary_metrics) -- its near-perfect score should not be quoted as "
            "the model's real detection accuracy; use the controlled_same_environment_eval instead.",
            "rssi_dbm-style calibrated dBm values are not used as a feature. This field already "
            "existed in the release's own derived file_level_features table (we did not compute or "
            "introduce it); we verified empirically that rssi_dbm == rssi - 95 for every row with "
            "zero variance in that offset, and the release's data dictionary documents `rssi` only "
            "as a driver-scale value, not a calibrated measurement -- so we exclude rssi_dbm and use "
            "only the documented driver-scale rssi_* statistics.",
            "Frequency coverage is limited to 2.4GHz and 5GHz Wi-Fi bands only.",
            "The unseen-location generalization check only covers the benign class (malicious is "
            "chamber-only, so no unseen-location test of jamming detection is possible with this data).",
        ],
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata to {METADATA_PATH}")

    # ---- Persist a small held-out demo sample set (for the frontend picker) ----
    # Drawn from controlled_df (not the confounded raw test split) so the
    # demo picker shows genuine same-environment benign AND malicious
    # examples rather than an environment-confounded mix.
    rng = np.random.RandomState(RANDOM_STATE)
    demo_idx = rng.choice(controlled_df.index, size=min(N_DEMO_SAMPLES, len(controlled_df)), replace=False)
    demo_df = controlled_df.loc[demo_idx]
    demo_X = encode_categoricals(demo_df, band_categories, scan_mode_categories)
    demo_proba = rf.predict_proba(demo_X)[:, 1]
    demo_pred = (demo_proba >= best_t).astype(int)

    samples = []
    for i, (idx, row) in enumerate(demo_df.iterrows()):
        samples.append({
            "sample_id": f"test_{idx}",
            "file_name": row["file_name"],
            "true_label": row["label"],
            "band": row["band"],
            "scan_mode": row["scan_mode"],
            "waveform": row["waveform"] if pd.notna(row["waveform"]) else None,
            "power_dbm": row["power_dbm"] if pd.notna(row["power_dbm"]) else None,
            "features": {c: float(row[c]) for c in FEATURE_COLUMNS},
            "model_probability_malicious": round(float(demo_proba[i]), 4),
            "model_prediction": "malicious" if demo_pred[i] == 1 else "benign",
        })
    with open(TEST_SAMPLES_PATH, "w") as f:
        json.dump({"threshold": float(best_t), "samples": samples}, f, indent=2)
    print(f"Saved {len(samples)} demo samples to {TEST_SAMPLES_PATH}")


if __name__ == "__main__":
    main()

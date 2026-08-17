import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "spectrum_occupancy_synthetic_33600.csv"
MODEL_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = BASE_DIR / "results"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target_available"
CATEGORICAL_FEATURES = ["state", "city", "service_type"]
NUMERIC_FEATURES = [
    "start_frequency_mhz",
    "end_frequency_mhz",
    "bandwidth_mhz",
    "hour_of_day",
    "day_of_week",
    "signal_power_dbm",
    "noise_floor_dbm",
    "snr_db",
]
GROUP_COL = "city"

def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

def main():
    df = pd.read_csv(DATA_PATH)
    if 'hour' in df.columns:
        df = df.rename(columns={"hour": "hour_of_day"})
    
    # Calculate P1 and P99 bounds for OOD metadata
    numeric_bounds = {}
    for col in NUMERIC_FEATURES:
        numeric_bounds[col] = {
            "p1": float(df[col].quantile(0.01)),
            "p99": float(df[col].quantile(0.99))
        }

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].astype(int)
    groups = df[GROUP_COL]

    # 1. SPLIT: Train(60), Validation(20), Test(20)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=RANDOM_STATE)
    train_idx, val_test_idx = next(gss.split(X, y, groups))
    
    X_train, y_train, g_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    X_temp, y_temp, g_temp = X.iloc[val_test_idx], y.iloc[val_test_idx], groups.iloc[val_test_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss_val.split(X_temp, y_temp, g_temp))
    
    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]
    
    print(f"Train samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    # 2. Random Forest Tuning (Uncalibrated, balanced weights) on Train
    rf_base = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    print("Tuning Random Forest on Train set...")
    rs = RandomizedSearchCV(rf_base, param_distributions=param_grid, n_iter=5, scoring='f1', cv=3, random_state=RANDOM_STATE, n_jobs=-1)
    rs.fit(X_train_transformed, y_train)
    best_rf = rs.best_estimator_

    # 3. Find optimal decision threshold on Validation set
    print("Optimizing Decision Threshold on Validation set...")
    best_t = 0.5
    best_f1 = 0
    probs_val = best_rf.predict_proba(X_val_transformed)[:, 1]
    for t in np.arange(0.1, 0.9, 0.05):
        preds_t = (probs_val >= t).astype(int)
        score = f1_score(y_val, preds_t, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    
    print(f"Optimal threshold found: {best_t:.2f} (Val F1: {best_f1:.4f})")
    
    # 4. Final Evaluation on Untouched Test Set
    print("Evaluating metrics on untouch Test set...")
    probs_test = best_rf.predict_proba(X_test_transformed)[:, 1]
    y_pred = (probs_test >= best_t).astype(int)
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs_test)),
        "pr_auc": float(average_precision_score(y_test, probs_test)),
        "brier_score": float(brier_score_loss(y_test, probs_test)),
        "false_available": int(fp),
        "false_occupied": int(fn)
    }

    # 5. Extract Feature Importances & Permutation Importance on Test set
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cats = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = NUMERIC_FEATURES + list(encoded_cats)
    
    importances = best_rf.feature_importances_
    feat_imp = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print("Calculating Permutation Importance on Test set...")
    perm_result = permutation_importance(best_rf, X_test_transformed, y_test, scoring='f1', n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
    perm_imp = sorted(zip(all_feature_names, perm_result.importances_mean), key=lambda x: x[1], reverse=True)
    
    print("\n--- TREE IMPURITY IMPORTANCE ---")
    for feat, imp in feat_imp[:5]:
        print(f"{feat:30} {imp:.4f}")
        
    print("\n--- PERMUTATION IMPORTANCE (F1) ---")
    for feat, imp in perm_imp[:5]:
        print(f"{feat:30} {imp:.4f}")

    # Save Pipeline using the Uncalibrated RF
    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", best_rf)])
    
    model_bundle = {
        "pipeline": final_pipeline,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    joblib.dump(model_bundle, MODEL_DIR / "wire_watcher_model.pkl")

    # Generate metadata JSON according to precise specs
    metadata = {
      "model_version": "v1.0",
      "training_date": datetime.utcnow().isoformat(),
      "dataset_name": "spectrum_occupancy_synthetic_33600.csv",
      "dataset_type": "synthetic",
      "training_samples": len(X_train),
      "validation_samples": len(X_val),
      "test_samples": len(X_test),
      "real_rf_validation": False,
      "data_source": "Synthetic",
      "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
      "preprocessing": "StandardScaler, OneHotEncoder",
      "algorithm": "RandomForestClassifier",
      "hyperparameters": {
          "n_estimators": best_rf.n_estimators,
          "max_depth": best_rf.max_depth,
          "min_samples_split": best_rf.min_samples_split,
          "min_samples_leaf": best_rf.min_samples_leaf,
          "class_weight": "balanced"
      },
      "calibration_method": "None (Uncalibrated)",
      "best_threshold": best_t,
      "metrics": metrics,
      "feature_importances": {k: float(v) for k, v in feat_imp[:15]},
      "permutation_importances": {k: float(v) for k, v in perm_imp[:15]},
      "training_bounds": numeric_bounds,
      "ood_methodology": "Strict boundary check against 1st and 99th percentile of training distributions for SNR and Signal Power.",
      "limitations": [
        "Training data is strictly synthetic.",
        "Categorical features may dominate unpredictably if not representing real propagation physics.",
        "No real RF occupancy measurements were used.",
        "Model serves as a pipeline prototype for spectrum availability estimation."
      ]
    }
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nSaved final model pipeline and model_metadata.json.")

if __name__ == "__main__":
    main()

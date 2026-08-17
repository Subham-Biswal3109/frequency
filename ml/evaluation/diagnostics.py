import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, balanced_accuracy_score,
    average_precision_score, brier_score_loss
)
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "spectrum_occupancy_synthetic_33600.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    if 'hour' in df.columns:
        df = df.rename(columns={'hour': 'hour_of_day'})
        
    TARGET_COL = "target_available"
    CATEGORICAL_FEATURES = ["state", "city", "service_type"]
    NUMERIC_FEATURES = [
        "start_frequency_mhz", "end_frequency_mhz", "bandwidth_mhz",
        "hour_of_day", "day_of_week", "signal_power_dbm",
        "noise_floor_dbm", "snr_db"
    ]
    
    print("--- 1. CLASS IMBALANCE ---")
    c0 = (df[TARGET_COL] == 0).sum()
    c1 = (df[TARGET_COL] == 1).sum()
    print(f"Class 0 (Occupied): {c0} ({(c0/len(df))*100:.2f}%)")
    print(f"Class 1 (Available): {c1} ({(c1/len(df))*100:.2f}%)")
    print(f"Imbalance ratio: {c0/c1:.2f}:1")
    
    # Split
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].astype(int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, df['city']))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
    ])
    X_tr = preprocessor.fit_transform(X_train)
    X_te = preprocessor.transform(X_test)
    
    models = {
        "LogReg": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        "GBM": GradientBoostingClassifier(random_state=42),
        "RF_Balanced": RandomForestClassifier(class_weight="balanced", n_estimators=100, max_depth=10, random_state=42),
    }
    
    print("\n--- MODEL TRAINING & BASE METRICS ---")
    fitted = {}
    metrics_list = []
    
    for name, m in models.items():
        m.fit(X_tr, y_train)
        fitted[name] = m
        preds = m.predict(X_te)
        probs = m.predict_proba(X_te)[:, 1]
        
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        metrics_list.append({
            "Model": name,
            "Acc": accuracy_score(y_test, preds),
            "Bal_Acc": balanced_accuracy_score(y_test, preds),
            "Prec": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, probs),
            "PR_AUC": average_precision_score(y_test, probs),
            "Brier": brier_score_loss(y_test, probs),
            "False_Avail(FP)": fp,
            "False_Occ(FN)": fn
        })
        
    # Add calibrations
    rf_iso = CalibratedClassifierCV(fitted["RF_Balanced"], cv=3, method="isotonic")
    rf_iso.fit(X_tr, y_train)
    fitted["RF_Isotonic"] = rf_iso
    
    rf_sig = CalibratedClassifierCV(fitted["RF_Balanced"], cv=3, method="sigmoid")
    rf_sig.fit(X_tr, y_train)
    fitted["RF_Sigmoid"] = rf_sig
    
    for name in ["RF_Isotonic", "RF_Sigmoid"]:
        m = fitted[name]
        preds = m.predict(X_te)
        probs = m.predict_proba(X_te)[:, 1]
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics_list.append({
            "Model": name,
            "Acc": accuracy_score(y_test, preds),
            "Bal_Acc": balanced_accuracy_score(y_test, preds),
            "Prec": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1": f1_score(y_test, preds, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, probs),
            "PR_AUC": average_precision_score(y_test, probs),
            "Brier": brier_score_loss(y_test, probs),
            "False_Avail(FP)": fp,
            "False_Occ(FN)": fn
        })
        
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(pd.DataFrame(metrics_list).to_string(index=False, float_format="%.4f"))

    print("\n--- 2. THRESHOLD ANALYSIS (RF_Balanced Uncalibrated) ---")
    probs_rf = fitted["RF_Balanced"].predict_proba(X_te)[:, 1]
    thresh_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    t_metrics = []
    for t in thresh_list:
        p_t = (probs_rf >= t).astype(int)
        cm = confusion_matrix(y_test, p_t, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        t_metrics.append({
            "Threshold": t,
            "Acc": accuracy_score(y_test, p_t),
            "Bal_Acc": balanced_accuracy_score(y_test, p_t),
            "Prec": precision_score(y_test, p_t, zero_division=0),
            "Recall": recall_score(y_test, p_t, zero_division=0),
            "F1": f1_score(y_test, p_t, zero_division=0),
            "FP(False Avail)": fp,
            "FN(False Occ)": fn
        })
    print(pd.DataFrame(t_metrics).to_string(index=False, float_format="%.4f"))

    print("\n--- 3 & 4. FEATURE IMPORTANCE (IMPURITY vs PERMUTATION) ---")
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cats = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = NUMERIC_FEATURES + list(encoded_cats)
    
    impurity = fitted["RF_Balanced"].feature_importances_
    
    print("Calculating permutation importance...")
    perm_result = permutation_importance(fitted["RF_Balanced"], X_te, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    
    feat_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Impurity": impurity,
        "Permutation": perm_result.importances_mean
    }).sort_values("Permutation", ascending=False).head(15)
    print(feat_df.to_string(index=False, float_format="%.4f"))

    print("\n--- 5 & 6. SNR SENSITIVITY TESTING (Using RF_Balanced) ---")
    base_data = pd.DataFrame([{
        "start_frequency_mhz": 1800.0,
        "end_frequency_mhz": 1810.0,
        "bandwidth_mhz": 10.0,
        "hour_of_day": 14,
        "day_of_week": 1,
        "signal_power_dbm": -100.0,
        "noise_floor_dbm": -100.0,
        "snr_db": 0.0,
        "state": "Karnataka",
        "city": "Bengaluru",
        "service_type": "4G LTE"
    }])
    
    sens_res = []
    signals = [-100, -95, -90, -85, -80, -75, -70, -65]
    for s in signals:
        test_df = base_data.copy()
        test_df["signal_power_dbm"] = s
        test_df["snr_db"] = s - (-100)
        X_tmp = preprocessor.transform(test_df)
        prob = fitted["RF_Balanced"].predict_proba(X_tmp)[0, 1]
        sens_res.append({
            "Signal": s,
            "Noise": -100,
            "SNR": s - (-100),
            "Prob_Avail": prob,
            "Prediction(t=0.5)": 1 if prob >= 0.5 else 0
        })
    print(pd.DataFrame(sens_res).to_string(index=False, float_format="%.4f"))
    
    print("\n--- 7 & 8. FREQUENCY SENSITIVITY ---")
    freqs = [900, 1800, 2100, 2300, 3500, 5000, 24000]
    freq_res = []
    for f in freqs:
        test_df = base_data.copy()
        test_df["start_frequency_mhz"] = f
        test_df["end_frequency_mhz"] = f + 10
        # Keep SNR at a mid-level where it's ambiguous
        test_df["signal_power_dbm"] = -90
        test_df["snr_db"] = 10
        X_tmp = preprocessor.transform(test_df)
        prob = fitted["RF_Balanced"].predict_proba(X_tmp)[0, 1]
        freq_res.append({
            "Freq": f,
            "Prob_Avail": prob
        })
    print(pd.DataFrame(freq_res).to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    main()

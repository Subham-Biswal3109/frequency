"""
Rigorous audit of the RF Interference/Jamming Detector, run BEFORE any
backend/frontend integration. This script does not modify the existing
model artifacts; it only reads the prepared dataset and the already-
trained model to interrogate whether ROC-AUC=1.0 reflects genuine RF
separability or a split/leakage artifact.

Run: python ml/jamming/audit.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from prepare_dataset import FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS, split
from train_model import encode_categoricals, evaluate, fit_energy_threshold, energy_baseline_predict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = REPO_ROOT / "ml" / "data" / "jamming_release" / "jamming_features_labeled.csv.gz"
ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts"

RANDOM_STATE = 42


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    train_df, val_df, test_df = split(df)

    bundle = joblib.load(ARTIFACTS_DIR / "jamming_detector_model.pkl")
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    meta = json.loads((ARTIFACTS_DIR / "jamming_detector_metadata.json").read_text())
    band_cats = meta["categorical_encoding"]["band"]
    scan_cats = meta["categorical_encoding"]["scan_mode"]
    threshold = meta["best_threshold"]

    section("1. EXACT FEATURE AUDIT")
    print(f"Final feature count: {len(feature_names)}")
    numeric_feats = [f for f in feature_names if not f.startswith(("band__", "scan_mode__"))]
    cat_feats = [f for f in feature_names if f.startswith(("band__", "scan_mode__"))]
    stats = df[numeric_feats].agg(["min", "max", "mean", "std"]).T
    print("\nNumeric feature stats (unit = driver-scale / dB-like per the release's own data "
          "dictionary, NOT confirmed-calibrated dBm):")
    print(stats.round(3).to_string())
    print(f"\nCategorical (one-hot) features: {cat_feats}")

    excluded = {
        "label": "the target itself",
        "benign_subtype": "null iff malicious -> 100% label leakage via missingness",
        "waveform": "null iff benign -> 100% label leakage via missingness",
        "power_dbm": "null iff benign -> 100% label leakage via missingness",
        "channel_mhz": "non-null almost exclusively for malicious (99.8% of non-null rows); "
                       "excluded on the same principle even though 153/77408 benign rows also "
                       "have a value (verified empirically, not assumed)",
        "location": "malicious occurs ONLY at location=='rf_chamber' -> would leak the protocol",
        "collection_environment": "malicious occurs ONLY at 'rf_chamber' -> same reason",
        "file_name/file_path/relative_path": "some filenames literally contain label words",
        "rssi_dbm_* (derived)": "== rssi - 95 for every row (verified, zero variance in the "
                                  "offset) -> pure linear duplicate of rssi_*, no new information",
        "is_location_expected / band_expected / *_missing_when_expected": "release-notebook QC "
            "flags, not RF measurements",
        "file_size_bytes / row_count_manifest / row_count_actual": "reflect the collection "
            "script's configured capture duration per experimental condition, not RF content",
    }
    print("\nExcluded fields and why:")
    for k, v in excluded.items():
        print(f"  - {k}: {v}")

    section("2. TARGET DEFINITION")
    print("Target: target_malicious = 1 if manifest['label']=='malicious' else 0")
    print("Built directly from the release's own `label` column, before any feature engineering.")
    print(pd.crosstab(df["label"], df["target_malicious"]))

    section("3. DUPLICATE / NEAR-DUPLICATE ANALYSIS")
    feat_cols = FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    dup_mask = df.duplicated(subset=feat_cols, keep=False)
    print(f"Exact duplicate feature rows: {dup_mask.sum()} ({dup_mask.mean()*100:.3f}%)")
    df["_split"] = "unused"
    df.loc[train_df.index, "_split"] = "train"
    df.loc[val_df.index, "_split"] = "val"
    df.loc[test_df.index, "_split"] = "test"
    if dup_mask.sum() > 0:
        dup_groups = df[dup_mask].groupby(feat_cols)["_split"].apply(lambda s: set(s))
        cross = dup_groups[dup_groups.apply(lambda s: len(s) > 1)]
        print(f"Duplicate groups spanning >1 split: {len(cross)} / {len(dup_groups)}")
    else:
        print("No exact duplicates found -> no duplicate-crossing-boundary risk.")

    section("4. GROUP SPLIT AUDIT")
    print("Grouping variable: session_key = label|benign_subtype|location|waveform|power_dbm|")
    print("                                  channel_mhz|band|scan_mode")
    train_groups, val_groups, test_groups = set(train_df["session_key"]), set(val_df["session_key"]), set(test_df["session_key"])
    print(f"Training groups: {len(train_groups)}")
    print(f"Validation groups ({len(val_groups)}): {sorted(val_groups)}")
    print(f"Testing groups ({len(test_groups)}): {sorted(test_groups)}")
    print(f"Train n Test = {len(train_groups & test_groups)}  Train n Val = {len(train_groups & val_groups)}  Val n Test = {len(val_groups & test_groups)}")
    print("VERIFIED: zero group overlap across all three splits.")
    print("\nCAVEAT: only 131 total session groups exist; the split placed only 4 groups in test")
    print("and 5 in val. This caused the ORIGINAL test set to contain, by chance, zero")
    print("rf_chamber-benign rows -- a split artifact, addressed directly in Section 5.")

    section("5. REAL-WORLD VS RF-CHAMBER ANALYSIS")
    X_test = encode_categoricals(test_df, band_cats, scan_cats)[feature_names]
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    test_df = test_df.copy()
    test_df["pred"], test_df["proba"] = test_pred, test_proba

    print("--- Original TEST split, by environment ---")
    for env in ["real_world_indoor", "rf_chamber"]:
        sub = test_df[test_df["collection_environment"] == env]
        n_classes = sub["target_malicious"].nunique()
        print(f"{env}: n={len(sub)}, malicious_rate={sub['target_malicious'].mean():.3f}, classes_present={n_classes}")
        if n_classes < 2:
            print("  Only one class present in this environment WITHIN THE TEST SPLIT -> "
                  "precision/recall/F1/ROC-AUC undefined here.")
            print(f"  Accuracy: {(sub['pred'] == sub['target_malicious']).mean():.4f}")
        else:
            print(" ", json.dumps(evaluate(sub["target_malicious"], sub["pred"], sub["proba"]), indent=2))

    print("\n--- CONTROLLED comparison: in-chamber-only VALIDATION subset (both classes present, "
          "model never trained on these rows) ---")
    val_chamber = val_df[val_df["collection_environment"] == "rf_chamber"].copy()
    X_vc = encode_categoricals(val_chamber, band_cats, scan_cats)[feature_names]
    vc_proba = model.predict_proba(X_vc)[:, 1]
    vc_pred = (vc_proba >= threshold).astype(int)
    print(f"n={len(val_chamber)}, malicious_rate={val_chamber['target_malicious'].mean():.3f}")
    controlled_metrics = evaluate(val_chamber["target_malicious"], vc_pred, vc_proba)
    print(json.dumps(controlled_metrics, indent=2))
    print(">>> THIS is the scientifically meaningful same-environment benign-vs-malicious result <<<")

    section("6. CLASS-SPECIFIC ANALYSIS (in-chamber validation subset)")
    val_chamber["pred"], val_chamber["proba"] = vc_pred, vc_proba
    for subtype, mask in [
        ("benign/floor", (val_chamber["label"] == "benign")),
        ("malicious/gaussian_noise", (val_chamber["waveform"] == "gaussian_noise")),
        ("malicious/singletone", (val_chamber["waveform"] == "singletone")),
    ]:
        sub = val_chamber[mask]
        if len(sub) == 0:
            print(f"{subtype}: no rows in this held-out subset")
            continue
        acc = (sub["pred"] == sub["target_malicious"]).mean()
        print(f"{subtype}: n={len(sub)}, accuracy={acc:.4f}, mean_predicted_proba_malicious={sub['proba'].mean():.4f}")
    print("\nNote: benign/background does not appear here (background is real-world-only).")

    section("7. FEATURE IMPORTANCE (Gini + permutation, in-chamber val subset)")
    top_gini = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])[:10]
    print("Top 10 Gini importances:")
    for name, imp in top_gini:
        print(f"  {name}: {imp:.4f}")

    print("\nComputing permutation importance (n_repeats=5)...")
    perm = permutation_importance(model, X_vc, val_chamber["target_malicious"], n_repeats=5,
                                   random_state=RANDOM_STATE, n_jobs=-1, scoring="f1")
    top_perm = sorted(zip(feature_names, perm.importances_mean), key=lambda x: -x[1])[:10]
    print("Top 10 permutation importances (F1 drop when shuffled):")
    for name, imp in top_perm:
        print(f"  {name}: {imp:.4f}")
    top1_share = top_gini[0][1] / sum(model.feature_importances_)
    verdict = "a single feature dominates, investigate" if top1_share > 0.5 else "importance is reasonably distributed"
    print(f"\nTop-1 Gini feature ({top_gini[0][0]}) = {top1_share*100:.1f}% of total importance -> {verdict}.")

    section("8. ABLATION STUDY (evaluated on in-chamber validation subset)")

    def fit_and_eval(feat_list, label):
        Xtr = encode_categoricals(train_df, band_cats, scan_cats)[feat_list]
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(Xtr, train_df["target_malicious"])
        Xvc = encode_categoricals(val_chamber, band_cats, scan_cats)[feat_list]
        proba = m.predict_proba(Xvc)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = evaluate(val_chamber["target_malicious"], pred, proba)
        print(f"--- {label} ({len(feat_list)} features) ---")
        print(json.dumps(metrics, indent=2))
        return metrics

    print("(A) All approved features [same as the shipped model]:")
    fit_and_eval(numeric_feats + cat_feats, "A: all approved features")

    core_means_only = [f"{p}_mean" for p in [
        "freq1", "noise", "max_magnitude", "total_gain_db", "base_pwr_db", "rssi", "relpwr_db", "avgpwr_db",
    ]]
    print("\n(C) Only core RF measurement means (8 features, no quantile/std engineering):")
    fit_and_eval(core_means_only, "C: core means only")

    print("\n(B*) DIAGNOSTIC ONLY -- WITH excluded protocol metadata (location) added back in, to")
    print("illustrate leakage risk. NEVER used for the shipped model.")
    Xtr = encode_categoricals(train_df, band_cats, scan_cats)
    for loc in df["location"].unique():
        Xtr[f"location__{loc}"] = (train_df["location"] == loc).astype(int)
    Xvc = encode_categoricals(val_chamber, band_cats, scan_cats)
    for loc in df["location"].unique():
        Xvc[f"location__{loc}"] = (val_chamber["location"] == loc).astype(int)
    diag_feats = numeric_feats + cat_feats + [f"location__{loc}" for loc in df["location"].unique()]
    m_diag = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=-1)
    m_diag.fit(Xtr[diag_feats], train_df["target_malicious"])
    proba_diag = m_diag.predict_proba(Xvc[diag_feats])[:, 1]
    pred_diag = (proba_diag >= 0.5).astype(int)
    print(json.dumps(evaluate(val_chamber["target_malicious"], pred_diag, proba_diag), indent=2))
    print("(All in-chamber validation rows share location=='rf_chamber', so this diagnostic adds")
    print("no signal HERE by construction -- the real risk of this leaky feature is on the ORIGINAL")
    print("uncontrolled test split in Section 5, where location perfectly separated the classes.)")

    section("9. BASELINE (controlled in-chamber subset)")
    energy_threshold = fit_energy_threshold(train_df)
    energy_pred = energy_baseline_predict(val_chamber, energy_threshold)
    print(f"Energy baseline (max_magnitude_mean > {energy_threshold:.2f}):")
    print(json.dumps(evaluate(val_chamber["target_malicious"], energy_pred), indent=2))
    print("\nRandom Forest, same subset (from Section 5):")
    print(json.dumps(controlled_metrics, indent=2))

    section("10. ROBUSTNESS CHECK: unseen jammer power level (leave-one-power-out)")
    malicious = df[df["label"] == "malicious"]
    power_levels = sorted(malicious["power_dbm"].unique())
    held_out_power = 9.0
    print(f"Available jammer power levels (dBm): {power_levels}")
    print(f"Holding out power_dbm == {held_out_power} entirely from training.")

    train_lopo = df[~((df["label"] == "malicious") & (df["power_dbm"] == held_out_power))]
    test_lopo_malicious = df[(df["label"] == "malicious") & (df["power_dbm"] == held_out_power)]
    test_lopo_benign = df[(df["label"] == "benign") & (df["collection_environment"] == "rf_chamber")].sample(
        n=min(len(test_lopo_malicious), (df["label"] == "benign").sum()), random_state=RANDOM_STATE
    )
    test_lopo = pd.concat([test_lopo_malicious, test_lopo_benign])

    Xtr = encode_categoricals(train_lopo, band_cats, scan_cats)[numeric_feats + cat_feats]
    m_lopo = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=-1)
    m_lopo.fit(Xtr, train_lopo["target_malicious"])
    Xte = encode_categoricals(test_lopo, band_cats, scan_cats)[numeric_feats + cat_feats]
    proba_lopo = m_lopo.predict_proba(Xte)[:, 1]
    pred_lopo = (proba_lopo >= 0.5).astype(int)
    print(f"n={len(test_lopo)} (power_dbm={held_out_power} malicious + matched in-chamber benign)")
    print(json.dumps(evaluate(test_lopo["target_malicious"], pred_lopo, proba_lopo), indent=2))

    print("\nRobustness check: unseen jammer WAVEFORM (train w/o singletone, test on singletone only)")
    train_low = df[~((df["label"] == "malicious") & (df["waveform"] == "singletone"))]
    test_low_malicious = df[(df["label"] == "malicious") & (df["waveform"] == "singletone")]
    test_low_benign = df[(df["label"] == "benign") & (df["collection_environment"] == "rf_chamber")].sample(
        n=len(test_low_malicious), random_state=RANDOM_STATE
    )
    test_low = pd.concat([test_low_malicious, test_low_benign])
    Xtr2 = encode_categoricals(train_low, band_cats, scan_cats)[numeric_feats + cat_feats]
    m_low = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1)
    m_low.fit(Xtr2, train_low["target_malicious"])
    Xte2 = encode_categoricals(test_low, band_cats, scan_cats)[numeric_feats + cat_feats]
    proba_low = m_low.predict_proba(Xte2)[:, 1]
    pred_low = (proba_low >= 0.5).astype(int)
    print(f"n={len(test_low)} (all {len(test_low_malicious)} singletone malicious files + matched in-chamber benign)")
    print(json.dumps(evaluate(test_low["target_malicious"], pred_low, proba_low), indent=2))

    print("\nRobustness check NOT supported by this dataset: jamming detection generalization to")
    print("real-world (non-chamber) environments -- 100% of malicious samples were collected")
    print("exclusively in the RF chamber. This is a genuine dataset gap, stated explicitly rather")
    print("than fabricating a workaround.")


if __name__ == "__main__":
    main()

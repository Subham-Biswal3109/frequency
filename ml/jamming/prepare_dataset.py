"""
Dataset preparation for the RF Interference/Jamming Detector.

This is an ADDITIVE module: it does not touch the existing spectrum
availability pipeline (ml/training/train_model.py, ml/data/spectrum_occupancy_synthetic_33600.csv,
ml/artifacts/wire_watcher_model.pkl) in any way.

SOURCE DATA
-----------
Input: derived/file_level_features.parquet from the "release_artifacts.zip"
dataset release, which contains one row per raw RF spectral-scan CSV file
(96,090 files), with per-file statistical summaries (mean/std/min/q25/median/
q75/max/skew/kurtosis/nunique) of 8 driver-reported spectral fields, plus a
`label` column: "benign" (background RF / RF-chamber floor noise) or
"malicious" (an active RF jammer — gaussian_noise or singletone waveform —
was transmitting into a specific channel at a configured power).

This is a JAMMING / INTERFERENCE DETECTION dataset, not a spectrum
occupancy dataset. See ml/jamming/README.md for the full provenance and
label-methodology writeup. This script only prepares data for that
detection task — it is never used to train or evaluate spectrum
availability predictions.

The original file_level_features.parquet (38MB, 96,090 rows x 108 cols) is
NOT committed to this repository per the project's "do not commit huge
datasets unnecessarily" rule. This script assumes it is available locally
(e.g. re-extracted from release_artifacts.zip) and writes a trimmed,
leakage-audited, already-labeled CSV into ml/data/jamming_release/, which
IS committed (it is compact: selected numeric summary stats only).

FEATURE SELECTION AND LEAKAGE AUDIT (Phase 4 / Phase 5)
--------------------------------------------------------
KEPT as features (real RF-signal-derived statistics, available at
inference time for any new capture regardless of how it was collected):
    freq1_*, noise_*, max_magnitude_*, total_gain_db_*, base_pwr_db_*,
    rssi_*, relpwr_db_*, avgpwr_db_*   (mean, std, min, q25, median, q75, max)
    band, scan_mode                    (categorical, present for every file)

EXCLUDED, with reasons:
    - file_path, relative_path, file_name: identifiers; several filenames
      literally contain the words "jamming"/"background" — using them (or
      any text/hash derived from them) would leak the label directly.
    - label, benign_subtype: benign_subtype is the target's own subclass
      and is null 100% of the time for malicious rows — its *missingness*
      alone perfectly predicts the label. Direct leakage.
    - waveform, power_dbm, channel_mhz: all three are non-null ONLY for
      malicious rows (they describe the jammer's own configuration). Their
      missingness pattern perfectly predicts the label. Direct leakage.
    - location, collection_environment: empirically, malicious==True occurs
      ONLY at location=="rf_chamber"/collection_environment=="rf_chamber";
      every real_world_indoor file is benign. A model could trivially use
      "which room was this collected in" as a proxy for the label — that
      is a property of the experiment's protocol, not of the RF signal,
      and would not generalize to a live deployment. Excluded.
    - is_location_expected, band_expected, band_missing_when_expected,
      location_missing_when_expected: release-notebook integrity/QC flags,
      not RF measurements.
    - file_size_bytes, row_count_manifest, row_count_actual: reflect the
      data-COLLECTION SCRIPT's configured capture duration/size for each
      condition, not the RF spectral content. Kept as documented context
      only (see the "context_columns" list below, excluded from
      X/features used to fit the model).
    - rssi_dbm_*: this derived field equals rssi - 95 for every single
      file (a constant offset, verified empirically), i.e. it is a purely
      linear rescaling of rssi_* and contains no additional information.
      Its name implies a calibrated dBm reading, but the data dictionary
      documents `rssi` only as "Driver scale" (not confirmed-calibrated
      dBm) and does not explain or justify the -95 offset. Per this
      project's rule against claiming unconfirmed dBm calibration, these
      columns are excluded from the feature set entirely.
    - *_skew, *_kurtosis, *_nunique: kept out of the default feature set
      to keep the model compact and interpretable; the mean/std/quantile
      statistics already capture the distributional shape needed for this
      classification task. (Not a leakage concern — a scope choice.)

GROUP-AWARE SPLITTING (Phase 6)
--------------------------------
Files collected under the identical experimental condition (same label,
subtype/waveform, power, channel, band, location) are highly correlated
with each other (they are repeated captures of the same physical setup).
Randomly shuffling individual files into train/val/test would let the
model see near-duplicate conditions in both train and test, producing an
overly optimistic evaluation. We build a `session_key` from exactly the
condition-defining columns (never used as a model feature) and use it as
the GroupShuffleSplit group, so an entire experimental condition falls
into exactly one split.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold  # StratifiedGroupKFold retained for reference/comparison in audit.py

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SOURCE_PARQUET = REPO_ROOT / "ml" / "jamming" / "file_level_features.parquet"
OUTPUT_DIR = REPO_ROOT / "ml" / "data" / "jamming_release"
OUTPUT_CSV = OUTPUT_DIR / "jamming_features_labeled.csv.gz"

RAW_FIELD_PREFIXES = [
    "freq1", "noise", "max_magnitude", "total_gain_db",
    "base_pwr_db", "rssi", "relpwr_db", "avgpwr_db",
]
STAT_SUFFIXES = ["mean", "std", "min", "q25", "median", "q75", "max"]

FEATURE_COLUMNS = [f"{field}_{stat}" for field in RAW_FIELD_PREFIXES for stat in STAT_SUFFIXES]
CATEGORICAL_FEATURE_COLUMNS = ["band", "scan_mode"]

# Columns kept in the output CSV for transparency/session-grouping/display,
# but NEVER passed to the model as features.
CONTEXT_COLUMNS = [
    "file_name", "label", "benign_subtype", "location", "waveform",
    "power_dbm", "channel_mhz", "collection_environment",
    "file_size_bytes", "row_count_actual",
]

SESSION_KEY_COLUMNS = [
    "label", "benign_subtype", "location", "waveform",
    "power_dbm", "channel_mhz", "band", "scan_mode",
]


def build_session_key(df: pd.DataFrame) -> pd.Series:
    """
    Groups files by their full experimental condition (label, subtype,
    location, waveform, power, channel, band, scan_mode). Files sharing a
    condition are treated as one indivisible session for splitting, since
    they are typically one continuous acquisition run and therefore highly
    correlated with each other.

    NOTE: an earlier version of this function further chunked large
    sessions into fixed-size pseudo-sessions (by file_name's sequential
    index) to force balanced train/val/test proportions across every
    (label, environment) stratum. That was reverted: chunking splits a
    single continuous recording run into adjacent time-slices that can
    land in different splits, which is itself a subtler form of the same
    session-correlation problem group-based splitting exists to prevent —
    adjacent slices of one continuous chamber run are not independent
    measurements. See ml/jamming/README.md's "Split methodology" section
    for the full discussion and prepare_dataset.build_controlled_eval_set()
    for how we instead get an honest, representative evaluation despite a
    handful of very large, indivisible sessions.
    """
    parts = df[SESSION_KEY_COLUMNS].astype(str).fillna("NA")
    return parts.apply(lambda row: "|".join(row.values), axis=1)


def prepare(source_parquet: Path = DEFAULT_SOURCE_PARQUET) -> pd.DataFrame:
    if not source_parquet.exists():
        raise FileNotFoundError(
            f"Source file not found: {source_parquet}\n"
            "This script expects derived/file_level_features.parquet from "
            "release_artifacts.zip. Extract the release and pass its path, "
            "e.g.: python prepare_dataset.py /path/to/file_level_features.parquet"
        )

    df = pd.read_parquet(source_parquet)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {source_parquet}")

    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Expected feature columns missing from source: {missing_features}")

    if df[FEATURE_COLUMNS].isna().any().any():
        na_counts = df[FEATURE_COLUMNS].isna().sum()
        raise ValueError(f"Unexpected NaNs in feature columns:\n{na_counts[na_counts > 0]}")

    df["target_malicious"] = (df["label"] == "malicious").astype(int)
    df["session_key"] = build_session_key(df)

    kept_columns = FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS + CONTEXT_COLUMNS + [
        "target_malicious", "session_key",
    ]
    out = df[kept_columns].copy()

    print(f"Session groups: {out['session_key'].nunique()} unique experimental conditions")
    print(f"Class balance: {out['target_malicious'].value_counts(normalize=True).to_dict()}")

    return out


def split(df: pd.DataFrame, test_size=0.15, val_size=0.15, seed=42):
    """
    Group-aware, label-stratified train/val/test split. Groups (whole
    experimental sessions, see build_session_key) never cross a split
    boundary, and StratifiedGroupKFold keeps the overall malicious/benign
    ratio comparable across splits.

    IMPORTANT LIMITATION (see ml/jamming/audit.py): because malicious
    samples exist ONLY in the RF chamber, and because a handful of very
    large sessions dominate each stratum, stratifying by label alone can
    — by chance — produce a test split with little or no rf_chamber-benign
    representation, making "benign vs malicious" partly conflate with
    "real-world vs chamber". This did happen with seed=42 on this dataset
    (verified in the audit). We do NOT try to force perfect environment
    balance into this split (the fix we tried — chunking large sessions
    into smaller pseudo-sessions — reintroduces within-run correlation
    between train and test, which is worse). Instead:
      - This split is used for standard train/val/test model fitting, and
        its raw test-set metrics are still reported (see metadata) —
        but are NOT the headline number when they don't include a
        same-environment comparison.
      - build_controlled_eval_set() below constructs the scientifically
        meaningful "same-environment benign-vs-malicious" evaluation
        directly, from whichever of val/test genuinely contains both
        classes within rf_chamber, and that IS the headline reported
        metric (see train_model.py).
    """
    n_splits_outer = round(1 / test_size)
    sgkf_outer = StratifiedGroupKFold(n_splits=n_splits_outer, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf_outer.split(df, y=df["target_malicious"], groups=df["session_key"]))
    trainval_df, test_df = df.iloc[trainval_idx], df.iloc[test_idx]

    n_splits_inner = round(1 / (val_size / (1 - test_size)))
    sgkf_inner = StratifiedGroupKFold(n_splits=n_splits_inner, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf_inner.split(
        trainval_df, y=trainval_df["target_malicious"], groups=trainval_df["session_key"]
    ))
    train_df, val_df = trainval_df.iloc[train_idx], trainval_df.iloc[val_idx]

    train_keys, val_keys, test_keys = set(train_df["session_key"]), set(val_df["session_key"]), set(test_df["session_key"])
    assert not (train_keys & val_keys), "Session leakage between train and val!"
    assert not (train_keys & test_keys), "Session leakage between train and test!"
    assert not (val_keys & test_keys), "Session leakage between val and test!"

    return train_df, val_df, test_df


def build_controlled_eval_set(val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Returns (controlled_df, source_split_name).

    Picks whichever of val_df/test_df contains BOTH classes within
    collection_environment=='rf_chamber' (the only environment where both
    classes exist at all) and returns that environment-restricted subset.
    This is the scientifically meaningful "same-environment benign vs
    malicious" evaluation set: every row in it was held out of training,
    and no environmental confound can inflate the score.

    Raises if NEITHER val nor test happens to contain both classes
    in-chamber — in that case the split must be regenerated with a
    different seed rather than silently reporting a confounded metric.
    """
    for name, d in [("test", test_df), ("val", val_df)]:
        chamber = d[d["collection_environment"] == "rf_chamber"]
        if chamber["target_malicious"].nunique() == 2:
            return chamber.copy(), name
    raise RuntimeError(
        "Neither val nor test contains both classes within rf_chamber — "
        "cannot build a controlled, environment-free evaluation set from this split. "
        "Try a different random seed in split()."
    )


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE_PARQUET
    df = prepare(src)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, compression="gzip")
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")

    train_df, val_df, test_df = split(df)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"train malicious rate={train_df['target_malicious'].mean():.3f}")
    print(f"val malicious rate={val_df['target_malicious'].mean():.3f}")
    print(f"test malicious rate={test_df['target_malicious'].mean():.3f}")

"""
Inference service for the RF Interference/Jamming Detector.

This is a SEPARATE, additive model from the spectrum-availability Random
Forest in backend/services/prediction.py. It does not share model state,
thresholds, or database tables with that service. See ml/jamming/README.md
for the full dataset provenance and label methodology.
"""

import json

import joblib
import pandas as pd

from backend.config import (
    JAMMING_MODEL_PATH, JAMMING_METADATA_PATH, JAMMING_TEST_SAMPLES_PATH,
)

try:
    print("Loading jamming detector model...")
    jamming_model_bundle = joblib.load(JAMMING_MODEL_PATH)
    print("Jamming detector model loaded successfully.")
except Exception as e:
    print(f"Failed to load jamming detector model: {e}")
    jamming_model_bundle = None

try:
    with open(JAMMING_METADATA_PATH, "r") as f:
        jamming_metadata = json.load(f)
except Exception as e:
    print(f"Failed to load jamming detector metadata: {e}")
    jamming_metadata = {}

try:
    with open(JAMMING_TEST_SAMPLES_PATH, "r") as f:
        jamming_test_samples = json.load(f)
except Exception as e:
    print(f"Failed to load jamming detector demo samples: {e}")
    jamming_test_samples = {"threshold": 0.5, "samples": []}


def get_jamming_model_bundle():
    return jamming_model_bundle


def get_jamming_metadata():
    return jamming_metadata


def get_jamming_test_samples():
    return jamming_test_samples["samples"]


def _encode_features(feature_dict: dict, band: str, scan_mode: str) -> list:
    """
    Builds the exact feature vector the model was trained on, in the exact
    column order saved at training time (jamming_model_bundle["feature_names"]).
    Missing raw stat features raise a clear error rather than silently
    defaulting, since a wrong/zero-filled RF statistic could look like a
    plausible but meaningless input to the model.
    """
    feature_names = jamming_model_bundle["feature_names"]
    band_categories = jamming_metadata.get("categorical_encoding", {}).get("band", [])
    scan_mode_categories = jamming_metadata.get("categorical_encoding", {}).get("scan_mode", [])

    row = dict(feature_dict)
    for cat in band_categories:
        row[f"band__{cat}"] = 1 if band == cat else 0
    for cat in scan_mode_categories:
        row[f"scan_mode__{cat}"] = 1 if scan_mode == cat else 0

    missing = [f for f in feature_names if f not in row]
    if missing:
        raise ValueError(f"Missing required features for jamming detector: {missing}")

    return [row[f] for f in feature_names]


def run_jamming_inference(feature_dict: dict, band: str, scan_mode: str):
    """Returns (prediction: 'benign'|'malicious', probability_malicious: float, threshold: float)."""
    if jamming_model_bundle is None:
        raise RuntimeError("Jamming detector model is not loaded.")

    threshold = jamming_metadata.get("best_threshold", 0.5)
    vector = _encode_features(feature_dict, band, scan_mode)
    feature_names = jamming_model_bundle["feature_names"]
    model = jamming_model_bundle["model"]
    row_df = pd.DataFrame([vector], columns=feature_names)
    probability = float(model.predict_proba(row_df)[0][1])
    prediction = "malicious" if probability >= threshold else "benign"
    return prediction, probability, threshold

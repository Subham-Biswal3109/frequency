"""
RF Interference/Jamming Detector API — a SEPARATE Flask blueprint. This
model detects whether an RF spectral capture looks like benign ambient
activity or an active jammer (gaussian-noise/single-tone), trained on real
experimental measurements (release_artifacts). It is NOT a spectrum
availability/occupancy model and must never be presented as one — see
ml/jamming/README.md.

Endpoints:
    GET  /api/jamming/model-info   — dataset/model metadata, metrics, limitations
    GET  /api/jamming/samples      — held-out demo samples (with true labels)
    POST /api/jamming/predict      — run inference on a sample_id or custom features
"""

import json

from flask import Blueprint, request, jsonify
from pydantic import ValidationError

from backend.api.schemas import JammingSampleRequest
from backend.services.jamming_detection import (
    get_jamming_metadata, get_jamming_test_samples, get_jamming_model_bundle,
    run_jamming_inference,
)

jamming_bp = Blueprint("jamming_bp", __name__)

DISCLAIMER = (
    "This model detects RF interference/jamming signatures in experimentally captured "
    "spectral data. It is a separate research model from Wire Watcher's spectrum "
    "availability predictor and does not estimate channel occupancy or availability."
)


@jamming_bp.route("/api/jamming/model-info", methods=["GET"])
def jamming_model_info():
    metadata = get_jamming_metadata()
    headline = metadata.get("headline_vs_supplementary_metrics", {})
    controlled = headline.get("controlled_same_environment_eval", {})
    raw_test = headline.get("raw_test_split_eval", {})
    return jsonify({
        "model_loaded": get_jamming_model_bundle() is not None,
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "task": metadata.get("task"),
        "dataset_name": metadata.get("dataset_name"),
        "dataset_type": metadata.get("dataset_type"),
        "training_samples": metadata.get("training_samples"),
        "validation_samples": metadata.get("validation_samples"),
        "test_samples": metadata.get("test_samples"),
        "num_session_groups": metadata.get("num_session_groups"),
        "algorithm": metadata.get("algorithm"),
        "best_threshold": metadata.get("best_threshold"),
        "split_methodology": metadata.get("split_methodology"),
        "threshold_tuning_methodology": metadata.get("threshold_tuning_methodology"),
        # PRIMARY reported research metric — same-environment, non-confounded.
        # Never surface raw_test_split_eval's ROC-AUC=1.0 as the headline number.
        "primary_controlled_metrics": {
            "description": controlled.get("description"),
            "n": controlled.get("n"),
            "class_distribution": controlled.get("class_distribution"),
            **(controlled.get("metrics") or {}),
        },
        # Supplementary only, clearly labeled as environment-confounded.
        "supplementary_raw_test_metrics": {
            "description": raw_test.get("description"),
            "n": raw_test.get("n"),
            "environment_composition": raw_test.get("environment_composition"),
            "is_environment_confounded": raw_test.get("is_environment_confounded"),
            **(raw_test.get("metrics") or {}),
        },
        "baseline_comparison": {
            "energy_baseline": (metadata.get("metrics") or {}).get("energy_baseline"),
            "logistic_regression": (metadata.get("metrics") or {}).get("logistic_regression_baseline"),
            "decision_tree": (metadata.get("metrics") or {}).get("decision_tree_baseline"),
        },
        "feature_importances": (metadata.get("feature_importances") or [])[:15],
        "limitations": metadata.get("limitations"),
        "disclaimer": DISCLAIMER,
    }), 200


@jamming_bp.route("/api/jamming/samples", methods=["GET"])
def jamming_samples():
    samples = get_jamming_test_samples()
    # Return a lightweight summary list (no full feature vectors) for a picker UI.
    summary = [
        {
            "sample_id": s["sample_id"],
            "file_name": s["file_name"],
            "true_label": s["true_label"],
            "band": s["band"],
            "scan_mode": s["scan_mode"],
            "waveform": s["waveform"],
            "power_dbm": s["power_dbm"],
        }
        for s in samples
    ]
    return jsonify({"samples": summary, "count": len(summary)}), 200


@jamming_bp.route("/api/jamming/predict", methods=["POST"])
def jamming_predict():
    try:
        data = request.json or {}
        validated = JammingSampleRequest(**data)
    except ValidationError as e:
        return jsonify({"error": "Invalid jamming detection input", "details": json.loads(e.json())}), 400
    except Exception as e:
        return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

    if get_jamming_model_bundle() is None:
        return jsonify({"error": "Jamming detector model failed to load on startup."}), 500

    try:
        if validated.sample_id:
            samples = {s["sample_id"]: s for s in get_jamming_test_samples()}
            sample = samples.get(validated.sample_id)
            if sample is None:
                return jsonify({"error": f"Unknown sample_id: {validated.sample_id}"}), 404

            prediction, probability, threshold = run_jamming_inference(
                sample["features"], sample["band"], sample["scan_mode"]
            )
            return jsonify({
                "sample_id": sample["sample_id"],
                "file_name": sample["file_name"],
                "true_label": sample["true_label"],
                "prediction": prediction,
                "probability_malicious": round(probability, 4),
                "threshold": threshold,
                "correct": prediction == sample["true_label"],
                "disclaimer": DISCLAIMER,
            }), 200
        else:
            prediction, probability, threshold = run_jamming_inference(
                validated.features, validated.band, validated.scan_mode
            )
            return jsonify({
                "prediction": prediction,
                "probability_malicious": round(probability, 4),
                "threshold": threshold,
                "disclaimer": DISCLAIMER,
            }), 200
    except ValueError as e:
        return jsonify({"error": "Invalid features", "details": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Jamming detection failed", "details": str(e)}), 500

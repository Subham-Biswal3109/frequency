import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from backend.app import app
from backend.services.jamming_detection import get_jamming_test_samples


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_jamming_model_info(client):
    response = client.get("/api/jamming/model-info")
    assert response.status_code == 200
    data = response.get_json()
    assert data["model_loaded"] is True
    assert data["task"] == "RF interference / jamming detection (benign vs malicious)"
    # Headline metric must be the controlled, non-confounded evaluation.
    primary = data["primary_controlled_metrics"]
    assert primary["roc_auc"] < 1.0  # never the confounded 1.0
    assert 0.8 < primary["f1"] < 0.9
    assert "chamber" in primary["description"].lower()
    # Confounded raw-test metrics must still be present, but clearly labeled.
    assert data["supplementary_raw_test_metrics"]["is_environment_confounded"] is True
    assert any("chamber" in lim.lower() for lim in data["limitations"])


def test_jamming_samples_list(client):
    response = client.get("/api/jamming/samples")
    assert response.status_code == 200
    data = response.get_json()
    assert data["count"] == len(data["samples"]) > 0
    sample = data["samples"][0]
    assert "sample_id" in sample and "true_label" in sample
    # Sample list must not leak the full feature vector (kept lightweight).
    assert "features" not in sample


def test_jamming_predict_by_sample_id(client):
    samples = get_jamming_test_samples()
    sample = samples[0]
    response = client.post("/api/jamming/predict", json={"sample_id": sample["sample_id"]})
    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction"] in ("benign", "malicious")
    assert 0.0 <= data["probability_malicious"] <= 1.0
    assert data["true_label"] == sample["true_label"]
    assert "correct" in data


def test_jamming_predict_unknown_sample_id(client):
    response = client.post("/api/jamming/predict", json={"sample_id": "does_not_exist"})
    assert response.status_code == 404


def test_jamming_predict_missing_input(client):
    response = client.post("/api/jamming/predict", json={})
    assert response.status_code == 400


def test_jamming_predict_incomplete_features(client):
    response = client.post("/api/jamming/predict", json={
        "features": {"freq1_mean": 1.0}, "band": "5GHz", "scan_mode": "passive",
    })
    assert response.status_code == 400


def test_jamming_predict_missing_band_for_custom_features(client):
    samples = get_jamming_test_samples()
    response = client.post("/api/jamming/predict", json={"features": samples[0]["features"]})
    assert response.status_code == 400


def test_jamming_predict_custom_features_full_vector(client):
    samples = get_jamming_test_samples()
    sample = samples[0]
    response = client.post("/api/jamming/predict", json={
        "features": sample["features"], "band": sample["band"], "scan_mode": sample["scan_mode"],
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction"] in ("benign", "malicious")


def test_jamming_never_calls_itself_availability_model(client):
    """Regression guard: the jamming API must never use occupancy language."""
    response = client.get("/api/jamming/model-info")
    data = response.get_json()
    disclaimer = data["disclaimer"].lower()
    assert "availability" in disclaimer  # disclaims that it's NOT the availability model
    assert "occupancy" in disclaimer or "availability" in disclaimer


def test_existing_predict_endpoint_unaffected_by_jamming_module(client):
    """Regression guard: registering the jamming blueprint must not break /api/predict."""
    payload = {
        "start_frequency_mhz": 1800.0, "end_frequency_mhz": 1810.0, "bandwidth_mhz": 10.0,
        "hour_of_day": 12, "day_of_week": 2, "signal_power_dbm": -90.0, "noise_floor_dbm": -100.0,
        "snr_db": 10.0, "state": "Maharashtra", "city": "Mumbai", "service_type": "4G LTE",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200


def test_existing_simulation_endpoint_unaffected_by_jamming_module(client):
    """Regression guard: registering the jamming blueprint must not break /api/simulation/run."""
    payload = {
        "start_frequency_mhz": 1800, "end_frequency_mhz": 1900, "channel_bandwidth_mhz": 10,
        "noise_floor_dbm": -100, "num_existing_users": 5, "requested_bandwidth_mhz": 10, "seed": 42,
    }
    response = client.post("/api/simulation/run", json=payload)
    assert response.status_code == 200

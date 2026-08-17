import pytest
import json
import os
import sys
from pathlib import Path

# Setup path so we can import backend
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from backend.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data["api"] == "ok"
    assert "model_loaded" in data
    assert "database_connected" in data

def test_model_info(client):
    response = client.get('/api/model-info')
    assert response.status_code == 200
    data = response.get_json()
    assert "algorithm" in data
    assert "data_source" in data
    assert "best_threshold" in data

def test_valid_prediction(client):
    payload = {
        "start_frequency_mhz": 1800.0,
        "end_frequency_mhz": 1810.0,
        "bandwidth_mhz": 10.0,
        "hour_of_day": 12,
        "day_of_week": 2,
        "signal_power_dbm": -90.0,
        "noise_floor_dbm": -100.0,
        "snr_db": 10.0,
        "state": "Maharashtra",
        "city": "Mumbai",
        "service_type": "4G LTE"
    }
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "probability" in data
    assert "confidence" in data
    assert "data_source" in data
    assert "ood_warning" in data
    assert "important_features" in data
    assert len(data["important_features"]) <= 3

def test_inconsistent_snr(client):
    # SNR should be signal - noise (e.g. -80 - -100 = 20)
    # We supply 50 here, which is invalid.
    payload = {
        "start_frequency_mhz": 1800.0,
        "end_frequency_mhz": 1810.0,
        "bandwidth_mhz": 10.0,
        "hour_of_day": 12,
        "day_of_week": 2,
        "signal_power_dbm": -80.0,
        "noise_floor_dbm": -100.0,
        "snr_db": 50.0, 
        "state": "Maharashtra",
        "city": "Mumbai",
        "service_type": "4G LTE"
    }
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Invalid RF input" in data["error"]
    assert "details" in data
    assert "SNR is inconsistent" in data["details"]

def test_ood_detection(client):
    # Extremely low signal (-150) should trigger OOD
    payload = {
        "start_frequency_mhz": 1800.0,
        "end_frequency_mhz": 1810.0,
        "bandwidth_mhz": 10.0,
        "hour_of_day": 12,
        "day_of_week": 2,
        "signal_power_dbm": -150.0,
        "noise_floor_dbm": -100.0,
        "snr_db": -50.0,
        "state": "Maharashtra",
        "city": "Mumbai",
        "service_type": "4G LTE"
    }
    response = client.post('/api/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["ood_warning"] is True
    assert "warning" in data
    assert "outside typical range" in data["warning"]
    assert data["confidence"] == "OOD / Unreliable"

def test_predictions_history(client):
    response = client.get('/api/predictions')
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    if len(data["predictions"]) > 0:
        first = data["predictions"][0]
        assert "signal_power_dbm" in first
        assert "data_source" in first
        assert "ood_status" in first

def test_spectrum_analyze(client):
    payload = {
        "center_freq_mhz": 1800.0,
        "bandwidth_mhz": 10.0,
        "signal_strength_dbm": -80.0,
        "noise_floor_dbm": -100.0,
        "state": "Maharashtra",
        "city": "Mumbai",
        "service_type": "4G LTE"
    }
    response = client.post('/api/spectrum/analyze', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "data_source" in data
    assert "spectrum_data" in data
    assert "extracted_features" in data
    assert data["extracted_features"]["snr_db"] > 0

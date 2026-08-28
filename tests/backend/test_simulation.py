import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from backend.app import app
from backend.rf.simulation.channel_generator import generate_channel_grid
from backend.rf.simulation.signal_generator import assign_existing_signals
from backend.rf.simulation.allocator import (
    find_candidates,
    rank_candidates,
    allocate,
    apply_allocation,
    resource_utilization,
    allocate_multi_user,
)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# --------------------------- channel_generator ---------------------------

def test_channel_grid_count_and_bounds():
    channels = generate_channel_grid(1800, 1900, 10)
    assert len(channels) == 10
    assert channels[0]["start_mhz"] == 1800
    assert channels[-1]["end_mhz"] == 1900


def test_channel_grid_non_overlapping():
    channels = generate_channel_grid(1800, 1855, 10)
    # 55 MHz / 10 MHz -> 5 full channels, remainder dropped
    assert len(channels) == 5
    for i in range(len(channels) - 1):
        assert channels[i]["end_mhz"] == channels[i + 1]["start_mhz"]


def test_channel_grid_invalid_range_raises():
    with pytest.raises(ValueError):
        generate_channel_grid(1900, 1800, 10)


def test_channel_grid_bandwidth_larger_than_range_raises():
    with pytest.raises(ValueError):
        generate_channel_grid(1800, 1805, 10)


# --------------------------- signal_generator ----------------------------

def test_signal_assignment_reproducible_with_seed():
    channels = generate_channel_grid(1800, 1900, 10)
    a = assign_existing_signals(channels, 4, seed=123)
    b = assign_existing_signals(channels, 4, seed=123)
    assert a == b


def test_signal_assignment_respects_count_cap():
    channels = generate_channel_grid(1800, 1850, 10)  # 5 channels
    assignments = assign_existing_signals(channels, 100, seed=1)
    assert len(assignments) == 5  # capped at number of channels


# ------------------------------- allocator --------------------------------

def _make_channels(states, snrs=None, ml_probs=None):
    """Helper: builds minimal channel dicts for allocator unit tests."""
    channels = []
    for i, state in enumerate(states):
        cid = i + 1
        channels.append({
            "channel_id": cid,
            "start_mhz": 1800 + i * 10,
            "end_mhz": 1810 + i * 10,
            "bandwidth_mhz": 10,
            "rf_state": "OCCUPIED" if state == "OCCUPIED" else "AVAILABLE",
            "state": state,
            "rf_snr_db": (snrs[i] if snrs else 5.0),
            "ml_probability": (ml_probs[i] if ml_probs else 0.8),
        })
    return channels


def test_allocate_selects_available_channel():
    channels = _make_channels(["OCCUPIED", "AVAILABLE", "OCCUPIED", "AVAILABLE"])
    selected, ranked, message = allocate(channels, 10)
    assert selected is not None
    assert selected["channel_ids"][0] in (2, 4)
    assert len(ranked) == 2


def test_allocate_no_suitable_channel():
    channels = _make_channels(["OCCUPIED", "OCCUPIED", "OCCUPIED"])
    selected, ranked, message = allocate(channels, 10)
    assert selected is None
    assert ranked == []
    assert "No suitable channel found" in message


def test_allocate_bandwidth_matching_requires_contiguous_group():
    channels = _make_channels(["AVAILABLE", "AVAILABLE", "OCCUPIED", "AVAILABLE"])
    selected, ranked, message = allocate(channels, 20)  # needs 2 contiguous 10 MHz channels
    assert selected is not None
    assert selected["channel_ids"] == [1, 2]
    assert selected["total_bandwidth_mhz"] == 20


def test_allocate_does_not_overlap_occupied():
    channels = _make_channels(["AVAILABLE", "OCCUPIED", "AVAILABLE"])
    selected, ranked, message = allocate(channels, 20)  # can't span the occupied channel
    assert selected is None


def test_apply_allocation_marks_channels_allocated():
    channels = _make_channels(["AVAILABLE", "AVAILABLE"])
    selected, ranked, _ = allocate(channels, 10)
    updated = apply_allocation(channels, selected)
    allocated_ids = {c["channel_id"] for c in updated if c["state"] == "ALLOCATED"}
    assert allocated_ids == set(selected["channel_ids"])
    # original list must be untouched (deep copy)
    assert all(c["state"] != "ALLOCATED" for c in channels)


def test_resource_utilization_sums_correctly():
    channels = _make_channels(["OCCUPIED", "AVAILABLE", "AVAILABLE"])
    util = resource_utilization(channels)
    assert util["total_mhz"] == 30
    assert util["occupied_mhz"] == 10
    assert util["available_mhz"] == 20
    assert util["allocated_mhz"] == 0


def test_multi_user_allocation_prevents_overlap():
    channels = _make_channels(["AVAILABLE", "AVAILABLE", "AVAILABLE"])
    users = [
        {"user_id": "A", "requested_bandwidth_mhz": 10},
        {"user_id": "B", "requested_bandwidth_mhz": 10},
        {"user_id": "C", "requested_bandwidth_mhz": 10},
        {"user_id": "D", "requested_bandwidth_mhz": 10},  # 4th user, only 3 channels exist
    ]
    result = allocate_multi_user(channels, users)
    successes = [r["success"] for r in result["user_results"]]
    assert successes == [True, True, True, False]
    allocated_ids = [
        cid for r in result["user_results"] if r["selected"] for cid in r["selected"]["channel_ids"]
    ]
    assert len(allocated_ids) == len(set(allocated_ids))  # no channel allocated twice


# --------------------------------- API ------------------------------------

def test_simulation_run_endpoint_basic(client):
    payload = {
        "start_frequency_mhz": 1800,
        "end_frequency_mhz": 1900,
        "channel_bandwidth_mhz": 10,
        "noise_floor_dbm": -100,
        "num_existing_users": 5,
        "requested_bandwidth_mhz": 10,
        "seed": 42,
    }
    response = client.post("/api/simulation/run", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["channels"]) == 10
    assert "allocation" in data
    assert "disclaimer" in data
    assert "spectrum_data" in data


def test_simulation_run_reproducible_with_seed(client):
    payload = {
        "start_frequency_mhz": 1800,
        "end_frequency_mhz": 1850,
        "channel_bandwidth_mhz": 10,
        "noise_floor_dbm": -100,
        "num_existing_users": 2,
        "requested_bandwidth_mhz": 10,
        "seed": 99,
    }
    r1 = client.post("/api/simulation/run", json=payload).get_json()
    r2 = client.post("/api/simulation/run", json=payload).get_json()
    states1 = [c["rf_state"] for c in r1["channels"]]
    states2 = [c["rf_state"] for c in r2["channels"]]
    assert states1 == states2


def test_simulation_run_invalid_range_returns_400(client):
    payload = {
        "start_frequency_mhz": 1900,
        "end_frequency_mhz": 1800,
        "channel_bandwidth_mhz": 10,
        "requested_bandwidth_mhz": 10,
    }
    response = client.post("/api/simulation/run", json=payload)
    assert response.status_code == 400


def test_simulation_run_multi_user_mode(client):
    payload = {
        "start_frequency_mhz": 1800,
        "end_frequency_mhz": 1900,
        "channel_bandwidth_mhz": 10,
        "noise_floor_dbm": -100,
        "num_existing_users": 3,
        "seed": 5,
        "mode": "multi_user",
        "users": [
            {"user_id": "A", "requested_bandwidth_mhz": 10},
            {"user_id": "B", "requested_bandwidth_mhz": 10},
        ],
    }
    response = client.post("/api/simulation/run", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "multi_user_allocation" in data
    assert len(data["multi_user_allocation"]["user_results"]) == 2


def test_simulation_run_multi_user_requires_users_list(client):
    payload = {
        "start_frequency_mhz": 1800,
        "end_frequency_mhz": 1900,
        "channel_bandwidth_mhz": 10,
        "mode": "multi_user",
    }
    response = client.post("/api/simulation/run", json=payload)
    assert response.status_code == 400


def test_snr_sweep_endpoint(client):
    response = client.post("/api/simulation/snr-sweep", json={})
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["points"]) == 7
    for point in data["points"]:
        assert "snr_db" in point
        assert "decision" in point


def test_existing_predict_endpoint_unaffected(client):
    """Regression guard: the new blueprint must not interfere with /api/predict."""
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
        "service_type": "4G LTE",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200

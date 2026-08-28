"""
Spectrum Simulation API — a SEPARATE Flask blueprint from the existing
backend/api/routes.py. Nothing in this file modifies the existing
/api/predict, /api/predictions, /api/health, /api/model-info or
/api/spectrum/analyze endpoints, the existing ML model, its threshold, or
the existing `availability_candidates` database table.

Endpoints:
    POST /api/simulation/run          — full environment + RF sensing + ML +
                                         allocation (single or multi-user)
    POST /api/simulation/snr-sweep    — SNR vs. availability-probability
                                         experiment (section 18)

IMPORTANT: This module demonstrates "simulated spectrum sensing and
availability-based channel allocation" for educational/engineering
purposes. It must never be described as the real-world mechanism used by
TRAI, DoT, or any telecom operator.
"""

import json
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from sqlalchemy import text

from backend.api.schemas import SimulationRequest, SnrSweepRequest
from backend.rf.simulation.spectrum_simulator import run_environment_simulation, run_snr_sweep
from backend.rf.simulation.allocator import allocate, apply_allocation, allocate_multi_user, resource_utilization
from backend.database.connection import get_db
from backend.database.models import Base, SpectrumSimulation

simulation_bp = Blueprint("simulation_bp", __name__)

DISCLAIMER = (
    "Simulation uses synthetic RF conditions for demonstration. Results do not represent "
    "live spectrum measurements or official spectrum allocation."
)

# Best-effort: create the new table if it doesn't exist yet. This never
# touches any other existing table (create_all with an explicit `tables`
# list only creates the tables named, and is a no-op if already present).
try:
    from backend.database.connection import engine
    Base.metadata.create_all(bind=engine, tables=[SpectrumSimulation.__table__])
except Exception as e:  # pragma: no cover - matches existing app's tolerant DB error handling
    print(f"Spectrum Simulation: could not ensure history table exists: {e}")


def _log_simulation(mode: str, config: dict, result_summary: dict, allocated_channel: dict | None):
    """Best-effort persistence, mirroring save_prediction_to_db's error tolerance."""
    try:
        db_gen = get_db()
        db = next(db_gen)
        row = SpectrumSimulation(
            created_at=datetime.now(timezone.utc),
            mode=mode,
            configuration_json=json.dumps(config, default=str),
            result_summary_json=json.dumps(result_summary, default=str),
            allocated_channel_json=json.dumps(allocated_channel, default=str) if allocated_channel else None,
        )
        db.add(row)
        db.commit()
    except Exception as e:
        print(f"Failed to log spectrum simulation: {e}")
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


@simulation_bp.route("/api/simulation/run", methods=["POST"])
def run_simulation():
    try:
        data = request.json or {}
        validated = SimulationRequest(**data)
    except ValidationError as e:
        return jsonify({"error": "Invalid simulation input", "details": json.loads(e.json())}), 400
    except Exception as e:
        return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

    try:
        environment = run_environment_simulation(
            start_frequency_mhz=validated.start_frequency_mhz,
            end_frequency_mhz=validated.end_frequency_mhz,
            channel_bandwidth_mhz=validated.channel_bandwidth_mhz,
            noise_floor_dbm=validated.noise_floor_dbm,
            num_existing_users=validated.num_existing_users,
            seed=validated.seed,
            state=validated.state,
            city=validated.city,
            service_type=validated.service_type,
        )
    except ValueError as e:
        return jsonify({"error": "Invalid spectrum configuration", "details": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Simulation failed", "details": str(e)}), 500

    channels = environment["channels"]
    response_payload = {
        "mode": validated.mode,
        "channels": channels,
        "spectrum_data": environment["spectrum_data"],
        "occupied_regions": environment["occupied_regions"],
        "available_regions": environment["available_regions"],
        "model_loaded": environment["model_loaded"],
        "noise_floor_dbm": environment["noise_floor_dbm"],
        "resource_utilization_before": resource_utilization(channels),
        "disclaimer": DISCLAIMER,
    }

    result_summary_for_log = {
        "num_channels": len(channels),
        "occupied": sum(1 for c in channels if c["rf_state"] == "OCCUPIED"),
        "available": sum(1 for c in channels if c["state"] == "AVAILABLE"),
    }
    allocated_channel_for_log = None

    if validated.mode == "multi_user":
        users = [u.model_dump() for u in (validated.users or [])]
        multi_result = allocate_multi_user(channels, users)
        response_payload["multi_user_allocation"] = multi_result
        response_payload["resource_utilization_after"] = multi_result["utilization_timeline"][-1]
        allocated_channel_for_log = [
            r["selected"] for r in multi_result["user_results"] if r["selected"]
        ] or None
    else:
        selected, ranked, message = allocate(channels, validated.requested_bandwidth_mhz)
        final_channels = apply_allocation(channels, selected) if selected else channels
        response_payload["allocation"] = {
            "requested_bandwidth_mhz": validated.requested_bandwidth_mhz,
            "success": selected is not None,
            "selected": selected,
            "top_candidates": ranked[:5],
            "message": message,
            "final_channels": final_channels,
        }
        response_payload["resource_utilization_after"] = resource_utilization(final_channels)
        allocated_channel_for_log = selected

    _log_simulation(validated.mode, validated.model_dump(), result_summary_for_log, allocated_channel_for_log)

    return jsonify(response_payload), 200


@simulation_bp.route("/api/simulation/snr-sweep", methods=["POST"])
def snr_sweep():
    try:
        data = request.json or {}
        validated = SnrSweepRequest(**data)
    except ValidationError as e:
        return jsonify({"error": "Invalid SNR sweep input", "details": json.loads(e.json())}), 400
    except Exception as e:
        return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

    try:
        result = run_snr_sweep(
            signal_power_dbm=validated.signal_power_dbm,
            start_frequency_mhz=validated.start_frequency_mhz,
            end_frequency_mhz=validated.end_frequency_mhz,
            bandwidth_mhz=validated.bandwidth_mhz,
            state=validated.state,
            city=validated.city,
            service_type=validated.service_type,
            snr_values_db=validated.snr_values_db,
        )
    except Exception as e:
        return jsonify({"error": "SNR sweep failed", "details": str(e)}), 500

    if not result["model_loaded"]:
        return jsonify({"error": "ML model failed to load on startup.", "model_loaded": False}), 500

    return jsonify({**result, "disclaimer": DISCLAIMER}), 200


@simulation_bp.route("/api/simulation/history", methods=["GET"])
def simulation_history():
    """Optional: most recent simulation runs logged to spectrum_simulations."""
    db_gen = get_db()
    db = next(db_gen)
    try:
        rows = db.execute(
            text(
                "SELECT simulation_id, created_at, mode, configuration_json, result_summary_json "
                "FROM spectrum_simulations ORDER BY created_at DESC LIMIT 50"
            )
        ).fetchall()
        history = [
            {
                "simulation_id": r[0],
                "created_at": r[1].isoformat() if r[1] else None,
                "mode": r[2],
                "configuration": json.loads(r[3]) if r[3] else None,
                "result_summary": json.loads(r[4]) if r[4] else None,
            }
            for r in rows
        ]
        return jsonify({"history": history}), 200
    except Exception as e:
        return jsonify({"error": "Failed to fetch simulation history", "details": str(e)}), 500
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass

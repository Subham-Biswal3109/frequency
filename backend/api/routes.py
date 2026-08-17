from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from sqlalchemy import text

from backend.api.schemas import PredictionRequest
from backend.database.connection import get_db
from backend.database.models import AvailabilityCandidate
from backend.services.prediction import (
    get_model_bundle,
    get_model_metadata,
    run_ml_inference,
    save_prediction_to_db
)
from backend.rf.rf_source import SimulatedRFSource
from backend.rf.spectrum_processor import compute_fft_psd
from backend.rf.noise_estimator import estimate_noise_floor
from backend.rf.peak_detector import detect_peaks, get_occupied_regions
from backend.rf.feature_extractor import extract_ml_features

api_bp = Blueprint("api_bp", __name__)

@api_bp.route("/api/predict", methods=["POST"])
def run_prediction():
    model_bundle = get_model_bundle()
    if model_bundle is None:
        return jsonify({"error": "ML model failed to load on startup.", "details": "Model bundle is None"}), 500

    try:
        # Validate input
        data = request.json
        validated_data = PredictionRequest(**data)
    except ValidationError as e:
        return jsonify({"error": "Invalid RF input", "details": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

    # OOD Warning Detection
    ood_warning = False
    warnings_list = []
    model_metadata = get_model_metadata()
    
    if "training_bounds" in model_metadata:
        bounds = model_metadata["training_bounds"]
        
        if "signal_power_dbm" in bounds:
            b = bounds["signal_power_dbm"]
            if validated_data.signal_power_dbm < b["p1"] or validated_data.signal_power_dbm > b["p99"]:
                ood_warning = True
                warnings_list.append(f"Signal Power ({validated_data.signal_power_dbm} dBm) is outside typical range ({b['p1']:.1f} to {b['p99']:.1f}).")
                
        if "snr_db" in bounds:
            b = bounds["snr_db"]
            if validated_data.snr_db < b["p1"] or validated_data.snr_db > b["p99"]:
                ood_warning = True
                warnings_list.append(f"SNR ({validated_data.snr_db} dB) is outside typical range ({b['p1']:.1f} to {b['p99']:.1f}).")

    # Prepare input for ML model exactly as expected
    ml_input = {
        "start_frequency_mhz": validated_data.start_frequency_mhz,
        "end_frequency_mhz": validated_data.end_frequency_mhz,
        "bandwidth_mhz": validated_data.bandwidth_mhz,
        "hour_of_day": validated_data.hour_of_day,
        "day_of_week": validated_data.day_of_week,
        "signal_power_dbm": validated_data.signal_power_dbm,
        "noise_floor_dbm": validated_data.noise_floor_dbm,
        "snr_db": validated_data.snr_db,
        "state": validated_data.state,
        "city": validated_data.city,
        "service_type": validated_data.service_type
    }

    try:
        # Call the ML inference service
        prediction_result, probability, best_t = run_ml_inference(ml_input)
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500
        
    # Calculate Confidence
    confidence = "Low"
    if probability is not None:
        dist_from_threshold = abs(probability - best_t)
        if ood_warning:
            confidence = "OOD / Unreliable"
        elif dist_from_threshold > 0.15:
            confidence = "High"
        elif dist_from_threshold > 0.05:
            confidence = "Medium"
        else:
            confidence = "Low"
            
    data_source = model_metadata.get("data_source", "Synthetic")

    # Persist the prediction to the database
    save_prediction_to_db(validated_data, prediction_result, probability, best_t, ood_warning, data_source)

    response_payload = {
        "prediction": prediction_result,
        "available": bool(prediction_result == 1),
        "probability": probability,
        "confidence": confidence,
        "data_source": data_source,
        "threshold": best_t,
        "features_used": ml_input,
        "important_features": list(model_metadata.get("feature_importances", {}).keys())[:3]
    }
    
    if ood_warning:
        response_payload["ood_warning"] = True
        response_payload["warning"] = "Input is outside the model's typical training distribution. Prediction reliability may be reduced. " + " ".join(warnings_list)
    else:
        response_payload["ood_warning"] = False

    return jsonify(response_payload), 200


@api_bp.route("/api/predictions", methods=["GET"])
def get_predictions():
    db_gen = get_db()
    db = next(db_gen)
    try:
        candidates = db.query(AvailabilityCandidate).order_by(AvailabilityCandidate.generated_at.desc()).limit(100).all()
        
        results = []
        for row in candidates:
            results.append({
                "id": row.candidate_id,
                "start_frequency_mhz": float(row.frequency_start_mhz) if row.frequency_start_mhz is not None else None,
                "end_frequency_mhz": float(row.frequency_end_mhz) if row.frequency_end_mhz is not None else None,
                "bandwidth_mhz": float(row.required_bandwidth_mhz) if row.required_bandwidth_mhz is not None else (float(row.frequency_end_mhz - row.frequency_start_mhz) if row.frequency_end_mhz and row.frequency_start_mhz else None),
                "city": row.district,
                "state": row.state,
                "service_type": row.required_service,
                "available": bool(row.recommendation_status == "recommended") if row.recommendation_status else False,
                "probability": float(row.predicted_availability_probability) if row.predicted_availability_probability is not None else None,
                "timestamp": row.generated_at.isoformat() if row.generated_at else None,
                "signal_power_dbm": float(row.signal_power_dbm) if row.signal_power_dbm is not None else None,
                "noise_floor_dbm": float(row.noise_floor_dbm) if row.noise_floor_dbm is not None else None,
                "snr_db": float(row.snr_db) if row.snr_db is not None else None,
                "ood_status": True if row.ood_status == "1" else False,
                "data_source": row.data_source
            })
        return jsonify({"predictions": results}), 200
    except Exception as e:
        return jsonify({"error": "Failed to fetch predictions", "details": str(e)}), 500
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


@api_bp.route("/api/health", methods=["GET"])
def health_check():
    db_connected = False
    try:
        db_gen = get_db()
        db = next(db_gen)
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        pass
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass

    model_bundle = get_model_bundle()
    model_loaded = model_bundle is not None
    status = "ok" if (model_loaded and db_connected) else "degraded"

    return jsonify({
        "status": status,
        "model_loaded": model_loaded,
        "database_connected": db_connected,
        "api": "ok",
        "model": "loaded" if model_loaded else "failed",
        "database": "connected" if db_connected else "failed"
    }), 200


@api_bp.route("/api/model-info", methods=["GET"])
def model_info():
    model_bundle = get_model_bundle()
    if model_bundle is None:
        return jsonify({"error": "Model not loaded", "details": "Model bundle is None"}), 500
        
    model_metadata = get_model_metadata()
    
    # Calculate KPIs from Database
    total_predictions = 0
    available_predictions = 0
    avg_probability = 0.0
    ood_count = 0
    
    db_gen = get_db()
    db = next(db_gen)
    try:
        from sqlalchemy import func
        stats = db.query(
            func.count(AvailabilityCandidate.candidate_id).label("total"),
            func.sum(func.cast(AvailabilityCandidate.recommendation_status == 'recommended', text("integer"))).label("available"),
            func.avg(AvailabilityCandidate.predicted_availability_probability).label("avg_prob"),
            func.sum(func.cast(AvailabilityCandidate.ood_status == '1', text("integer"))).label("ood")
        ).first()
        
        if stats and stats.total > 0:
            total_predictions = stats.total
            available_predictions = int(stats.available or 0)
            avg_probability = float(stats.avg_prob or 0.0)
            ood_count = int(stats.ood or 0)
    except Exception as e:
        print(f"Failed to load KPIs: {e}")
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass

    return jsonify({
        "algorithm": model_metadata.get("algorithm", "Random Forest Classifier"),
        "model_version": model_metadata.get("model_version", "v1.0"),
        "training_date": model_metadata.get("training_date", ""),
        "dataset_name": model_metadata.get("dataset_name", ""),
        "dataset_type": model_metadata.get("dataset_type", "synthetic"),
        "training_samples": model_metadata.get("training_samples", 0),
        "validation_samples": model_metadata.get("validation_samples", 0),
        "test_samples": model_metadata.get("test_samples", 0),
        "real_rf_validation": model_metadata.get("real_rf_validation", False),
        "data_source": model_metadata.get("data_source", "Synthetic"),
        "best_threshold": model_metadata.get("best_threshold", 0.5),
        "calibrated": False,
        "features": model_metadata.get("features", []),
        "feature_importances": model_metadata.get("feature_importances", {}),
        "limitations": model_metadata.get("limitations", []),
        "kpis": {
            "total_predictions": total_predictions,
            "available_predictions": available_predictions,
            "occupied_predictions": total_predictions - available_predictions,
            "avg_probability": avg_probability,
            "ood_count": ood_count
        }
    }), 200

@api_bp.route("/api/spectrum/analyze", methods=["POST"])
def analyze_spectrum():
    try:
        data = request.json or {}
        center_freq_mhz = float(data.get("center_freq_mhz", 1800.0))
        bandwidth_mhz = float(data.get("bandwidth_mhz", 10.0))
        signal_strength_dbm = float(data.get("signal_strength_dbm", -75.0))
        noise_floor_dbm_input = float(data.get("noise_floor_dbm", -100.0))
        
        # 1. Generate RF Signal (Simulated)
        source = SimulatedRFSource(
            center_freq_mhz=center_freq_mhz,
            bandwidth_mhz=bandwidth_mhz,
            signal_strength_dbm=signal_strength_dbm,
            noise_floor_dbm=noise_floor_dbm_input
        )
        rx_signal, sample_rate_mhz = source.get_signal()
        
        # 2. Compute FFT / PSD
        freqs_mhz, psd_dbm = compute_fft_psd(rx_signal, sample_rate_mhz, center_freq_mhz)
        
        # 3. Estimate Noise Floor
        estimated_noise_floor = estimate_noise_floor(psd_dbm)
        
        # 4. Detect Peaks
        peaks = detect_peaks(freqs_mhz, psd_dbm, estimated_noise_floor)
        
        # 5. Extract Features
        ml_features = extract_ml_features(center_freq_mhz, bandwidth_mhz, peaks, estimated_noise_floor, location_info=data)
        
        # Downsample arrays for frontend visualization to avoid huge JSON payload
        downsample_factor = max(1, len(freqs_mhz) // 200)
        freqs_downsampled = freqs_mhz[::downsample_factor].tolist()
        psd_downsampled = psd_dbm[::downsample_factor].tolist()

        return jsonify({
            "data_source": source.get_source_name(),
            "frequency_range": {
                "start_mhz": center_freq_mhz - (bandwidth_mhz / 2),
                "end_mhz": center_freq_mhz + (bandwidth_mhz / 2)
            },
            "noise_floor_dbm": round(estimated_noise_floor, 2),
            "detected_signals": peaks,
            "occupied_regions": get_occupied_regions(peaks),
            "available_regions": [],
            "spectrum_data": {
                "frequencies": [round(f, 3) for f in freqs_downsampled],
                "power_dbm": [round(p, 2) for p in psd_downsampled]
            },
            "extracted_features": ml_features
        }), 200

    except Exception as e:
        return jsonify({"error": "Spectrum analysis failed", "details": str(e)}), 500

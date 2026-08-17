import sys
from pathlib import Path
import json
from datetime import datetime

# We will move ml/predict.py to ml/inference/predict.py later
ML_DIR = Path(__file__).resolve().parent.parent.parent / "ml"
sys.path.append(str(ML_DIR))
try:
    from inference import predict
except ImportError:
    import predict

from backend.config import METADATA_PATH
from backend.database.models import AvailabilityCandidate
from backend.database.connection import get_db

# Load model and metadata as singletons
try:
    print("Loading ML model...")
    model_bundle = predict.load_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    model_bundle = None

try:
    print("Loading model metadata...")
    with open(METADATA_PATH, "r") as f:
        model_metadata = json.load(f)
    print("Metadata loaded successfully.")
except Exception as e:
    print(f"Failed to load metadata: {e}")
    model_metadata = {}

def get_model_bundle():
    return model_bundle

def get_model_metadata():
    return model_metadata

def run_ml_inference(ml_input):
    """Executes the ML prediction and applies the threshold."""
    if model_bundle is None:
        raise RuntimeError("ML model is not loaded.")
        
    best_t = model_metadata.get("best_threshold", 0.5)
    prediction_result, probability = predict.predict(ml_input, model_bundle)
    
    if probability is not None:
        prediction_result = 1 if probability >= best_t else 0
        
    return prediction_result, probability, best_t

def save_prediction_to_db(validated_data, prediction_result, probability, best_t, ood_warning, data_source):
    """Persists the prediction result and inputs to MySQL."""
    try:
        db_gen = get_db()
        db = next(db_gen)
        
        candidate = AvailabilityCandidate(
            frequency_start_mhz=validated_data.start_frequency_mhz,
            frequency_end_mhz=validated_data.end_frequency_mhz,
            region=validated_data.region or validated_data.state,
            state=validated_data.state,
            district=validated_data.city,
            latitude=validated_data.latitude,
            longitude=validated_data.longitude,
            required_bandwidth_mhz=validated_data.bandwidth_mhz,
            required_service=validated_data.service_type,
            predicted_availability_probability=probability if probability is not None else (1.0 if prediction_result == 1 else 0.0),
            recommendation_status='recommended' if prediction_result == 1 else 'review_required',
            generated_at=datetime.utcnow(),
            model_version=model_metadata.get("model_version", "v1.0"),
            signal_power_dbm=validated_data.signal_power_dbm,
            noise_floor_dbm=validated_data.noise_floor_dbm,
            snr_db=validated_data.snr_db,
            threshold_applied=best_t,
            ood_status="1" if ood_warning else "0",
            data_source=data_source
        )
        db.add(candidate)
        db.commit()
    except Exception as e:
        print(f"Failed to save to database: {e}")
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass

import argparse
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "wire_watcher_model.pkl"

def load_model(model_path=MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    bundle = joblib.load(model_path)
    return bundle

def predict(input_data: dict, bundle: dict):
    """
    input_data: dictionary containing feature values for a single prediction
                (must match feature_columns in the bundle)
    bundle: loaded model bundle
    
    Returns:
    prediction (int), probability (float)
    """
    pipe = bundle["pipeline"]
    feature_columns = bundle["feature_columns"]
    
    # Convert input dict to DataFrame
    df = pd.DataFrame([input_data])
    
    # Ensure all required features are present
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required features: {missing_cols}")
        
    X = df[feature_columns]
    
    prediction = int(pipe.predict(X)[0])
    
    probability = None
    if hasattr(pipe, "predict_proba"):
        probability = float(pipe.predict_proba(X)[0, 1])
        
    return prediction, probability

def main():
    parser = argparse.ArgumentParser(description="Run a single prediction using the saved Wire Watcher model.")
    parser.add_argument("--start_freq", type=float, required=True, help="Start frequency in MHz")
    parser.add_argument("--end_freq", type=float, required=True, help="End frequency in MHz")
    parser.add_argument("--bandwidth", type=float, required=True, help="Bandwidth in MHz")
    parser.add_argument("--hour", type=int, required=True, help="Hour of day (0-23)")
    parser.add_argument("--day_of_week", type=int, required=True, help="Day of week (0-6)")
    parser.add_argument("--signal_power", type=float, required=True, help="Signal power in dBm")
    parser.add_argument("--noise_floor", type=float, required=True, help="Noise floor in dBm")
    parser.add_argument("--snr", type=float, required=True, help="Signal-to-Noise Ratio in dB")
    parser.add_argument("--state", type=str, required=True, help="State name")
    parser.add_argument("--city", type=str, required=True, help="City name")
    parser.add_argument("--service", type=str, required=True, help="Service type")
    
    args = parser.parse_args()
    
    # Bundle input into dictionary matching the feature columns
    input_data = {
        "start_frequency_mhz": args.start_freq,
        "end_frequency_mhz": args.end_freq,
        "bandwidth_mhz": args.bandwidth,
        "hour_of_day": args.hour,
        "day_of_week": args.day_of_week,
        "signal_power_dbm": args.signal_power,
        "noise_floor_dbm": args.noise_floor,
        "snr_db": args.snr,
        "state": args.state,
        "city": args.city,
        "service_type": args.service
    }
    
    print("Loading model...")
    bundle = load_model()
    
    print("Running prediction...")
    pred, prob = predict(input_data, bundle)
    
    print("-" * 40)
    print("PREDICTION RESULT")
    print("-" * 40)
    print(f"Target available (0=No, 1=Yes): {pred}")
    if prob is not None:
        print(f"Probability (Confidence): {prob:.2%}")
    print("-" * 40)

if __name__ == "__main__":
    main()

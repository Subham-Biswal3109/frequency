import json
from pathlib import Path
from inference import predict

BASE_DIR = Path(__file__).resolve().parent.parent

def evaluate_scenarios(model_bundle, best_t, bounds):
    print("\n==============================================")
    print("      REALISTIC SCENARIO EVALUATION ENGINE")
    print("==============================================\n")
    
    base = {
        "start_frequency_mhz": 1800.0,
        "end_frequency_mhz": 1810.0,
        "bandwidth_mhz": 10.0,
        "hour_of_day": 14,
        "day_of_week": 2,
        "state": "Maharashtra",
        "city": "Mumbai",
        "service_type": "4G LTE"
    }

    scenarios = [
        {"name": "1. Noise Dominated", "mods": {"signal_power_dbm": -103.0, "noise_floor_dbm": -100.0, "snr_db": -3.0}},
        {"name": "2. Weak Signal", "mods": {"signal_power_dbm": -96.0, "noise_floor_dbm": -103.0, "snr_db": 7.0}},
        {"name": "3. Intermediate Signal", "mods": {"signal_power_dbm": -80.0, "noise_floor_dbm": -100.0, "snr_db": 20.0}},
        {"name": "4. Strong Signal", "mods": {"signal_power_dbm": -70.0, "noise_floor_dbm": -103.0, "snr_db": 33.0}},
        {"name": "5. Very Strong Signal", "mods": {"signal_power_dbm": -60.0, "noise_floor_dbm": -100.0, "snr_db": 40.0}},
        {"name": "6. Below Noise Floor", "mods": {"signal_power_dbm": -105.0, "noise_floor_dbm": -100.0, "snr_db": -5.0}},
        {"name": "7. Different Frequency (3500 MHz)", "mods": {"start_frequency_mhz": 3500.0, "end_frequency_mhz": 3520.0, "bandwidth_mhz": 20.0, "signal_power_dbm": -96.0, "noise_floor_dbm": -103.0, "snr_db": 7.0}},
        {"name": "8. Different Location (Delhi)", "mods": {"state": "Delhi", "city": "Delhi", "signal_power_dbm": -96.0, "noise_floor_dbm": -103.0, "snr_db": 7.0}},
        {"name": "9. Different Time (3 AM)", "mods": {"hour_of_day": 3, "signal_power_dbm": -96.0, "noise_floor_dbm": -103.0, "snr_db": 7.0}}
    ]

    for s in scenarios:
        d = dict(base)
        d.update(s["mods"])
        
        # Check OOD
        ood = False
        if d["signal_power_dbm"] < bounds["signal_power_dbm"]["p1"] or d["signal_power_dbm"] > bounds["signal_power_dbm"]["p99"]:
            ood = True
        if d["snr_db"] < bounds["snr_db"]["p1"] or d["snr_db"] > bounds["snr_db"]["p99"]:
            ood = True

        _, prob = predict.predict(d, model_bundle)
        pred = "AVAILABLE" if prob >= best_t else "OCCUPIED"
        
        print(f"{s['name']}")
        print(f"  Freq: {d['start_frequency_mhz']} MHz | BW: {d['bandwidth_mhz']} MHz | Time: {d['hour_of_day']}:00 | Loc: {d['city']}")
        print(f"  Signal: {d['signal_power_dbm']} dBm | Noise: {d['noise_floor_dbm']} dBm | SNR: {d['snr_db']} dB")
        print(f"  Model Probability: {prob:.4f}")
        print(f"  Prediction: {pred}")
        print(f"  OOD Status: {'YES' if ood else 'NO'}\n")

def run_sensitivity(model_bundle, best_t):
    print("\n==============================================")
    print("      AUTOMATED SENSITIVITY TESTING")
    print("==============================================\n")
    
    base = {
        "start_frequency_mhz": 1800.0,
        "end_frequency_mhz": 1810.0,
        "bandwidth_mhz": 10.0,
        "hour_of_day": 14,
        "day_of_week": 2,
        "noise_floor_dbm": -100.0,
        "state": "Maharashtra",
        "city": "Mumbai",
        "service_type": "4G LTE"
    }

    # Signal Power / SNR Sensitivity
    print("--- 1. Signal Power / SNR Sensitivity ---")
    signals = [-100, -95, -90, -85, -80, -75, -70, -65]
    for s in signals:
        d = dict(base)
        d["signal_power_dbm"] = float(s)
        d["snr_db"] = float(s - d["noise_floor_dbm"])
        _, prob = predict.predict(d, model_bundle)
        pred = "AVAILABLE" if prob >= best_t else "OCCUPIED"
        print(f"Signal: {s:4} dBm | SNR: {d['snr_db']:4} dB | Prob(Avail): {prob:.4f} | {pred}")

    # Frequency Sensitivity
    print("\n--- 2. Frequency Sensitivity (SNR Fixed at 10 dB) ---")
    freqs = [900, 1800, 2100, 2300, 3500, 5000]
    for f in freqs:
        d = dict(base)
        d["start_frequency_mhz"] = float(f)
        d["end_frequency_mhz"] = float(f + 10)
        d["signal_power_dbm"] = -90.0
        d["snr_db"] = 10.0
        _, prob = predict.predict(d, model_bundle)
        pred = "AVAILABLE" if prob >= best_t else "OCCUPIED"
        print(f"Freq: {f:4} MHz | Prob(Avail): {prob:.4f} | {pred}")

    # Noise Sensitivity
    print("\n--- 3. Noise Floor Sensitivity (Signal Fixed at -90 dBm) ---")
    noises = [-110, -105, -100, -95, -90]
    for n in noises:
        d = dict(base)
        d["noise_floor_dbm"] = float(n)
        d["signal_power_dbm"] = -90.0
        d["snr_db"] = float(-90.0 - n)
        _, prob = predict.predict(d, model_bundle)
        pred = "AVAILABLE" if prob >= best_t else "OCCUPIED"
        print(f"Noise: {n:4} dBm | SNR: {d['snr_db']:4} dB | Prob(Avail): {prob:.4f} | {pred}")

    # Time Sensitivity
    print("\n--- 4. Time Sensitivity (SNR Fixed at 10 dB) ---")
    hours = [0, 4, 8, 12, 16, 20]
    for h in hours:
        d = dict(base)
        d["hour_of_day"] = h
        d["signal_power_dbm"] = -90.0
        d["snr_db"] = 10.0
        _, prob = predict.predict(d, model_bundle)
        pred = "AVAILABLE" if prob >= best_t else "OCCUPIED"
        print(f"Hour: {h:02d}:00 | Prob(Avail): {prob:.4f} | {pred}")


def main():
    try:
        model_bundle = predict.load_model()
        with open(BASE_DIR / "artifacts" / "model_metadata.json") as f:
            meta = json.load(f)
            best_t = meta.get("best_threshold", 0.5)
            bounds = meta.get("training_bounds", {})
    except Exception as e:
        print(f"Failed to load model or metadata: {e}")
        return

    evaluate_scenarios(model_bundle, best_t, bounds)
    run_sensitivity(model_bundle, best_t)

if __name__ == "__main__":
    main()

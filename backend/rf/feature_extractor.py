from datetime import datetime

def extract_ml_features(center_freq_mhz: float, bandwidth_mhz: float, peaks: list[dict], noise_floor_dbm: float, location_info: dict = None) -> dict:
    """
    Maps RF layer outputs to the specific schema required by the existing ML model.
    """
    now = datetime.now()
    
    # Use the primary peak for SNR / Power, or default to noise if no signal
    if peaks:
        primary = max(peaks, key=lambda p: p["power_dbm"])
        signal_power = primary["power_dbm"]
        snr = primary["snr_db"]
    else:
        signal_power = noise_floor_dbm
        snr = 0.0
        
    if location_info is None:
        location_info = {
            "state": "Maharashtra",
            "city": "Mumbai",
            "service_type": "4G LTE"
        }
        
    ml_features = {
        "start_frequency_mhz": center_freq_mhz - (bandwidth_mhz / 2),
        "end_frequency_mhz": center_freq_mhz + (bandwidth_mhz / 2),
        "bandwidth_mhz": bandwidth_mhz,
        "signal_power_dbm": round(signal_power, 2),
        "noise_floor_dbm": round(noise_floor_dbm, 2),
        "snr_db": round(snr, 2),
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "state": location_info.get("state", "Unknown"),
        "city": location_info.get("city", "Unknown"),
        "service_type": location_info.get("service_type", "Unknown")
    }
    
    return ml_features

import numpy as np

def detect_peaks(freqs_mhz: np.ndarray, psd_dbm: np.ndarray, noise_floor_dbm: float, detection_margin_db: float = 10.0) -> list[dict]:
    """
    Detects signal peaks above the noise floor threshold.
    """
    threshold = noise_floor_dbm + detection_margin_db
    peaks = []
    
    # Simple threshold-based peak detection
    # A real implementation might use scipy.signal.find_peaks
    above_thresh = psd_dbm > threshold
    
    if np.any(above_thresh):
        # We just find the global max for now to represent the primary carrier
        # For a full multi-carrier system, we would group contiguous indices above threshold
        max_idx = np.argmax(psd_dbm)
        peak_power = psd_dbm[max_idx]
        
        if peak_power > threshold:
            # Estimate bandwidth by counting adjacent bins above a -3dB or threshold mark
            # (Simplified for the prototype: assumed 10% of total bandwidth approx)
            freq_res = freqs_mhz[1] - freqs_mhz[0]
            bw = np.sum(psd_dbm > (peak_power - 3.0)) * freq_res
            
            peaks.append({
                "frequency_mhz": float(freqs_mhz[max_idx]),
                "power_dbm": float(peak_power),
                "bandwidth_mhz": max(float(bw), 1.0),
                "snr_db": float(peak_power - noise_floor_dbm)
            })
            
    return peaks

def get_occupied_regions(peaks: list[dict]) -> list[dict]:
    regions = []
    for p in peaks:
        regions.append({
            "start_mhz": p["frequency_mhz"] - (p["bandwidth_mhz"] / 2),
            "end_mhz": p["frequency_mhz"] + (p["bandwidth_mhz"] / 2)
        })
    return regions

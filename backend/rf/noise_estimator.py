import numpy as np

def estimate_noise_floor(psd_dbm: np.ndarray) -> float:
    """
    Robust statistical estimation of the noise floor.
    We use the median of the PSD to avoid bias from signal peaks.
    """
    return float(np.median(psd_dbm))

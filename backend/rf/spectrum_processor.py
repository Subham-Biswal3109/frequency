import numpy as np

def compute_fft_psd(rx_signal: np.ndarray, sample_rate_mhz: float, center_freq_mhz: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Processes raw RF signal and computes FFT and Power Spectral Density (PSD).
    Returns (frequency_axis_mhz, psd_dbm)
    """
    num_samples = len(rx_signal)
    
    # Compute FFT
    fft_result = np.fft.fftshift(np.fft.fft(rx_signal))
    
    # Calculate power in mW, normalizing by num_samples
    power_mw = (np.abs(fft_result) / num_samples) ** 2
    
    # Avoid log of zero
    power_mw = np.maximum(power_mw, 1e-20)
    psd_dbm = 10 * np.log10(power_mw)
    
    # Frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(num_samples, d=1/(sample_rate_mhz * 1e6)))
    freqs_mhz = center_freq_mhz + (freqs / 1e6)
    
    return freqs_mhz, psd_dbm

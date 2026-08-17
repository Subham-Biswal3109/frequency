import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def generate_simulated_rf_signal(
    center_freq_mhz=1800.0,
    bandwidth_mhz=10.0,
    signal_strength_dbm=-75.0,
    noise_floor_dbm=-100.0,
    num_samples=2048,
    sample_rate_mhz=20.0
):
    """
    Generates a simulated time-domain RF signal consisting of white noise
    and a CW (Continuous Wave) carrier to demonstrate ECE signal processing concepts.
    """
    # 1. Generate Thermal Noise
    # Convert noise floor from dBm to linear power (mW), then to amplitude
    noise_power_mw = 10 ** (noise_floor_dbm / 10.0)
    noise_amplitude = np.sqrt(noise_power_mw)
    
    # Complex white Gaussian noise
    noise = (np.random.normal(0, noise_amplitude, num_samples) + 
             1j * np.random.normal(0, noise_amplitude, num_samples)) / np.sqrt(2)
             
    # 2. Generate Carrier Signal (if signal strength is above noise significantly)
    signal_power_mw = 10 ** (signal_strength_dbm / 10.0)
    signal_amplitude = np.sqrt(signal_power_mw)
    
    # Time vector
    t = np.arange(num_samples) / (sample_rate_mhz * 1e6)
    
    # Let's put the signal slightly offset from the center
    offset_freq = 2e6 # 2 MHz offset
    signal = signal_amplitude * np.exp(1j * 2 * np.pi * offset_freq * t)
    
    # Combined RF signal
    rx_signal = signal + noise
    
    return rx_signal, sample_rate_mhz

def process_rf_signal(rx_signal, sample_rate_mhz, center_freq_mhz, bandwidth_mhz):
    """
    Processes the raw RF signal (FFT / PSD), detects the signal power and noise floor,
    and formats the output for the ML pipeline.
    """
    num_samples = len(rx_signal)
    
    # 1. Compute FFT and Power Spectral Density (PSD)
    fft_result = np.fft.fftshift(np.fft.fft(rx_signal))
    
    # Calculate power in mW, then convert to dBm
    # (Dividing by num_samples to normalize the FFT)
    power_mw = (np.abs(fft_result) / num_samples) ** 2
    
    # Avoid log of zero
    power_mw = np.maximum(power_mw, 1e-20)
    psd_dbm = 10 * np.log10(power_mw)
    
    # Frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(num_samples, d=1/(sample_rate_mhz * 1e6)))
    freqs_mhz = center_freq_mhz + (freqs / 1e6)
    
    # 2. ECE Parameter Estimation
    # Estimate noise floor using the median of the PSD (robust to peaks)
    estimated_noise_floor = np.median(psd_dbm)
    
    # Detect signal peak
    estimated_signal_power = np.max(psd_dbm)
    
    # Calculate SNR
    snr = estimated_signal_power - estimated_noise_floor
    
    # 3. Create Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_mhz, psd_dbm, color='blue', alpha=0.7)
    plt.axhline(estimated_noise_floor, color='red', linestyle='--', label=f'Noise Floor ({estimated_noise_floor:.1f} dBm)')
    plt.plot(freqs_mhz[np.argmax(psd_dbm)], estimated_signal_power, 'go', label=f'Signal Peak ({estimated_signal_power:.1f} dBm)')
    
    # Highlight band of interest
    plt.axvspan(center_freq_mhz - bandwidth_mhz/2, center_freq_mhz + bandwidth_mhz/2, color='green', alpha=0.1, label=f'{bandwidth_mhz} MHz Band')
    
    plt.title("SIMULATED RF SIGNAL - Power Spectral Density (PSD)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dBm)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("spectrum_plot.png", dpi=150)
    plt.close()
    
    # 4. Standard Wire Watcher Feature Schema mapping
    now = datetime.now()
    ml_features = {
        "timestamp": now.isoformat(),
        "start_frequency_mhz": center_freq_mhz - (bandwidth_mhz / 2),
        "end_frequency_mhz": center_freq_mhz + (bandwidth_mhz / 2),
        "bandwidth_mhz": bandwidth_mhz,
        "signal_power_dbm": round(float(estimated_signal_power), 2),
        "noise_floor_dbm": round(float(estimated_noise_floor), 2),
        "snr_db": round(float(snr), 2),
        "hour_of_day": now.hour,
        "day_of_week": now.weekday(),
        "latitude": 19.0760,  # Default mock
        "longitude": 72.8777, # Default mock
        "state": "Maharashtra",
        "city": "Mumbai",
        "service_type": "4G LTE"
    }
    
    return ml_features

if __name__ == "__main__":
    print("--- Wire Watcher ECE Signal Processor Demonstration ---")
    print("1. Generating Simulated Time-Domain RF Signal...")
    rx, fs = generate_simulated_rf_signal(signal_strength_dbm=-75, noise_floor_dbm=-100)
    
    print("2. Processing Signal (FFT / PSD / Feature Extraction)...")
    features = process_rf_signal(rx, fs, center_freq_mhz=1800.0, bandwidth_mhz=10.0)
    
    print("3. Exported Visualization to 'spectrum_plot.png'")
    print("\n--- Extracted Standard ML Feature Schema ---")
    for k, v in features.items():
        print(f"{k}: {v}")
    print("\nNote: Data Source is SIMULATED. Do not use for real-world measurements.")

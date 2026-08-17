import numpy as np
from abc import ABC, abstractmethod

class RFDataSource(ABC):
    @abstractmethod
    def get_signal(self) -> tuple[np.ndarray, float]:
        """Returns time-domain IQ signal (complex) and sample rate in MHz."""
        pass
        
    @abstractmethod
    def get_source_name(self) -> str:
        pass

class SimulatedRFSource(RFDataSource):
    """
    Generates a simulated time-domain RF signal consisting of white noise
    and a CW (Continuous Wave) carrier to demonstrate ECE signal processing concepts.
    """
    def __init__(self, center_freq_mhz=1800.0, bandwidth_mhz=10.0, signal_strength_dbm=-75.0, noise_floor_dbm=-100.0, num_samples=2048, sample_rate_mhz=20.0):
        self.center_freq_mhz = center_freq_mhz
        self.bandwidth_mhz = bandwidth_mhz
        self.signal_strength_dbm = signal_strength_dbm
        self.noise_floor_dbm = noise_floor_dbm
        self.num_samples = num_samples
        self.sample_rate_mhz = sample_rate_mhz

    def get_signal(self) -> tuple[np.ndarray, float]:
        # 1. Generate Thermal Noise
        noise_power_mw = 10 ** (self.noise_floor_dbm / 10.0)
        noise_amplitude = np.sqrt(noise_power_mw)
        
        noise = (np.random.normal(0, noise_amplitude, self.num_samples) + 
                 1j * np.random.normal(0, noise_amplitude, self.num_samples)) / np.sqrt(2)
                 
        # 2. Generate Carrier Signal
        signal_power_mw = 10 ** (self.signal_strength_dbm / 10.0)
        signal_amplitude = np.sqrt(signal_power_mw)
        
        t = np.arange(self.num_samples) / (self.sample_rate_mhz * 1e6)
        
        # Offset signal by some random fraction of bandwidth for realism
        offset_freq = (np.random.uniform(-0.4, 0.4) * self.bandwidth_mhz) * 1e6
        signal = signal_amplitude * np.exp(1j * 2 * np.pi * offset_freq * t)
        
        rx_signal = signal + noise
        return rx_signal, self.sample_rate_mhz
        
    def get_source_name(self) -> str:
        return "SIMULATED"

from pydantic import BaseModel, Field, model_validator
from typing import Optional

class PredictionRequest(BaseModel):
    start_frequency_mhz: float = Field(..., gt=0, description="Start frequency in MHz")
    end_frequency_mhz: float = Field(..., gt=0, description="End frequency in MHz")
    bandwidth_mhz: float = Field(..., gt=0, description="Bandwidth in MHz")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    signal_power_dbm: float = Field(..., description="Signal power in dBm")
    noise_floor_dbm: float = Field(..., description="Noise floor in dBm")
    snr_db: float = Field(..., description="Signal-to-Noise Ratio in dB")
    state: str = Field(..., min_length=1, description="State name")
    city: str = Field(..., min_length=1, description="City name")
    service_type: str = Field(..., min_length=1, description="Service type")
    
    # Optional fields for DB storage
    region: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    @model_validator(mode='after')
    def check_physical_consistency(self):
        # 1. Bandwidth / Frequency logical checks
        if self.start_frequency_mhz >= self.end_frequency_mhz:
            raise ValueError("start_frequency_mhz must be less than end_frequency_mhz")
            
        # 2. SNR strict physical check: SNR(dB) = Signal(dBm) - Noise(dBm)
        expected_snr = self.signal_power_dbm - self.noise_floor_dbm
        if abs(self.snr_db - expected_snr) > 0.01:
            raise ValueError(f"SNR is inconsistent with signal power and noise floor. Expected {expected_snr}, got {self.snr_db}.")
            
        return self

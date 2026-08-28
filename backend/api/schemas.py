from pydantic import BaseModel, Field, model_validator
from typing import Optional, List

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


class SimulationUser(BaseModel):
    """One entry in a multi-user allocation request (Spectrum Simulation module)."""
    user_id: Optional[str] = None
    requested_bandwidth_mhz: float = Field(..., gt=0, description="Requested bandwidth in MHz")


class SimulationRequest(BaseModel):
    """
    Request schema for the new, separate Spectrum Simulation module
    (POST /api/simulation/run). This does not replace or alter
    PredictionRequest / /api/predict in any way.
    """
    start_frequency_mhz: float = Field(..., gt=0, description="Start of the simulated band in MHz")
    end_frequency_mhz: float = Field(..., gt=0, description="End of the simulated band in MHz")
    channel_bandwidth_mhz: float = Field(..., gt=0, description="Width of each simulated channel in MHz")
    noise_floor_dbm: float = Field(-100.0, description="Simulated ambient noise floor in dBm")
    num_existing_users: int = Field(0, ge=0, le=200, description="Number of pre-existing occupying signals to place")
    seed: Optional[int] = Field(None, description="Random seed for reproducible simulation runs")

    mode: str = Field("ml_assisted", description="basic | ml_assisted | multi_user")
    requested_bandwidth_mhz: Optional[float] = Field(
        None, gt=0, description="Bandwidth requested by a new single user (basic/ml_assisted modes)"
    )
    users: Optional[List[SimulationUser]] = Field(
        None, description="Sequential user requests for multi_user mode"
    )

    state: str = Field("Maharashtra", description="State passed through to the existing ML model's location features")
    city: str = Field("Mumbai", description="City passed through to the existing ML model's location features")
    service_type: str = Field("4G LTE", description="Service type passed through to the existing ML model")

    @model_validator(mode='after')
    def check_consistency(self):
        if self.end_frequency_mhz <= self.start_frequency_mhz:
            raise ValueError("end_frequency_mhz must be greater than start_frequency_mhz")
        if self.channel_bandwidth_mhz > (self.end_frequency_mhz - self.start_frequency_mhz):
            raise ValueError("channel_bandwidth_mhz cannot be larger than the total frequency range")
        if self.mode not in ("basic", "ml_assisted", "multi_user"):
            raise ValueError("mode must be one of: basic, ml_assisted, multi_user")
        if self.mode == "multi_user" and not self.users:
            raise ValueError("multi_user mode requires a non-empty 'users' list")
        if self.mode != "multi_user" and self.requested_bandwidth_mhz is None:
            raise ValueError("requested_bandwidth_mhz is required for basic/ml_assisted modes")
        return self


class SnrSweepRequest(BaseModel):
    """Request schema for POST /api/simulation/snr-sweep (section 18 experiment)."""
    signal_power_dbm: float = Field(-80.0, description="Fixed signal power in dBm used across the sweep")
    start_frequency_mhz: float = Field(1800.0, gt=0)
    end_frequency_mhz: float = Field(1810.0, gt=0)
    bandwidth_mhz: float = Field(10.0, gt=0)
    state: str = Field("Maharashtra")
    city: str = Field("Mumbai")
    service_type: str = Field("4G LTE")
    snr_values_db: Optional[List[float]] = Field(None, description="Custom SNR sweep points; defaults to 0-30dB in 5dB steps")

    @model_validator(mode='after')
    def check_consistency(self):
        if self.end_frequency_mhz <= self.start_frequency_mhz:
            raise ValueError("end_frequency_mhz must be greater than start_frequency_mhz")
        return self

from sqlalchemy import Column, BigInteger, String, Numeric, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class AvailabilityCandidate(Base):
    __tablename__ = "availability_candidates"

    candidate_id = Column(BigInteger, primary_key=True, autoincrement=True)
    band_id = Column(String(32))
    frequency_start_mhz = Column(Numeric(12, 3), nullable=False)
    frequency_end_mhz = Column(Numeric(12, 3), nullable=False)
    region = Column(String(150))
    state = Column(String(100))
    district = Column(String(100))
    latitude = Column(Numeric(10, 7))
    longitude = Column(Numeric(10, 7))
    required_bandwidth_mhz = Column(Numeric(12, 3))
    required_service = Column(String(100))
    assignment_status = Column(String(40), default='not_verified', nullable=False)
    predicted_availability_probability = Column(Numeric(6, 5))
    interference_score = Column(Numeric(8, 5))
    suitability_score = Column(Numeric(8, 5))
    recommendation_status = Column(String(40), default='review_required')
    generated_at = Column(DateTime)
    model_version = Column(String(100))
    
    # RF inputs and ML tracking fields
    signal_power_dbm = Column(Numeric(10, 3))
    noise_floor_dbm = Column(Numeric(10, 3))
    snr_db = Column(Numeric(10, 3))
    threshold_applied = Column(Numeric(6, 5))
    ood_status = Column(String(10)) 
    data_source = Column(String(50))

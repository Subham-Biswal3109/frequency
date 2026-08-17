CREATE INDEX idx_assignment_band
ON frequency_assignments(band_id);

CREATE INDEX idx_assignment_frequency
ON frequency_assignments(
    start_frequency_mhz,
    end_frequency_mhz
);

CREATE INDEX idx_assignment_location
ON frequency_assignments(
    state,
    district
);

CREATE INDEX idx_observation_band
ON occupancy_observations(band_id);

CREATE INDEX idx_observation_frequency
ON occupancy_observations(
    frequency_start_mhz,
    frequency_end_mhz
);

CREATE INDEX idx_observation_time
ON occupancy_observations(observation_time);

CREATE INDEX idx_observation_location
ON occupancy_observations(
    state,
    district
);

CREATE INDEX idx_candidate_band
ON availability_candidates(band_id);

CREATE INDEX idx_candidate_location
ON availability_candidates(
    state,
    district
);

CREATE INDEX idx_candidate_probability
ON availability_candidates(
    predicted_availability_probability
);

CREATE INDEX idx_training_band
ON ml_training_samples(band_id);

CREATE INDEX idx_training_time
ON ml_training_samples(observation_time);

CREATE INDEX idx_training_target
ON ml_training_samples(target_available);

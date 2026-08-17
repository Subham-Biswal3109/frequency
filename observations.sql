
CREATE TABLE occupancy_observations (

    observation_id BIGINT NOT NULL AUTO_INCREMENT,

    band_id VARCHAR(32),

    frequency_start_mhz DECIMAL(12,3) NOT NULL,

    frequency_end_mhz DECIMAL(12,3) NOT NULL,

    center_frequency_mhz DECIMAL(12,3),

    bandwidth_mhz DECIMAL(12,3),

    region VARCHAR(150),

    state VARCHAR(100),

    district VARCHAR(100),

    latitude DECIMAL(10,7),

    longitude DECIMAL(10,7),

    observation_time DATETIME NOT NULL,

    signal_power_dbm DECIMAL(8,3),

    noise_floor_dbm DECIMAL(8,3),

    snr_db DECIMAL(8,3),

    occupancy_ratio DECIMAL(6,5),

    interference_level VARCHAR(40),

    measurement_source VARCHAR(255),

    data_type VARCHAR(50)
        DEFAULT 'real',

    created_at TIMESTAMP
        DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (observation_id),

    CONSTRAINT fk_observation_band
        FOREIGN KEY (band_id)
        REFERENCES spectrum_bands(band_id)
        ON UPDATE CASCADE
        ON DELETE SET NULL,

    CONSTRAINT chk_observation_frequency
        CHECK (frequency_start_mhz < frequency_end_mhz),

    CONSTRAINT chk_occupancy_ratio
        CHECK (
            occupancy_ratio IS NULL
            OR (
                occupancy_ratio >= 0
                AND occupancy_ratio <= 1
            )
        ),

    CONSTRAINT chk_observation_latitude
        CHECK (
            latitude IS NULL
            OR (latitude >= -90 AND latitude <= 90)
        ),

    CONSTRAINT chk_observation_longitude
        CHECK (
            longitude IS NULL
            OR (longitude >= -180 AND longitude <= 180)
        )

) ENGINE=InnoDB;

CREATE TABLE ml_training_samples (

    sample_id BIGINT NOT NULL AUTO_INCREMENT,

    band_id VARCHAR(32),

    observation_time DATETIME,

    start_frequency_mhz DECIMAL(12,3),

    end_frequency_mhz DECIMAL(12,3),

    bandwidth_mhz DECIMAL(12,3),

    hour_of_day TINYINT,

    day_of_week TINYINT,

    region VARCHAR(150),

    state VARCHAR(100),

    district VARCHAR(100),

    signal_power_dbm DECIMAL(8,3),

    noise_floor_dbm DECIMAL(8,3),

    snr_db DECIMAL(8,3),

    occupancy_ratio DECIMAL(6,5),

    interference_score DECIMAL(8,5),

    target_available TINYINT,

    target_occupancy_ratio DECIMAL(6,5),

    label_source VARCHAR(255),

    data_type VARCHAR(50)
        DEFAULT 'real',

    created_at TIMESTAMP
        DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (sample_id),

    CONSTRAINT fk_training_band
        FOREIGN KEY (band_id)
        REFERENCES spectrum_bands(band_id)
        ON UPDATE CASCADE
        ON DELETE SET NULL,

    CONSTRAINT chk_training_target
        CHECK (
            target_available IS NULL
            OR target_available IN (0,1)
        ),

    CONSTRAINT chk_training_occupancy
        CHECK (
            target_occupancy_ratio IS NULL
            OR (
                target_occupancy_ratio >= 0
                AND target_occupancy_ratio <= 1
            )
        )

) ENGINE=InnoDB;

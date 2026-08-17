-- ============================================================
-- WIRE WATCHER - MYSQL DATABASE
-- Final Year Project
-- ============================================================

-- ------------------------------------------------------------
-- 1. CREATE DATABASE
-- ------------------------------------------------------------

DROP DATABASE IF EXISTS wire_watcher;

CREATE DATABASE wire_watcher
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE wire_watcher;


-- ============================================================
-- 2. SPECTRUM BANDS
-- ============================================================
-- Stores regulatory/reference frequency bands.
--
-- IMPORTANT:
-- regulatory_status != assignment_status != occupancy_status
--
-- Do NOT assume that an "allocated" band is currently available.
-- ============================================================

CREATE TABLE spectrum_bands (

    band_id VARCHAR(32) NOT NULL,

    start_frequency_mhz DECIMAL(12,3) NOT NULL,

    end_frequency_mhz DECIMAL(12,3) NOT NULL,

    candidate_service VARCHAR(100),

    regulatory_status VARCHAR(80) NOT NULL,

    assignment_status VARCHAR(40)
        NOT NULL DEFAULT 'not_verified',

    occupancy_status VARCHAR(40)
        NOT NULL DEFAULT 'not_verified',

    country VARCHAR(80)
        DEFAULT 'India',

    source_reference VARCHAR(120),

    source_location VARCHAR(80),

    notes TEXT,

    created_at TIMESTAMP
        DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMP
        DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (band_id),

    CONSTRAINT chk_band_frequency
        CHECK (start_frequency_mhz < end_frequency_mhz)

) ENGINE=InnoDB;


-- ============================================================
-- 3. FREQUENCY ASSIGNMENTS
-- ============================================================
-- Stores actual/verified frequency assignments.
--
-- Example:
-- 1800-1810 MHz
-- Operator = XYZ
-- State = Karnataka
-- Status = assigned
-- ============================================================

CREATE TABLE frequency_assignments (

    assignment_id BIGINT NOT NULL AUTO_INCREMENT,

    band_id VARCHAR(32),

    start_frequency_mhz DECIMAL(12,3) NOT NULL,

    end_frequency_mhz DECIMAL(12,3) NOT NULL,

    service_type VARCHAR(100),

    assignee_name VARCHAR(255),

    operator_type VARCHAR(100),

    region VARCHAR(150),

    state VARCHAR(100),

    district VARCHAR(100),

    latitude DECIMAL(10,7),

    longitude DECIMAL(10,7),

    assignment_status VARCHAR(40)
        NOT NULL DEFAULT 'not_verified',

    assignment_date DATE,

    expiry_date DATE,

    source_name VARCHAR(255),

    source_url TEXT,

    last_verified_at DATETIME,

    notes TEXT,

    created_at TIMESTAMP
        DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMP
        DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (assignment_id),

    CONSTRAINT fk_assignment_band
        FOREIGN KEY (band_id)
        REFERENCES spectrum_bands(band_id)
        ON UPDATE CASCADE
        ON DELETE SET NULL,

    CONSTRAINT chk_assignment_frequency
        CHECK (start_frequency_mhz < end_frequency_mhz),

    CONSTRAINT chk_latitude
        CHECK (
            latitude IS NULL
            OR (latitude >= -90 AND latitude <= 90)
        ),

    CONSTRAINT chk_longitude
        CHECK (
            longitude IS NULL
            OR (longitude >= -180 AND longitude <= 180)
        ),

    CONSTRAINT chk_assignment_dates
        CHECK (
            expiry_date IS NULL
            OR assignment_date IS NULL
            OR expiry_date >= assignment_date
        )

) ENGINE=InnoDB;


-- ============================================================
-- 4. OCCUPANCY OBSERVATIONS
-- ============================================================
-- Stores actual/simulated spectrum measurements.
--
-- This is one of the most important tables for the ML model.
-- ============================================================

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


-- ============================================================
-- 5. AVAILABILITY CANDIDATES
-- ============================================================
-- Stores frequency recommendations generated by the system.
--
-- Example:
--
-- 2400-2410 MHz
-- Probability = 0.91
-- Suitability = 0.88
-- Recommendation = recommended
-- ============================================================

CREATE TABLE availability_candidates (

    candidate_id BIGINT NOT NULL AUTO_INCREMENT,

    band_id VARCHAR(32),

    frequency_start_mhz DECIMAL(12,3) NOT NULL,

    frequency_end_mhz DECIMAL(12,3) NOT NULL,

    region VARCHAR(150),

    state VARCHAR(100),

    district VARCHAR(100),

    latitude DECIMAL(10,7),

    longitude DECIMAL(10,7),

    required_bandwidth_mhz DECIMAL(12,3),

    required_service VARCHAR(100),

    assignment_status VARCHAR(40)
        NOT NULL DEFAULT 'not_verified',

    predicted_availability_probability DECIMAL(6,5),

    interference_score DECIMAL(8,5),

    suitability_score DECIMAL(8,5),

    recommendation_status VARCHAR(40)
        DEFAULT 'review_required',

    generated_at DATETIME,

    model_version VARCHAR(100),

    created_at TIMESTAMP
        DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (candidate_id),

    CONSTRAINT fk_candidate_band
        FOREIGN KEY (band_id)
        REFERENCES spectrum_bands(band_id)
        ON UPDATE CASCADE
        ON DELETE SET NULL,

    CONSTRAINT chk_candidate_frequency
        CHECK (frequency_start_mhz < frequency_end_mhz),

    CONSTRAINT chk_availability_probability
        CHECK (
            predicted_availability_probability IS NULL
            OR (
                predicted_availability_probability >= 0
                AND predicted_availability_probability <= 1
            )
        ),

    CONSTRAINT chk_suitability_score
        CHECK (
            suitability_score IS NULL
            OR (
                suitability_score >= 0
                AND suitability_score <= 1
            )
        ),

    CONSTRAINT chk_candidate_latitude
        CHECK (
            latitude IS NULL
            OR (latitude >= -90 AND latitude <= 90)
        ),

    CONSTRAINT chk_candidate_longitude
        CHECK (
            longitude IS NULL
            OR (longitude >= -180 AND longitude <= 180)
        )

) ENGINE=InnoDB;


-- ============================================================
-- 6. ML TRAINING SAMPLES
-- ============================================================
-- Stores ML-ready feature records.
--
-- target_available:
--     0 = unavailable
--     1 = available
--
-- IMPORTANT:
-- Do not create fake labels and present them as real-world
-- spectrum availability in your final FYP.
-- ============================================================

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


-- ============================================================
-- 7. INDEXES
-- ============================================================

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


-- ============================================================
-- 8. VIEW: BAND SUMMARY
-- ============================================================
-- Useful for your dashboard.
-- ============================================================

CREATE VIEW v_spectrum_band_summary AS

SELECT

    sb.band_id,

    sb.start_frequency_mhz,

    sb.end_frequency_mhz,

    (
        sb.end_frequency_mhz
        - sb.start_frequency_mhz
    ) AS bandwidth_mhz,

    sb.candidate_service,

    sb.regulatory_status,

    sb.assignment_status,

    sb.occupancy_status,

    sb.country,

    COUNT(DISTINCT fa.assignment_id)
        AS assignment_count,

    COUNT(DISTINCT oo.observation_id)
        AS observation_count

FROM spectrum_bands sb

LEFT JOIN frequency_assignments fa
    ON sb.band_id = fa.band_id

LEFT JOIN occupancy_observations oo
    ON sb.band_id = oo.band_id

GROUP BY

    sb.band_id,
    sb.start_frequency_mhz,
    sb.end_frequency_mhz,
    sb.candidate_service,
    sb.regulatory_status,
    sb.assignment_status,
    sb.occupancy_status,
    sb.country;


-- ============================================================
-- 9. VIEW: LATEST AVAILABILITY RECOMMENDATIONS
-- ============================================================

CREATE VIEW v_availability_recommendations AS

SELECT

    ac.candidate_id,

    ac.band_id,

    ac.frequency_start_mhz,

    ac.frequency_end_mhz,

    ac.region,

    ac.state,

    ac.district,

    ac.required_bandwidth_mhz,

    ac.required_service,

    ac.predicted_availability_probability,

    ac.interference_score,

    ac.suitability_score,

    ac.recommendation_status,

    ac.model_version,

    ac.generated_at

FROM availability_candidates ac

WHERE ac.recommendation_status
    IN ('recommended', 'review_required');


-- ============================================================
-- 10. VIEW: ML TRAINING DATA
-- ============================================================

CREATE VIEW v_ml_training_data AS

SELECT

    mts.sample_id,

    mts.band_id,

    mts.observation_time,

    mts.start_frequency_mhz,

    mts.end_frequency_mhz,

    mts.bandwidth_mhz,

    mts.hour_of_day,

    mts.day_of_week,

    mts.region,

    mts.state,

    mts.district,

    mts.signal_power_dbm,

    mts.noise_floor_dbm,

    mts.snr_db,

    mts.occupancy_ratio,

    mts.interference_score,

    mts.target_available,

    mts.target_occupancy_ratio,

    mts.data_type

FROM ml_training_samples mts;


-- ============================================================
-- 11. VERIFY DATABASE
-- ============================================================

SHOW TABLES;

SELECT
    TABLE_NAME
FROM
    information_schema.TABLES
WHERE
    TABLE_SCHEMA = 'wire_watcher'
ORDER BY
    TABLE_NAME;


-- ============================================================
-- END
-- ============================================================
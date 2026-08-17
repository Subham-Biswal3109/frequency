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

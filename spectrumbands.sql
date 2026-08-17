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

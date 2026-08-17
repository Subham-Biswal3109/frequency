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
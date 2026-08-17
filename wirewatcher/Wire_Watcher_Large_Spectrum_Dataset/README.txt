WIRE WATCHER — LARGE DATASET PACKAGE
=======================================

This package is deliberately split into:
1) regulatory reference data
2) synthetic observation data for ML/software pipeline development
3) synthetic candidate data for dashboard/database testing

SIZE
----
spectrum_occupancy_synthetic_33600.csv = 33,600 rows
synthetic_assignment_candidates_5000.csv = 5,000 rows
spectrum_bands_reference_25.csv = 25 rows

CRITICAL WARNING
----------------
The 33,600 observation rows and 5,000 candidate rows are SYNTHETIC.
They are not measurements from Indian spectrum monitoring and must not be
presented in the FYP as real-world observations.

Why synthetic data?
-------------------
Public official sources provide regulatory allocation information, but they
do not provide a simple public nationwide table of all currently unassigned
frequencies with time/location occupancy labels. NFAP-2025 is a regulatory
allocation framework and explicitly says it does not itself grant the right
to use spectrum; frequency assignment is separately required from WPC unless
exempted.

Use the synthetic data to:
- build and test your database
- test frontend charts
- test APIs
- test the ML pipeline
- debug preprocessing and model code

For final ML claims, replace the synthetic observation data with real
measurements or a properly sourced public spectrum-occupancy dataset.

Suggested target:
target_available = 1/0

Suggested features:
frequency, bandwidth, state, city, hour, day_of_week, signal_power,
noise_floor, SNR, occupancy_ratio, interference_score.

Recommended first models:
- Logistic Regression baseline
- Random Forest
- Gradient Boosting / XGBoost if allowed
Compare them using a held-out test set and appropriate metrics.

Do not train and test on the same records.

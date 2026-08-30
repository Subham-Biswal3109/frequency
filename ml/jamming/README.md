# RF Interference/Jamming Detector — Module Documentation

**This is a SEPARATE model and task from Wire Watcher's Spectrum Availability
predictor.** It does not predict "Available" or "Occupied" spectrum. It
classifies **benign vs. malicious (jamming) RF activity**, trained on real
experimental measurements, and must never be presented or reasoned about as
an occupancy/availability model.

| | Spectrum Availability Model | RF Interference/Jamming Detector |
|---|---|---|
| Target | Available / Occupied | Benign / Malicious |
| Dataset | `spectrum_occupancy_synthetic_33600.csv` — **synthetic**, used for initial pipeline development | `release_artifacts` — **real, experimentally measured** RF spectral-scan captures |
| Endpoint | `POST /api/predict` | `POST /api/jamming/predict` |
| Model file | `ml/artifacts/wire_watcher_model.pkl` | `ml/artifacts/jamming_detector_model.pkl` |

## Dataset provenance

`release_artifacts.zip` contains 96,090 RF spectral-scan capture files (not
included raw in this repo — only their manifest and derived per-file feature
summaries are). Each file is labeled `benign` (background ambient RF or
RF-chamber floor noise) or `malicious` (an active jammer — gaussian-noise or
single-tone waveform — transmitting into a specific channel at a configured
power). Files were collected across 2.4GHz and 5GHz Wi-Fi bands, in 3
real-world indoor locations and one RF chamber.

**Critically: this is a jamming/interference dataset, not a spectrum
occupancy dataset.** `benign` RF activity can still represent an occupied
channel (e.g. ordinary Wi-Fi traffic); `malicious` specifically means an
active jammer was present. We do not equate these with "available"/"occupied".

## Feature set (60 features, all from `ml/jamming/prepare_dataset.py`)

Mean / std / min / q25 / median / q75 / max of 8 raw driver-scale fields —
`freq1, noise, max_magnitude, total_gain_db, base_pwr_db, rssi, relpwr_db,
avgpwr_db` — plus one-hot `band` (2.4GHz/5GHz) and `scan_mode`
(active/passive). These are **driver-scale / dB-like values, not confirmed-
calibrated dBm** per the release's own data dictionary.

**Explicitly excluded**, with reasons documented in `prepare_dataset.py` and
verified empirically (not assumed): `benign_subtype`, `waveform`,
`power_dbm`, `channel_mhz` (each is a perfect or near-perfect proxy for the
label via its missingness pattern), `location`/`collection_environment`
(malicious occurs only at `rf_chamber`), filenames, QC/protocol metadata, and
the release's own derived `rssi_dbm` field (verified to equal `rssi − 95` for
every row — an undocumented constant offset, not a validated calibration).

## Split methodology and the ROC-AUC=1.0 finding

Splitting uses whole, undivided experimental sessions as groups (a
continuous recording run is never divided between train/val/test — see
`ml/jamming/prepare_dataset.py::split`). With only 131 such sessions, several
very large, this split can — by chance — produce a test set that is 100%
"environment-confounded" (e.g. all real-world rows benign, all chamber rows
malicious in that split), which is exactly what happened with this dataset's
seed. In that split, ROC-AUC reads as 1.0, but this partly reflects "which
environment" rather than genuine RF-content separability.

**We did not force artificial balance by subdividing large sessions into
smaller chunks** — that would let temporally-adjacent (and therefore
correlated) slices of the *same* recording run cross the train/test
boundary, which is a subtler violation of independence than the imbalance it
would fix. True independence was prioritized over balanced classes.

Instead, `build_controlled_eval_set()` identifies whichever split (val or
test) happens to contain both classes within `collection_environment ==
'rf_chamber'` — the only environment where both classes exist — and treats
that as the primary, scientifically meaningful evaluation.

## Reported metrics — controlled (primary) vs. raw test (supplementary)

| Metric | Controlled (primary) | Raw test split (supplementary, confounded) |
|---|---|---|
| n | 15,512 (in-chamber only) | 16,527 |
| Accuracy | 0.946 | 0.978 |
| Precision | 0.744 | 0.894 |
| Recall | 0.999 | 1.0 |
| **F1** | **0.853** | 0.944 |
| **ROC-AUC** | **0.987** | 1.0 |
| PR-AUC | 0.915 | 1.0 |

**The controlled metrics (F1=0.853, ROC-AUC=0.987, PR-AUC=0.915) are the
primary reported research result.** The raw test split's near-perfect score
is never presented as the headline metric — it is shown transparently in
`GET /api/jamming/model-info` labeled as `supplementary_raw_test_metrics`
with `is_environment_confounded: true`.

Baselines on the raw test split: energy threshold (max_magnitude_mean) F1=0.158;
Logistic Regression F1=0.451; Decision Tree F1=0.905. Random Forest is the
best-performing model and is used as the shipped classifier.

## Known limitations (surfaced in the API and UI)

- Malicious (jamming) samples exist **only** in the RF chamber — this dataset
  cannot support any test of jamming-detection generalization to real-world
  environments, and no such claim is made.
- A dedicated unseen-location check (benign-only, since malicious is
  chamber-only) found the model's predicted-benign rate collapses on a
  location held out of training — generalization to genuinely new physical
  premises is weak and should be treated as an open problem.
- Frequency coverage is limited to 2.4GHz/5GHz Wi-Fi bands.
- Values are driver-scale, not calibrated dBm.

## Reproducing training

```bash
cd ml/jamming
python prepare_dataset.py /path/to/file_level_features.parquet   # regenerates ml/data/jamming_release/jamming_features_labeled.csv.gz
python train_model.py                                             # trains + saves ml/artifacts/jamming_detector_*
python audit.py                                                   # re-runs the full leakage/confounding audit
```

`file_level_features.parquet` (the release's own derived per-file feature
table, ~38MB) is not committed to this repository; only the compact, already
feature-engineered, gzip-compressed training CSV is (`jamming_features_labeled.csv.gz`,
~17MB).

## API

- `GET /api/jamming/model-info` — dataset, metrics (controlled + supplementary), limitations
- `GET /api/jamming/samples` — held-out demo samples with true labels (no full feature vectors)
- `POST /api/jamming/predict` — `{"sample_id": "..."}` or `{"features": {...}, "band": "...", "scan_mode": "..."}`

## Frontend

- `/jamming` — dedicated page, visually distinct from Spectrum Simulation/Prediction, sample picker + result card using "Benign RF Activity" / "Potential Jamming / Malicious RF Activity" language (never "Available"/"Occupied")
- `/model` — extended with a second, clearly separated panel summarizing this model alongside the existing Spectrum Availability model

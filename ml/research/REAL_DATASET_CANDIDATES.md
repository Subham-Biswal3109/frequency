# Real RF/Spectrum Dataset Candidates — Feasibility & Ranking

**Status: research/investigation only. No production ML model, backend,
frontend, database, or the existing synthetic dataset was modified to
produce this document.**

**Goal:** find an authentic, publicly available, real-measured dataset with
an explicit, non-invented occupancy/availability ground-truth label that
could legitimately replace or supplement
`spectrum_occupancy_synthetic_33600.csv` for Wire Watcher's **Spectrum
Availability** model (Available/Occupied).

Throughout this document, **[VERIFIED]** marks a fact taken directly from a
primary source page/paper I fetched or a search result quoting that source.
**[INFERENCE]** marks my own reasoning/judgment, not a stated fact. Where I
could not confirm something, it is marked **"insufficient evidence"** rather
than assumed.

---

## Candidate 1 — A Multi-Environment Real-World Multi-Band RF Dataset for Spectrum Sensing and Occupancy Analysis

| Field | Value |
|---|---|
| Dataset | A Multi-Environment Real-World Multi-Band RF Dataset for Spectrum Sensing and Occupancy Analysis for Allocations or Sharing |
| Official source | IEEE DataPort |
| Paper | **[INFERENCE]** No peer-reviewed paper found citing this dataset yet — it was created 2026-08-28, essentially brand new. A companion GitHub repo ("FedSense") is linked from the sister dataset (Candidate 2), suggesting a paper may be in preparation. |
| DOI | **[VERIFIED]** `10.21227/5cc8-wg20` |
| Institution | **[VERIFIED]** Énergie Matériaux Télécommunications Research Centre, Institut national de la recherche scientifique (INRS-ÉMT), Montréal, Canada — a real Canadian university research institute. Authors Atik Mahabub and Shervin Vakili both have ORCID iDs linked from the page. |
| Frequency range | **[VERIFIED]** Seven allocations: `ISM_2400` (2400–2483.5 MHz), `UNII_1` (5150–5250), `UNII_2A_DFS` (5250–5350), `UNII_2C_DFS` (5470–5725), `UNII_3_5800` (5725–5850), `UNII_4_ITS` (5850–5925), `UNII_5_6E_EDGE` (5925–5995 MHz) |
| 2.4 GHz | **[VERIFIED]** Yes — `ISM_2400` explicitly covers the full Wi-Fi 2.4 GHz band |
| 5 GHz | **[VERIFIED]** Yes — five separate 5 GHz allocations covering Wi-Fi channels 36–165, including DFS |
| Number of samples | **[VERIFIED]** 7 sessions (environments) × ≈1.02×10⁶ complex samples per dwell, continuous PSD (1024 bins/row) for the whole session per environment, plus raw SigMF I/Q recordings. Exact frame count depends on the conversion script's `--frame-len`; not a single fixed "N samples" figure. |
| Raw IQ available | **[VERIFIED]** Yes — SigMF 1.0.0 format, int16 interleaved I/Q, with JSON sidecar metadata |
| PSD/spectrum available | **[VERIFIED]** Yes — a continuous float16 PSD record (1024 bins/row) covering all seven bands for each full session |
| Channel information | **[VERIFIED]** Per-dwell CSV index with timestamp, band label, center frequency, sample rate, RX gain, noise-floor estimate, occupancy fraction, burst statistics |
| Occupancy label | **[VERIFIED]** Yes — explicit binary occupancy label, plus a numeric occupancy percentage reported per band per environment (0.0%–55.6% across the corpus) |
| Label definition | **[VERIFIED]** "Occupancy is the fraction of PSD bins in the band exceeding the per-tile noise floor by 6 dB, over the whole session" — a documented energy-detection threshold, not an arbitrary/undisclosed one. For the raw-I/Q framed tier, labels come from a "dead-zone energy detector" with explicitly exposed `--hi-db`/`--lo-db` threshold flags in the released converter script. |
| Ground-truth methodology | **[VERIFIED]** Real over-the-air RF measurement with a real SDR receiver; occupancy is a **derived** label (energy detection relative to measured noise floor), not a directly-observed ground truth (e.g. not confirmed against actual AP association logs). This matches the standard, literature-accepted energy-detection convention for spectrum-sensing datasets (see the Cheema & Salous reference in Candidate 3), and is unusually transparent about its exact formula and threshold — but it is still a threshold-based derived label, with the dataset's own documentation candidly warning that "a broadband impulsive event" was captured in one session that "energy detectors routinely mislabel." |
| Hardware | **[VERIFIED]** Single Ettus USRP B205mini-i SDR, dual-band 2.4/5/5.8 GHz omnidirectional antenna, per-tile calibrated RX gain (recorded in `gain_cal_wifi.json`) |
| Locations | **[VERIFIED]** Seven real Montréal-area sites: INRS-ÉMT office, a public park, STM Place-des-Arts metro station, a residential apartment, and three sessions described as Concordia University campus (indoor, library) — **note:** the dataset's own overview table lists environment 7 as "Campus Outside... Concordia University grounds," but the same page's detailed write-up for that environment is titled "Campus Outside (McGill University)." **[VERIFIED — internal inconsistency found in the source, not introduced by me.]** This should be clarified with the dataset authors before citing the exact institution for that one session. |
| Time information | **[VERIFIED]** Exact session date/time windows given per environment (all in August 2026), sample rate (15 or 25 MS/s), per-dwell timestamps in SigMF metadata |
| License | **[VERIFIED — a real finding, not an omission on my part]** The dataset's own "License" section literally reads `[license, e.g. CC BY 4.0]` — an unfilled template placeholder. **The license is currently unspecified.** This must be resolved (e.g., by contacting the authors) before any research use, redistribution, or publication reuse. |
| Can we download it | **[VERIFIED]** "This dataset requires an IEEE DataPort Subscription to access." Six per-environment ZIP files ranging from 1.99 GB (Library) to 35.64 GB (Indoor Office); not freely downloadable without a subscription (institutional or paid). |
| Can we use it for research | **[INFERENCE]** Likely yes once a subscription is available and the license is clarified with the authors — IEEE DataPort is a legitimate, established academic data repository and the authors explicitly invite contact via private message. Not yet a "yes" in absolute terms given the missing license field. |
| Compatibility with our current model | **[INFERENCE]** High. Numeric per-band occupancy %, explicit threshold formula, and both PSD and per-dwell tabular index map cleanly onto a supervised binary classification task very similar to Wire Watcher's existing `signal_power_dbm`/`noise_floor_dbm`/`snr_db` feature style — though power values here are **explicitly dBFS, not calibrated dBm** (see below), so Wire Watcher's existing `signal_power_dbm` feature semantics would need to be relabeled/reinterpreted, not blindly reused. |
| Major limitations | (1) License unspecified; (2) requires paid/institutional IEEE DataPort access; (3) power values are **explicitly documented as "Relative dB re: ADC full scale (dBFS) — NOT calibrated to dBm"** — cannot be used as calibrated dBm without independent calibration; (4) brand-new (Aug 2026), zero independent citations/replications yet; (5) very large per-environment files (up to 35.64 GB); (6) the location-labeling inconsistency noted above; (7) label is threshold-derived, not independently ground-truthed against a known transmitter log. |

## Candidate 2 — A Real-World Multi-Band RF Dataset from a Mass-Gathering Event for Spectrum Analysis Framework

| Field | Value |
|---|---|
| Dataset | A Real-World Multi-Band RF Dataset from a Mass-Gathering Event for Spectrum Analysis Framework |
| Official source | IEEE DataPort |
| Paper | **[VERIFIED]** Companion GitHub repository linked: `github.com/atikmahabub42/FedSense` — suggests an associated paper (possibly on federated learning) exists or is in preparation; I did not locate a separate peer-reviewed publication citing this exact dataset. |
| DOI | **[VERIFIED]** `10.21227/f2rv-ms26` |
| Institution | **[VERIFIED]** Same as Candidate 1 — INRS-ÉMT, Montréal, Canada; same two authors |
| Frequency range | **[VERIFIED]** 2.4 GHz ISM band plus 5, 5.8, and lower 6 GHz U-NII allocations (43 swept tiles across 7 bands total, per the dataset description) |
| 2.4 GHz | **[VERIFIED]** Yes |
| 5 GHz | **[VERIFIED]** Yes (5/5.8 GHz explicitly named; raw I/Q specifically recorded for `ISM_2400`, `UNII_1`, and `UNII_3_5800`) |
| Number of samples | **[VERIFIED]** 92,843 labeled complex-baseband frames of 32,768 samples each (raw-I/Q tier); continuous PSD record for the full 51-minute session across all 43 tiles (PSD tier) |
| Raw IQ available | **[VERIFIED]** Yes, for 3 of the 7 bands (storing I/Q continuously for all 43 tiles was reported as infeasible) |
| PSD/spectrum available | **[VERIFIED]** Yes, for all 7 bands, full session |
| Channel information | **[VERIFIED]** Per-dwell CSV index (timing, tuning, gain, spectral occupancy, burst statistics, duty cycle) |
| Occupancy label | **[VERIFIED]** Yes — binary occupied/vacant label per frame |
| Label definition | **[VERIFIED]** "Binary occupancy labels derived by a conservative dead-zone energy detector," with an exact printed class split: 10,384 occupied (11.18%) vs. 82,459 vacant (88.82%) at the paper's default framing settings. The documentation explicitly warns that changing the frame length or crop position changes the correct label, and quantifies a specific failure mode (~25% label noise in the positive class under a specific mismatch scenario) — an unusually rigorous, transparent treatment of label validity. |
| Ground-truth methodology | **[VERIFIED]** Same energy-detection family as Candidate 1; a real, naturally-occurring congestion event (≈100,000-person fireworks festival) rather than a synthesized or staged scenario — "nothing in the corpus is synthesised" per the abstract. Still a derived (not independently verified) label. |
| Hardware | **[VERIFIED]** Same as Candidate 1 — single Ettus USRP B205mini-i, dual-band 2.4/5/5.8 GHz omnidirectional antenna |
| Locations | **[VERIFIED]** One location: Montréal waterfront, during the "International des Feux Loto-Québec" fireworks festival |
| Time information | **[VERIFIED]** 51-minute continuous acquisition bracketing the fireworks display; exact UTC timestamps in SigMF sidecars |
| License | **[INFERENCE]** Not explicitly stated on the page text I retrieved (no license section reproduced in the abstract/instructions section); given the sibling dataset's placeholder issue, **treat as unconfirmed until directly verified**, not assumed to be open. |
| Can we download it | **[VERIFIED]** Single 8.73 GB ZIP; same "Subscription Required" gate as Candidate 1 |
| Can we use it for research | **[INFERENCE]** Same caveats as Candidate 1 — plausible but license must be confirmed first |
| Compatibility with our current model | **[INFERENCE]** High for a single-scenario, imbalanced-class training/validation set; its real 11.2% occupancy prior is scientifically valuable for realistic class-imbalance handling (directly relevant given Wire Watcher's Phase 7 requirement to not rely on accuracy alone under imbalance) |
| Major limitations | (1) Single event/location — no cross-environment generalization signal on its own (pairs well with Candidate 1 for that); (2) same subscription/access barrier; (3) license unconfirmed; (4) same dBFS-not-dBm caveat expected (not explicitly restated on this specific page, but the acquisition chain is identical to Candidate 1's, which documents the same convention); (5) derived, not independently verified, label. |

## Candidate 3 — Cheema & Salous, "Spectrum Occupancy Measurements and Analysis in 2.4 GHz WLAN" (MDPI Electronics, 2019)

| Field | Value |
|---|---|
| Dataset | No standalone dataset name — a peer-reviewed measurement paper |
| Official source | MDPI *Electronics*, and Durham Research Online (DRO), Durham University's institutional repository |
| Paper | **[VERIFIED]** Cheema, A.A., Salous, S. (2019). "Spectrum Occupancy Measurements and Analysis in 2.4 GHz WLAN." *Electronics*, 8(9), 1011 |
| DOI | **[VERIFIED]** `10.3390/electronics8091011` |
| Institution | **[VERIFIED]** Durham University (paper hosted on `dro.dur.ac.uk`, Durham's own repository) |
| Frequency range | **[VERIFIED]** 2.4 GHz WLAN band specifically |
| 2.4 GHz | **[VERIFIED]** Yes — this is the paper's entire focus |
| 5 GHz | **[VERIFIED]** No |
| Number of samples | **[INFERENCE]** Not stated as a discrete sample count in the abstract/search snippets available to me; the paper reports statistical distributions (lognormal, gamma) fit to idle-time-window durations, implying a large continuous time-series recording rather than a fixed sample count. Insufficient evidence to state an exact N. |
| Raw IQ available | **[INFERENCE]** No indication of raw IQ release; the described system is a "custom-designed wideband sensing engine" recording received power directly, not IQ samples. |
| PSD/spectrum available | **[INFERENCE]** The paper describes power-vs-time recordings and derived idle/busy time series, not a published PSD file dataset for download. |
| Channel information | **[VERIFIED]** Indoor environment, omnidirectional and directional antenna configurations tested separately |
| Occupancy label | **[VERIFIED]** Yes — explicit busy/idle channel-state label |
| Label definition | **[VERIFIED]** "A custom-designed wideband sensing engine records the received power of signals, and its performance is presented to select the decision threshold required to define the channel state (busy/idle)" — i.e., a documented, methodologically-justified energy-detection threshold (the paper explicitly discusses detector performance in selecting this threshold, referencing the classical Urkowitz 1967 energy-detection formulation per its reference list). |
| Ground-truth methodology | **[VERIFIED]** Real over-the-air 2.4 GHz WLAN measurement in an indoor environment; threshold chosen with reference to detector performance analysis, a defensible, literature-grounded methodology. |
| Hardware | **[INFERENCE]** "Custom-designed wideband sensing engine" — exact hardware model not confirmed from the snippets available; would require reading the full paper PDF for a receiver spec sheet. |
| Locations | **[VERIFIED]** Indoor environment (single site; exact institution/room not confirmed from available snippets) |
| Time information | **[INFERENCE]** "Different network traffic loads" were tested; exact date/duration not confirmed from available snippets. |
| License | **[VERIFIED]** Open access, Creative Commons Attribution License ("This is an open access article distributed under the Creative Commons Attribution License which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited") |
| Can we download it | **[VERIFIED]** The **paper** is freely downloadable (MDPI open access + Durham's DRO mirror). **[INSUFFICIENT EVIDENCE]** for a separate, standalone raw measurement dataset file — I found no data-availability statement or supplementary dataset link in the sources retrieved. **Treat this as a methodology reference, not a confirmed downloadable ML training dataset**, unless the authors are contacted directly and confirm raw data can be shared. |
| Can we use it for research | **[VERIFIED]** Yes, for citing methodology (CC BY permits this freely) |
| Compatibility with our current model | **[INFERENCE]** Low as a direct data source (no confirmed raw data download) but **high as methodological justification** for using an energy-detection-threshold-based occupancy label on any dataset that provides raw power/PSD but no label — this is exactly the kind of literature citation Phase 3 of the original ElectroSense investigation asked for. |
| Major limitations | No confirmed downloadable dataset; 2.4 GHz only (no 5 GHz); single indoor site; sample size/duration not confirmed from available sources. |

## Candidate 4 — NSF/Shared Spectrum Company Spectrum Occupancy Measurements (McHenry et al., 2004–2005)

| Field | Value |
|---|---|
| Dataset | NSF Spectrum Occupancy Measurements (multi-city campaign) |
| Official source | Shared Spectrum Company report to NSF; widely cited in subsequent literature (e.g., ACM MobiCom 2009 "Mining spectrum usage data") |
| Paper | **[VERIFIED]** M. A. McHenry, "NSF spectrum occupancy measurements project summary," Shared Spectrum Company Report, August 2005 — a foundational, extremely widely-cited reference in the cognitive-radio literature |
| DOI | **[INSUFFICIENT EVIDENCE]** No DOI located for the original report from the sources reviewed. |
| Institution | **[VERIFIED]** Shared Spectrum Company, under NSF funding |
| Frequency range | **[INFERENCE]** Commonly cited as covering roughly 30 MHz–3 GHz across multiple measurement campaigns/cities (based on how this study is described in downstream citing papers); I could not directly confirm the exact per-city band list from a primary source in this investigation. |
| 2.4 GHz | **[INFERENCE]** Likely included, given the stated upper range, but not independently confirmed from a primary source in this investigation. |
| 5 GHz | **[INFERENCE]** Uncertain/likely not, if the upper bound was ~3 GHz for most campaigns. |
| Number of samples | **[INSUFFICIENT EVIDENCE]** |
| Raw IQ available | **[INSUFFICIENT EVIDENCE]** |
| PSD/spectrum available | **[INFERENCE]** The original campaign is generally described in the literature as reporting occupancy **percentages by band**, not raw PSD/IQ for redistribution. |
| Channel information | **[INFERENCE]** Reported at the band level (e.g., "cellular band," "ISM band"), per multiple secondary citations. |
| Occupancy label | **[VERIFIED]** Yes, in the sense that "spectrum occupancy" percentage by band is exactly what this classic study reports and is famous for. |
| Label definition | **[INSUFFICIENT EVIDENCE]** for the exact threshold/methodology from a primary source in this investigation; widely referenced as an energy-detection-based measurement campaign in the surrounding literature. |
| Ground-truth methodology | **[INFERENCE]** Real measurement campaign, multiple US cities; considered a foundational reference dataset in the field. |
| Hardware | **[INSUFFICIENT EVIDENCE]** |
| Locations | **[INFERENCE]** Multiple U.S. cities (commonly cited figure is around 6 locations); not independently confirmed here. |
| Time information | **[VERIFIED]** ~2004–2005 |
| License | **[INSUFFICIENT EVIDENCE]** |
| Can we download it | **[INSUFFICIENT EVIDENCE]** — I found no working direct download link for a raw dataset file in this investigation; it appears to circulate primarily as **summary tables and percentages quoted in later papers**, not as a raw, ML-ready file. |
| Can we use it for research | **[INFERENCE]** As a **citation for typical real-world occupancy statistics** (e.g., to sanity-check whether a derived label's occupancy rate is realistic), yes. As a raw training dataset, **insufficient evidence it is downloadable at all today.** |
| Compatibility with our current model | **[INFERENCE]** Low as a direct data source; useful only as a literature benchmark. |
| Major limitations | Likely not directly downloadable as raw/ML-ready data (only summary statistics found); band coverage for 5 GHz unconfirmed; over 20 years old, so real-world spectrum usage patterns (especially Wi-Fi density) have changed enormously since 2004–2005. |

## Candidate 5 — NSF-funded "Dataset for Spectrum Coexistence in Passive Sensing and Wireless Communication"

| Field | Value |
|---|---|
| Dataset | Dataset for Spectrum Coexistence in Passive Sensing and Wireless Communication |
| Official source | NSF Public Access Repository (`par.nsf.gov`) |
| Paper | **[INFERENCE]** Associated with NSF-funded research on L-band passive microwave radiometry coexistence with 5G; exact paper title not confirmed from the snippet retrieved. |
| DOI | **[INSUFFICIENT EVIDENCE]** — not confirmed from the retrieved snippet; would need to open the `par.nsf.gov` record directly. |
| Institution | **[INFERENCE]** A U.S. university research group under NSF funding (exact institution not confirmed from the snippet). |
| Frequency range | **[VERIFIED]** The protected L-band, 1400–1427 MHz (a passive remote-sensing/radiometry protected band) |
| 2.4 GHz | **[VERIFIED]** **No.** |
| 5 GHz | **[VERIFIED]** **No.** |
| Number of samples | **[INSUFFICIENT EVIDENCE]** |
| Raw IQ available | **[INSUFFICIENT EVIDENCE]** |
| PSD/spectrum available | **[INFERENCE]** Likely yes ("calibrated power spectra" is referenced in a related paper describing a similar Dryad-hosted dataset from what appears to be the same research area, though that specific quote was found for a *different*, 2.7/4.4 GHz dataset — **do not conflate the two**; flagged here only to show the general practice in this NSF-funded research area of releasing calibrated spectra via Dryad.) |
| Channel information | **[INSUFFICIENT EVIDENCE]** for this specific dataset |
| Occupancy label | **[INSUFFICIENT EVIDENCE]** — the retrieved description discusses out-of-band emission coexistence, not an explicit binary occupancy label for this exact dataset. |
| Label definition | **[INSUFFICIENT EVIDENCE]** |
| Ground-truth methodology | **[INFERENCE]** Real passive radiometer measurement, a legitimate and well-documented measurement science context, but for an entirely different application (protecting scientific passive sensing from telecom out-of-band emissions), not Wi-Fi channel availability. |
| Hardware | **[INSUFFICIENT EVIDENCE]** |
| Locations | **[INSUFFICIENT EVIDENCE]** |
| Time information | **[INSUFFICIENT EVIDENCE]** |
| License | **[INFERENCE]** NSF Public Access Repository entries are generally openly accessible, but the specific dataset license was not confirmed. |
| Can we download it | **[INFERENCE]** Likely, via the `par.nsf.gov` record or a linked repository, but not directly confirmed in this investigation. |
| Can we use it for research | **[INFERENCE]** Yes, in principle, for its own stated purpose (spectrum coexistence in passive sensing). |
| Compatibility with our current model | **[VERIFIED — disqualifying]** **Wrong frequency band entirely (1400–1427 MHz, not 2.4/5 GHz Wi-Fi)** and a different application domain (protecting a scientific passive-sensing allocation from telecom interference, not Wi-Fi channel availability prediction). |
| Major limitations | Band and application mismatch make this **not usable** for Wire Watcher's stated goal, regardless of its data quality. Included here specifically as a "real, well-funded, real dataset that nonetheless fails the frequency-relevance criterion" — precisely the kind of dataset the task instructions warned not to recommend "merely because it contains RF data." |

---

## Scoring (0–10 per criterion)

| Criterion | C1: Multi-Env (IEEE DataPort) | C2: Mass-Gathering (IEEE DataPort) | C3: Cheema & Salous (MDPI) | C4: NSF/SSC (2004–05) | C5: NSF L-band Coexistence |
|---|---|---|---|---|---|
| A. Authenticity/provenance | 8 | 8 | 9 | 7 | 7 |
| B. Occupancy-label quality | 7 | 7 | 7 | 4 | 2 |
| C. Frequency relevance | 10 | 9 | 6 | 4 | 0 |
| D. Dataset size | 8 | 6 | 3 | 4 | 3 |
| E. Documentation | 10 | 9 | 6 | 3 | 3 |
| F. Reproducibility | 6 | 6 | 3 | 2 | 4 |
| G. Compatibility with our project | 8 | 7 | 4 | 2 | 0 |
| H. Research-paper suitability | 7 | 7 | 8 | 6 | 5 |
| **Total (/80)** | **64** | **59** | **46** | **32** | **24** |

**Scoring notes [INFERENCE, my own judgment]:**
- C1/C2 score highest on documentation and frequency relevance precisely because their authors were unusually transparent about exact thresholds, calibration caveats, and even their own dataset's failure modes — a strong positive signal of scientific rigor, even though they are very new and unconfirmed on licensing.
- C1/C2 lose points on reproducibility (F) because of the paid-subscription access barrier and the unresolved license — a project cannot be fully "reproducible" for others if access itself is gated and unclear.
- C3 scores highest on authenticity/provenance and research-paper suitability (peer-reviewed, CC BY, a real university) but loses heavily on dataset size/compatibility because no confirmed raw dataset download exists.
- C4 and C5 score low overall for this project specifically — not because they are poor science, but because they either lack confirmed public raw data (C4) or target the wrong frequency band entirely (C5).

---

## Final recommendation

### 1. BEST replacement candidate
**Candidate 1 — "A Multi-Environment Real-World Multi-Band RF Dataset for Spectrum Sensing and Occupancy Analysis" (IEEE DataPort, DOI 10.21227/5cc8-wg20).** It is the only candidate found with **both** 2.4 GHz and 5 GHz coverage, an **explicit, formula-documented occupancy label**, real multi-environment diversity (office/residential/campus/library/metro/park) needed for a defensible train/test generalization story, and unusually rigorous, self-critical documentation (it flags its own DFS-channel quirks, dataset inconsistencies, and detector failure modes). **Before using it:** (a) resolve IEEE DataPort subscription access, (b) get the license field clarified directly from the authors (it is currently an unfilled placeholder), and (c) never present its dBFS power values as calibrated dBm.

### 2. BEST supplementary dataset
**Candidate 2 — "A Real-World Multi-Band RF Dataset from a Mass-Gathering Event" (IEEE DataPort, DOI 10.21227/f2rv-ms26).** Same authors, same hardware, same methodology as Candidate 1, but a single dramatic, realistically-imbalanced (11.2% occupied) scenario — excellent as an **additional evaluation set** to test how a model trained on Candidate 1's calmer environments generalizes to an extreme-congestion event, rather than as a primary training set on its own.

### 3. BEST fallback
**Candidate 3 — Cheema & Salous (2019), MDPI *Electronics*, Durham University.** Not usable as a downloadable dataset (insufficient evidence one exists), but it is the strongest available **peer-reviewed literature citation** for justifying an energy-detection busy/idle threshold methodology specifically for 2.4 GHz WLAN — useful to cite in Wire Watcher's own documentation regardless of which raw dataset is ultimately used, and worth directly emailing the authors to ask whether raw measurements can be shared.

### 4. Datasets that should NOT be used, and why
- **Candidate 4 (NSF/Shared Spectrum Company, 2004–2005):** no confirmed downloadable raw dataset was found (appears to survive only as summary statistics in later papers), and it is over 20 years old — Wi-Fi spectrum usage patterns have changed too much since then to be representative today.
- **Candidate 5 (NSF L-band passive-sensing coexistence dataset):** real, credible, NSF-funded science — but for the **wrong frequency band and the wrong application entirely** (radio-astronomy-adjacent passive sensing coexistence, not Wi-Fi channel availability). Including it in a Wi-Fi occupancy training pipeline would misrepresent what the data actually measures, exactly the mistake this investigation was told to avoid.
- **ElectroSense / SpecScape (from the prior investigation):** reconfirmed here as **not suitable**, per the previously delivered `ml/research/ELECTROSENSE_FEASIBILITY.md` — signs of project abandonment, no confirmed 2.4/5 GHz coverage, no India coverage, and no occupancy label.

**Overall recommendation:** pursue Candidate 1 as the primary real-data replacement candidate, using Candidate 2 as a stress-test supplementary set, once (a) IEEE DataPort access is secured and (b) the dataset's license is confirmed directly with the authors. Do not proceed with training until both of those are resolved — an unresolved license is a real blocker for a research paper, not a formality.

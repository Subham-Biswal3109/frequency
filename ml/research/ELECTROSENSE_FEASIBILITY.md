# ElectroSense / SpecScape Feasibility Investigation

**Status: research/investigation only. No production code, model, backend,
frontend, or database was touched to produce this document.**

**Scope:** evaluate whether real-world RF measurements from ElectroSense
(and, secondarily, SpecScape) can replace `spectrum_occupancy_synthetic_33600.csv`
as the training data for Wire Watcher's existing **Spectrum Availability**
model (Available/Occupied). This is a separate question from the
RF Interference/Jamming Detector already integrated from `release_artifacts`
— see `ml/jamming/README.md` for that unrelated model.

---

## 1. API source and provenance

- **ElectroSense** ("Electrosense: Open and Big Spectrum Data", Rajendran et
  al., IEEE Communications Magazine, 2018) is a crowd-sourced RF spectrum
  monitoring network built on low-cost RTL-SDR-class sensors, originally
  developed by IMDEA Networks Institute (Madrid) with EU/armasuisse funding.
- Official resources reviewed: `github.com/electrosense/api-examples`,
  `electrosense.org`, `electrosense.networks.imdea.org` (the current
  IMDEA-hosted project page), and the associated IEEE ComMag paper and
  follow-on papers (e.g. INFOCOM 2023 spectrum-classification framework).
- **Critical finding — project status.** The current official IMDEA-hosted
  page (`electrosense.networks.imdea.org`) describes the network entirely in
  **past tense**: *"The ElectroSense network **was** a crowd-sourcing
  initiative... It **used** small radio sensors... The initiative's goal
  **was** to sense..."* This phrasing on the project's own current landing
  page is consistent with the network being described as a **historical/
  concluded** effort rather than an actively operated live service.
- A public discussion thread ("Electrosense project undergoing
  abandonment?", `groups.google.com/g/electrosense-general`) reports the
  number of "online" sensors visible on the live map had fallen to **56**,
  and that emails to the project's contact address went unanswered.
- I could not directly browse the live sensor map or the Swagger API spec
  page (`electrosense.org/app.html#!/api-spec`) from this environment (the
  site blocks automated fetches), so I cannot personally confirm today's
  exact live sensor count or API uptime. **This must be independently
  verified by manually visiting the site and attempting a real API call
  before any commitment is made** — this report should not be treated as
  proof the API is fully dead, only as strong, multi-sourced evidence of
  decline that raises the bar for due diligence.

## 2. Endpoints

Per the `api-examples` repository's own description: *"The Electrosense
REST API allows retrieval of raw and aggregated spectrum data as well as
information about sensors."* Based on this description and the
architecture described in the IEEE ComMag paper, the API exposes
(at a conceptual level) sensor discovery, raw/aggregated spectrum
measurement retrieval, and historical measurement queries.

**I was not able to directly fetch and verify the exact URL paths, request
parameter names, or field-level JSON response schema** — GitHub's raw file
view and the interactive Swagger spec page were both inaccessible from this
environment. Academic papers that *used* the API describe its behavior
(see §5) but do not reproduce the raw request/response JSON. **This is a
gap, not an assumption filled in — do not proceed on the assumption that
the schema below is complete; the `python/` examples in the repo and the
live Swagger spec must be read directly before writing any integration
code.**

## 3. Authentication

The `api-examples` repository and downstream tooling (e.g. `apd.aggregation`-
style sensor aggregators generally, and ElectroSense's own documentation as
referenced in secondary sources) is consistent with an **API-key-based**
authentication model, requested via account registration on the ElectroSense
portal. I could not directly confirm the exact header name or key format
from primary sources in this environment. **Insufficient evidence to state
the exact authentication mechanism with certainty — verify directly against
the live `python/` example scripts before integration.**

## 4. Request examples

**Insufficient evidence.** I could not fetch the `python/` subdirectory of
`api-examples` (blocked by the site's automated-access policy) to extract
literal example request code. Do not fabricate example calls; a developer
must clone the repository directly (`git clone
https://github.com/electrosense/api-examples`) and read `python/*.py`.

## 5. Response schema / measurements actually available

From an independent academic paper that *used* the live ElectroSense API
directly for a classification task (Distributed Deep Learning Models for
Wireless Signal Classification with Low-Cost Spectrum Sensors, arXiv:1707.08908):

- Spectral resolution: **~10 kHz**
- Time resolution: **60 seconds** per aggregated reading
- Underlying FFT size: 256 bins, sensor ADC bit-width 8 → effective ~12-bit
  after processing
- **Dynamic range: theoretical 74 dB, practical ~60–65 dB** — described as
  a receiver dynamic-range characteristic, not a statement that returned
  values are calibrated absolute dBm.
- Sensors used in that study had **omnidirectional antennas, deployed
  indoors**, following sequential full-spectrum scanning (not a fixed
  narrowband dwell).
- Multiple FFT vectors (5) are averaged server-side to reduce thermal noise
  before the aggregated value is exposed via the API — i.e. the "aggregated
  spectrum" endpoint returns a **noise-reduced power/magnitude estimate**,
  not a raw single-snapshot reading.

**On calibration:** none of the sources reviewed explicitly document an
end-to-end calibration procedure that would justify treating returned power
values as calibrated absolute dBm across sensors. Per this project's rule,
**such values must NOT be presented as calibrated dBm** unless a specific
sensor's calibration certificate/procedure is located and verified.

**Timestamp, sensor location, and metadata:** the underlying architecture
paper confirms sensor registration includes antenna details and location,
visible on the sensor map, and that the API surfaces sensor status/location
information alongside spectrum data. Exact field names again require direct
inspection of the live spec.

## 6. Frequency coverage

- The live network's sensors are described in the primary IEEE ComMag paper
  as targeting **DC to 6 GHz** capability at the hardware level (RTL-SDR +
  up/downconverter), which would in principle span both the 2.4 GHz and
  5 GHz Wi-Fi bands.
- However, the one **concretely verifiable, downloadable ElectroSense
  dataset I could locate** (Zenodo record 7521246, "ElectroSense PSD
  Spectrum Dataset", published alongside the INFOCOM 2023 spectrum-
  classification paper) explicitly covers **24 MHz – 1.7 GHz only**
  (RTL-SDR full-spectrum sweep in 2 MHz chunks), labeled for licensed bands
  such as FM — **this published dataset does NOT cover 2.4 GHz or 5 GHz at
  all.**
- **Conclusion: hardware capability up to 6 GHz is claimed at the network-
  architecture level, but the only real, inspectable dataset artifact found
  in this investigation does not include the Wi-Fi ISM bands Wire Watcher
  needs.** Whether any *individual currently-online* sensor is actually
  configured to scan 2.4/5 GHz right now cannot be confirmed without live
  API access.

## 7. India coverage

**No evidence of any India-based ElectroSense sensor was found in any
resource reviewed** (official pages, GitHub, IEEE/arXiv papers, the Zenodo
dataset, or community discussion). Every concrete deployment reference found
names **European** locations (Madrid/IMDEA, St. Gallen/Switzerland, "47
sensors across Europe"). Per the explicit instruction not to assume
coverage: **India coverage is unconfirmed and, based on available evidence,
likely absent or negligible.** This must be verified against the live
sensor map directly (not assumed from this document) before any commitment.

## 8. 2.4/5 GHz feasibility

Given §6, feasibility for exactly Wire Watcher's target bands is
**unconfirmed and doubtful based on available evidence** — the one
inspectable published dataset stops at 1.7 GHz. If a live sensor happens to
be configured for 2.4/5 GHz scanning, that would need to be confirmed
sensor-by-sensor via the live API (if it is even still operational).

## 9. Available/Occupied label feasibility

**No Available/Occupied (or any binary occupancy) label is provided by
ElectroSense.** The API and its published datasets provide **raw/aggregated
PSD-style measurements and sensor metadata only** — occupancy is not a
field in any source reviewed. This matches the API's own self-description
("retrieval of raw and aggregated spectrum data... information about
sensors") — it is a **measurement** service, not a **classification** or
**labeling** service.

### Proposed labeling methodology, IF pursued

A scientifically defensible energy-detection occupancy label from
measured PSD would follow the classical spectrum-sensing energy-detector
formulation (see e.g. Digham, Alouini & Simon, "On the Energy Detection of
Unknown Signals over Fading Channels," IEEE Trans. Commun., 2007, and the
general spectrum-sensing survey literature, e.g. Yucek & Arslan, "A Survey
of Spectrum Sensing Algorithms for Cognitive Radio Applications," IEEE
Commun. Surveys & Tutorials, 2009):

```
occupied  if  measured_power(f) > noise_floor_estimate(f) + margin_dB
available otherwise
```

This is a **derived, threshold-based label**, not a ground-truth
measurement — it would need to be documented in exactly this way (formula,
margin value and its justification, and the fact that it is an estimation
method with a known false-alarm/miss-detection tradeoff), matching the
literature's own treatment of energy detection as an imperfect, threshold-
sensitive method. **This project's rule against inventing thresholds means
any concrete `margin_dB` value chosen would itself need either (a) a
noise-floor characterization of the specific sensor hardware, or (b) a
citation to a specific paper using that exact margin for the exact same
sensor/ADC characteristics** — neither of which I located in this
investigation. This section is therefore a **methodology sketch**, not a
ready-to-implement recipe.

## 10. Proposed labeling methodology (summary)

Energy-detection thresholding against a per-sensor, per-frequency
estimated noise floor, following classical spectrum-sensing literature —
but only implementable once (a) the API is confirmed live, (b) a specific
sensor's noise-floor behavior is characterized, and (c) a margin is
justified from either hardware characterization or a directly-comparable
published study. **Not implementable from the evidence gathered so far.**

## 11. Dataset size feasibility

Cannot be assessed without live API access and rate-limit documentation
(neither of which I could verify). The one confirmed downloadable archive
(Zenodo 7521246) totals measurements from 47 sensors × 6 hours each, in the
wrong frequency range for this project. **Insufficient evidence** to
estimate a realistic 2.4/5 GHz sample count from ElectroSense today.

## 12. Licensing / provenance

- The `PSD-technology-classification-framework` companion code repository
  is released under a **BSD 3-Clause License**; its associated Zenodo
  dataset requires **citation** of the INFOCOM 2023 paper.
- No explicit terms-of-use for the *live* ElectroSense API (rate limits,
  redistribution rights, commercial-use restrictions) were located in this
  investigation. **Insufficient evidence** — must be read directly from the
  live portal's terms before any data collection.

## 13. Risks and limitations (summary)

1. **Project viability risk (high):** current official page uses past
   tense; independent community reports describe apparent abandonment
   (56 online sensors, unresponsive maintainers).
2. **Band-coverage risk (high):** the only verifiable published dataset
   stops at 1.7 GHz; 2.4/5 GHz availability is unconfirmed.
3. **Geographic-coverage risk (high):** zero evidence of India sensors;
   all confirmed deployments are European.
4. **Label risk (high):** no occupancy label exists; any derived label
   requires a defensible, hardware-specific noise-floor/margin
   justification not available from this investigation alone.
5. **Calibration risk (medium):** no documented end-to-end dBm calibration
   found; values should be treated as uncalibrated/relative unless proven
   otherwise for a specific sensor.
6. **Schema/documentation risk (medium):** exact endpoints, parameters, and
   response fields could not be directly verified in this environment due
   to site access restrictions — must be manually confirmed before any
   integration work begins.
7. **SpecScape risk (high, disqualifying for now):** SpecScape (NSF
   CCRI-funded, University of Wisconsin–Madison) is, per its own project
   page and its inclusion as a "demo" at IEEE DySPAN **2026**, still an
   **active hardware/software research prototype** ("SpecTile" vehicle-
   mounted sensor kits, prototype Android app) with **no evidence found of
   a public REST API, registration process, or downloadable historical
   dataset**. It is not currently usable as a data source for this project
   at all, real-world or otherwise.

## 14. FINAL VERDICT

### ElectroSense: **NOT SUITABLE** (at this time, based on available evidence)

Disqualifying combination: strong evidence of project decline/possible
abandonment, unconfirmed 2.4/5 GHz coverage (the only inspectable dataset
stops at 1.7 GHz), zero evidence of India coverage, and no occupancy label
(requiring a threshold methodology this investigation could not fully
justify without further hardware-specific evidence). This is a "not
suitable given current evidence" verdict, not a permanent scientific
impossibility — a live, hands-on check of the API (which this investigation
could not perform due to environment restrictions) could change this
assessment, but the burden of proof is high given the abandonment signals.

### SpecScape: **NOT SUITABLE** (too early-stage; no public data access exists)

---

## Concise recommendation

**Do not proceed with ElectroSense or SpecScape as a replacement for the
synthetic availability dataset right now.** The evidence gathered points to
real, multi-part risk (project viability, band coverage, geographic
coverage, and missing ground-truth labels) that would need independent,
hands-on verification before any integration work is justified — and even
in the best case, deriving a defensible occupancy label from raw PSD would
require hardware-specific noise-floor characterization this investigation
could not supply from documentation alone.

**Suggested path instead:** keep `spectrum_occupancy_synthetic_33600.csv`
honestly labeled as synthetic (as it already is), and if a *real* dataset
is wanted for the availability model specifically, search for one that
directly ships an occupancy/availability ground-truth label in the 2.4/5 GHz
range (rather than raw PSD requiring you to invent a threshold) — for
example, published cognitive-radio spectrum-occupancy measurement datasets
from IEEE DySPAN/INFOCOM-adjacent research groups that explicitly define
their occupancy methodology, similar in spirit to how `release_artifacts`
gave Wire Watcher's jamming detector a real, directly-usable label rather
than a derived one. I have not identified a specific such dataset in this
investigation — that would be the next research task if you want to pursue
a real dataset for the availability model.

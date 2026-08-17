# Wire Watcher — Project Review 1 Notes

> **Purpose:** Concise preparation notes for the first project review.  
> **Focus:** ECE signal processing + ML pipeline + dataset understanding + deployment.  
> **Source basis:** Project documentation and the review phases covered in this conversation. Where the documentation does not explicitly support a detail, it is marked rather than invented.

---

# 1. PROJECT IN ONE MINUTE

## What is Wire Watcher?

Wire Watcher is a spectrum-monitoring and spectrum-availability prototype.

The system combines:

1. **RF signal simulation**
2. **DSP / spectrum analysis**
3. **Machine-learning based availability prediction**
4. **Flask backend API**
5. **Web frontend**
6. **Database storage**

The current ML model predicts whether a frequency band is:

- **0 → Occupied**
- **1 → Available**

### One-line explanation

> Wire Watcher is a prototype system that analyzes simulated RF/IQ data using DSP techniques and uses a supervised Random Forest classifier to estimate spectrum availability from RF, temporal, geographic and service-related features.

---

# 2. IMPORTANT PROJECT LIMITATIONS

These are important to state honestly.

- The primary ML dataset is **synthetic**, not collected from real RF hardware.
- The current RF source is a **simulated I/Q source**.
- There is currently no physical SDR ingestion.
- The ML model therefore learns patterns from synthetic data, not real-world RF propagation.
- The current peak detector identifies the **strongest global spectral peak**, not all carriers.
- The current model has useful ranking capability but limited minority-class precision/recall.
- Real SDR measurements and real occupancy labels are needed for real-world validation.

### Safe reviewer statement

> “Our current implementation is a prototype. We use synthetic data and simulated I/Q to validate the complete pipeline. Real SDR measurements would be required before claiming real-world spectrum-prediction performance.”

---

# 3. COMPLETE SYSTEM ARCHITECTURE

```text
User / Frontend
      |
      +----------------------+
      |                      |
      v                      v
RF Parameters          Prediction Request
      |                      |
      v                      v
Simulated RF Source      Flask API
      |                      |
      v                      v
Complex I/Q          Preprocessing + ML Model
      |                      |
      v                      v
FFT -> PSD -> Noise    Availability Probability
Floor -> Peak Detection       |
      |                       v
      v                Available / Occupied
Spectrum Visualization        |
                              v
                         Database
```

---

# 4. PHASE 1 — PROJECT INTRODUCTION

## Problem

Spectrum is a limited resource. A spectrum-monitoring system should determine which frequency regions contain signal activity and which may be available.

## Proposed solution

Wire Watcher demonstrates:

- RF signal generation
- Frequency-domain analysis
- Noise-floor estimation
- Peak detection
- ML-based availability prediction
- Web-based visualization
- API and database integration

## Important distinction

**DSP and ML do different jobs.**

- **DSP:** analyzes/characterizes the signal and spectrum.
- **ML:** uses selected features to estimate availability.

---

# 5. PHASE 2/3 — RF + DSP FUNDAMENTALS

## I/Q signal

I/Q means:

- **I = In-phase**
- **Q = Quadrature**

A complex baseband signal can be represented conceptually as:

`s(t) = I(t) + jQ(t)`

I/Q preserves amplitude and phase information.

## Important project detail

The current project does **not** receive I/Q from a physical antenna/SDR.

It uses a simulated RF source.

### Answer if asked: “Where is the real RF hardware?”

> “We currently use a simulated I/Q source. Physical SDR integration such as RTL-SDR or HackRF is planned future work.”

---

# 6. SAMPLING

A continuous signal is converted into discrete samples so a computer can process it.

Sampling rate means:

> Number of samples taken per second.

The current documented RF simulation uses a **20 MHz sampling rate**.

---

# 7. FFT

## What is FFT?

FFT = Fast Fourier Transform.

It efficiently converts a sampled signal from:

**Time domain → Frequency domain**

Why?

Because spectrum monitoring is interested in:

> Which frequencies contain signal energy?

### Viva answer

> “We use FFT to transform the sampled I/Q signal into the frequency domain so that signal energy can be analyzed across frequency.”

The documented implementation uses a **2048-point FFT**.

---

# 8. FFTSHIFT

`fftshift` rearranges the FFT output so that the zero-frequency component is centered.

Conceptually:

```text
Negative frequencies | 0 | Positive frequencies
```

### Viva answer

> “fftshift centers the zero-frequency component, making the spectrum easier to interpret and visualize.”

---

# 9. PSD

PSD = **Power Spectral Density**

It represents how signal power is distributed across frequency.

The project calculates PSD from the FFT magnitude and represents the result in dB/dBm for visualization.

### Graph interpretation

- X-axis → Frequency
- Y-axis → Spectral power
- High peaks → strong signal energy
- Background → noise

---

# 10. NOISE FLOOR

The noise floor represents the general background spectral power.

The current project estimates it using:

`noise_floor = median(PSD)`

### Why median?

Strong signal peaks can distort an average. Median is more robust to a small number of strong peaks.

### Viva answer

> “We estimate the noise floor using the median PSD value because the median is less affected by strong spectral peaks.”

---

# 11. PEAK DETECTION

The current detector uses:

**Noise floor + 10 dB**

as the detection margin.

If the strongest spectral peak exceeds that margin, it is detected.

### IMPORTANT LIMITATION

The current implementation detects the **single strongest global peak** above the threshold.

It does NOT perform complete multi-carrier peak detection.

### Correct statement

> “Our current detector identifies the strongest spectral peak above a 10 dB noise margin. Multi-carrier peak detection is a future improvement.”

---

# 12. DSP PIPELINE

```text
Simulated RF Signal
       |
       v
Complex I/Q
       |
       v
2048-point FFT
       |
       v
fftshift
       |
       v
PSD
       |
       +----> Median PSD -> Noise Floor
       |
       +----> Peak Detection
       |
       v
Spectrum Visualization
```

---

# 13. PHASE 4 — ML FUNDAMENTALS

## Machine Learning

Traditional programming:

`Rules + Input -> Output`

Machine learning:

`Examples -> Learning -> Model -> Prediction`

The model learns patterns from training examples.

---

# 14. YOUR ML PROBLEM

Wire Watcher uses:

> **Supervised binary classification**

### Supervised learning

The training data contains:

- Inputs/features
- Known correct target/label

### Binary classification

There are two output classes:

- `0 = Occupied`
- `1 = Available`

### One-sentence answer

> “We formulate spectrum availability as a supervised binary classification problem where RF and contextual features are used to predict whether a band is occupied or available.”

---

# 15. DATASET

Primary dataset:

` spectrum_occupancy_synthetic_33600.csv `

Documented properties:

- **33,600 records**
- **19 columns**
- **0 missing/null values**
- Synthetic data
- Target: `target_available`

## Class distribution

| Class | Meaning | Records | Percentage |
|---|---|---:|---:|
| 0 | Occupied | 32,326 | 96.21% |
| 1 | Available | 1,274 | 3.79% |

Approximate imbalance:

**25.4 : 1**

---

# 16. WHAT IS A SAMPLE?

One row = one spectrum-availability observation.

A row conceptually contains:

- Frequency
- Bandwidth
- Signal power
- Noise floor
- SNR
- Time
- Location
- Service type
- Availability target

---

# 17. PRODUCTION ML FEATURES

The documented production model uses **11 features**.

## Numerical — 8

1. `start_frequency_mhz`
2. `end_frequency_mhz`
3. `bandwidth_mhz`
4. `hour_of_day`
5. `day_of_week`
6. `signal_power_dbm`
7. `noise_floor_dbm`
8. `snr_db`

## Categorical — 3

9. `state`
10. `city`
11. `service_type`

## Target

`target_available`

- 0 → Occupied
- 1 → Available

---

# 18. FEATURE VS TARGET

### Feature

An input used by the model.

Examples:

- SNR
- Signal power
- Frequency
- Bandwidth
- City

### Target

The value the model tries to predict.

In this project:

`target_available`

---

# 19. DATASET ANALYSIS

Before training, we need to understand the data.

Typical analysis:

1. Dataset dimensions
2. Data types
3. Missing values
4. Target/class distribution
5. Feature distributions
6. Feature meaning
7. Relationships/importance
8. Leakage checks
9. Feature selection

Documented findings:

- 33,600 records
- 19 columns
- 0 missing/null values
- Strong class imbalance
- SNR and signal power dominate reported feature importance
- Target leakage was identified

---

# 20. TARGET LEAKAGE ⭐⭐⭐

## Definition

Target leakage happens when an input feature gives the model information that directly or indirectly reveals the target in a way that would not legitimately be available at prediction time.

## Project example

`occupancy_ratio` was used to generate the target using:

`target_available = 1 if occupancy_ratio < 0.25`

Therefore:

```text
occupancy_ratio
      |
      v
Target-generation rule
      |
      v
target_available
```

Including `occupancy_ratio` lets the model learn the synthetic generation rule instead of learning useful RF relationships.

---

# 21. LEAKAGE EXPERIMENT

Including leakage-related information produced:

**F1 ≈ 0.994**

This looked excellent but was artificial.

Therefore the production feature set excludes:

- `occupancy_ratio`
- `interference_score`

### Strong viva answer

> “We identified target leakage because occupancy_ratio was directly used to generate target_available. Including it gave an artificially high F1 of about 0.994. We therefore removed occupancy_ratio and interference_score from the production feature set.”

---

# 22. WHY SYNTHETIC DATA?

The current dataset is synthetic.

Reason:

- The current project does not have a physical SDR measurement system supplying empirical RF observations.
- Synthetic data allows development and demonstration of the full pipeline.

### Important limitation

The model learns:

**Synthetic relationships**

not:

**Real-world RF propagation behavior**

### Future improvement

Collect real measurements using SDR hardware and build a real occupancy-labelled dataset.

---

# 23. PHASE 6 — ML TRAINING PIPELINE

Complete training flow:

```text
Dataset
   |
   v
Dataset Analysis
   |
   v
Remove Leakage Features
   |
   v
GroupShuffleSplit
   |
   +----> 60% Training
   +----> 20% Validation
   +----> 20% Test
   |
   v
Preprocessing
   |
   +----> Numeric -> StandardScaler
   |
   +----> Categorical -> OneHotEncoder
   |
   v
Random Forest
   |
   v
RandomizedSearchCV
   |
   v
Best Configuration
   |
   v
Final Evaluation
   |
   v
wire_watcher_model.pkl
```

---

# 24. TRAIN / VALIDATION / TEST

## Training

Used to learn model patterns.

## Validation

Used during development/tuning.

## Test

Held out for final evaluation.

### Easy analogy

- Training = learning
- Validation = practice/checking
- Test = final exam

---

# 25. WHY GROUPSHUFFLESPLIT?

The project uses **city as the grouping variable**.

The intention is to prevent records from the same city from being spread across training/evaluation groups.

This tests:

> **Geographic generalization**

### Viva answer

> “We use city-based grouping so the model is evaluated on geographic groups that were not used in training, reducing the chance that it simply memorizes city-specific patterns.”

---

# 26. PREPROCESSING

## Numerical features

Use:

**StandardScaler**

Conceptually:

`z = (x - mean) / standard deviation`

Purpose:

> Standardize numerical features.

## Categorical features

Use:

**OneHotEncoder**

Example:

`City = Mumbai / Delhi / Bengaluru`

becomes indicator columns such as:

- City_Mumbai
- City_Delhi
- City_Bengaluru

## ColumnTransformer

Applies the correct preprocessing to each feature group.

---

# 27. IMPORTANT SCALING POINT

Do NOT say:

> “Random Forest requires StandardScaler.”

Better:

> “Our preprocessing pipeline standardizes numerical features using StandardScaler. Random Forest itself is not inherently dependent on feature scaling.”

---

# 28. HYPERPARAMETER TUNING

A **hyperparameter** is a configuration chosen for the model rather than learned directly as a model parameter.

Examples:

- Number of trees
- Maximum tree depth
- Class weighting

The project uses:

**RandomizedSearchCV**

to search hyperparameter combinations using cross-validation.

---

# 29. CLASS IMBALANCE

Dataset:

- Occupied = 96.21%
- Available = 3.79%

Problem:

A model could get high accuracy by mostly predicting Occupied.

Solution used in training:

`class_weight = "balanced"`

This gives greater importance to the minority class during training.

Important:

> Balanced weighting does NOT create new samples. It changes class weighting during learning.

---

# 30. PHASE 7 — RANDOM FOREST

## Decision Tree

A decision tree makes sequential feature-based decisions.

Example:

```text
SNR > 15?
  |
  +-- Yes -> Signal Power > -80?
  |
  +-- No  -> ...
```

Each question is a **split**.

---

# 31. GINI IMPURITY

Gini impurity measures how mixed the classes are in a group.

Example:

- 100% Occupied → pure
- 50% Occupied / 50% Available → highly mixed

Trees prefer useful splits that separate the classes.

---

# 32. RANDOM FOREST

Random Forest = **ensemble of decision trees**.

Instead of one tree:

```text
Data -> Tree -> Prediction
```

we use many:

```text
             Input
               |
     +---------+---------+
     |         |         |
   Tree 1    Tree 2    Tree 3
     |         |         |
     +---- ... +---------+
               |
        Combined output
```

The randomness comes from sampling data/features so trees learn somewhat different structures.

---

# 33. YOUR RANDOM FOREST CONFIGURATION

Documented selected configuration:

- `n_estimators = 200`
- `max_depth = 20`
- `class_weight = balanced`

## Meaning

### 200 trees

The forest contains 200 decision trees.

### max_depth = 20

Limits the maximum depth/complexity of each tree.

### balanced

Addresses the strong class imbalance.

---

# 34. WHY RANDOM FOREST?

Good answer:

> “We selected Random Forest because this is a binary classification problem with numerical and categorical features, and Random Forest can model nonlinear relationships and interactions between features. It also provides feature-importance information.”

Do not claim it is universally the best algorithm.

---

# 35. FEATURE IMPORTANCE

Reported Gini importance:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `snr_db` | 0.363 |
| 2 | `signal_power_dbm` | 0.223 |
| 3 | `noise_floor_dbm` | 0.073 |
| 4 | `hour_of_day` | 0.072 |
| 5 | `day_of_week` | 0.053 |

## Interpretation

SNR is the strongest reported feature, followed by signal power.

This has an intuitive ECE interpretation:

`Signal + Noise -> SNR -> Signal detectability`

### Important

Do NOT say:

> “SNR determines 36.3% of the prediction.”

Say:

> “SNR has the highest reported Gini importance.”

---

# 36. PERMUTATION IMPORTANCE

Permutation importance works conceptually by:

1. Shuffle one feature.
2. Re-evaluate performance.
3. See how much performance changes.

The project reports that SNR and signal power dominate, while geographic features have near-zero permutation importance.

This means location is currently not a major contributor in the synthetic model.

---

# 37. PHASE 8 — MODEL EVALUATION

The final model is evaluated on a held-out test set of approximately **8,400 samples**.

Documented results:

| Metric | Result |
|---|---:|
| Accuracy | **92.76%** |
| Balanced Accuracy | **73.16%** |
| Precision | **33.23%** |
| Recall | **51.50%** |
| F1 | **0.4039** |
| ROC-AUC | **0.9061** |
| PR-AUC | **0.3381** |
| Brier Score | **0.0431** |
| False Positives | **414** |
| False Negatives | **194** |

---

# 38. WHY ACCURACY IS MISLEADING

Because:

**96.21% = Occupied**

A dummy model that always predicts Occupied could achieve about 96.21% accuracy.

Therefore:

> Accuracy alone is not enough.

The project uses F1 as the primary model-selection metric because of the strong class imbalance.

---

# 39. CONFUSION MATRIX

For Class 1 = Available:

### True Positive (TP)

Actually Available + predicted Available.

### False Positive (FP)

Actually Occupied + predicted Available.

### False Negative (FN)

Actually Available + predicted Occupied.

### True Negative (TN)

Actually Occupied + predicted Occupied.

---

# 40. PRECISION

Formula:

`Precision = TP / (TP + FP)`

Question:

> “When the model says Available, how often is it correct?”

Project result:

**33.23%**

Interpretation:

Only about one-third of positive/Available predictions were actually Available in the documented test result.

---

# 41. RECALL

Formula:

`Recall = TP / (TP + FN)`

Question:

> “Out of all actually Available cases, how many did we find?”

Project result:

**51.50%**

Interpretation:

The model identifies about half of the genuinely Available cases.

---

# 42. F1 SCORE ⭐⭐⭐

Formula:

`F1 = 2 × Precision × Recall / (Precision + Recall)`

F1 balances precision and recall.

Project:

**F1 = 0.4039**

### Why low?

Because:

- Precision = 33.23%
- Recall = 51.50%

The minority-class performance is limited.

### Best explanation

> “Our F1 is only 0.4039 because the Available class is rare and the model currently has limited precision and recall for that class. This is a prototype result rather than production-level performance.”

---

# 43. BALANCED ACCURACY

Balanced Accuracy gives equal consideration to the two classes.

Documented:

**73.16%**

Reported:

- Sensitivity/Recall = 51.5%
- Specificity = 94.8%

Conceptually:

`Balanced Accuracy = (Sensitivity + Specificity) / 2`

This is more informative than ordinary accuracy for imbalanced data.

---

# 44. SPECIFICITY

Specificity asks:

> “Of all actually Occupied cases, how many did we correctly identify as Occupied?”

Documented:

**94.8%**

So the model is much stronger at recognizing the majority Occupied class than identifying Available cases.

---

# 45. ROC-AUC

Project:

**ROC-AUC = 0.9061**

ROC-AUC evaluates how well the model ranks positive/Available cases above negative/Occupied cases across thresholds.

### Important distinction

ROC-AUC is about:

**Ranking capability**

F1 is about:

**Classification performance at a chosen threshold**

Therefore high ROC-AUC and lower F1 are not automatically contradictory.

---

# 46. PR-AUC

Project:

**PR-AUC = 0.3381**

PR-AUC summarizes the precision-recall relationship.

It is particularly useful when the positive class is rare.

Here:

**Available = 3.79%**

So PR-AUC gives another view of minority-class performance.

---

# 47. BRIER SCORE

Project:

**Brier Score = 0.0431**

Brier score measures probability prediction error.

Lower is better.

It asks:

> “How close are the predicted probabilities to the actual outcomes?”

Important limitation:

The current Random Forest probabilities are not additionally calibrated using Platt scaling or isotonic regression.

---

# 48. FALSE POSITIVE VS FALSE NEGATIVE

## False Positive

Model:

**Available**

Reality:

**Occupied**

Project:

**414**

Potential concern:

> Incorrectly treating an occupied band as available.

## False Negative

Model:

**Occupied**

Reality:

**Available**

Project:

**194**

Meaning:

> The system misses an available opportunity.

The acceptable balance depends on the actual deployment requirements.

---

# 49. PHASE 9 — DECISION THRESHOLD

The model produces an availability probability.

Example:

`P(Available) = 0.72`

A threshold converts probability into a class.

The project uses:

**Threshold = 0.40**

So:

```text
P(Available) >= 0.40
          |
          v
       Available

P(Available) < 0.40
          |
          v
       Occupied
```

---

# 50. WHY NOT 0.50?

0.50 is a common default, but it is not always optimal.

Your project searched thresholds from:

**0.10 to 0.90**

and selected the threshold that maximized F1 on the held-out validation data.

Selected:

**0.40**

### Key idea

Changing the threshold changes:

- Precision
- Recall
- F1
- Number of positive predictions

---

# 51. THRESHOLD TRADE-OFF

Lower threshold:

```text
More Available predictions
        |
        +--> Recall may increase
        +--> False positives may increase
        +--> Precision may decrease
```

Higher threshold:

```text
Fewer Available predictions
        |
        +--> Precision may increase
        +--> Recall may decrease
```

The project selected **0.40** because it maximized the chosen F1 objective on validation data.

---

# 52. PHASE 10 — MODEL DEPLOYMENT

The trained model is serialized as:

`wire_watcher_model.pkl`

The bundle contains the production preprocessing/model and relevant metadata.

The Flask backend loads the model when the service starts.

---

# 53. RUNTIME PREDICTION FLOW

```text
User enters RF/context values
          |
          v
Frontend
          |
          v
POST /api/predict
          |
          v
Input validation
          |
          v
Preprocessing
          |
          v
Random Forest
          |
          v
P(Available)
          |
          v
Threshold = 0.40
          |
          v
Available / Occupied
          |
          v
Store prediction record
```

---

# 54. WHAT IS INFERENCE?

Inference means:

> Using an already-trained model to predict on new data.

Training:

`Dataset -> Model`

Inference:

`New input -> Existing model -> Prediction`

The model is NOT retrained every time the user clicks Predict.

---

# 55. FLASK API

The ML model is exposed through a Flask backend.

Documented endpoint:

`/api/predict`

The endpoint:

- Receives prediction input
- Validates it
- Runs the ML model
- Returns prediction/probability information
- Stores prediction records in the database

---

# 56. PHASE 11 — OOD / SAFETY CHECKS

OOD = **Out Of Distribution**

Meaning:

> Input is significantly different from what the model saw during training.

Example:

Training SNR range:

`~11–32 dB`

User enters:

`SNR = 100 dB`

That input is outside the learned operating range.

The model may still produce a prediction, but it should not automatically be trusted.

---

# 57. PROJECT OOD IDEA

The model bundle stores training statistics/range information.

Documented examples include percentile bounds for:

- Signal power
- SNR

The system can use these checks to identify unusual inputs.

### Safe answer

> “OOD checking helps identify inputs that are outside the distribution or operating range represented by the training data, so the system can avoid treating every model prediction as equally trustworthy.”

---

# 58. MODEL CONFIDENCE

Probability is not the same as certainty.

Example:

`P(Available) = 0.72`

means:

> The model assigns 0.72 probability to the Available class.

It does NOT mean:

> There is a 72% guarantee.

This is why probability calibration and OOD checks matter.

---

# 59. PHASE 12 — COMPLETE END-TO-END ARCHITECTURE

```text
                    USER
                      |
                      v
               React Frontend
                      |
        +-------------+-------------+
        |                           |
        v                           v
   RF Simulation              Prediction Input
        |                           |
        v                           v
     Complex I/Q               Flask API
        |                           |
        v                           v
       FFT                     Validation
        |                           |
        v                           v
       PSD                    Preprocessing
        |                           |
        +----> Spectrum             v
              Visualization    Random Forest
                                    |
                                    v
                             Probability
                                    |
                                    v
                              Threshold 0.40
                                    |
                                    v
                           Available / Occupied
                                    |
                                    v
                               Database
```

---

# 60. DSP VS ML

This distinction should be crystal clear.

## DSP

Input:

**Complex I/Q**

Operations:

- FFT
- fftshift
- PSD
- Noise floor
- Peak detection

Output:

**Spectrum information**

## ML

Input:

**Feature vector**

Examples:

- Frequency
- Signal power
- Noise floor
- SNR
- Time
- Location
- Service

Output:

**Availability probability/class**

### One-line answer

> “DSP characterizes the RF spectrum; ML uses selected features to estimate availability.”

---

# 61. PHASE 13 — DATASET VIVA QUESTIONS

## Q: What dataset did you use?

> `spectrum_occupancy_synthetic_33600.csv`

## Q: How many records?

> 33,600.

## Q: How many columns?

> 19.

## Q: Is it real?

> No, it is synthetic.

## Q: Any missing values?

> No, documented missing/null count is zero.

## Q: What is the target?

> `target_available`.

## Q: What does 0 mean?

> Occupied.

## Q: What does 1 mean?

> Available.

---

# 62. MORE DATASET QUESTIONS

## Q: Why is the dataset imbalanced?

> It contains 96.21% Occupied and 3.79% Available samples.

## Q: Why is imbalance a problem?

> A model can achieve high accuracy by mostly predicting the majority class.

## Q: How did you handle it?

> `class_weight="balanced"` and F1-based model selection.

## Q: What is target leakage?

> When an input contains information that directly or indirectly reveals the target.

## Q: Did your dataset have leakage?

> Yes. `occupancy_ratio` was used to generate the target.

## Q: What did you do?

> Removed `occupancy_ratio` and `interference_score` from the production feature set.

---

# 63. ML VIVA QUESTIONS

## Q: What type of ML?

> Supervised binary classification.

## Q: What algorithm?

> Random Forest Classifier.

## Q: Why Random Forest?

> Handles nonlinear relationships/interactions, works with mixed feature types through preprocessing, and provides feature-importance information.

## Q: How many trees?

> 200.

## Q: Maximum depth?

> 20.

## Q: What is training?

> Learning patterns from labelled examples.

## Q: What is inference?

> Using the trained model to predict on new input.

## Q: What is a feature?

> An input variable used by the model.

## Q: What is a target?

> The value the model is trying to predict.

---

# 64. TRAINING QUESTIONS

## Q: Why split the dataset?

> To evaluate generalization on unseen data.

## Q: Split ratio?

> 60% training, 20% validation, 20% test.

## Q: Why group by city?

> To reduce geographic leakage and evaluate geographic generalization.

## Q: What is preprocessing?

> Preparing raw data for the model.

## Q: Why OneHotEncoder?

> Converts categorical variables into numerical indicator features.

## Q: Why StandardScaler?

> Standardizes numerical features in the preprocessing pipeline.

## Q: What is RandomizedSearchCV?

> A method for searching hyperparameter combinations using cross-validation.

---

# 65. EVALUATION QUESTIONS

## Q: Why not just report accuracy?

> Because the dataset is highly imbalanced.

## Q: Accuracy?

> 92.76%.

## Q: Precision?

> 33.23%.

## Q: Recall?

> 51.50%.

## Q: F1?

> 0.4039.

## Q: ROC-AUC?

> 0.9061.

## Q: PR-AUC?

> 0.3381.

## Q: Brier?

> 0.0431.

## Q: FP?

> 414.

## Q: FN?

> 194.

---

# 66. THE MOST IMPORTANT REVIEW QUESTION

## “Your accuracy is 92.76%. Is your model good?”

Best answer:

> “Accuracy is high, but because 96.21% of our data is Occupied, accuracy alone is misleading. For the minority Available class, precision is 33.23%, recall is 51.50% and F1 is 0.4039. At the same time, ROC-AUC is 0.9061, showing strong ranking capability. So I would describe the current model as a functional prototype with useful ranking performance but limited minority-class classification performance. Real-world data and further improvement are required before production deployment.”

---

# 67. THE SECOND MOST IMPORTANT REVIEW QUESTION

## “Why did you remove occupancy_ratio?”

Best answer:

> “We identified target leakage. The synthetic target was generated directly from occupancy_ratio < 0.25. Including occupancy_ratio gave an artificially high F1 of approximately 0.994 because the model could learn the target-generation rule. We therefore removed occupancy_ratio and interference_score from the production feature set.”

---

# 68. THE THIRD MOST IMPORTANT REVIEW QUESTION

## “Why use synthetic data?”

Best answer:

> “The current system does not have a physical SDR measurement pipeline, so we used synthetic data to validate the complete software and ML pipeline. We clearly treat the resulting model as a prototype. Real SDR measurements and empirical occupancy labels would be needed for real-world validation.”

---

# 69. THE FOURTH MOST IMPORTANT REVIEW QUESTION

## “Explain your project from input to output.”

Best answer:

> “The system can generate simulated complex I/Q RF data and process it through FFT, PSD, noise-floor estimation and peak detection for spectrum visualization. Separately, the ML pipeline takes 11 RF, temporal, geographic and service-related features, preprocesses them, and feeds them into a Random Forest classifier. The model produces an availability probability. A threshold of 0.40 converts that probability into Available or Occupied, and the backend can store the prediction.”

---

# 70. THE FIFTH MOST IMPORTANT REVIEW QUESTION

## “What would you improve?”

Strong answers:

1. Collect real RF/SDR data.
2. Build real occupancy labels.
3. Improve multi-carrier peak detection.
4. Improve minority-class precision/recall.
5. Calibrate predicted probabilities.
6. Validate across real geographic environments.
7. Retrain using real measurements.
8. Continue evaluating OOD behavior.

---

# 71. WHAT NOT TO SAY

Avoid these statements:

❌ “Our model is 92.76% accurate, so it is excellent.”

❌ “The model understands real RF signals.”

❌ “SNR alone determines the prediction.”

❌ “Random Forest requires StandardScaler.”

❌ “The model detects all active carriers.”

❌ “The 99.4% F1 proves the model is excellent.”

❌ “Our synthetic dataset represents real-world RF perfectly.”

Instead, be precise and acknowledge limitations.

---

# 72. EMERGENCY 60-SECOND EXPLANATION

If the reviewer suddenly says:

> “Explain your project.”

Say:

> “Wire Watcher is a spectrum-monitoring prototype combining DSP and machine learning. On the signal-processing side, we generate simulated complex I/Q data and use a 2048-point FFT, fftshift, PSD calculation, median-based noise-floor estimation and peak detection to visualize spectrum activity. On the ML side, we formulate spectrum availability as a supervised binary classification problem. Our dataset contains 33,600 synthetic records. The production model uses 11 features covering frequency, bandwidth, signal power, noise floor, SNR, time, state, city and service type. We identified target leakage in occupancy_ratio and removed it along with interference_score. We use a city-grouped 60/20/20 split, StandardScaler for numerical features, OneHotEncoder for categorical features and a balanced Random Forest with 200 trees and maximum depth 20. The final model produces an availability probability, and a validation-optimized threshold of 0.40 converts it into Available or Occupied. On the held-out test set, accuracy is 92.76%, but because of class imbalance the more important metrics are 33.23% precision, 51.50% recall and F1 0.4039, while ROC-AUC is 0.9061. We consider it a prototype because the data is synthetic and real SDR data is needed for real-world validation.”

---

# 73. FINAL MEMORY MAP

```text
PROJECT
  |
  +-- RF / DSP
  |     |
  |     +-- I/Q
  |     +-- Sampling
  |     +-- FFT
  |     +-- PSD
  |     +-- Noise Floor
  |     +-- Peak Detection
  |
  +-- DATASET
  |     |
  |     +-- 33,600 rows
  |     +-- 19 columns
  |     +-- Synthetic
  |     +-- 96.21% Occupied
  |     +-- 3.79% Available
  |     +-- Leakage analysis
  |
  +-- ML
  |     |
  |     +-- Supervised
  |     +-- Binary Classification
  |     +-- 11 features
  |     +-- Random Forest
  |     +-- 200 trees
  |     +-- Depth 20
  |     +-- Balanced weights
  |
  +-- TRAINING
  |     |
  |     +-- 60/20/20
  |     +-- Group by City
  |     +-- StandardScaler
  |     +-- OneHotEncoder
  |     +-- RandomizedSearchCV
  |
  +-- EVALUATION
  |     |
  |     +-- Accuracy 92.76%
  |     +-- Precision 33.23%
  |     +-- Recall 51.50%
  |     +-- F1 0.4039
  |     +-- ROC-AUC 0.9061
  |     +-- PR-AUC 0.3381
  |     +-- Brier 0.0431
  |
  +-- DECISION
  |     |
  |     +-- Probability
  |     +-- Threshold 0.40
  |     +-- Available / Occupied
  |
  +-- DEPLOYMENT
        |
        +-- Flask
        +-- /api/predict
        +-- .pkl model bundle
        +-- Database
        +-- Frontend
```

---

# 74. LAST-MINUTE REVISION — MEMORIZE THESE

If you have very little time, memorize these exact facts:

**Dataset:** 33,600 synthetic records, 19 columns.

**Target:** `target_available`  
0 = Occupied, 1 = Available.

**Production features:** 11.

**Class distribution:**  
96.21% Occupied / 3.79% Available.

**Leakage:**  
`occupancy_ratio` directly contributed to target generation → removed.  
`interference_score` also excluded.

**Split:**  
60/20/20, grouped by city.

**Preprocessing:**  
StandardScaler + OneHotEncoder + ColumnTransformer.

**Algorithm:**  
Random Forest Classifier.

**Model:**  
200 trees, max depth 20, balanced class weights.

**Primary metric:**  
F1 because of severe class imbalance.

**Test metrics:**  
Accuracy 92.76%  
Balanced Accuracy 73.16%  
Precision 33.23%  
Recall 51.50%  
F1 0.4039  
ROC-AUC 0.9061  
PR-AUC 0.3381  
Brier 0.0431.

**Errors:**  
414 FP, 194 FN.

**Threshold:**  
0.40, selected by validation F1 search.

**Model file:**  
`wire_watcher_model.pkl`

**Biggest limitation:**  
Synthetic data; no real SDR data yet.

**Biggest future improvement:**  
Real SDR measurements + real occupancy labels + retraining/validation.

---

# 75. REMEMBER THE STORY, NOT JUST THE NUMBERS

The best review performance will come from explaining the logic:

```text
Why spectrum monitoring?
        ↓
Need to understand RF activity
        ↓
DSP analyzes the spectrum
        ↓
Need to estimate availability
        ↓
Use supervised ML
        ↓
Dataset has severe imbalance
        ↓
Analyze and remove leakage
        ↓
Train Random Forest
        ↓
Evaluate using appropriate metrics
        ↓
Optimize threshold
        ↓
Deploy through Flask
        ↓
Prototype limitations identified
        ↓
Real RF data is the next major step
```

> **Your goal tomorrow is not to pretend the model is perfect. Your goal is to demonstrate that you understand what you built, why you made each major decision, what the results mean, and what the current limitations are.**

# Wire Watcher

Wire Watcher is a **Spectrum Availability Estimation Prototype** built for an ECE Final Year Project. It combines physical RF (Radio Frequency) signal-processing concepts, Machine Learning, robust REST API services, and an interactive React-based dashboard. 

The primary goal of this project is to simulate realistic RF environments, extract their features (via FFT/PSD analysis), and intelligently predict whether a specific frequency band is available or occupied using a trained Random Forest classifier.

---

## Key Features

1. **Simulated RF Environment & Signal Processing**
   - Generates realistic IQ time-domain data consisting of thermal white noise and CW (Continuous Wave) carriers.
   - Computes the Fast Fourier Transform (FFT) and Power Spectral Density (PSD).
   - Robustly estimates the RF Noise Floor and detects signal peaks.

2. **Machine Learning Inference**
   - Incorporates a pre-trained **Random Forest Model** to predict spectrum availability based on physical parameters (SNR, Signal Power, Bandwidth).
   - Dynamically calculates prediction **Confidence Levels** (High, Medium, Low) and triggers Out-Of-Distribution (OOD) warnings for anomalous inputs.

3. **Interactive Lovable Dashboard (React)**
   - **Prediction Engine**: A UI to configure RF properties and query the model.
   - **Live Spectrum Chart**: Visualizes the PSD, Noise Floor, and carrier signals dynamically.
   - **Monitoring Dashboard**: Displays live aggregate KPI statistics (e.g., Total monitored records, OOD counts) directly pulled from MySQL.
   - **History Table**: Logs historical RF predictions including computed SNR, Power, and Data Source.

4. **Modular Architecture**
   - A clean separation of concerns: `backend` (Flask + MySQL), `frontend` (React + Vite), and `ml` (Data Science, Evaluation, and Simulation scripts).

---

## Project Structure

```text
wirewatcher/
├── backend/                  # Flask REST API and Database Services
│   ├── api/                  # Routes and Pydantic validation schemas
│   ├── database/             # SQLAlchemy ORM models and connection pool
│   ├── rf/                   # RF Signal Processing Module (FFT, PSD, Noise, Peaks)
│   ├── services/             # ML prediction execution and database logging
│   ├── app.py                # Main Flask entry point
│   └── config.py             # Environment and paths configuration
│
├── frontend/                 # Lovable/React Web Interface
│   ├── src/                  
│   │   ├── components/       # Reusable UI components and charts
│   │   ├── hooks/            # React hooks for API data fetching
│   │   ├── routes/           # Application pages (Predict, History, Monitoring)
│   │   ├── services/         # API integration layer
│   │   └── types/            # TypeScript definitions for the API
│   └── package.json          # Node dependencies
│
├── ml/                       # Machine Learning & Signal Processing Core
│   ├── artifacts/            # Pickled Random Forest model and JSON metadata
│   ├── data/                 # Raw dataset CSVs
│   ├── evaluation/           # Scripts to assess model metrics and diagnostics
│   ├── inference/            # Prediction wrappers for the Flask backend
│   ├── simulation/           # Scenarios and ECE RF Signal visualizations
│   └── training/             # Model training pipelines
│
├── tests/                    
│   └── backend/              # Pytest suite for API endpoints and physical bounds
│
├── .env                      # Database configuration
└── README.md                 # Project Documentation
```

---

## Prerequisites & Installation

### 1. Database Configuration
Ensure a local MySQL instance (e.g., via XAMPP) is running.
Configure your database credentials inside the `.env` file at the root of the project. Make sure the database exists in your MySQL instance.

### 2. Backend Setup (Flask API)
The backend runs the ML inference and provides data to the frontend.
```bash
# Navigate to the backend directory
cd backend

# (Optional) Activate a virtual environment
# python -m venv venv
# source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the Flask API
python app.py
```
The server will start on `http://localhost:5000` and automatically load the pre-trained ML model artifact from `ml/artifacts`.

### 3. Frontend Setup (React Dashboard)
The interactive dashboard uses Node.js.
```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
Open your browser and navigate to the local host URL provided by Vite (usually `http://localhost:8080`).

---

## API Endpoints

The Flask backend exposes the following primary REST endpoints:

* **`GET /api/health`**: Returns API, Database, and ML model loading statuses.
* **`GET /api/model-info`**: Returns current metadata of the loaded ML model (Training samples, important features) and Live Database KPI Statistics (total predictions, OOD anomalies).
* **`GET /api/predictions`**: Fetches the 100 most recent predictions stored historically in the MySQL database.
* **`POST /api/spectrum/analyze`**: Simulates an RF environment, performs the FFT to calculate PSD, extracts noise/signal peaks, and returns the raw spectrum graph data for visualization.
* **`POST /api/predict`**: Accepts an RF JSON payload (either extracted from the RF Simulation or provided manually), queries the ML model, maps confidence thresholds, and persists the transaction to MySQL.

### RF Interference/Jamming Detector (separate model, separate task)

A second, independent Random Forest model trained on **real, experimentally
measured** RF spectral-scan captures (`release_artifacts`), classifying
**benign vs. malicious (jamming) RF activity** — distinct from spectrum
availability. See [`ml/jamming/README.md`](ml/jamming/README.md) for full
dataset provenance, methodology, and the controlled-vs-confounded evaluation
writeup.

* **`GET /api/jamming/model-info`**: Dataset provenance, controlled validation metrics (F1=0.853, ROC-AUC=0.987, PR-AUC=0.915), and limitations.
* **`GET /api/jamming/samples`**: Held-out demo samples with true labels.
* **`POST /api/jamming/predict`**: Runs the detector on a sample or custom feature vector.

Frontend: `/jamming` page (separate from Spectrum Simulation/Prediction).

---

## Development & Testing

### Running the Test Suite
To verify the application behaves correctly against physical constraints and API contracts, a full Pytest suite is provided.
```bash
cd tests/backend
python -m pytest test_wirewatcher.py
```

### Running the Physical ML Simulation (CLI)
To execute the physical simulated tests or extract a Power Spectral Density (PSD) chart directly via Python CLI without the frontend:
```bash
cd ml/simulation
python rf_signal_processor.py
python scenario_engine.py
```

---

> **Note on Data Provenance:** The underlying Random Forest model was trained on synthetic ECE spectrum data. While it accurately maps physical laws (like SNR decay) and runs true statistical inference, it is explicitly branded as a **simulated prototype** and does not fabricate fake real-world measurements. All visualizations clearly denote the data source as `SIMULATED` or `SYNTHETIC`.

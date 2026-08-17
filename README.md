# Wire Watcher

A spectrum availability estimation prototype combining RF signal-processing concepts, machine learning, Flask API services, MySQL storage, and an interactive React web dashboard.

## Project Structure

The codebase is organized cleanly into distinct modules based on responsibility.

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
│   │   ├── components/       # Reusable UI components
│   │   ├── hooks/            # React hooks for API data fetching
│   │   ├── routes/           # Application pages
│   │   ├── services/         # API integration layer
│   │   └── types/            # TypeScript definitions for the API
│   └── package.json          # Node dependencies
│
├── ml/                       # Machine Learning & Signal Processing 
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
└── README.md
```

## Running the Application

### 1. Database
Ensure a local MySQL instance (e.g., via XAMPP) is running and the database specified in `.env` exists. 

### 2. Backend (Flask)
Start the REST API server:
```bash
cd backend
python app.py
```
The server will run on `http://localhost:5000` and automatically load the ML model artifact from `ml/artifacts`.

### 3. Frontend (React)
Start the interactive Lovable dashboard:
```bash
cd frontend
npm run dev
```

### 4. Running the Tests
To verify the application behaves as expected, run the Pytest suite:
```bash
cd tests/backend
python -m pytest test_wirewatcher.py
```

### 5. Running the ML Simulation
To execute the physical simulated tests or extract a Power Spectral Density (PSD) chart:
```bash
cd ml/simulation
python rf_signal_processor.py
python scenario_engine.py
```

## API Endpoints

* **`GET /api/health`**: Returns API, Database, and ML model load status.
* **`GET /api/model-info`**: Returns current metadata of the loaded ML model (Training samples, parameters) and live KPI stats.
* **`GET /api/predictions`**: Fetches the 100 most recent predictions stored in the database.
* **`POST /api/predict`**: Accepts an RF JSON payload, performs physical bounds validation, queries the ML model, and persists the transaction to MySQL.
* **`POST /api/spectrum/analyze`**: Simulates an RF environment, performs FFT to calculate PSD, extracts noise/signal peaks, and returns the raw spectrum data.

> **Note:** The underlying ML model was trained on synthetic data. This system acts as a prototype for full end-to-end evaluation.

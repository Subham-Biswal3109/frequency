import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Base directory references
BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "ml"
MODEL_DIR = ML_DIR / "artifacts"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
MODEL_PATH = MODEL_DIR / "wire_watcher_model.pkl"

# RF Interference/Jamming Detector — a SEPARATE, additive model trained on
# real experimental RF measurements (release_artifacts). Does not replace
# or interact with the spectrum-availability model paths above.
JAMMING_MODEL_PATH = MODEL_DIR / "jamming_detector_model.pkl"
JAMMING_METADATA_PATH = MODEL_DIR / "jamming_detector_metadata.json"
JAMMING_TEST_SAMPLES_PATH = MODEL_DIR / "jamming_detector_test_samples.json"

WEB_DIR = BASE_DIR / "web" # Fallback if serving static UI

# Database configuration
MYSQL_USER = os.getenv("DB_USER", "root")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD", "")
MYSQL_HOST = os.getenv("DB_HOST", "localhost")
MYSQL_PORT = os.getenv("DB_PORT", "3306")
MYSQL_DATABASE = os.getenv("DB_NAME", "wire_watcher")

SQLALCHEMY_DATABASE_URI = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)

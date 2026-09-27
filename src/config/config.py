from pathlib import Path

# ======================
# PROJECT ROOT
# ======================
ROOT_DIR = Path(__file__).resolve().parents[2]

# ======================
# DATA PATHS
# ======================
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "Dataset.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# ======================
# REPRODUCIBILITY
# ======================
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ======================
# MODEL VERSIONS
# ======================
RATING_MODEL_VERSION = "v1.0"
CUISINE_MODEL_VERSION = "v1.0"
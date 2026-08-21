from pathlib import Path

# ==============================
# Project Root Directory
# ==============================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==============================
# Data Directories
# ==============================

RAW_DATA = PROJECT_ROOT / "data" / "raw"

TRAIN_DIR = RAW_DATA / "train"
VALIDATION_DIR = RAW_DATA / "validation"
TEST_DIR = RAW_DATA / "test"

CLEANED_DATA = PROJECT_ROOT / "data" / "cleaned"
FEATURE_DATA = PROJECT_ROOT / "data" / "features"

# ==============================
# Output Directories
# ==============================

PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# ==============================
# Create folders if they don't exist
# ==============================

for folder in [
    CLEANED_DATA,
    FEATURE_DATA,
    PLOTS_DIR,
    REPORTS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)
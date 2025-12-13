from pathlib import Path

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Variables
MAX_DATE = "2024-01-31"
MIN_DATE = "2024-01-01"

# ot useful features

COLUMNS_TO_DROP = [
    "is_active",
    "marketing_consent",
    "first_booking",
    "existing_customer",
    "last_seen",
    "domain",
    "country",
    "visited_learn_more_before_booking",
    "visited_faq",
]

# identifiers / leakage
ID_COLUMNS = [
    "lead_id",
    "customer_code",
    "date_part",
]

ALL_COLUMNS_TO_DROP = ID_COLUMNS + COLUMNS_TO_DROP

COLUMNS_TO_CLEAN = [
    "lead_indicator",
    "lead_id",
    "customer_code",
]

COLUMNS_REQUIRED = [
    "lead_indicator",
    "lead_id",
]

COLUMNS_TO_OBJECT = [
    "lead_id",
    "lead_indicator",
    "customer_group",
    "onboarding",
    "source",
    "customer_code",
]

CAT_COLUMNS = ["customer_group", 
               "onboarding", 
               "bin_source", 
               "source",
]
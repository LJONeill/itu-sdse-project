# Imports

from pathlib import Path
import os
import datetime

# Make the folders for the docker container storage within dagger pipeline
os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Data
DATA_DIR = PROJ_ROOT / "data"

# Data/
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# External data
SCALER_PATH = EXTERNAL_DATA_DIR / "scaler.pkl"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Variables

RANDOM_STATE = 42
MAX_DATE = "2024-01-31"
MIN_DATE = "2024-01-01"

SOURCE_COLUMN = "source"
SOURCE_VALUE = "signup"

TARGET_COLUMN = "lead_indicator"
EXPERIMENT_NAME = datetime.datetime.now().strftime("%Y_%B_%d")

# Columns to drop
"""Columns that are not relevant for modelling and dropped earlier
in the notebook but never added back or used in downstream tasks"""

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
    "lead_id",
    "customer_code",
    "date_part"
]

# Columns for data validations

VALIDATION_COLUMNS = [
    "lead_indicator",
    "lead_id",
    "customer_code",
]

COLUMNS_TO_OBJECT = [
    "lead_indicator",
    "customer_group",
    "onboarding",
    "source",
]


CAT_COLUMNS = [
    "customer_group", 
    "onboarding", 
    "bin_source", 
    "source",
]


SOURCE_BIN_VALUES = ["li", "organic", "signup", "fb"]

SOURCE_BIN_MAPPING = {
    "li": "socials",
    "fb": "socials",
    "organic": "group1",
    "signup": "group1",
}

BIN_SOURCE_COLUMN = "bin_source"
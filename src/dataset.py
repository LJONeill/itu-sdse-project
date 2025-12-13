# Imports

# Standard libraries
from pathlib import Path
import datetime
import json

# Third-party libraries
import pandas as pd
import mlflow
import typer
from loguru import logger

from config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    MIN_DATE,
    MAX_DATE,
)

app = typer.Typer()

# Paths

INPUT_PATH: Path = RAW_DATA_DIR / "raw_data.csv" #constant should be capitalized
CLEANED_DATA_PATH: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"
DATE_LIMITS_PATH: Path = INTERIM_DATA_DIR / "date_limits.json"

# This is a constant?

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

# Functions

# Load data

def load_data(path: Path) -> pd.DataFrame:
    """Load data from CSV."""
    return pd.read_csv(path)

# Date limits

def define_dates(min_date, max_date):
    """Define the date limits in datetime format.

    Checks whether min and max dates are provided. If not, max_date
    defaults to today's date and min_date defaults to 2024-01-01.
    """
    if max_date is None:
        max_date = pd.to_datetime(
            datetime.datetime.now().date()
        ).date()
    else:
        max_date = pd.to_datetime(max_date).date()

    if min_date is not None:
        min_date = pd.to_datetime(min_date).date()
    else:
        min_date = pd.to_datetime("2024-01-01").date()

    return min_date, max_date

# Limit data 

def filter_data_by_date (data, min_date, max_date):
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]
    return data

# Storing dates

def store_date_limits(min_date, max_date):
    """Store min and max date limits as a JSON file."""

    date_limits = {
        "min_date": str(min_date),
        "max_date": str(max_date),
    }

    with open(date_limits_path, "w") as f:
        json.dump(date_limits, f)

# Drop columns from data

def drop_columns(data, columns_to_drop):
    """Drop specified columns from the dataset."""

    return data.drop(
        columns=columns_to_drop,
        axis=1,
    )

# Docker main script
@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    """Run the data processing pipeline."""

    logger.info("Processing started")

    # Load data
    data = load_data(input_path)

    # Define and apply date limits
    parsed_min_date, parsed_max_date = define_dates(min_date, max_date)
    data = filter_data_by_date(data, parsed_min_date, parsed_max_date)

    # Store actual date limits
    actual_min_date = data["date_part"].min()
    actual_max_date = data["date_part"].max()
    store_date_limits(actual_min_date, actual_max_date)

    # Drop columns and save cleaned data
    data = drop_columns(data, COLUMNS_TO_DROP)
    data.to_csv(output_path, index=False)

    # MLflow tracking
    with mlflow.start_run():
        mlflow.log_param("min_date", str(actual_min_date))
        mlflow.log_param("max_date", str(actual_max_date))
        mlflow.log_artifact(output_path)
        mlflow.log_artifact(date_limits_path)

    logger.success("Processing done")


if __name__ == "__main__":
    app()




# Imports

# Standard libraries
from pathlib import Path
import datetime
import json

# Third-party libraries
import pandas as pd
import numpy as np
import mlflow
import typer
from loguru import logger

from src.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    MIN_DATE,
    MAX_DATE,
    SOURCE_COLUMN,
    SOURCE_VALUE,
    COLUMNS_TO_DROP,
    VALIDATION_COLUMNS,
    TARGET_COLUMN, 
)

app = typer.Typer()

# Paths

INPUT_PATH: Path = RAW_DATA_DIR / "raw_data.csv"
CLEANED_DATA_PATH: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"
DATE_LIMITS_PATH: Path = INTERIM_DATA_DIR / "date_limits.json"
TARGET_DISTRIBUTION_PATH: Path = INTERIM_DATA_DIR / "target_distribution.json"


# Functions

# Load data
def load_data(path: Path) -> pd.DataFrame:
    """Load data from CSV."""
    return pd.read_csv(path)

# Date limits
def define_dates(min_date, max_date):
    """Define the date limits in datetime format."""
    if max_date is None:
        max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        max_date = pd.to_datetime(max_date).date()

    if min_date is None:
        min_date = pd.to_datetime("2024-01-01").date()
    else:
        min_date = pd.to_datetime(min_date).date()

    return min_date, max_date

# Limit data by date
def filter_data_by_date(
    data: pd.DataFrame,
    min_date,
    max_date,
) -> pd.DataFrame:
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    return data[
        (data["date_part"] >= min_date)
        & (data["date_part"] <= max_date)
    ]

def store_date_limits(
    min_date,
    max_date,
    output_path: Path = DATE_LIMITS_PATH,
) -> None:
    """
    Store the applied minimum and maximum date limits
    as a JSON artifact for reproducibility.
    """
    date_limits = {
        "min_date": str(min_date),
        "max_date": str(max_date),
    }
 
    with open(output_path, "w") as f:
        json.dump(date_limits, f, indent=2)
        
# Data cleaning

# Drop rows with missing values for given columns
def drop_missing_values(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Replace empty strings with NaN in the given columns
    and drop rows with missing values in those columns.
    """
    for col in columns:
        if col in data.columns:
            data[col] = data[col].replace("", np.nan)

    return data.dropna(subset=columns)

# Filter by source
def filter_by_source(
    data: pd.DataFrame,
    source_column: str = SOURCE_COLUMN,
    source_value: str = SOURCE_VALUE,
) -> pd.DataFrame:
    """Filter dataset by a specific source value."""
    return data[data[source_column] == source_value]

# Target distribution
def compute_target_distribution(
    data: pd.DataFrame,
    target_column: str,
) -> pd.Series:
    """
    Compute normalized target value distribution
    and store it as a JSON artifact.
    """
    target_dist = data[target_column].value_counts(normalize=True)

    with open(TARGET_DISTRIBUTION_PATH, "w") as f:
        json.dump(target_dist.to_dict(), f)

    return target_dist

# Drop columns
def drop_columns(
    data: pd.DataFrame,
    columns_to_drop: list[str],
) -> pd.DataFrame:
    """Drop specified columns from the dataset."""
    return data.drop(columns=columns_to_drop)


# Docker main script
@app.command()
def main(
    input_path: Path = INPUT_PATH,
    output_path: Path = CLEANED_DATA_PATH,
):
    """Run the data processing pipeline."""

    logger.info("Processing started")

    # 1. Load data
    data = load_data(input_path)

    # 2. Define date limits
    min_date, max_date = define_dates(
        MIN_DATE,
        MAX_DATE,
    )

    # 3. Filter data by date
    data = filter_data_by_date(
        data,
        min_date,
        max_date,
    )

    # 4. Store date limits
    store_date_limits(
        min_date,
        max_date,
    )

    # 5. Handle missing values (uses VALIDATION_COLUMNS)
    data = drop_missing_values(
        data,
        VALIDATION_COLUMNS,
    )

    # 6. Filter by source
    data = filter_by_source(
        data,
        SOURCE_COLUMN,
        SOURCE_VALUE,
    )

    # 7. Compute target distribution
    compute_target_distribution(
    data,
    target_column=TARGET_COLUMN,
    )

    # 8. Drop columns (after validation + usage)
    data = drop_columns(
        data,
        COLUMNS_TO_DROP,
    )

    # 9. Write cleaned data
    data.to_csv(output_path, index=False)

    # MLflow tracking
    with mlflow.start_run():
        mlflow.log_param("min_date", str(min_date))
        mlflow.log_param("max_date", str(max_date))
        mlflow.log_artifact(output_path)
        mlflow.log_artifact(DATE_LIMITS_PATH)
        mlflow.log_artifact(TARGET_DISTRIBUTION_PATH)

    logger.success("Processing done")


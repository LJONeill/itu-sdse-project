# Imports

# Standard libraries
from pathlib import Path
import datetime
import json

# Third-party libraries
import pandas as pd
import mlflow

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, 
from config import max_date, min_date #maybe change to capital? 

# Config + paths

INPUT_PATH: Path = RAW_DATA_DIR / "raw_data.csv" #constant should be capitalized
CLEANED_DATA_PATH: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"
DATE_LIMITS_PATH: Path = INTERIM_DATA_DIR / "date_limits.json"

# Functions

# Load data

def load_data(path: Path) -> pd.DataFrame:
    """Load data from CSV."""
    return pd.read_csv(path)

# Date limits

def define_dates(min_date, max_date):
    """Define the date limits in datetime format."""
    if not max_date: # I am not sure it makes sense to have this check as the first line in the function?
        max_date = pd.to_datetime(datetime.datetime.now().date()).date()
    else:
        max_date = pd.to_datetime(max_date).date()

    min_date = pd.to_datetime(min_date).date()
    return min_date, max_date

# Limit data 

def filter_data_by_date (data, min_date, max_date):
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]
    return data

# Storing dates

def store_date_limits(min_date, max_date):
    date_limits = {
        "min_date": str(min_date),
        "max_date": str(max_date)
    }
    
    with open(date_limits_path, "w") as f:
    json.dump(date_limits, f)

# Drop columns from data

def drop_columns(data, columns_to_drop):
    data = data.drop(
    [
     "is_active", 
     "marketing_consent", 
     "first_booking", 
     "existing_customer", 
     "last_seen", 
     "domain", 
     "country", 
     "visited_learn_more_before_booking", 
     "visited_faq"
     ],
    axis=1
    )
    return data


# Docker main script
@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()


#Mlflow

with mlflow.start_run():
    mlflow.log_param("min_date", str(min_date))
    mlflow.log_param("max_date", str(max_date))
    mlflow.log_artifact(cleaned_data_path)
    mlflow.log_artifact(date_limits_path)



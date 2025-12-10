# Imports

from pathlib import Path
import pandas as pd
import datetime
import json

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, max_date, min_date, EXTERNAL_DATA_DIR


# Config + paths

input_path: Path = RAW_DATA_DIR / "raw_data.csv"
cleaned_data_path: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"
date_limits_path: Path = INTERIM_DATA_DIR / "date_limits.json"

# Functions

# Load data

def load_data(path: Path) -> pd.DataFrame:
    """Load data from a CSV file located at the given path."""
    return pd.read_csv(path)

# Define the date limits in datetime format
if not max_date:
    max_date = pd.to_datetime(datetime.datetime.now().date()).date()
else:
    max_date = pd.to_datetime(max_date).date()

min_date = pd.to_datetime(min_date).date()

# Limit data by the above date bounds
data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

min_date = data["date_part"].min()
max_date = data["date_part"].max()
date_limits = {"min_date": str(min_date), "max_date": str(max_date)}

# Write out date limits
with open(date_limits_path, "w") as f:
    json.dump(date_limits, f)

# Drop columns from data
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

# Write out cleaned data
data.to_csv(cleaned_data_path, index=False)
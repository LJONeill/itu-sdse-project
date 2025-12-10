from pathlib import Path

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, max_date, min_date, EXTERNAL_DATA_DIR

import pandas as pd
import datetime
import json

# Paths
input_path: Path = RAW_DATA_DIR / "raw_data.csv"
cleaned_data_path: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"
date_limits_path: Path = INTERIM_DATA_DIR / "date_limits.json"

# Load data
data = pd.read_csv(input_path)

# Load data function version

def load_data(input_path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")
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
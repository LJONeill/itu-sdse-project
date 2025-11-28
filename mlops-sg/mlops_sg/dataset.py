from pathlib import Path

from mlops_sg.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, max_date, min_date, EXTERNAL_DATA_DIR


import pandas as pd
import datetime
import json
import numpy as np
import joblib


# ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
input_path: Path = RAW_DATA_DIR / "raw_data.csv",
training_data_path: Path = PROCESSED_DATA_DIR / "training_data.csv",
cleaned_data_path: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"
training_gold_path: Path = PROCESSED_DATA_DIR / "training_gold.csv",
date_limits_path: Path = INTERIM_DATA_DIR / "date_limits.json",
outlier_summary_path: Path = INTERIM_DATA_DIR / "outlier_summary.csv",
cat_missing_impute_path: Path = INTERIM_DATA_DIR / "cat_missing_impute.csv",
scaler_path: Path = EXTERNAL_DATA_DIR / "scaler.pkl",
column_drift_path: Path = INTERIM_DATA_DIR / "columns_drift.json"

data = pd.read_csv(input_path)

if not max_date:
    max_date = pd.to_datetime(datetime.datetime.now().date()).date()
else:
    max_date = pd.to_datetime(max_date).date()

min_date = pd.to_datetime(min_date).date()

# Time limit data
data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]

min_date = data["date_part"].min()
max_date = data["date_part"].max()
date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
with open(date_limits_path, "w") as f:
    json.dump(date_limits, f)

data = data.drop(
    ["is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"],
    axis=1
)

data = data.drop(
    ["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
    axis=1
)

data.to_csv(cleaned_data_path, index=False)
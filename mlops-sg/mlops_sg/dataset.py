from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from mlops_sg.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, max_date, min_date

import pandas as pd
import datetime
import json

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "raw_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "processed_data.csv",
    date_limits: Path = INTERIM_DATA_DIR / "date_limits.json"
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

data = pd.read_csv(RAW_DATA_DIR/"raw_data.csv")

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
with open(INTERIM_DATA_DIR/"date_limits.json", "w") as f:
    json.dump(date_limits, f)
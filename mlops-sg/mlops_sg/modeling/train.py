from pathlib import Path

from mlops_sg.config import MODELS_DIR, PROCESSED_DATA_DIR

import datetime
import mlflow
import pandas as pd

# ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
training_gold_path: Path = PROCESSED_DATA_DIR / "training_gold.csv",
features_path: Path = PROCESSED_DATA_DIR / "features.csv",
labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
model_path: Path = MODELS_DIR / "model.pkl",

# defined variables for use throughout
def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = training_gold_path
data_version = "00000"
experiment_name = current_date

mlflow.set_experiment(experiment_name)




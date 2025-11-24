from pathlib import Path

from mlops_sg.config import MODELS_DIR, PROCESSED_DATA_DIR

import datetime
import mlflow

# ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
training_gold_path: Path = PROCESSED_DATA_DIR / "training_gold.csv",
features_path: Path = PROCESSED_DATA_DIR / "features.csv",
labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
model_path: Path = MODELS_DIR / "model.pkl",

current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_gold_path = training_gold_path
data_version = "00000"
experiment_name = current_date

mlflow.set_experiment(experiment_name)


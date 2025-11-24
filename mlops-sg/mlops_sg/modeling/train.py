from pathlib import Path

from mlops_sg.config import MODELS_DIR, PROCESSED_DATA_DIR

# ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
features_path: Path = PROCESSED_DATA_DIR / "features.csv",
labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
model_path: Path = MODELS_DIR / "model.pkl",



# Imports

from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRFClassifier
from scipy.stats import uniform, randint
from dataset import load_data
from typing import Literal

import datetime
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json

from config import (
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
    TARGET_COLUMN, 
)

# Paths

# Input
INPUT_PATH: Path = PROCESSED_DATA_DIR / "training_gold.csv"

# Intermediate data
features_path: Path = INTERIM_DATA_DIR / "features.csv"
labels_path: Path = INTERIM_DATA_DIR/ "labels.csv"
xgboost_model_path: Path = INTERIM_DATA_DIR / "xgboost_model.pkl"
lr_model_path: Path = INTERIM_DATA_DIR / "lr_model.pkl"
column_list_path: Path = INTERIM_DATA_DIR / "columns_list.json"

# Output data
model_results_path: Path = PROCESSED_DATA_DIR /  "model_results.json"

# Classes and functions

# Helper functions

# Creating dummy columns
def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

# Load data
def load_data(path: Path) -> pd.DataFrame:
    """Load data from CSV."""
    return pd.read_csv(path)

# Data type split:

def data_type_split(
        data: pd.DataFrame,
        columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    for col in columns:
        if col in data.columns:
            cat_cols = columns

    other_vars = data.drop(cat_cols, axis=1)

    return cat_cols, other_vars

# Dummy column creation

def create_dummy_cols(df, col):
    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)
        
    return data = pd.concat([other_vars, cat_vars], axis=1)

# Float conversion

def float_conversion(data):
    for col in data:
        data[col] = data[col].astype("float64")
    return data

# Split data

def split_data(
        data: pd.DataFrame,
        columns: list[str]
) -> pd.DataFrame:
    
    y = data[columns]
    X = data.drop([columns], axis=1)

    return y, X

# Split data into train and test sets

def train_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y
    )

    return X_train, X_test, y_train, y_test

# Build class (not sure how this is used)
class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]
    
# Model training

# I am gonna leave the model here - but it should be in cofig.py
model = XGBRFClassifier(random_state=42)
params = {
    "learning_rate": uniform(1e-2, 3e-1),
    "min_split_loss": uniform(0, 10),
    "max_depth": randint(3, 10),
    "subsample": uniform(0, 1),
    "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
    "eval_metric": ["aucpr", "error"]
}

# Setup grid search

def setup_grid_search(model, params):
    model_grid = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10)
    model_grid.fit(X_train, y_train)
    return model_grid

# These will all be ran in main runner
best_model_xgboost_params = model_grid.best_params_
print("Best xgboost params")
pprint(best_model_xgboost_params)

y_pred_train = model_grid.predict(X_train)
y_pred_test = model_grid.predict(X_test)
print("Accuracy train", accuracy_score(y_pred_train, y_train ))
print("Accuracy test", accuracy_score(y_pred_test, y_test))

# also in main runner

conf_matrix = confusion_matrix(y_test, y_pred_test)
print("Test actual/predicted\n")
print(pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
print("Classification report\n")
print(classification_report(y_test, y_pred_test),'\n')

conf_matrix = confusion_matrix(y_train, y_pred_train)
print("Train actual/predicted\n")
print(pd.crosstab(y_train, y_pred_train, rownames=['Actual'], colnames=['Predicted'], margins=True),'\n')
print("Classification report\n")
print(classification_report(y_train, y_pred_train),'\n')

# Also in main runner
xgboost_model = model_grid.best_estimator_
xgboost_model_path = "./artifacts/lead_model_xgboost.json"
xgboost_model.save_model(xgboost_model_path)

model_results = {
    xgboost_model_path: classification_report(y_train, y_pred_train, output_dict=True)
}

# Docker main script
@app.command()
def main(
    input_path: Path = INPUT_PATH,
    output_path: Path = model_results_path,
):
    """Run the data processing pipeline."""


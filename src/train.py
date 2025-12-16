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
    CAT_COLUMNS
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

accuracy_scores_path: Path = INTERIM_DATA_DIR / "accuracy_scores.json"

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

    return cat_vars, other_vars

# Dummy column creation

def cast_to_category(df, col):
    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
    return cat_vars

def concat_vars(data, 
                other_vars, 
                cat_vars
) -> pd.DataFrame:
    data = pd.concat([other_vars, cat_vars], axis=1)
    return data 

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

def get_best_model_params(model_grid):
    best_model_xgboost_params = model_grid.best_params_

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    artifact = {
        "accuracy_train": accuracy_score(y_pred_train, y_train),
        "accuracy_test": accuracy_score(y_pred_test, y_test),
    }

    with open(accuracy_scores_path, "w") as f:
        json.dump(artifact, f)

    

    return best_model_xgboost_params, y_pred_test, y_pred_train


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
    """Model training pipeline."""

    logger.info("Processing started")

    # 1. Load data
    data = load_data(input_path)

    # 2. split data
    cat_cols, other_vars = data_type_split(
        data, 
        CAT_COLUMNS
    )

    # 3. Cast categorical variables to category type
    cat_vars = cast_to_category(data, cat_vars)

    # 4. Concatenate categorical and other variables
    data = concat_vars(
        data, 
        other_vars, 
        cat_vars
    )

    # 5. convert into float
    data = float_conversion(data)

    # 6. Split data into features and target
    y, X = split_data(data, TARGET_COLUMN)

    # 7. Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # 8. Get model grid
    model_grid = setup_grid_search(model, params)

    # 9. Get best parameters
    best_model_xgboost_params = model_grid.best_params_


    # 10. Get confusion matrix
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)

    # 11. Get model
    xgboost_model = model_grid.best_estimator_

    # MLflow tracking
    with mlflow.start_run():
        mlflow.log_param("param_distributions", str(params))
        mlflow.log_param("Best xgboost params", str(best_model_xgboost_params))
        mlflow.log_artifact(accuracy_scores_path)

if __name__ == "__main__":
    app()








    












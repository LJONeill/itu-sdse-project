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

# Build class
class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]
    

def prepare_data_for_models(data: pd.DataFrame):

    X, y = separate_feats_labels(data=data)
    return perform_train_test_split(X=X, y=y)

def setup_grid_search(
        model_class_choice: Literal["xgboost", "lr"],
        xgboost_params: dict,
        lr_params: dict,
        ):

    if model_class_choice == "xgboost":
        model = XGBRFClassifier()
        params = xgboost_params
    elif model_class_choice == "lr":
        model = LogisticRegression()
        params = lr_params
    else:
        print("Error: Invalid model choice entered, please choose between xgboost or lr")

    model_grid = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_jobs=-1,
        verbose=3,
        n_iter=10,
        cv=10,
        )
    
    return model_grid
    
def make_model_predictions(
        model_results: dict,
        data,
        model_class_choice: Literal["xgboost", "lr"] = "lr",
        ):

    if model_class_choice == "xgboost":
        model_path = xgboost_model_path
    elif model_class_choice == "lr":
        model_path = lr_model_path
    else:
        print(error_mcc)

    X_train, X_test, y_train, y_test = prepare_data_for_models(data=data)

    model_grid = setup_grid_search(model_class_choice=model_class_choice)

    model_grid.fit(X_train, y_train)

    y_pred_train = model_grid.predict(X_train)
  
    model_results[model_path] = classification_report(y_train, y_pred_train, output_dict=True)

    return model_grid, model_results, X_test, y_test

def train_and_save_model(
        model_class_choice: Literal["xgboost", "lr"],
        model_results: dict,
        data,
        ):
    if model_class_choice == "xgboost":
        model_path = xgboost_model_path
    elif model_class_choice == "lr":
        model_path = lr_model_path
    else:
        print(error_mcc)

    with mlflow.start_run(experiment_id=experiment_id) as run:

        model_grid, model_results, X_test, y_test = make_model_predictions(model_results=model_results, data=data)

        best_model = model_grid.best_estimator_

        y_pred_test = model_grid.predict(X_test)

    if model_class_choice == "xgboost":
        best_model.save_model(model_path)
    elif model_class_choice == "lr":
        mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
        mlflow.log_artifacts("artifacts", artifact_path="model")
        mlflow.log_param("data_version", "00000")
        mlflow.pyfunc.log_model('model', python_model=lr_wrapper(best_model))
        joblib.dump(value=best_model, filename=model_path)
    else:
        print(error_mcc)
    
    return model_results

# Error message for invalid model choice
error_mcc = "Error: Invalid model_class_choice, choose xgboost or lr"

# Define experiment name
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
experiment_name = current_date

# Start mlflow tracking
mlflow.set_experiment(experiment_name)
mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Parameters by model
xgboost_params = {
    "learning_rate": uniform(1e-2, 3e-1),
    "min_split_loss": uniform(0, 10),
    "max_depth": randint(3, 10),
    "subsample": uniform(0, 1),
    "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
    "eval_metric": ["aucpr", "error"]
    }

lr_params = {
    'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    'penalty':  ["none", "l1", "l2", "elasticnet"],
    'C' : [100, 10, 1.0, 0.1, 0.01]
    }

# Load data
data = load_data(INPUT_PATH)

#initialise results dictionary
model_results = dict()

train_and_save_model(
    "xgboost", 
    model_results=model_results,
    data=data,
    )

train_and_save_model(
    "lr", 
    model_results=model_results,
    data=data,
    )

# Store model results
with open(model_results_path, 'w+') as results_file:
    json.dump(model_results, results_file)
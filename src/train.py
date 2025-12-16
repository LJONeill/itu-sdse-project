# Imports

from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRFClassifier
from scipy.stats import uniform, randint
from typing import Literal


import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import json
import typer
import warnings
from loguru import logger
import mlflow.pyfunc

from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import cohen_kappa_score, f1_score
import matplotlib.pyplot as plt
import joblib

from config import (
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
    TARGET_COLUMN, 
    CAT_COLUMNS,
    RANDOM_STATE,
    EXPERIMENT_NAME as experiment_name,
)

app = typer.Typer()

# Paths

# Input
INPUT_PATH: Path = PROCESSED_DATA_DIR / "training_gold.csv"

# Intermediate data
features_path: Path = INTERIM_DATA_DIR / "features.csv"
labels_path: Path = INTERIM_DATA_DIR/ "labels.csv"
xgboost_model_path: Path = INTERIM_DATA_DIR / "lead_model_xgboost.json"
lr_model_path: Path = INTERIM_DATA_DIR / "lr_model.pkl"
column_list_path: Path = INTERIM_DATA_DIR / "columns_list.json"

accuracy_scores_path: Path = INTERIM_DATA_DIR / "accuracy_scores.json"
classification_reports_path: Path = INTERIM_DATA_DIR / "classification_reports.json"

# Output data
model_results_path: Path = PROCESSED_DATA_DIR /  "model_results.json"


# Classes and functions
class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

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
            cat_vars = data[cat_cols]
    other_vars = data.drop(cat_cols, axis=1)

    return cat_vars, other_vars


def categorical_dummy_variables(
    categorical_variables: pd.DataFrame,
    other_variables: pd.DataFrame,
) -> pd.DataFrame:
    
    for col in categorical_variables.columns:
        categorical_variables[col] = categorical_variables[col].astype("category")

    for col in list(categorical_variables.columns):
        categorical_variables = create_dummy_cols(categorical_variables, col)

    data = pd.concat([other_variables, categorical_variables], axis=1)

    for col in data.columns:
        data[col] = data[col].astype("float64")

    return data


# Split data into train and test sets
def split_data(data: pd.DataFrame, target_column: str):
    y = data[target_column]
    X = data.drop([target_column], axis=1)
    return y, X
    
# Model training

# I am gonna leave the model here - but it should be in cofig.py
model = XGBRFClassifier(random_state=RANDOM_STATE)
params = {
    "learning_rate": uniform(1e-2, 3e-1),
    "min_split_loss": uniform(0, 10),
    "max_depth": randint(3, 10),
    "subsample": uniform(0, 1),
    "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
    "eval_metric": ["aucpr", "error"]
}

# Setup grid search

def setup_grid_search(model, params, X_train, y_train):
    model_grid = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_jobs=-1,
        verbose=3,
        n_iter=10,
        cv=10,
    )
    model_grid.fit(X_train, y_train)
    return model_grid


# Get best model params

def get_best_model_params(model_grid, X_train, X_test, y_train, y_test):
    best_model_xgboost_params = model_grid.best_params_

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)

    artifact = {
        "accuracy_train": accuracy_score(y_pred_train, y_train),
        "accuracy_test": accuracy_score(y_pred_test, y_test),
    }

    with open(accuracy_scores_path, "w") as f:
        json.dump(artifact, f)

    

    return best_model_xgboost_params, y_pred_train, y_pred_test


# Get confusion matrices ... also need to log this?

def get_confusion_matrix(
        y_test, 
        y_pred_test, 
        y_train, 
        y_pred_train
):
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)

    report_test = pd.crosstab(
        y_test,
        y_pred_test,
        rownames=["Actual"],
        colnames=["Predicted"],
        margins=True,
    )

    report_train = pd.crosstab(
        y_train,
        y_pred_train,
        rownames=["Actual"],
        colnames=["Predicted"],
        margins=True,
    )

    classification_reports = {
        "confusion_matrix_test": conf_matrix_test.tolist(),
        "confusion_matrix_train": conf_matrix_train.tolist(),
        "classification_report_test": report_test.to_dict(),
        "classification_report_train": report_train.to_dict(),
    }

    with open(classification_reports_path, "w") as f:
        json.dump(classification_reports, f, indent=2)

    return conf_matrix_test, conf_matrix_train


# Also in main runner

def get_xgboost_model(model_grid, model_results):
    xgboost_model = model_grid.best_estimator_

    with open(xgboost_model_path, "w") as f:
        json.dump(xgboost_model, f)

    return xgboost_model

# In config?

lr_model = LogisticRegression(max_iter=1000)

lr_params = {
    "solver": ["lbfgs", "liblinear"],
    "penalty": ["l2"],
    "C": uniform(0.01, 100),
}

@app.command()
def main(
    input_path: Path = INPUT_PATH,
    output_path: Path = model_results_path,
):
    """Model training pipeline."""

    logger.info("Processing started")

    
    # Data preparation 
    
    data = load_data(input_path)

    cat_vars, other_vars = data_type_split(data, CAT_COLUMNS)
    data = categorical_dummy_variables(cat_vars, other_vars)

    y, X = split_data(data, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=RANDOM_STATE,
        test_size=0.15,
        stratify=y,
    )

    
    # XGBoost training 
    
    model_grid = setup_grid_search(model, params, X_train, y_train)

    best_model_xgboost_params, y_pred_train, y_pred_test = get_best_model_params(
        model_grid,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    conf_matrix_test, conf_matrix_train = get_confusion_matrix(
        y_test,
        y_pred_test,
        y_train,
        y_pred_train,
    )


    # MLflow + Logistic 
    mlflow.set_experiment(experiment_name)
   
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)

    with mlflow.start_run(experiment_id=experiment_id):

        lr_grid = setup_grid_search(lr_model, lr_params, X_train, y_train)
        best_lr_model = lr_grid.best_estimator_

        y_pred_train = best_lr_model.predict(X_train)
        y_pred_test = best_lr_model.predict(X_test)

        # Metrics (same as notebook)
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred_test))
        mlflow.log_metric(
            "kappa_test",
            cohen_kappa_score(y_test, y_pred_test),
        )

        # Params
        mlflow.log_params(lr_grid.best_params_)
        mlflow.log_param("data_version", "00000")

        # Save model 
        joblib.dump(best_lr_model, lr_model_path)
        mlflow.log_artifact(lr_model_path, artifact_path="model")

        mlflow.pyfunc.log_model(
            artifact_path="model_pyfunc",
            python_model=lr_wrapper(best_lr_model),
        )

    logger.info("Training completed")

if __name__ == "__main__":
    main()
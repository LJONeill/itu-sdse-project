# Imports

# Standard libraries
from pathlib import Path
import json

# Third-party libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import mlflow
import typer
from loguru import logger

from config import (
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
    SCALER_PATH,
    COLUMNS_TO_OBJECT,
    SOURCE_COLUMN,
    SOURCE_BIN_VALUES,
    SOURCE_BIN_MAPPING,
    BIN_SOURCE_COLUMN,
)

app = typer.Typer()

# Input / output datasets
INPUT_PATH: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"
OUTPUT_PATH: Path = PROCESSED_DATA_DIR / "training_gold.csv"

# Interim artifacts
COLUMN_SPLIT_PATH = INTERIM_DATA_DIR / "columns_split.json"
COLUMN_DRIFT_PATH = INTERIM_DATA_DIR / "columns_drift.json"
OUTLIER_SUMMARY_PATH: Path = INTERIM_DATA_DIR / "outlier_summary.json"
CAT_MISSING_IMPUTE_PATH: Path = INTERIM_DATA_DIR / "cat_missing_impute.csv"

# Helper functions
def describe_numeric_col(x):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats. 
    """
    return pd.Series(
        [x.count(), x.isnull().sum(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x, method="mean"):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
        method (str): Values: "mean", "median"
    """
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        x = x.fillna(x.mode()[0])
    return x

def load_data(path: Path) -> pd.DataFrame:
    """Load data from CSV."""
    return pd.read_csv(path)

# Feature engineering steps
def cast_columns_to_object(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    for col in columns:
        if col in data.columns:
            data[col] = data[col].astype("object")
    return data

def split_categorical_and_continuous(
    data: pd.DataFrame,
    artifact_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cont_vars = data.loc[
        :,
        (data.dtypes == "float64") | (data.dtypes == "int64"),
    ]
    cat_vars = data.loc[:, data.dtypes == "object"]

    # Replace notebook print statements with artifact
    artifact = {
        "continuous_columns": list(cont_vars.columns),
        "categorical_columns": list(cat_vars.columns),
    }

    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)

    return cont_vars, cat_vars

def clip_continuous_outliers_and_save_summary(
    cont_vars: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    Clip continuous variables using ±2 std 
    and save summary statistics as a JSON artifact.
    """

    cont_vars = cont_vars.apply(
        lambda x: x.clip(
            lower=x.mean() - 2 * x.std(),
            upper=x.mean() + 2 * x.std(),
        )
    )

    
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    summary_dict = outlier_summary.to_dict(orient="index")

    with open(output_path, "w") as f:
        json.dump(summary_dict, f, indent=2)

    return cont_vars

def save_categorical_mode_snapshot(
    cat_vars: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """
    Compute categorical mode values 
    and store them as a JSON artifact.
    """

    cat_missing_impute = cat_vars.mode(
        numeric_only=False,
        dropna=True,
    )

    snapshot = cat_missing_impute.to_dict(orient="records")

    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    return cat_missing_impute

def fit_and_scale_continuous_variables(
    cont_vars: pd.DataFrame,
    scaler_path: Path,
) -> pd.DataFrame:
    """
    Fit a MinMaxScaler on continuous variables,
    save the scaler, and return scaled data.
    """
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)

    joblib.dump(scaler, scaler_path)

    cont_vars_scaled = pd.DataFrame(
        scaler.transform(cont_vars),
        columns=cont_vars.columns,
    )

    return cont_vars_scaled

def recombine_categorical_and_continuous(
    cat_vars: pd.DataFrame,
    cont_vars: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recombine categorical and continuous variables
    after independent preprocessing.
    """
    cat_vars = cat_vars.reset_index(drop=True)
    cont_vars = cont_vars.reset_index(drop=True)

    return pd.concat([cat_vars, cont_vars], axis=1)

def bin_source_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Bin the source column exactly as in the notebook,
    while preserving the 'Others' category.
    """
    data = data.copy()

    data[BIN_SOURCE_COLUMN] = data[SOURCE_COLUMN]

    data.loc[
        ~data[SOURCE_COLUMN].isin(SOURCE_BIN_VALUES),
        BIN_SOURCE_COLUMN,
    ] = "Others"

    data[BIN_SOURCE_COLUMN] = (
        data[BIN_SOURCE_COLUMN]
        .map(SOURCE_BIN_MAPPING)
        .fillna("Others")
    )

    return data

def save_data_drift(
    data: pd.DataFrame,
    columns_drift_path: Path,
) -> None:
    """
    Save final training feature columns as a JSON artifact
    (data drift reference).
    """
    with open(columns_drift_path, "w") as f:
        json.dump(list(data.columns), f, indent=2)

# Docker main script
@app.command()
def main(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """
    Run feature engineering pipeline and persist training gold data.
    """

    logger.info("Feature processing started")

    # 1. Load cleaned data
    data = load_data(input_path)

    # 2. Cast categorical columns
    data = cast_columns_to_object(
        data,
        COLUMNS_TO_OBJECT,
    )

    # 3. Split categorical and continuous
    cont_vars, cat_vars = split_categorical_and_continuous(
        data,
        COLUMN_SPLIT_PATH,
    )

    # 4. Clip outliers + save outlier summary (JSON)
    cont_vars = clip_continuous_outliers_and_save_summary(
        cont_vars,
        OUTLIER_SUMMARY_PATH,
    )

    # 5. Save categorical mode snapshot (JSON)
    save_categorical_mode_snapshot(
        cat_vars,
        CAT_MISSING_IMPUTE_PATH,
    )

    # 6. Impute missing values
    cont_vars = cont_vars.apply(impute_missing_values)
    cat_vars = cat_vars.apply(impute_missing_values)

    # 7. Scale continuous variables (+ save scaler)
    cont_vars = fit_and_scale_continuous_variables(
        cont_vars,
        SCALER_PATH,
    )

    # 8. Recombine categorical and continuous features
    data = recombine_categorical_and_continuous(
        cat_vars,
        cont_vars,
    )

    # 9. Bin source column
    data = bin_source_column(data)

    # 10. Save data drift artifact (final feature schema)
    save_data_drift(
        data,
        COLUMN_DRIFT_PATH,
    )

    # 11. Persist training gold data (ONLY CSV WRITE IN FEATURES)
    data.to_csv(output_path, index=False)

    logger.info(
        "Training gold data saved",
        path=str(output_path),
        rows=data.shape[0],
        features=data.shape[1],
    )

    # 12. MLflow logging
    with mlflow.start_run():
        mlflow.log_param("rows", data.shape[0])
        mlflow.log_param("features", data.shape[1])

        mlflow.log_artifact(output_path)
        mlflow.log_artifact(OUTLIER_SUMMARY_PATH)
        mlflow.log_artifact(CAT_MISSING_IMPUTE_PATH)
        mlflow.log_artifact(COLUMN_DRIFT_PATH)
        mlflow.log_artifact(SCALER_PATH)

    logger.success("Feature processing done")

if __name__ == "__main__":
    app()
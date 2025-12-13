# Imports

# Standard libraries
from pathlib import Path
import json

# Third-party libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

from config import (
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
    CLEANED_DATA_PATH,
    TRAINING_GOLD_PATH,
    SCALER_PATH,
    COLUMNS_TO_CLEAN,
    COLUMNS_REQUIRED,
    COLUMNS_TO_OBJECT,
)

# Paths

# Processed data paths
#TRAINING_DATA_PATH: Path = PROCESSED_DATA_DIR / "training_data.csv"
#CLEANED_DATA_PATH: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"
#TRAINING_GOLD_PATH: Path = PROCESSED_DATA_DIR / "training_gold.csv"
# removing cause already in config.py


# Interim data paths
OUTLIER_SUMMARY_PATH: Path = INTERIM_DATA_DIR / "outlier_summary.csv"
CAT_MISSING_IMPUTE_PATH: Path = INTERIM_DATA_DIR / "cat_missing_impute.csv"
COLUMN_DRIFT_PATH: Path = INTERIM_DATA_DIR / "columns_drift.json"

# External artifacts
# SCALER_PATH: Path = EXTERNAL_DATA_DIR / "scaler.pkl" 
#Moved to config


def describe_numeric_col(x: pd.Series) -> pd.Series:
    """Return basic descriptive statistics for a numeric Series."""
    return pd.Series(
        [x.count(), x.isnull().sum(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"],
    )


def impute_missing_values(x: pd.Series, method: str = "mean") -> pd.Series:
    """Impute missing values in a Series."""
    if x.dtype in ("float64", "int64"):
        x = x.fillna(x.mean() if method == "mean" else x.median())
    else:
        x = x.fillna(x.mode()[0])

    return x


def create_dummy_cols(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Create dummy variables for a categorical column."""
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    return df.drop(columns=[col])


# Load data
def load_data(path: Path) -> pd.DataFrame:
    """Load data from CSV."""
    return pd.read_csv(path)

# Replace empty cells with nan

def replace_empty_with_nan(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Replace empty strings with NaN in specified columns."""
    for col in columns:
        data[col] = data[col].replace("", np.nan)

    return data


# Drop all records with NA in the given variables
def drop_rows_with_missing_values(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Drop rows with missing values in specified columns."""
    return data.dropna(subset=columns)



# Change data types to object
def columns_to_object(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Change specified columns to object dtype."""
    for col in columns:
        data[col] = data[col].astype("object")

    return data


# Splitting con and cat

def split_continuous_and_categorical(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into continuous and categorical variables."""
    cont_vars = data.loc[
        :,
        (data.dtypes == "float64") | (data.dtypes == "int64"),
    ]
    cat_vars = data.loc[:, data.dtypes == "object"]

    return cont_vars, cat_vars


# Find outliers in continuous variables

def handle_continuous_outliers(
    cont_vars: pd.DataFrame,
    summary_path: Path,
) -> pd.DataFrame:
    """Clipping outliers in continuous variables and saving a summary."""
    cont_vars = cont_vars.apply(
        lambda x: x.clip(
            lower=x.mean() - 2 * x.std(),
            upper=x.mean() + 2 * x.std(),
        )
    )

    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv(summary_path)

    return cont_vars

def save_categorical_imputation_values(
    cat_vars: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Saving values for categorical variables before imputation."""
    cat_missing_impute = cat_vars.mode(
        numeric_only=False,
        dropna=True,
    )
    cat_missing_impute.to_csv(output_path)

    return cat_missing_impute

# imputation on continues variables

def impute_continueus_variables(
    cont_vars: pd.DataFrame,
) -> pd.DataFrame:
    """Impute missing values in continuous variables."""
    return cont_vars.apply(impute_missing_values)

# Imputation on categorical variables

def impute_categorical_variables(
    cat_vars: pd.DataFrame,
    customer_code_col: str = "customer_code",
) -> pd.DataFrame:
    """Impute missing values in categorical variables."""
    cat_vars = cat_vars.copy()
    cat_vars.loc[cat_vars[customer_code_col].isna(), customer_code_col] = "None"

    return cat_vars.apply(impute_missing_values)


# Make and save scaler for continuous variables

def scaler( #needs another name
    cont_vars: pd.DataFrame,
    scaler_path: Path,
) -> MinMaxScaler:
    """Fit a MinMaxScaler on continuous variables and save it."""
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    joblib.dump(scaler, scaler_path)

    return scaler

def scale_continuous_variables( #should this be a function on its own or just a part of the main flow? 
    cont_vars: pd.DataFrame,
    scaler: MinMaxScaler,
) -> pd.DataFrame:
    """Scale continuous variables."""
    return pd.DataFrame(
        scaler.transform(cont_vars),
        columns=cont_vars.columns,
    )

def recombine_categorical_and_continuous(
    cat_vars: pd.DataFrame,
    cont_vars: pd.DataFrame,
) -> pd.DataFrame:
    """Recombine categorical and continuous variables."""
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)

    return pd.concat([cat_vars, cont_vars], axis=1)


def store_data_and_columns(
    data: pd.DataFrame,
    columns_path: Path,
    data_path: Path,
) -> None:
    """Store dataset columns and data to disk."""
    data_columns = list(data.columns)

    with open(columns_path, "w+") as f:
        json.dump(data_columns, f)

    data.to_csv(data_path, index=False)


def bin_source_category(data: pd.DataFrame) -> pd.DataFrame:
    """Perform category binning for the source column."""
    data["bin_source"] = data["source"]

    values_list = ["li", "organic", "signup", "fb"]
    data.loc[~data["source"].isin(values_list), "bin_source"] = "Others"

    mapping = {
        "li": "socials",
        "fb": "socials",
        "organic": "group1",
        "signup": "group1",
    }

    data["bin_source"] = data["source"].map(mapping)

    return data


# Drop columns
#data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)
# I decided this should be moved to dataset.py, no reason to drop columns twice

def one_hot_encode_categorical_variables(
    data: pd.DataFrame,
    cat_cols: list[str],
) -> pd.DataFrame:
    """One-hot encode specified categorical columns."""
    cat_vars = data[cat_cols]
    other_vars = data.drop(cat_cols, axis=1)

    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    return pd.concat([other_vars, cat_vars], axis=1)


# Reconcatenate continuous and one hot encoded variables
def recombine_and_cast_to_float(
    other_vars: pd.DataFrame,
    cat_vars: pd.DataFrame,
) -> pd.DataFrame:
    """Recombine variables and cast all columns to float."""
    data = pd.concat([other_vars, cat_vars], axis=1)

    for col in data:
        data[col] = data[col].astype("float64")

    return data


def save_features_data(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save feature dataset to CSV."""
    data.to_csv(output_path, index=False)


#Docker main script:

@app.command() 
def main(
    input_path: Path = CLEANED_DATA_PATH, #still think it should be filtered
    output_path: Path = TRAINING_GOLD_PATH,
):
    """Run the data processing pipeline."""

    logger.info("Processing started")

    # Load data
    data = load_data(input_path)

        # MLflow tracking
    with mlflow.start_run():
        mlflow.log_param()
        mlflow.log_artifact()

    logger.success("Processing done")


if __name__ == "__main__":
    app()
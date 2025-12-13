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
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    COLUMNS_TO_CLEAN,
    COLUMNS_REQUIRED,
    COLUMNS_TO_OBJECT,
)

# Paths

# Processed data paths
TRAINING_DATA_PATH: Path = PROCESSED_DATA_DIR / "training_data.csv"
CLEANED_DATA_PATH: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"
TRAINING_GOLD_PATH: Path = PROCESSED_DATA_DIR / "training_gold.csv"

# Interim data paths
OUTLIER_SUMMARY_PATH: Path = INTERIM_DATA_DIR / "outlier_summary.csv"
CAT_MISSING_IMPUTE_PATH: Path = INTERIM_DATA_DIR / "cat_missing_impute.csv"
COLUMN_DRIFT_PATH: Path = INTERIM_DATA_DIR / "columns_drift.json"

# External artifacts
SCALER_PATH: Path = EXTERNAL_DATA_DIR / "scaler.pkl"


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

# Change customer code from NA to None, then impute all categorical variables
cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
cat_vars = cat_vars.apply(impute_missing_values)

# Make and save scaler for continuous variables
scaler = MinMaxScaler()
scaler.fit(cont_vars)
joblib.dump(value=scaler, filename=scaler_path)

# Perform scaling on continuous variables
cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

# Recombine categorical and continuous data
cont_vars = cont_vars.reset_index(drop=True)
cat_vars = cat_vars.reset_index(drop=True)
data = pd.concat([cat_vars, cont_vars], axis=1)

# Store variables and data
data_columns = list(data.columns)
with open(column_drift_path,'w+') as f:           
    json.dump(data_columns,f)
    
data.to_csv(training_data_path, index=False)

# Perform category binning
data['bin_source'] = data['source']
values_list = ['li', 'organic','signup','fb']
data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
mapping = {'li' : 'socials', 
           'fb' : 'socials', 
           'organic': 'group1', 
           'signup': 'group1'
           }

data['bin_source'] = data['source'].map(mapping)

# Drop columns
data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

# One hot encode categorical variables
cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
cat_vars = data[cat_cols]

other_vars = data.drop(cat_cols, axis=1)

for col in cat_vars:
    cat_vars[col] = cat_vars[col].astype("category")
    cat_vars = create_dummy_cols(cat_vars, col)

# Reconcatenate continuous and one hot encoded variables
data = pd.concat([other_vars, cat_vars], axis=1)

for col in data:
    data[col] = data[col].astype("float64")

# Write out features data
data.to_csv(training_gold_path, index=False)
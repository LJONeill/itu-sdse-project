from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from mlops_sg.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR, EXTERNAL_DATA_DIR

import numpy as np
import pandas as pd
import joblib
import json

training_data_path: Path = PROCESSED_DATA_DIR / "training_data.csv",
cleaned_data_path: Path = PROCESSED_DATA_DIR / "cleaned_data.csv"
training_gold_path: Path = PROCESSED_DATA_DIR / "training_gold.csv",
outlier_summary_path: Path = INTERIM_DATA_DIR / "outlier_summary.csv",
cat_missing_impute_path: Path = INTERIM_DATA_DIR / "cat_missing_impute.csv",
scaler_path: Path = EXTERNAL_DATA_DIR / "scaler.pkl",
column_drift_path: Path = INTERIM_DATA_DIR / "columns_drift.json"

# Defined variables for use throughout

def describe_numeric_col(x):
    """
    Parameters:
        x (pd.Series): Pandas col to describe.
    Output:
        y (pd.Series): Pandas series with descriptive stats. 
    """
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
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

data = pd.read_csv(cleaned_data_path)

data["lead_indicator"].replace("", np.nan, inplace=True)
data["lead_id"].replace("", np.nan, inplace=True)
data["customer_code"].replace("", np.nan, inplace=True)

data = data.dropna(axis=0, subset=["lead_indicator"])
data = data.dropna(axis=0, subset=["lead_id"])

data = data[data.source == "signup"]
result=data.lead_indicator.value_counts(normalize = True)

print("Target value counter")
for val, n in zip(result.index, result):
    print(val, ": ", n)

vars = [
    "lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"
]

for col in vars:
    data[col] = data[col].astype("object")
    print(f"Changed {col} to object type")

cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
cat_vars = data.loc[:, (data.dtypes=="object")]

#print("\nContinuous columns: \n")
#print(list(cont_vars.columns), indent=4)
#print("\n Categorical columns: \n")
#print(list(cat_vars.columns), indent=4)

cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                             upper = (x.mean()+2*x.std())))
outlier_summary = cont_vars.apply(describe_numeric_col).T
outlier_summary.to_csv(outlier_summary_path)

cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
cat_missing_impute.to_csv(cat_missing_impute_path)

cont_vars = cont_vars.apply(impute_missing_values)
cont_vars.apply(describe_numeric_col).T

cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
cat_vars = cat_vars.apply(impute_missing_values)
cat_vars.apply(lambda x: pd.Series([x.count(), x.isnull().sum()], index = ['Count', 'Missing'])).T

scaler = MinMaxScaler()
scaler.fit(cont_vars)

joblib.dump(value=scaler, filename=scaler_path)
print("Saved scaler in artifacts")

cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

cont_vars = cont_vars.reset_index(drop=True)
cat_vars = cat_vars.reset_index(drop=True)
data = pd.concat([cat_vars, cont_vars], axis=1)
print(f"Data cleansed and combined.\nRows: {len(data)}")

data_columns = list(data.columns)
with open(column_drift_path,'w+') as f:           
    json.dump(data_columns,f)
    
data.to_csv(training_data_path, index=False)

data['bin_source'] = data['source']
values_list = ['li', 'organic','signup','fb']
data.loc[~data['source'].isin(values_list),'bin_source'] = 'Others'
mapping = {'li' : 'socials', 
           'fb' : 'socials', 
           'organic': 'group1', 
           'signup': 'group1'
           }

data['bin_source'] = data['source'].map(mapping)

data.to_csv(training_gold_path, index=False)
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlops_sg.config import MODELS_DIR, PROCESSED_DATA_DIR
from xgboost import XGBRFClassifier
from scipy.stats import uniform
from scipy.stats import randint

import datetime
import mlflow
import pandas as pd

# ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
data_gold_path: Path = PROCESSED_DATA_DIR / "training_gold.csv",
features_path: Path = PROCESSED_DATA_DIR / "features.csv",
labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
model_path: Path = MODELS_DIR / "model.pkl",

# defined variables for use throughout
def create_dummy_cols(df, col):
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_version = "00000"
experiment_name = current_date

mlflow.set_experiment(experiment_name)

data = pd.read_csv(data_gold_path)

data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)

cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
cat_vars = data[cat_cols]

other_vars = data.drop(cat_cols, axis=1)

for col in cat_vars:
    cat_vars[col] = cat_vars[col].astype("category")
    cat_vars = create_dummy_cols(cat_vars, col)

data = pd.concat([other_vars, cat_vars], axis=1)

for col in data:
    data[col] = data[col].astype("float64")
    print(f"Changed column {col} to float")

y = data["lead_indicator"]
X = data.drop(["lead_indicator"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y)

model = XGBRFClassifier(random_state=42)
params = {
    "learning_rate": uniform(1e-2, 3e-1),
    "min_split_loss": uniform(0, 10),
    "max_depth": randint(3, 10),
    "subsample": uniform(0, 1),
    "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
    "eval_metric": ["aucpr", "error"]
}

model_grid = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10)

model_grid.fit(X_train, y_train)

best_model_xgboost_params = model_grid.best_params_
print("Best xgboost params")
pprint(best_model_xgboost_params)

y_pred_train = model_grid.predict(X_train)
y_pred_test = model_grid.predict(X_test)
print("Accuracy train", accuracy_score(y_pred_train, y_train ))
print("Accuracy test", accuracy_score(y_pred_test, y_test))

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

xgboost_model = model_grid.best_estimator_
xgboost_model.save_model(model_path)

model_results = {
    model_path: classification_report(y_train, y_pred_train, output_dict=True)
}
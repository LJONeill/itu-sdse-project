from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, f1_score
from sklearn.linear_model import LogisticRegression
from mlops_sg.config import MODELS_DIR, PROCESSED_DATA_DIR
from xgboost import XGBRFClassifier
from scipy.stats import uniform, randint

import datetime
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json

# ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
data_gold_path: Path = PROCESSED_DATA_DIR / "training_gold.csv",
features_path: Path = PROCESSED_DATA_DIR / "features.csv",
labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
model_path: Path = MODELS_DIR / "model.pkl",
lr_model_path: Path = MODELS_DIR / "lr_model.pkl",
column_list_path: Path = MODELS_DIR / "columns_list.json",
model_results_path: Path = MODELS_DIR /  "model_results.json",


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

class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id) as run:
    model = LogisticRegression()

    params = {
              'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
              'penalty':  ["none", "l1", "l2", "elasticnet"],
              'C' : [100, 10, 1.0, 0.1, 0.01]
    }
    model_grid = RandomizedSearchCV(model, param_distributions= params, verbose=3, n_iter=10, cv=3)
    model_grid.fit(X_train, y_train)

    best_model = model_grid.best_estimator_

    y_pred_train = model_grid.predict(X_train)
    y_pred_test = model_grid.predict(X_test)


    # log artifacts
    mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test))
    mlflow.log_artifacts("artifacts", artifact_path="model")
    mlflow.log_param("data_version", "00000")
    
    # store model for model interpretability
    joblib.dump(value=model, filename=lr_model_path)
        
    # Custom python model for predicting probability 
    mlflow.pyfunc.log_model('model', python_model=lr_wrapper(model))


model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

best_model_lr_params = model_grid.best_params_

print("Best lr params")
pprint(best_model_lr_params)

print("Accuracy train:", accuracy_score(y_pred_train, y_train ))
print("Accuracy test:", accuracy_score(y_pred_test, y_test))

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

model_results[lr_model_path] = model_classification_report
print(model_classification_report["weighted avg"]["f1-score"])

with open(column_list_path, 'w+') as columns_file:
    columns = {'column_names': list(X_train.columns)}
    print(columns)
    json.dump(columns, columns_file)

print('Saved column list to ', column_list_path)

with open(model_results_path, 'w+') as results_file:
    json.dump(model_results, results_file)
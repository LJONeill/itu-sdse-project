from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, f1_score
from sklearn.linear_model import LogisticRegression
from ..config import MODELS_DIR, PROCESSED_DATA_DIR
from xgboost import XGBRFClassifier
from scipy.stats import uniform, randint

import datetime
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json

# Paths
data_gold_path: Path = PROCESSED_DATA_DIR / "training_gold.csv"
features_path: Path = PROCESSED_DATA_DIR / "features.csv"
labels_path: Path = PROCESSED_DATA_DIR / "labels.csv"
xgboost_model_path: Path = MODELS_DIR / "xgboost_model.pkl"
lr_model_path: Path = MODELS_DIR / "lr_model.pkl"
column_list_path: Path = MODELS_DIR / "columns_list.json"
model_results_path: Path = MODELS_DIR /  "model_results.json"

# Define experiment name
current_date = datetime.datetime.now().strftime("%Y_%B_%d")
data_version = "00000"
experiment_name = current_date

# Start mlflow tracking
mlflow.set_experiment(experiment_name)

# Load the data
data = pd.read_csv(data_gold_path)

# Separate features from labels
y = data["lead_indicator"]
X = data.drop(["lead_indicator"], axis=1)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.15, stratify=y)

# Define xgboost model
model = XGBRFClassifier(random_state=42)

# Define hyperparameter possibilities
params = {
    "learning_rate": uniform(1e-2, 3e-1),
    "min_split_loss": uniform(0, 10),
    "max_depth": randint(3, 10),
    "subsample": uniform(0, 1),
    "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
    "eval_metric": ["aucpr", "error"]
}

# Define grid search with given xgboost model and hyperparameters
model_grid = RandomizedSearchCV(model, param_distributions=params, n_jobs=-1, verbose=3, n_iter=10, cv=10)

# Perform grid search
model_grid.fit(X_train, y_train)

# Make predictions
y_pred_train = model_grid.predict(X_train)
y_pred_test = model_grid.predict(X_test)

# Output the best xgboost model
xgboost_model = model_grid.best_estimator_
xgboost_model.save_model(xgboost_model_path)

# Initiate set of all model results and add xgboost results
model_results = {
    xgboost_model_path: classification_report(y_train, y_pred_train, output_dict=True)
}

# Create a class with a predict function to show model results
class lr_wrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

# Initiate experiment tracking
mlflow.sklearn.autolog(log_input_examples=True, log_models=False)

# Initiate experiment
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# run logistic regression model training
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

# Generate model report stats
model_classification_report = classification_report(y_test, y_pred_test, output_dict=True)

# Store model report stats to earlier created set
model_results[lr_model_path] = model_classification_report

# Store training data
with open(column_list_path, 'w+') as columns_file:
    columns = {'column_names': list(X_train.columns)}
    print(columns)
    json.dump(columns, columns_file)

# Store model results
with open(model_results_path, 'w+') as results_file:
<<<<<<< HEAD
    json.dump(model_results, results_file)
=======
    json.dump(model_results, results_file)
>>>>>>> fbc6e4a (docs(train): added predict, removed parameter storing)

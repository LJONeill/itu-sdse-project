from pathlib import Path
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from train import experiment_name

from ..config import MODELS_DIR, PROCESSED_DATA_DIR

import datetime
import time
import json
import pandas as pd
import mlflow

# Paths
model_results_path: Path = MODELS_DIR /  "model_results.json"

artifact_path = "model"
model_name = "lead_model"
client = MlflowClient()

def identify_best_experiment(experiment_name):

    experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]

    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1
    ).iloc[0]

    return experiment_best

def get_production_model():

    prod_model = [model for model in client.search_model_versions(f"name='{model_name}'") if dict(model)['current_stage']=='Production']

    if len(prod_model)==0:
        return print('No model in production')
    else:
        prod_model_run_id = dict(prod_model[0])['run_id']
        
    return prod_model, prod_model_run_id

def wait_until_ready(model_name, model_version):
    for _ in range(10):
        model_version_details = client.get_model_version(
          name=model_name,
          version=model_version,
        )
        status = ModelVersionStatus.from_string(model_version_details.status)
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)
        
def identify_and_register_best_model(
        experiment_best,
        prod_model,
        prod_model_run_id
        ):
    
    model_status = {}
    model_details = {}
    run_id = None

    train_model_score = experiment_best["metrics.f1_score"]

    prod_data, _ = mlflow.get_run(prod_model_run_id)
    prod_model_score = prod_data[1]["metrics.f1_score"]

    model_status["current"] = train_model_score
    model_status["prod"] = prod_model_score

    if train_model_score>prod_model_score:
        run_id = experiment_best["run_id"]

    if run_id is not None:
        

        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=run_id,
            artifact_path=artifact_path
        )
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        wait_until_ready(model_details.name, model_details.version)
        model_details = dict(model_details)

    return model_details

def wait_for_deployment(model_name, model_version, stage='Staging'):
    status = False
    while not status:
        model_version_details = dict(
            client.get_model_version(name=model_name,version=model_version)
            )
        if model_version_details['current_stage'] == stage:
            print(f'Transition completed to {stage}')
            status = True
            break
        else:
            time.sleep(2)
    return status

model_version = 1

model_version_details = dict(client.get_model_version(name=model_name,version=model_version))
model_status = True
if model_version_details['current_stage'] != 'Staging':
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Staging", 
        archive_existing_versions=True
    )
    model_status = wait_for_deployment(model_name, model_version, 'Staging')
else:
    print('Model already in staging')

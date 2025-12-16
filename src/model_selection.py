# Imports
import time
import mlflow
from pathlib import Path
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

from config import (
    PROCESSED_DATA_DIR,
    EXPERIMENT_NAME as experiment_name,
)

# Constants
ARTIFACT_PATH = "model"
MODEL_NAME = "lead_model"

# Paths
model_results_path: Path = PROCESSED_DATA_DIR / "model_results.json"

# Helper functions


def wait_until_ready(client, model_name, model_version):
    for _ in range(10):
        mv = client.get_model_version(name=model_name, version=model_version)
        status = ModelVersionStatus.from_string(mv.status)
        print(f"Model status: {ModelVersionStatus.to_string(status)}")
        if status == ModelVersionStatus.READY:
            return
        time.sleep(1)


def identify_best_experiment(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1,
    )

    return runs.iloc[0]


def get_production_model_run_id(client, model_name):
    prod_models = [
        m for m in client.search_model_versions(f"name='{model_name}'")
        if dict(m)["current_stage"] == "Production"
    ]

    if not prod_models:
        return None

    return dict(prod_models[0])["run_id"]


def register_best_model(client, run_id):
    model_uri = f"runs:/{run_id}/{ARTIFACT_PATH}"

    model_details = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    wait_until_ready(client, model_details.name, model_details.version)
    return model_details.version


def transition_to_staging(client, model_name, model_version):
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Staging",
        archive_existing_versions=True,
    )

    while True:
        mv = dict(client.get_model_version(name=model_name, version=model_version))
        if mv["current_stage"] == "Staging":
            print("Transition completed to Staging")
            break
        time.sleep(2)


# Main


def main():
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # 1. Identify best training run
    best_run = identify_best_experiment(experiment_name)
    best_f1 = best_run["metrics.f1_score"]
    best_run_id = best_run["run_id"]

    # 2. Identify current production model (if any)
    prod_run_id = get_production_model_run_id(client, MODEL_NAME)

    register_new_model = False

    if prod_run_id is None:
        print("No model in production — registering first model")
        register_new_model = True
    else:
        prod_run = mlflow.get_run(prod_run_id)
        prod_f1 = prod_run.data.metrics["f1_score"]

        if best_f1 > prod_f1:
            print("New model outperforms production — registering")
            register_new_model = True
        else:
            print("Production model performs better — skipping registration")

    # 3. Register & promote
    if register_new_model:
        model_version = register_best_model(client, best_run_id)
        transition_to_staging(client, MODEL_NAME, model_version)


if __name__ == "__main__":
    main()

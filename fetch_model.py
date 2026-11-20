"""During Docker build, bake the newest FINISHED MLflow run's model into /app/model.

DagsHub credentials arrive ONLY as temporary BuildKit secrets. The model
is downloaded at build time so the running FastAPI container needs no token.
"""

import json
import os
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts

EXPERIMENT_NAME = "RandomForest_V1.0.1"
DESTINATION = Path("/app/model")


def model_artifact_path(client: MlflowClient, run_id: str) -> str | None:
    """Find the MLflow 2.x model folder, including the notebook's custom names."""
    entries = client.list_artifacts(run_id)
    directories = sorted(
        (entry for entry in entries if entry.is_dir),
        key=lambda entry: (entry.path != "model", entry.path),
    )
    for directory in directories:
        # MLflow model folders always include an MLmodel configuration file.
        children = client.list_artifacts(run_id, directory.path)
        if any(child.path == f"{directory.path}/MLmodel" for child in children):
            return directory.path
    return None


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        raise SystemExit("Missing MLFLOW_TRACKING_URI Docker build argument.")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(
            f"Experiment {EXPERIMENT_NAME!r} not found on DagsHub. Run train.py first."
        )

    # Newest *successful* run with a model, not just the newest run of any kind.
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.end_time DESC"],
        max_results=100,
    )
    selected = None
    for run in runs:
        artifact_path = model_artifact_path(client, run.info.run_id)
        if artifact_path:
            selected = (run, artifact_path)
            break

    if selected is None:
        raise SystemExit(
            f"No completed run with an MLflow model artifact in {EXPERIMENT_NAME!r}. "
            "Train and log at least one model before building."
        )

    run, artifact_path = selected
    temporary_download_dir = Path("/tmp/mlflow-model-download")
    temporary_download_dir.mkdir(parents=True, exist_ok=True)
    local_path = download_artifacts(
        run_id=run.info.run_id,
        artifact_path=artifact_path,
        dst_path=str(temporary_download_dir),
    )
    downloaded = Path(local_path)
    if not (downloaded / "MLmodel").is_file():
        raise SystemExit(f"Downloaded artifact {artifact_path!r} is not an MLflow model.")

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    shutil.copytree(downloaded, DESTINATION)
    shutil.rmtree(temporary_download_dir)
    (DESTINATION / "source.json").write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "run_id": run.info.run_id,
                "artifact_path": artifact_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Fail the image build early if the model is incompatible with installed libs.
    loaded_model = mlflow.pyfunc.load_model(str(DESTINATION))
    print(f"Model downloaded and validated: run_id={run.info.run_id}")
    del loaded_model


if __name__ == "__main__":
    main()

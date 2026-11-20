"""Train the two RandomForest digit classifiers from MLFlowSetup.ipynb.

Logs MLflow params, accuracy, model artifacts, and registered-model versions
into the DagsHub tracking server. Set MLFLOW_TRACKING_URI,
MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD before running.
"""

import os

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

EXPERIMENT_NAME = "RandomForest_V1.0.1"
REGISTERED_MODEL_NAME = "digits-random-forest"


def main() -> None:
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise SystemExit("Set MLFLOW_TRACKING_URI to your DagsHub MLflow URL first.")
    if not all(os.environ.get(key) for key in ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")):
        raise SystemExit("Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD (DagsHub token) first.")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)

    digits = load_digits()  # The example notebook uses this built-in 64-feature dataset.
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, random_state=42
    )

    for n_estimators, max_depth in ((20, 5), (100, 10)):
        run_name = f"rf_n{n_estimators}_depth{max_depth}"
        with mlflow.start_run(run_name=run_name) as run:
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            accuracy = accuracy_score(y_test, predictions)

            mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
            mlflow.log_metric("accuracy", float(accuracy))
            signature = infer_signature(x_test[:5], predictions[:5])
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=x_test[:1],
                registered_model_name=REGISTERED_MODEL_NAME,
            )
            print(f"Run {run.info.run_id}: {run_name}, accuracy={accuracy:.4f}")

    print("Training completed. Both model versions were logged to DagsHub.")


if __name__ == "__main__":
    main()

"""Route and validation tests that mock out loading the remote MLflow model."""

from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app import app


class FakeModel:
    def predict(self, values):
        assert isinstance(values, np.ndarray)
        assert values.shape == (1, 64)
        return np.array([7])


def test_single_predict_endpoint():
    with patch("app.mlflow.pyfunc.load_model", return_value=FakeModel()):
        with TestClient(app) as client:
            response = client.post("/predict", json={"features": [0] * 64})
            assert response.status_code == 200
            assert response.json() == {"prediction": 7}
            assert client.get("/docs").status_code == 404
            assert client.get("/openapi.json").status_code == 404
            assert client.get("/predict").status_code == 405


def test_rejects_incorrect_feature_count():
    with patch("app.mlflow.pyfunc.load_model", return_value=FakeModel()):
        with TestClient(app) as client:
            response = client.post("/predict", json={"features": [0] * 4})
            assert response.status_code == 422

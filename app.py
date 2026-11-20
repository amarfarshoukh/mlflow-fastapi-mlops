"""FastAPI with exactly one route: POST /predict (no docs/OpenAPI routes)."""

from contextlib import asynccontextmanager

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(application: FastAPI):
    # The artifact was baked into the Docker image at build time.
    application.state.model = mlflow.pyfunc.load_model("/app/model")
    yield
    del application.state.model


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class PredictionInput(BaseModel):
    # sklearn.datasets.load_digits is an 8x8 image -> 64 numeric features.
    features: list[float] = Field(min_length=64, max_length=64)


@app.post("/predict")
def predict(request: PredictionInput) -> dict[str, int]:
    features = np.asarray([request.features], dtype=np.float64)
    prediction = app.state.model.predict(features)
    return {"prediction": int(prediction[0])}

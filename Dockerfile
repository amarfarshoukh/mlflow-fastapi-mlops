# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py fetch_model.py ./

# MLFLOW_TRACKING_URI is public repository metadata, NOT a credential.
ARG MLFLOW_TRACKING_URI
# Change on every GitHub run: avoids Docker caching an older model download.
ARG MODEL_REFRESH=local

# Download the newest completed model from DagsHub DURING THE IMAGE BUILD.
# Credentials are ephemeral BuildKit secrets, never image ENV or build args.
RUN --mount=type=secret,id=dagshub_username,required=true \
    --mount=type=secret,id=dagshub_token,required=true \
    test -n "$MLFLOW_TRACKING_URI" && \
    echo "Refreshing the baked model for build ${MODEL_REFRESH}" && \
    MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
    MLFLOW_TRACKING_USERNAME="$(cat /run/secrets/dagshub_username)" \
    MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/dagshub_token)" \
    python fetch_model.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

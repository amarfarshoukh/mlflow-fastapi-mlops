# MLOps assignment: MLflow + DagsHub + FastAPI + Docker + GitHub Actions

This repository implements the provided `15-MLFlow.pdf` instructions using the
`MLFlowSetup.ipynb` example (scikit-learn digits + two RandomForest runs), and
adds a single FastAPI route (`POST /predict`). **The Docker build downloads the
latest completed run's MLflow model from DagsHub and bakes it into the image.**
The running container does not need DagsHub credentials or internet access.

## What is included

- `train.py` logs two RandomForest experiments (params, accuracy, model artifact)
  to your DagsHub MLflow experiment, and registers their models.
- `fetch_model.py` selects the most recently finished run containing an MLflow
  model, and downloads that model into `/app/model` at Docker build time.
- `app.py` serves only `POST /predict`; docs and OpenAPI endpoints are disabled.
- `Dockerfile` uses BuildKit secrets to download the model *during build*.
- `.github/workflows/build.yml` builds and publishes the image to GHCR on
  pushes to `main` and on manual `workflow_dispatch`.
- `tests/test_app.py` tests `/predict` with a mocked model.

**This is based on the notebook's `sklearn.datasets.load_digits()` dataset.**
The `raw_dataset.csv` inside the supplied ZIP is unrelated to that example and
is not needed. The separate DVC/Azure PDF is not needed for this assignment.
Do not commit either PDF, a DagsHub token, or any Azure SAS token to a public repo.

## 1. Create GitHub + DagsHub repositories

1. Create a **PUBLIC** GitHub repository (example name: `mlflow-fastapi-mlops`).
2. On DagsHub, create a repository linked to that GitHub repository.
3. In DagsHub's **Experiments** tab, locate its MLflow tracking URL, usually:
   `https://dagshub.com/YOUR_DAGSHUB_OWNER/YOUR_DAGSHUB_REPO.mlflow`.
4. Create a DagsHub access token from your DagsHub account settings. Do not
   paste it into any committed file or screenshot.

## 2. Set PowerShell environment variables (locally; do not commit)

Install Python 3.11 first, to match the Docker image. Run from the project root
(replace placeholder values with your actual ones):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
$env:MLFLOW_TRACKING_URI = "https://dagshub.com/YOUR_DAGSHUB_OWNER/YOUR_DAGSHUB_REPO.mlflow"
$env:MLFLOW_TRACKING_USERNAME = "YOUR_DAGSHUB_USERNAME"
$secureToken = Read-Host "DagsHub access token (hidden)" -AsSecureString
$env:MLFLOW_TRACKING_PASSWORD = [System.Net.NetworkCredential]::new("", $secureToken).Password
Remove-Variable secureToken
```

The token input is hidden, and the token stays in this PowerShell session.
**Never echo it in a recording or paste it into a repo.** If desired, close the terminal after
training to clear the environment variables.

## 3. Log models to DagsHub

```powershell
python train.py
python -m pytest -q
```

The experiment name is `RandomForest_V1.0.1` (matching the sample notebook).
The run trained second (`n_estimators=100`, `max_depth=10`) is selected as
"latest" unless you later log a newer completed model run in that experiment.
Check the DagsHub **Experiments** MLflow UI to confirm the two model artifacts.

## 4. Set GitHub Actions settings

In the GitHub repo go to **Settings > Secrets and variables > Actions**:

- Under **Variables**, create `MLFLOW_TRACKING_URI` with your DagsHub MLflow URL.
- Under **Repository secrets**, create `DAGSHUB_USERNAME` (your DagsHub login).
- Under **Repository secrets**, create `DAGSHUB_TOKEN` (your DagsHub access token).

The workflow uses the built-in `${{ secrets.GITHUB_TOKEN }}` to publish to
**GitHub Container Registry** (`ghcr.io`). No Docker Hub credentials needed.
GitHub packages can have separate visibility settings from the repository;
publish package visibility separately if your instructor needs to pull it.

## 5. Push the public repo and run the workflow

Initialize Git and push to the empty GitHub repo (PowerShell):

```powershell
git init
git branch -M main
git add .
git commit -m "MLflow FastAPI assignment"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/mlflow-fastapi-mlops.git
git push -u origin main
```

Open **GitHub > Actions**. The workflow `Build FastAPI image with latest MLflow
model` should run on the push and show a successful image build and publish.
You can also use **Run workflow** to rebuild with the latest DagsHub model.
The `MODEL_REFRESH` build argument makes the model download execute again even
if BuildKit caching is enabled.

## 6. Run the Docker image locally

After the workflow succeeds, use your real image name. GitHub normalizes GHCR
image names to lowercase, e.g. `ghcr.io/username/mlflow-fastapi-mlops:latest`.
Depending on package visibility, `docker pull` may require `docker login ghcr.io`.

```powershell
docker pull ghcr.io/YOUR_GITHUB_USERNAME/mlflow-fastapi-mlops:latest
docker run --rm -p 8000:8000 ghcr.io/YOUR_GITHUB_USERNAME/mlflow-fastapi-mlops:latest
```

In a **second** PowerShell terminal:

```powershell
$body = @{ features = @(0) * 64 } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri 'http://localhost:8000/predict' -Method Post -ContentType 'application/json' -Body $body
```

Expected JSON shape: `{"prediction": 0}` (the integer class can be any digit
from 0 to 9 for a different input). Input must have exactly 64 numeric entries.

## Optional: build locally instead of using GHCR

Docker Desktop with BuildKit must be enabled. The model is downloaded as part of
building the image, so local build also needs network access and your DagsHub
credentials. Docker CLI can read the secrets from environment variables:

```powershell
$env:DOCKER_BUILDKIT = "1"
docker build --build-arg "MLFLOW_TRACKING_URI=$env:MLFLOW_TRACKING_URI" --build-arg "MODEL_REFRESH=$(Get-Date -Format yyyyMMddHHmmss)" --secret id=dagshub_username,env=MLFLOW_TRACKING_USERNAME --secret id=dagshub_token,env=MLFLOW_TRACKING_PASSWORD -t mlflow-fastapi-local .
docker run --rm -p 8000:8000 mlflow-fastapi-local
```

## Submit

Submit the **public** GitHub repository URL, which contains both required
files: `.github/workflows/build.yml` and `Dockerfile`. Before submitting,
verify the Actions run shows a green check and that `POST /predict` responds.

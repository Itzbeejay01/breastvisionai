# BreastVisionAI

BreastVisionAI is a Django + React application for breast-imaging research. It runs as one Docker container and serves the compiled React interface, Django API, SQLite database, uploaded media, and TensorFlow prediction models.

## Stop the application

If the container is running:

```bash
docker stop breastvisionai
```

To remove the stopped container as well:

```bash
docker rm breastvisionai
```

The named `breastvisionai-data` and `breastvisionai-media` volumes are not removed by those commands. Remove them only when you intentionally want to erase the database, users, prediction history, and uploads:

```bash
docker volume rm breastvisionai-data breastvisionai-media
```

## Requirements

- Docker Desktop (Windows, macOS, or Linux)
- At least 6 GB of free memory recommended for TensorFlow and the model files

No local Python, Django, Node.js, npm, or virtual environment installation is required.

## Build the image

From the repository root:

```bash
docker build -t breastvisionai .
```

The build installs Python and frontend dependencies, compiles the React UI, collects static files, and validates all five Keras model files plus the GradientBoosting meta-learner. The build can take several minutes.

## Start the container

The first start runs Django migrations, creates or updates the administrator, validates the three PSO-selected models used by predictions, and then starts Gunicorn. Model loading can take a few minutes on CPU.

### PowerShell

```powershell
docker run --name breastvisionai `
  -p 10000:10000 `
  -v breastvisionai-data:/app/data `
  -v breastvisionai-media:/app/media `
  -e DJANGO_SECRET_KEY="secret-key-123@" `
  -e DJANGO_DEBUG="true" `
  -e ALLOWED_HOSTS="localhost,127.0.0.1" `
  -e DJANGO_SUPERUSER_USERNAME="Olumide A.T." `
  -e DJANGO_SUPERUSER_EMAIL="admin@breastvisionai.com" `
  -e DJANGO_SUPERUSER_PASSWORD="Password123@" `
  breastvisionai
```

### macOS/Linux shell

```bash
docker run --name breastvisionai \
  -p 10000:10000 \
  -v breastvisionai-data:/app/data \
  -v breastvisionai-media:/app/media \
  -e DJANGO_SECRET_KEY='secret-key-123@' \
  -e DJANGO_DEBUG='true' \
  -e ALLOWED_HOSTS='localhost,127.0.0.1' \
  -e DJANGO_SUPERUSER_USERNAME='Olumide A.T.' \
  -e DJANGO_SUPERUSER_EMAIL='admin@breastvisionai.com' \
  -e DJANGO_SUPERUSER_PASSWORD='Password123@' \
  breastvisionai
```

Open <http://localhost:10000/> after the startup messages show that Gunicorn has started. Sign in with the administrator values supplied to `docker run`.

## Check status and logs

In another terminal:

```bash
docker ps
docker logs -f breastvisionai
curl http://localhost:10000/api/health/
```

The health response should be `{"status":"ok"}`. A successful startup log includes messages for migrations, administrator verification, model artifact validation, and Gunicorn.

If the container exits, inspect the complete startup error:

```bash
docker logs breastvisionai
```

Common causes are insufficient Docker memory, a missing model artifact in the build context, or missing `DJANGO_SUPERUSER_USERNAME`/`DJANGO_SUPERUSER_PASSWORD` values.

## Restart and update

Restart an existing container without rebuilding:

```bash
docker restart breastvisionai
```

After changing source code or model files, rebuild and replace the container:

```bash
docker build -t breastvisionai .
docker stop breastvisionai
docker rm breastvisionai
```

Then run the start command again. The named volumes preserve the SQLite database and uploaded media across container replacement.

## Model behavior

The image contains five Keras models: EfficientNet, DenseNet, ResNet, VGG16, and Xception. The configured PSO ensemble uses EfficientNet, ResNet, and VGG16 for predictions. Docker validates all five during the image build and validates the selected three plus the persisted GradientBoosting meta-learner at each container start.

The meta-learner is serialized with scikit-learn 1.6.1, so that version is pinned in `requirements.txt`. Changing it may make the `.joblib` artifact unreadable.

## Data and security

- SQLite and uploads are persisted in the named Docker volumes shown above.
- Replace the example secret key and administrator password before exposing the service outside a local machine.
- `DJANGO_DEBUG=true` is useful for local troubleshooting but should be `false` for deployment.
- This is a research tool and is not a substitute for clinical diagnosis.

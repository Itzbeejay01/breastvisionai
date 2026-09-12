# BreastVisionAI

An academically-structured and production-ready Django + TensorFlow/Keras project for breast imaging research and deployment.

## Features
- Django backend with Django REST Framework, CORS, and WhiteNoise
- TensorFlow (Keras included), scikit-learn, and medical imaging stack (SimpleITK, NiBabel, OpenCV)
- Organized datasets, models, results, adversarial experiments, and modular `src` code
- GPU auto-detection via TensorFlow with graceful CPU fallback

## Prerequisites
- Python 3.10 or 3.11 recommended
- macOS or Linux. For CUDA GPU, ensure NVIDIA drivers + CUDA/cuDNN compatible with your TensorFlow version

## Quickstart

## Deploy the Streamlit app for free

This repository includes a standalone Streamlit interface in
[`streamlit_app.py`](streamlit_app.py). It reuses the existing trained models
and does not require Django or the React frontend.

1. Push this repository to a GitHub repository. Keep the `models/`,
   `PSO_Result/`, and `Stacking_Result/` directories in the repository; the
   Streamlit app needs them at runtime.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/), sign in with
   GitHub, and choose **Create app**.
3. Select your repository and branch, set the main file to
   `streamlit_app.py`, and deploy. The root `requirements.txt` is installed
   automatically.
4. In **Advanced settings**, choose Python 3.11. TensorFlow 2.16.2 is pinned
   in `requirements.txt` and requires a supported Python version; Python 3.14
   cannot install it. Streamlit requires deleting and redeploying an app if
   you need to change its Python version later.

The first start can take several minutes because TensorFlow installs and the
three Keras models are loaded. Subsequent visits reuse the cached models while
the app instance is running. Streamlit Community Cloud apps may sleep when
unused, so a cold start is expected.

Run the same app locally with:

```bash
streamlit run streamlit_app.py
```

The free deployment is suitable for demonstrations and research support. It
is not a substitute for clinical diagnosis, and uploaded images should not be
treated as permanently stored medical records.

## Deploy the original React/Django platform on Render

For the original interface, use the included `Dockerfile` and `render.yaml`,
not `streamlit_app.py`. This deploys the React/Vite frontend, Django API, and
TensorFlow models as one Render Web Service. Django serves the compiled React
application from `breastvisionai-ui/dist`.

In Render, choose **New → Blueprint**, connect this repository, and apply the
`render.yaml` configuration. Render will build the Docker image, build the
React frontend, run migrations, and start Django with Gunicorn.

The deployment administrator is created from the `DJANGO_SUPERUSER_USERNAME`,
`DJANGO_SUPERUSER_EMAIL`, and `DJANGO_SUPERUSER_PASSWORD` environment
variables in `render.yaml`. Change these values in Render before using the
application. The generated password is available in the service environment
settings; it is not copied from the local `db.sqlite3`.

The free Render service is intended for demonstrations and may sleep after
inactivity. SQLite storage is also ephemeral on free instances, so use a
managed PostgreSQL database before relying on saved users, uploads, or
prediction history.

### 1) Setup (first time only)

```bash
# From project root
cd /Applications/MAMP/htdocs/BreastVisionAI

# Backend: create and activate a virtual environment, install dependencies
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Backend: apply migrations
python manage.py migrate

# Create the administrator account used to sign in to the React application
python manage.py createsuperuser

# The command prompts for username, email, and password. Use that account at
# the React UI login screen.

# Frontend: install UI dependencies (run in a separate terminal, or after deactivating the venv)
cd breastvisionai-ui
npm install
cd ..
```

> **Note:** `npm install` skips dev dependencies when `NODE_ENV=production` is set in
> your shell. If the UI build fails with "vite: not found", run
> `NODE_ENV=development npm install`.

### Large model and NumPy files (Git LFS)

The trained Keras/weights files and `.npy` artifacts are configured for Git
Large File Storage (LFS). GitHub blocks regular Git files larger than 100 MiB
and recommends LFS for binary files. Install and initialize it before staging
these files:

```bash
# macOS with Homebrew
brew install git-lfs
git lfs install

# Confirm the configured files are handled by LFS
git check-attr filter -- models/*.keras model_training/checkpoints/*.h5 results/*.npy

# Stage and upload new files normally
git add .gitattributes models model_training/checkpoints results pso_cache Stacking_Result
git commit -m "Store model artifacts with Git LFS"
git push origin main
```

If this repository has already been pushed and the model files are present in
regular Git history, migrate those existing files once before pushing the new
history:

```bash
git lfs migrate import --include="*.keras,*.h5,*.npy,*.npz,*.joblib,*.pkl"
git push --force-with-lease origin main
git lfs push --all origin
```

The migration rewrites commit history, so coordinate it with anyone else who
has cloned the repository. GitHub Free currently includes 10 GiB of LFS
storage and 10 GiB of monthly LFS bandwidth; additional usage may require a
paid data pack or plan.

### 2) Run backend and frontend together (one command)

From the project root:

```bash
./dev.sh
```

This starts:

- Django API at http://localhost:8000/
- Vite dev server at http://localhost:3000/ (proxies `/api` and `/media` to the backend)

Open http://localhost:8000/ or http://localhost:3000/ for the React landing
page, then "Access Platform" to sign in with the administrator account and
reach the dashboard (http://localhost:8000/dashboard). Authentication is
required before any image is uploaded or processed. Port 8000 serves the
compiled React UI through Django; port 3000 is the Vite development UI with
hot reload. Django remains behind the scenes for secure session authentication
and data services.

Press `Ctrl+C` to stop both.

### 3) Or run them separately

**Backend only:**

```bash
cd /Applications/MAMP/htdocs/BreastVisionAI
source .venv/bin/activate
python manage.py runserver 0.0.0.0:8000
```

**Frontend only:**

```bash
cd breastvisionai-ui
NODE_ENV=development npm run dev
```

### 4) Verify TensorFlow device detection

```bash
python main.py
```

## Project Structure
```
BreastVisionAI/
├── adversarial/
│   └── PGD/
├── datasets/
│   ├── INBreast/
│   ├── CBIS-DDSM/
│   └── MRI/
├── models/
│   ├── VGG16/
│   ├── ResNet50/
│   ├── EfficientNet/
│   ├── DenseNet/
│   └── Inception/
├── results/
│   ├── metrics/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── logs/
├── src/
│   ├── preprocessing/
│   ├── training/
│   ├── evaluation/
│   └── ensemble/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Notes
- TensorFlow automatically uses available GPUs. If none are found, it falls back to CPU without error. See `main.py`.
- For production, use `WhiteNoise` for static files and a WSGI server (e.g., gunicorn or uwsgi) behind a reverse proxy. Environment variables are loaded via `python-dotenv`.

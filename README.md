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

```bash
# From project root
cd /Applications/MAMP/htdocs/BreastVisionAI

# 1) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# 2) Install dependencies
pip install -r requirements.txt

# 3) Initialize Django project and API app (first-time only)
django-admin startproject breastvisionai .
python manage.py startapp api

# 4) Apply migrations and run server
python manage.py migrate
python manage.py runserver 0.0.0.0:8000

# 5) Verify TensorFlow device detection
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

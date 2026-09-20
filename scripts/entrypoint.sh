#!/bin/sh
set -eu

echo "Applying Django migrations..."
python manage.py migrate --noinput

echo "Creating/verifying deployment administrator..."
python scripts/create_superuser.py

echo "Validating prediction model artifacts..."
python scripts/verify_runtime.py

echo "Starting Gunicorn on port ${PORT:-10000}..."
exec python -m gunicorn breastvisionai.wsgi:application \
  --bind "0.0.0.0:${PORT:-10000}" \
  --workers "${GUNICORN_WORKERS:-1}" \
  --timeout "${GUNICORN_TIMEOUT:-300}"

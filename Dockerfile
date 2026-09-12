FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime libraries for TensorFlow/OpenCV, plus Node for the Vite build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY breastvisionai-ui/package*.json ./breastvisionai-ui/
RUN cd breastvisionai-ui && npm ci --include=dev

COPY . .

RUN cd breastvisionai-ui && npm run build
RUN python manage.py collectstatic --noinput
RUN mkdir -p media
# Do not bake the local development database into the production image.
RUN rm -f db.sqlite3

EXPOSE 10000

CMD ["sh", "-c", "python manage.py migrate && python scripts/create_superuser.py && python -m gunicorn breastvisionai.wsgi:application --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 300"]

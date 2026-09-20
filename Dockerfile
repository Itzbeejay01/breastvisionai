FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /app/breastvisionai-ui

COPY breastvisionai-ui/package*.json ./
RUN --mount=type=cache,target=/root/.npm,sharing=locked \
    npm ci --include=dev

COPY breastvisionai-ui/ ./
RUN npm run build


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime libraries for TensorFlow/OpenCV. The frontend is built in the
# separate Node stage above, so Node/npm are not needed in the final image.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    sed -i \
    -e 's|http://deb.debian.org|https://deb.debian.org|g' \
    -e 's|http://security.debian.org|https://security.debian.org|g' \
    /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && apt-get clean

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .
# .dockerignore excludes the local dist directory, so copy the build artifact
# explicitly from the frontend stage after copying the application source.
COPY --from=frontend-builder /app/breastvisionai-ui/dist ./breastvisionai-ui/dist

RUN python manage.py collectstatic --noinput
RUN mkdir -p /app/data media
# Fail the image build if any required model artifact or its meta-learner is
# missing/corrupt. The container entrypoint repeats the required-model check.
RUN python scripts/verify_runtime.py --all
# Do not bake the local development database into the production image.
RUN rm -f db.sqlite3

EXPOSE 10000

ENV DJANGO_DATA_DIR=/app/data \
    DJANGO_MEDIA_ROOT=/app/media

RUN chmod +x scripts/entrypoint.sh
ENTRYPOINT ["/app/scripts/entrypoint.sh"]

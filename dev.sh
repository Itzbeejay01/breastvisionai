#!/usr/bin/env bash
# Start the Django backend and Vite frontend together.
# Usage: ./dev.sh
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Keep local development aligned with the container startup sequence. Set
# DJANGO_SUPERUSER_USERNAME and DJANGO_SUPERUSER_PASSWORD before running this
# script (for example by exporting them from a local, untracked .env file).
cd "$ROOT_DIR"
"$PYTHON_BIN" manage.py migrate
"$PYTHON_BIN" scripts/create_superuser.py

# Kill both processes on Ctrl+C / exit
trap 'kill 0' EXIT INT TERM

echo "Starting Django backend on :$BACKEND_PORT ..."
(cd "$ROOT_DIR" && "$PYTHON_BIN" manage.py runserver 0.0.0.0:$BACKEND_PORT) &
BACKEND_PID=$!

echo "Starting Vite frontend on :$FRONTEND_PORT ..."
(cd "$ROOT_DIR/breastvisionai-ui" && NODE_ENV=development npm run dev -- --port $FRONTEND_PORT) &
FRONTEND_PID=$!

echo
echo "Backend:  http://localhost:$BACKEND_PORT/"
echo "Frontend: http://localhost:$FRONTEND_PORT/  (proxies /api and /media to backend)"
echo "Press Ctrl+C to stop both."
echo

wait

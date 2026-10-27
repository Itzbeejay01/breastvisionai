#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="$ROOT_DIR/breastvisionai-ui"
CONTAINER_NAME="${CONTAINER_NAME:-breastvisionai}"

if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container '$CONTAINER_NAME' does not exist."
  exit 1
fi

if [ ! -d "$UI_DIR/node_modules" ]; then
  echo "Frontend dependencies are not installed locally."
  echo "Run npm install once in $UI_DIR, then rerun this script."
  exit 1
fi

echo "Building the frontend with the existing local dependencies..."
(cd "$UI_DIR" && npm run build)

echo "Copying only the compiled frontend into '$CONTAINER_NAME'..."
docker cp "$UI_DIR/dist/." "$CONTAINER_NAME:/app/breastvisionai-ui/dist/"

echo "Frontend updated. No Python packages, models, database, or uploads were changed."
echo "Refresh the browser with cache bypass (Cmd+Shift+R or Ctrl+Shift+R)."

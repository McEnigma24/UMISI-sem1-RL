#!/usr/bin/env bash
# shellcheck disable=SC1091

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(cd "$SCRIPT_DIR/.." && pwd)/config"

clear; set +euo pipefail
docker build -f "$DOCKERFILE" --build-arg "PY_TAG=${PY_TAG}" --build-arg "PY_VERSION=${PY_VERSION}" -t "$DOCKER_IMAGE" "$L5_ROOT"
docker image prune -f

echo "DOCKER_GPUS: $DOCKER_GPUS"
sleep 1

clear; set +euo pipefail
docker run --rm -it \
  $DOCKER_RUN_FLAGS \
  -v "$L5_ROOT:/workspace" \
  -w /workspace \
  -p 6006:6006 \
  -e MUJOCO_GL="${MUJOCO_GL:-egl}" \
  "$DOCKER_IMAGE" \
  bash

docker container prune -f
clear

#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$(cd "$SCRIPT_DIR/.." && pwd)/config"

clear
docker build -f "$DOCKERFILE" -t "$DOCKER_IMAGE" "$L5_ROOT"
docker image prune -f

clear
docker run --rm -it \
  --gpus all \
  -v "$L5_ROOT:/workspace" \
  -w /workspace \
  -p 6006:6006 \
  -e MUJOCO_GL="${MUJOCO_GL:-egl}" \
  "$DOCKER_IMAGE" \
  bash

docker container prune -f
clear

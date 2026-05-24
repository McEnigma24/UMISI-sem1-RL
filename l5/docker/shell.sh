#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/config"

clear 2>/dev/null || true

# Interaktywna powłoka bez entrypointu (xvfb) — jak ``enter_dev_container.sh``.
docker run --rm -it \
  -v "$L5_ROOT:/workspace" \
  -w /workspace \
  -e MUJOCO_GL="${MUJOCO_GL:-egl}" \
  --entrypoint bash \
  "$DOCKER_IMAGE"

docker container prune -f >/dev/null 2>&1 || true
clear 2>/dev/null || true

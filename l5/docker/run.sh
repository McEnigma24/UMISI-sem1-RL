#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/config"

if [[ $# -eq 0 ]]; then
  set -- python train.py
fi

clear 2>/dev/null || true

docker run --rm -it \
  -v "$L5_ROOT:/workspace" \
  -w /workspace \
  -e MUJOCO_GL="${MUJOCO_GL:-egl}" \
  -e SKIP_XVFB="${SKIP_XVFB:-0}" \
  "$DOCKER_IMAGE" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/config"

clear 2>/dev/null || true

docker run --rm -it \
  -v "$L5_ROOT:/workspace" \
  -w /workspace \
  -p "${TENSORBOARD_PORT:-6006}:6006" \
  --entrypoint tensorboard \
  "$DOCKER_IMAGE" \
  --logdir runs --bind_all "$@"

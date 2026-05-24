#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/config"

clear 2>/dev/null || true

docker build -f "$DOCKERFILE" -t "$DOCKER_IMAGE" "$L5_ROOT"
docker image prune -f

clear 2>/dev/null || true
echo "✅ Obraz: $DOCKER_IMAGE"

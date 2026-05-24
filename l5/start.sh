#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$(cd "$SCRIPT_DIR/.." && pwd)/config"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"



python train.py

# exec xvfb-run -a -s "-screen 0 1920x1080x24" python train.py "$@"

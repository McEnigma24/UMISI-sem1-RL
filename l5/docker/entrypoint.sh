#!/usr/bin/env bash
set -euo pipefail

# Trening kończy się ewaluacją z ``render_mode="human"`` (GLFW) — bez wyświetlacza
# potrzebny jest wirtualny framebuffer (xvfb).
if [[ "${SKIP_XVFB:-0}" == "1" ]]; then
  exec "$@"
fi

exec xvfb-run -a -s "-screen 0 1920x1080x24" "$@"

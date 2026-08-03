#!/usr/bin/env bash
# Local dev launcher. `tools/package.py` writes the submission's own run.sh.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
PY="${BOT_PYTHON:-$(command -v python3 || command -v python)}"
exec "$PY" -u -m bot.main

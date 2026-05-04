#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -d ".venv_local" ]]; then
    python3 -m venv .venv_local
  fi
  # shellcheck disable=SC1091
  source .venv_local/bin/activate
fi

python3 - <<'PY'
import importlib.util
import subprocess
import sys

required = {
    "numpy": "numpy",
    "yaml": "pyyaml",
    "zmq": "pyzmq",
    "fastapi": "fastapi",
}

missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing missing dependencies:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("All required test dependencies are already installed.")
PY

PYTHONPATH=src python3 -m unittest discover -s tests -v

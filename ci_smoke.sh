#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  VENV_DIR="$(mktemp -d "${TMPDIR:-/tmp}/aegis-lab-ci.XXXXXX")"
  trap 'rm -rf "$VENV_DIR"' EXIT
  python3 -m venv "$VENV_DIR"
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check

python - <<'PY'
import aegis_lab
import aegis_lab.quantization.calibration
import aegis_lab.quantization.exporter
import aegis_lab.scheduler.engine
import aegis_lab.hardware.thermal
print("package import smoke: ok")
PY

python -m unittest \
  tests.unit.test_quantization_unit \
  tests.unit.test_thermal_unit \
  -v

python -m unittest discover -s tests/unit -p 'test_neurosurgery_*.py' -v

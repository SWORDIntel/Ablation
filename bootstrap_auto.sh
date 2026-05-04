#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --tests)
      RUN_TESTS=1
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: $0 [--tests]"
      exit 2
      ;;
  esac
done

if [[ ! -d ".venv_local" ]]; then
  python3 -m venv .venv_local
fi

# shellcheck disable=SC1091
source .venv_local/bin/activate

python3 -m pip install --upgrade pip >/dev/null
python3 -m pip install -e ".[dev]" >/dev/null
python3 -m pip install numpy pyyaml pyzmq fastapi >/dev/null

echo "Bootstrap auto-install complete."
echo "Environment: $ROOT_DIR/.venv_local"

if [[ "$RUN_TESTS" -eq 1 ]]; then
  ./run_tests_autoinstall.sh
else
  echo "Run tests with: ./run_tests_autoinstall.sh"
fi

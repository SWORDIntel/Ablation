#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

case "${1:-bootstrap}" in
  bootstrap)
    shift || true
    ./bootstrap_auto.sh "$@"
    ;;
  test)
    shift || true
    ./bootstrap_auto.sh
    ./run_tests_autoinstall.sh "$@"
    ;;
  *)
    echo "Usage: $0 [bootstrap|test] [--tests]"
    echo "Examples:"
    echo "  $0 bootstrap"
    echo "  $0 bootstrap --tests"
    echo "  $0 test"
    exit 2
    ;;
esac

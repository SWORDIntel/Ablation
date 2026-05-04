#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ARCHIVE_DIR="$ROOT_DIR/.runtime/openvino-2022.3-archive/l_openvino_toolkit_ubuntu20_2022.3.2.9279.e2c7e4d7b4d_x86_64"
COMPAT_LIB_DIR="$ROOT_DIR/.runtime/compat-libs/root/usr/lib/x86_64-linux-gnu"
PYTHON_BIN="$ROOT_DIR/.venvs/openvino2022/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python runtime: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$ARCHIVE_DIR" ]]; then
  echo "Missing OpenVINO archive runtime: $ARCHIVE_DIR" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$COMPAT_LIB_DIR:$ARCHIVE_DIR/runtime/lib/intel64:$ARCHIVE_DIR/runtime/3rdparty/hddl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ARCHIVE_DIR/python/python3.10:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ROOT_DIR/.venvs/openvino2022/bin${PATH:+:$PATH}"

if [[ $# -eq 0 ]]; then
  exec "$PYTHON_BIN"
fi

if [[ "$1" == "--shell" ]]; then
  exec bash --noprofile --norc
fi

exec "$@"

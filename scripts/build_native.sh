#!/bin/bash
set -e
echo "Building native components with stable ABI override..."
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cd src/native/vpu_core
cargo build --release
echo "Native build complete."

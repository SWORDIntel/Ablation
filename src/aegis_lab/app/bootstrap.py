import os
import sys
import subprocess
from pathlib import Path

LIBZE_PATH = "/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1"

def check_level_zero():
    """Checks if the Intel Level Zero GPU driver is installed."""
    return os.path.exists(LIBZE_PATH)

def offer_installation():
    """Provides instructions or helpers to install Level Zero."""
    print("Intel Level Zero GPU driver (libze_intel_gpu.so.1) is missing.")
    print("\nOptions for installation:")
    print("1) Install via APT (Ubuntu/Debian):")
    print("   sudo apt update && sudo apt install -y intel-level-zero-gpu intel-opencl-icd")
    print("\n2) Build from source (oneapi-src/level-zero):")
    print("   git clone https://github.com/oneapi-src/level-zero.git")
    print("   cd level-zero && mkdir build && cd build")
    print("   cmake .. && cmake --build . --config Release --target install")
    print("\nNote: You may need to add the Intel Graphics package repository first.")

def bootstrap():
    """Main bootstrap entry point."""
    if not check_level_zero():
        offer_installation()
        return False
    return True

if __name__ == "__main__":
    if bootstrap():
        print("Intel Level Zero environment verified.")
    else:
        sys.exit(1)

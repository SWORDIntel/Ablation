import hashlib
import os
import ctypes
from pathlib import Path
from typing import Union

# Try to load QIHSE for hardware-accelerated hashing
_QIHSE_LIB = None
try:
    # Look for libqihse in standard locations
    _possible_paths = [
        os.path.abspath("QIHSE/qihse/libqihse.so"),
        os.path.abspath("../QIHSE/qihse/libqihse.so"),
        "/usr/local/lib/libqihse.so"
    ]
    for p in _possible_paths:
        if os.path.exists(p):
            _QIHSE_LIB = ctypes.CDLL(p)
            _QIHSE_LIB.qihse_intel_hw_hash.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int
            ]
            _QIHSE_LIB.qihse_intel_hw_hash.restype = ctypes.c_int
            break
except Exception:
    pass

def sha256_optimized(data: bytes) -> str:
    """Computes SHA256 hash using Intel hardware acceleration if available."""
    if _QIHSE_LIB:
        try:
            hash_out = (ctypes.c_byte * 32)()
            res = _QIHSE_LIB.qihse_intel_hw_hash(
                data, len(data), ctypes.byref(hash_out), 0 # 0 = SHA-256
            )
            if res == 0:
                return "".join(f"{b & 0xFF:02x}" for b in hash_out)
        except Exception:
            pass
    return hashlib.sha256(data).hexdigest()

def hash_file(file_path: Union[Path, str], chunk_size: int = 1024 * 1024) -> str:
    """Computes the SHA256 hash of a file with large chunks for SIMD optimization."""
    # Increase chunk_size to 1MB to leverage hardware prefetching/SIMD better
    sha256 = hashlib.sha256()
    # If we have QIHSE, we can't easily use it for streaming without a more complex API,
    # so we'll use hashlib for streaming but we ensured QIHSE is available for smaller blobs.
    # Note: Modern hashlib/OpenSSL uses AVX-512 for SHA256 automatically if present.
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def hash_directory(dir_path: Union[Path, str]) -> str:
    """
    Computes a deterministic SHA256 hash for a directory's contents.
    Files are sorted by relative path to ensure consistency.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{dir_path} is not a directory.")

    sha256 = hashlib.sha256()
    
    # Sort files to ensure deterministic hashing
    for p in sorted(dir_path.rglob("*")):
        if p.is_file():
            # Hash the relative path
            rel_path = p.relative_to(dir_path).as_posix()
            sha256.update(rel_path.encode('utf-8'))
            
            # Hash the file content
            file_hash = hash_file(p)
            sha256.update(file_hash.encode('utf-8'))
            
    return sha256.hexdigest()

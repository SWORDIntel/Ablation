import os
import json
from pathlib import Path
from typing import Union, Dict, Any
from .hashing import hash_directory

def get_run_directory(base_dir: Union[str, Path], project_id: str, job_id: str) -> Path:
    """Returns the standardized run directory path."""
    return Path(base_dir) / "runs" / project_id / job_id

def ensure_run_layout(run_dir: Union[str, Path]) -> None:
    """Creates the standard directory layout for a run."""
    run_dir = Path(run_dir)
    directories = [
        "manifests",
        "configs",
        "logs/workers",
        "artifacts/source_refs",
        "artifacts/activations",
        "artifacts/projections",
        "artifacts/atoms",
        "artifacts/deltas",
        "artifacts/eval",
        "artifacts/calibration",
        "artifacts/quant",
        "artifacts/reports",
        "exports"
    ]
    for d in directories:
        (run_dir / d).mkdir(parents=True, exist_ok=True)

def atomic_promote_export(
    temp_export_dir: Union[str, Path],
    final_export_path: Union[str, Path],
    manifest_data: Dict[str, Any]
) -> str:
    """
    Executes the atomic promotion algorithm defined in Section 5.3.
    1. hash all contents (assumes files are written to temp directory)
    2. write final manifest
    3. fsync temp contents where possible
    4. atomically rename temp export directory to final export path
    """
    temp_export_dir = Path(temp_export_dir)
    final_export_path = Path(final_export_path)

    if not temp_export_dir.exists() or not temp_export_dir.is_dir():
        raise ValueError(f"Temporary export directory {temp_export_dir} does not exist or is not a directory.")

    # 1. Hash all contents
    dir_hash = hash_directory(temp_export_dir)
    
    # 2. Write final manifest
    manifest_path = temp_export_dir / "final_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # 3. Fsync temp contents where possible
    if hasattr(os, 'O_DIRECTORY'):
        try:
            fd = os.open(str(temp_export_dir), os.O_RDONLY | os.O_DIRECTORY)
            os.fsync(fd)
            os.close(fd)
        except OSError:
            pass

    # 4 & 5. Atomically rename temp export directory to final export path
    final_export_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(temp_export_dir), str(final_export_path))

    return dir_hash

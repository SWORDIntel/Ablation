import os
import shutil
from pathlib import Path
from typing import Union, Optional, Dict, Any

from .hashing import hash_file
from framewerx.aegis_lab.state.db import AegisState

class ArtifactStore:
    """
    Content-addressed artifact storage using SHA256.
    Integrated with unified QIHSE backend via AegisState for metadata persistence.
    """
    def __init__(self, base_dir: Union[str, Path], state: Optional[AegisState] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.state = state

    def _get_storage_path(self, sha256_hash: str) -> Path:
        """Determines the path in the store based on the hash."""
        # Using a 2-level directory structure based on the hash prefix
        prefix1 = sha256_hash[:2]
        prefix2 = sha256_hash[2:4]
        return self.base_dir / prefix1 / prefix2 / sha256_hash

    def put_file(self, source_path: Union[str, Path], move: bool = False, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Stores a file in the content-addressed store.
        Returns the SHA256 hash and registers it in the unified state.
        """
        source_path = Path(source_path)
        file_hash = hash_file(source_path)
        target_path = self._get_storage_path(file_hash)

        if not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(str(source_path), str(target_path))
            else:
                shutil.copy2(str(source_path), str(target_path))
        elif move:
            os.remove(source_path)

        # Register artifact metadata in unified QIHSE state if available
        if self.state:
            meta = metadata or {}
            meta.update({
                "original_path": str(source_path),
                "file_size": target_path.stat().st_size,
                "content_hash": file_hash
            })
            self.state.register_artifact(file_hash, meta)

        return file_hash

    def get_file(self, sha256_hash: str, destination_path: Union[str, Path]) -> None:
        """
        Retrieves a file from the store to the given destination.
        """
        storage_path = self._get_storage_path(sha256_hash)
        if not storage_path.exists():
            raise FileNotFoundError(f"Artifact with hash {sha256_hash} not found in store.")
        
        destination_path = Path(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(storage_path), str(destination_path))

    def exists(self, sha256_hash: str) -> bool:
        """Checks if an artifact exists in the store."""
        # Check filesystem first
        fs_exists = self._get_storage_path(sha256_hash).exists()
        if fs_exists:
            return True
            
        # Optional: check QIHSE state if FS fails (e.g. for small artifacts stored directly)
        if self.state:
            return self.state.get_artifact_metadata(sha256_hash) is not None
        return False
        
    def get_metadata(self, sha256_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves artifact metadata from the unified state."""
        if self.state:
            return self.state.get_artifact_metadata(sha256_hash)
        return None

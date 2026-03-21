import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

from aegis_lab.state.db import AegisState
from aegis_lab.artifacts.store import ArtifactStore
from aegis_lab.state.qihse_wrapper import QIHSE, QihseVectorDBBackend

class AtomRegistry:
    """
    Immutable storage and indexing of behavioral atoms.
    Fully integrated with the unified QIHSE backend via AegisState.
    """
    
    def __init__(self, state: AegisState, artifact_store: ArtifactStore):
        self.state = state
        self.artifact_store = artifact_store
        # Unified QIHSE is now managed by AegisState

    def register_atom(
        self,
        atom_id: str,
        atom_type: str,
        content_hash: str,
        architecture_scope: str,
        prompt_suite_hash: str,
        extraction_config_hash: str,
        validation_status: str,
        artifact_id: str,
        semantic_vector: List[float],
        parent_hash: Optional[str] = None,
        confidence_score: Optional[float] = None,
        created_by_job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Registers an immutable atom record and indexes it for semantic search.
        The actual tensor artifact should already be saved in ArtifactStore.
        """
        now = datetime.utcnow().isoformat()
        
        atom_record = {
            "atom_id": atom_id,
            "atom_type": atom_type,
            "content_hash": content_hash,
            "parent_hash": parent_hash,
            "architecture_scope": architecture_scope,
            "prompt_suite_hash": prompt_suite_hash,
            "extraction_config_hash": extraction_config_hash,
            "validation_status": validation_status,
            "confidence_score": confidence_score,
            "artifact_id": artifact_id,
            "created_by_job_id": created_by_job_id,
            "created_at": now
        }
        
        # Save to durable state using the unified registry method
        self.state.register_atom(atom_record, semantic_vector=semantic_vector)
        
        return atom_record

    def search_atoms(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Semantic search of atoms using the unified QIHSE backend.
        """
        return self.state.search_atoms(query_vector, top_k=top_k)

    def get_atom_record(self, atom_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves atom metadata from durable state.
        """
        return self.state.get_atom(atom_id)

    def retrieve_atom_artifact(self, atom_id: str, destination_path: str) -> None:
        """
        Retrieves the actual atom tensor/weights from the ArtifactStore to the given path.
        """
        atom = self.get_atom_record(atom_id)
        if not atom:
            raise ValueError(f"Atom {atom_id} not found in registry.")
        
        artifact_id = atom["artifact_id"]
        # ArtifactStore uses sha256 hashes for artifact retrieval
        self.artifact_store.get_file(artifact_id, destination_path)

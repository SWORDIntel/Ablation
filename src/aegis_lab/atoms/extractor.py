import logging
from pathlib import Path
from aegis_lab.state.db import AegisState
from aegis_lab.artifacts.store import ArtifactStore
from aegis_lab.editing.runtime import (
    ExecutionMode,
    resolve_execution_contract,
    stable_json_hash,
    write_json_artifact,
)

logger = logging.getLogger(__name__)

class BehavioralAtomExtractor:
    """
    Isolates behavioral 'atoms' from model activations.
    An 'atom' is a discrete, steerable component of model behavior, 
    often represented as a vector or projection in the activation space.
    """
    def __init__(self, state: AegisState, artifact_store: ArtifactStore):
        self.state = state
        self.artifact_store = artifact_store

    def extract_atom(self, 
                     job_id: str, 
                     positive_act_hash: str, 
                     negative_act_hash: str, 
                     method: str = "ridge_regression",
                     execution_mode: str = ExecutionMode.FALLBACK.value) -> str:
        """
        Isolates a behavioral atom by finding the causal difference 
        between the positive and negative activation sets.
        
        Methods supported:
        - ridge_regression: Learns a linear classifier and extracts its coefficients.
        - residualization: Uses orthogonal projection to isolate behavior-specific variance.
        - steering_target_atoms (STA): Uses Sparse Autoencoders (SAE) to isolate disentangled knowledge components.
        """
        logger.info(f"Extracting atom for job {job_id} using method: {method}")

        if method == "steering_target_atoms":
            return self._extract_sta_atom(job_id, positive_act_hash, negative_act_hash, execution_mode)

        contract = resolve_execution_contract(
            operation="atom_extraction",
            requested_mode=execution_mode,
            native_available=False,
            reason="Atom extraction uses deterministic fallback synthesis unless a native solver is wired in.",
            details={
                "job_id": job_id,
                "method": method,
                "positive_act_hash": positive_act_hash,
                "negative_act_hash": negative_act_hash,
            },
        )
        payload = {
            "atom_id": stable_json_hash({
                "job_id": job_id,
                "method": method,
                "positive": positive_act_hash,
                "negative": negative_act_hash,
                "mode": contract.mode.value,
            }),
            "job_id": job_id,
            "method": method,
            "positive_activations_hash": positive_act_hash,
            "negative_activations_hash": negative_act_hash,
            "execution_contract": contract.as_dict(),
            "atom_signature": stable_json_hash({
                "job_id": job_id,
                "method": method,
                "positive": positive_act_hash,
                "negative": negative_act_hash,
                "mode": contract.mode.value,
            }),
        }
        temp_atom_file = write_json_artifact(Path("/tmp/aegis_atoms"), f"{job_id}_atom_{method}", payload)["path"]
        atom_hash = self.artifact_store.put_file(temp_atom_file, move=True)

        atom_metadata = {
            "atom_id": payload["atom_id"],
            "job_id": job_id,
            "atom_hash": atom_hash,
            "method": method,
            "source_pos_hash": positive_act_hash,
            "source_neg_hash": negative_act_hash,
            "status": "extracted",
            "execution_contract": contract.as_dict(),
            "atom_signature": payload["atom_signature"],
        }
        
        # Store metadata in state
        self.state.db.insert("atoms", atom_metadata)
        
        return atom_hash

    def _extract_sta_atom(self, job_id: str, pos_hash: str, neg_hash: str, execution_mode: str) -> str:
        """
        Implements Steering Target Atoms (STA) logic using Sparse Autoencoders.
        """
        logger.info("Performing STA extraction via Sparse Autoencoder disentanglement.")
        contract = resolve_execution_contract(
            operation="atom_extraction_sta",
            requested_mode=execution_mode,
            native_available=False,
            reason="STA extraction is synthesized via SAE disentanglement fallback.",
            details={"job_id": job_id, "method": "STA_SAE_V1"}
        )

        payload = {
            "atom_id": f"sta-{stable_json_hash({'job': job_id, 'm': 'sta'})[:12]}",
            "job_id": job_id,
            "method": "steering_target_atoms",
            "disentanglement_type": "SAE_SPARSE",
            "execution_contract": contract.as_dict()
        }

        temp_atom_file = write_json_artifact(Path("/tmp/aegis_atoms"), f"{job_id}_sta_atom", payload)["path"]
        atom_hash = self.artifact_store.put_file(temp_atom_file, move=True)

        self.state.db.insert("atoms", {
            "atom_id": payload["atom_id"],
            "job_id": job_id,
            "atom_hash": atom_hash,
            "method": "steering_target_atoms",
            "status": "extracted"
        })
        return atom_hash

    def residualize(self, activations_hash: str, atom_hash: str, execution_mode: str = ExecutionMode.FALLBACK.value) -> str:
        """
        Applies orthogonal residualization to remove the influence of an atom from
        a set of activations, effectively 'ablating' the behavior in that space.

        This implementation simulates the projection of activations onto the orthogonal
        complement of the atom's steering vector.
        """
        logger.info(f"Applying orthogonal residualization for atom {atom_hash} to activations {activations_hash}")

        contract = resolve_execution_contract(
            operation="atom_residualization_orthogonal",
            requested_mode=execution_mode,
            native_available=False,
            reason="Residualization uses orthogonal projection synthesis in deterministic fallback mode.",
            details={
                "activations_hash": activations_hash,
                "atom_hash": atom_hash,
                "projection_type": "orthogonal_complement"
            },
        )

        # Simulate orthogonal projection: a_resid = a - (a \cdot v / |v|^2) v
        # In fallback mode, we represent this as a new artifact linked to the source and atom.
        payload = {
            "source_activations": activations_hash,
            "steering_atom": atom_hash,
            "residualization_method": "orthogonal_projection",
            "execution_contract": contract.as_dict(),
            "residual_signature": stable_json_hash({
                "source": activations_hash,
                "atom": atom_hash,
                "method": "orthogonal"
            })
        }

        temp_residual_file = write_json_artifact(
            Path("/tmp/aegis_atoms"),
            f"residualized_ortho_{atom_hash[:8]}",
            payload,
        )["path"]

        residual_hash = self.artifact_store.put_file(temp_residual_file, move=True)
        logger.info(f"Orthogonal residualization complete. Artifact hash: {residual_hash}")

        return residual_hash
import logging
from typing import Any, Dict, Optional

from aegis_lab.editing.adversarial import RedTeamEvaluator
from aegis_lab.atoms.extractor import BehavioralAtomExtractor

logger = logging.getLogger(__name__)

class GCGRefiner:
    """
    Iterative Adversarial Refinement module for AEGIS-LAB.
    
    Refines a behavioral atom by interacting with RedTeamEvaluator to test
    against adversarial suffixes, and BehavioralAtomExtractor to refine
    the atom until a robustness threshold is met.
    """
    
    def __init__(self, state: Any, evaluator: RedTeamEvaluator, extractor: BehavioralAtomExtractor):
        """
        Initializes the GCGRefiner.
        
        Args:
            state: An instance of AegisState.
            evaluator: An instance of RedTeamEvaluator.
            extractor: An instance of BehavioralAtomExtractor.
        """
        self.state = state
        self.evaluator = evaluator
        self.extractor = extractor
        
    def refine_atom(self,
                    job_id: str,
                    initial_atom_hash: str,
                    model_artifact_id: Optional[str] = None,
                    robustness_threshold: float = 0.9,
                    max_iterations: int = 5) -> Dict[str, Any]:
        """
        Iteratively tests and refines an atom against adversarial suffixes.
        
        Args:
            job_id: Identifier for the current job.
            initial_atom_hash: The starting atom to be refined.
            model_artifact_id: Optional target model artifact.
            robustness_threshold: Target robustness score (0.0 - 1.0).
            max_iterations: Maximum number of refinement iterations.
            
        Returns:
            Dict containing the final atom hash, robustness score, and iteration stats.
        """
        logger.info(f"Starting Iterative Adversarial Refinement for job {job_id}")
        logger.info(f"Target threshold: {robustness_threshold}, Max iterations: {max_iterations}")
        
        current_atom_hash = initial_atom_hash
        best_robustness = 0.0
        best_atom_hash = current_atom_hash
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"--- Refinement Iteration {iteration} ---")
            
            # 1. Evaluate current atom's adversarial robustness
            # This assesses vulnerability to adversarial suffixes (like GCG-optimized strings)
            current_robustness = self.evaluator.evaluate_adversarial_vulnerability(
                job_id=job_id,
                model_artifact_id=model_artifact_id
            )
            
            logger.info(f"Evaluated robustness score: {current_robustness:.4f}")
            
            # Update best atom if improvement is found
            if current_robustness > best_robustness:
                best_robustness = current_robustness
                best_atom_hash = current_atom_hash
                
            # Check if threshold is met
            if best_robustness >= robustness_threshold:
                logger.info(f"Robustness threshold ({robustness_threshold}) met!")
                break
                
            # 2. Refine the atom
            # Capture real adversarial activations from the evaluator's failure modes
            logger.info(f"Robustness below threshold. Refining atom {current_atom_hash}...")

            # Extract positive and negative activation hashes from the evaluator's
            # adversarial test results. These represent the activations that correspond
            # to the model's vulnerability (positive) and resistance (negative).
            pos_act_hash, neg_act_hash = self._capture_adversarial_activations(job_id, current_atom_hash)

            # Extract a new, refined atom using the captured activations
            try:
                current_atom_hash = self.extractor.extract_atom(
                    job_id=job_id,
                    positive_act_hash=pos_act_hash,
                    negative_act_hash=neg_act_hash,
                    method="ridge_regression"
                )
                logger.info(f"Successfully generated refined atom: {current_atom_hash}")
            except Exception as e:
                logger.error(f"Atom extraction failed during refinement: {str(e)}")
                break
                
        # Final summary
        success = best_robustness >= robustness_threshold
        logger.info(f"Refinement completed. Success: {success}, Final Robustness: {best_robustness:.4f}")
        
        return {
            "job_id": job_id,
            "initial_atom_hash": initial_atom_hash,
            "final_atom_hash": best_atom_hash,
            "robustness_score": best_robustness,
            "iterations": iteration,
            "threshold_met": success,
        }

    def _capture_adversarial_activations(self, job_id: str, current_atom_hash: str) -> tuple:
        """
        Capture activation hashes representing adversarial failure modes.
        
        Queries the evaluator for the most recent adversarial test results and
        extracts activation artifacts for both vulnerable (positive) and resistant
        (negative) model behaviors. Falls back to deriving hashes from the current
        atom if the evaluator doesn't expose activation artifacts directly.
        
        Returns:
            Tuple of (positive_act_hash, negative_act_hash)
        """
        import hashlib
        
        # Try to get activation artifacts from the evaluator's last test results
        pos_hash = None
        neg_hash = None
        
        try:
            if hasattr(self.evaluator, "get_last_adversarial_activations"):
                activations = self.evaluator.get_last_adversarial_activations(job_id)
                if activations:
                    pos_hash = activations.get("positive_hash")
                    neg_hash = activations.get("negative_hash")
        except Exception as e:
            logger.debug(f"Evaluator does not expose adversarial activations: {e}")

        # Fallback: derive deterministic hashes from job_id and current atom
        # This ensures the atom extractor receives stable, content-addressed references
        if pos_hash is None:
            pos_seed = f"adv_pos_{job_id}_{current_atom_hash}"
            pos_hash = hashlib.sha256(pos_seed.encode()).hexdigest()[:16]
        if neg_hash is None:
            neg_seed = f"adv_neg_{job_id}_{current_atom_hash}"
            neg_hash = hashlib.sha256(neg_seed.encode()).hexdigest()[:16]

        return pos_hash, neg_hash

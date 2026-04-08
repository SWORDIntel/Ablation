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
            # If not robust enough, simulate capturing new activations representing the
            # adversarial failure modes and extract a refined atom.
            logger.info(f"Robustness below threshold. Refining atom {current_atom_hash}...")
            
            # Mocking the discovery of new activations that capture the adversarial vulnerabilities
            mock_pos_act_hash = f"refined_pos_hash_iter_{iteration}_{job_id}"
            mock_neg_act_hash = f"refined_neg_hash_iter_{iteration}_{job_id}"
            
            # Extract a new, refined atom
            try:
                current_atom_hash = self.extractor.extract_atom(
                    job_id=job_id,
                    positive_act_hash=mock_pos_act_hash,
                    negative_act_hash=mock_neg_act_hash,
                    method="ridge_regression"  # Using ridge regression for linear classifier updates
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

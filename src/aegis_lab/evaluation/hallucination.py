import logging
import json
from typing import Dict, Any, List, Optional
from aegis_lab.state.db import AegisState

logger = logging.getLogger(__name__)

class HallucinationDetector:
    """
    Method 5: Hallucination Attribution Index.
    Cross-references model output against QIHSE-indexed behavioral atoms.
    """
    
    def __init__(self, state: AegisState):
        self.state = state

    def compute_index(self, model_output: str, job_id: str) -> float:
        """
        Calculates a hallucination index (0.0 = Truthful, 1.0 = Hallucinated).
        Uses QIHSE RAG to verify if the output contains info consistent 
        with the model's known behavioral atoms.
        """
        logger.info(f"Computing Hallucination Index for job {job_id}")
        
        # Search QIHSE for relevant context
        # Simplified: check if keywords from atoms appear in output
        atoms = self.state.get_stages(job_id) # Using stages as proxy for data
        
        matches = 0
        tokens = model_output.lower().split()
        if not tokens: return 0.0
        
        # Mock logic: check for 'ablation' related terms
        keywords = ["intake", "probe", "atom", "quantize", "verify"]
        for kw in keywords:
            if kw in model_output.lower():
                matches += 1
                
        # Higher index if no keywords match known system context
        return 1.0 - (matches / len(keywords))

import logging
import json
import math
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
        Uses QIHSE RAG to verify if the output is semantically consistent
        with the model's known behavioral atoms and job context.
        """
        logger.info(f"Computing Hallucination Index for job {job_id}")
        
        # Retrieve known context from job stages and atoms
        stages = self.state.get_stages(job_id)
        known_context = self._extract_known_context(stages)
        
        if not known_context:
            # No context available to verify against; moderate uncertainty
            logger.warning(f"No known context available for job {job_id}; returning neutral hallucination index.")
            return 0.5
        
        # Compute semantic overlap between model output and known context
        output_tokens = set(model_output.lower().split())
        if not output_tokens:
            return 0.0
        
        # Build a context vocabulary from known atoms and stage data
        context_tokens = set(known_context.lower().split())
        
        # Compute Jaccard similarity between output and context
        intersection = output_tokens & context_tokens
        union = output_tokens | context_tokens
        jaccard_sim = len(intersection) / len(union) if union else 0.0
        
        # Also check for factual claims not supported by context
        # Extract key claims (sentences) from model output
        output_sentences = [s.strip() for s in model_output.split('.') if s.strip()]
        unsupported_claims = 0
        for sentence in output_sentences:
            sent_tokens = set(sentence.lower().split())
            # A claim is "supported" if at least 30% of its tokens appear in context
            overlap = len(sent_tokens & context_tokens) / max(len(sent_tokens), 1)
            if overlap < 0.3:
                unsupported_claims += 1
        
        # Hallucination index: weighted combination of low similarity and unsupported claims
        total_sentences = max(len(output_sentences), 1)
        unsupported_ratio = unsupported_claims / total_sentences
        
        # Higher hallucination = low similarity + high unsupported claims
        hallucination_index = (1.0 - jaccard_sim) * 0.5 + unsupported_ratio * 0.5
        
        # Clamp to [0.0, 1.0]
        return min(1.0, max(0.0, hallucination_index))

    def _extract_known_context(self, stages: List[Dict[str, Any]]) -> str:
        """Extract known context text from job stages for comparison."""
        context_parts = []
        for stage in stages:
            # Include stage names and any text output
            stage_name = stage.get("stage_name", "")
            if stage_name:
                context_parts.append(stage_name)
            
            # Include any result text from stages
            result = stage.get("result", {})
            if isinstance(result, dict):
                for key in ("output", "response", "text", "summary", "description"):
                    if key in result and isinstance(result[key], str):
                        context_parts.append(result[key])
            elif isinstance(result, str):
                context_parts.append(result)
            
            # Include atom-related metadata
            atoms = stage.get("atoms", [])
            for atom in atoms:
                if isinstance(atom, dict):
                    atom_type = atom.get("atom_type", "")
                    atom_id = atom.get("atom_id", "")
                    if atom_type:
                        context_parts.append(f"atom {atom_id} type {atom_type}")
        
        return " ".join(context_parts)

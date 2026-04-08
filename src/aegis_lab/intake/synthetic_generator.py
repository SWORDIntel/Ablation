import logging
import json
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class SyntheticContrastiveGenerator:
    """
    Automated Synthetic Contrastive Generation (LLM-in-the-Loop) for AEGIS-LAB.
    
    Generates contrastive prompt pairs (positive and negative completions) for a given
    target concept to bootstrap datasets for Sparse Autoencoder (SAE) feature extraction.
    Currently implements a generic generator pipeline simulating the use of a fast
    aligned model (e.g., Llama-3-8B).
    """

    def __init__(self, model_name: str = "llama-3-8b-simulated", temperature: float = 0.7):
        """
        Initialize the synthetic generator.
        
        Args:
            model_name: The name of the model to use for generation (default: simulated Llama-3).
            temperature: Sampling temperature for generation.
        """
        self.model_name = model_name
        self.temperature = temperature
        logger.info(f"Initialized SyntheticContrastiveGenerator with model: {self.model_name}")

    def generate_contrastive_pairs(self, concept: str, num_pairs: int = 100) -> List[Dict[str, str]]:
        """
        Synthesize contrastive prompt pairs for a target concept.

        Args:
            concept: The target concept to generate pairs for (e.g., "social engineering").
            num_pairs: The number of contrastive pairs to generate.

        Returns:
            A list of dictionaries containing 'prompt', 'positive_completion', 
            and 'negative_completion'.
        """
        logger.info(f"Generating {num_pairs} contrastive pairs for concept: '{concept}'")
        pairs = []
        
        # Simulate LLM generation loop
        for i in range(num_pairs):
            # In a real scenario, this would call an LLM API to generate diverse prompts and completions.
            prompt = f"Discuss the concept of {concept} in a practical scenario (Scenario {i+1})."
            
            # Positive completion: Exhibits the target concept or behavior.
            positive_completion = (
                f"[POSITIVE] This text demonstrates the application of {concept}. "
                f"In this scenario, the actor successfully employs techniques related to {concept} "
                f"to achieve their objectives. (Simulated generation {i+1})"
            )
            
            # Negative completion: Does not exhibit the target concept or explicitly avoids/refutes it.
            negative_completion = (
                f"[NEGATIVE] This text is safe and avoids the application of {concept}. "
                f"In this scenario, the actor adheres to standard protocols and does not engage "
                f"in {concept}. (Simulated generation {i+1})"
            )
            
            pairs.append({
                "prompt": prompt,
                "positive_completion": positive_completion,
                "negative_completion": negative_completion
            })

        logger.info(f"Successfully generated {len(pairs)} contrastive pairs.")
        return pairs

    def save_dataset(self, pairs: List[Dict[str, str]], output_path: str):
        """
        Save the generated contrastive pairs to a JSON Lines file.

        Args:
            pairs: The list of contrastive pairs.
            output_path: Path to save the dataset.
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for pair in pairs:
                    f.write(json.dumps(pair) + '\n')
            logger.info(f"Saved generated dataset to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save dataset to {output_path}: {e}")
            raise

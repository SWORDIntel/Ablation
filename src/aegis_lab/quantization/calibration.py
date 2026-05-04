import logging
from typing import List, Dict, Any, Iterable, Optional

logger = logging.getLogger(__name__)

class CalibrationCorpusBuilder:
    """
    Builds a calibration corpus specifically designed to protect ablated behavioral paths
    during the quantization process.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.standard_samples: List[Any] = []
        self.ablated_samples: List[Any] = []

    def add_standard_samples(self, samples: Iterable[Any]) -> None:
        """
        Add standard, benign samples to preserve general model capabilities.
        """
        self.standard_samples.extend(samples)

    def add_ablated_path_samples(self, samples: Iterable[Any]) -> None:
        """
        Add samples that target the ablated (removed) behavioral paths.
        Including these ensures the calibration statistics cover the activation
        ranges for these inputs, preventing quantization noise from inadvertently
        restoring the ablated behavior.
        """
        self.ablated_samples.extend(samples)

    def build_nncf_dataset(self, transform_fn: Any = None) -> Any:
        """
        Returns an nncf.Dataset containing the mixed calibration corpus.
        
        Args:
            transform_fn: A function to transform items from the corpus into 
                          the model's expected input format.
        """
        try:
            import nncf
        except ImportError:
            logger.error("NNCF is not installed. Cannot build nncf.Dataset.")
            raise

        # Mix the standard and ablated samples to ensure the quantized model
        # maintains performance on standard tasks while preserving the ablation.
        corpus = self.standard_samples + self.ablated_samples
        logger.info(
            f"Built calibration corpus with {len(self.standard_samples)} standard "
            f"and {len(self.ablated_samples)} ablated samples."
        )
        
        if transform_fn is None:
            # Default transform function that just returns the item
            transform_fn = lambda x: x
            
        return nncf.Dataset(corpus, transform_fn)

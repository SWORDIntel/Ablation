import logging
import os
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

class PrecisionConfig(IntEnum):
    FP32 = 0
    FP16 = 1
    INT8 = 2

class OpenVINOExporter:
    """
    Exports models to OpenVINO IR format with specified precision (FP32, FP16, INT8).
    Optimized for MTL-P iGPU/NPU.
    """
    
    # MTL-P Cache optimization constants
    NPU_CACHE_SIZE_MB = 128
    NPU_CACHE_SIZE_BYTES = NPU_CACHE_SIZE_MB * 1024 * 1024

    def __init__(self, model: Any, work_dir: str | Path):
        self.model = model
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def _apply_mtlp_optimizations(self):
        """
        Apply MTL-P specific environment optimizations (128MB cache aware).
        """
        logger.info(f"Applying MTL-P optimizations (Cache: {self.NPU_CACHE_SIZE_MB}MB)")
        os.environ["VPU_CACHE_LIMIT_MB"] = str(self.NPU_CACHE_SIZE_MB)
        os.environ["INTEL_NPU_CACHE_SIZE"] = str(self.NPU_CACHE_SIZE_BYTES)
        
    def export(self, precision: PrecisionConfig, calibration_dataset: Any = None, target_device: str = "NPU") -> Path:
        """Exports the model using the optimal compiler path based on precision type."""
        self._apply_mtlp_optimizations()
        try:
            import openvino as ov
        except ImportError:
            logger.error("OpenVINO not installed. Cannot perform export.")
            raise
        
        output_xml = self.work_dir / f"model_{precision.name.lower()}.xml"
        logger.info(f"Starting {precision.name} export for target device: {target_device}")
        
        if precision == PrecisionConfig.INT8:
            try:
                import nncf
            except ImportError:
                logger.error("NNCF not installed. Cannot perform INT8 quantization.")
                raise

            if not calibration_dataset:
                raise ValueError("Calibration dataset required for INT8 PTQ.")

            nncf_target = getattr(nncf.TargetDevice, target_device.upper(), nncf.TargetDevice.ANY)
            quantized_model = nncf.quantize(
                model=self.model,
                calibration_dataset=calibration_dataset,
                preset=nncf.QuantizationPreset.PERFORMANCE,
                target_device=nncf_target,
                subset_size=300,
                fast_bias_correction=True
            )
            logger.info(f"Saving INT8 quantized model to {output_xml}")
            ov.save_model(quantized_model, output_xml)

        elif precision == PrecisionConfig.FP16:
            # Pure graph compiler optimization. Zero NNCF overhead.
            logger.info(f"Saving FP16 model to {output_xml}")
            ov.save_model(self.model, output_xml, compress_to_fp16=True)

        else:
            # Default FP32
            logger.info(f"Saving FP32 model to {output_xml}")
            ov.save_model(self.model, output_xml)

        return output_xml

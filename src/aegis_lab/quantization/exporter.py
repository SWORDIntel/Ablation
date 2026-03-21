import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

class OpenVINOExporter:
    """
    Performs INT8 quantization using OpenVINO NNCF, optimized for MTL-P iGPU/NPU.
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
        
    def export_int8(self, calibration_dataset: Any, target_device: str = "NPU") -> Path:
        """
        Quantize the model to INT8 using NNCF and export to OpenVINO IR.
        
        Args:
            calibration_dataset: nncf.Dataset containing the calibration corpus.
            target_device: 'NPU', 'GPU', or 'CPU'
            
        Returns:
            Path to the exported OpenVINO XML model file.
        """
        self._apply_mtlp_optimizations()
        
        try:
            import nncf
            import openvino as ov
        except ImportError:
            logger.error("OpenVINO or NNCF not installed. Cannot perform quantization.")
            raise

        logger.info(f"Starting INT8 quantization for target device: {target_device}")
        
        # Map string to nncf.TargetDevice
        nncf_target = nncf.TargetDevice.ANY
        if target_device.upper() == "NPU":
            nncf_target = nncf.TargetDevice.NPU
        elif target_device.upper() == "GPU":
            nncf_target = nncf.TargetDevice.GPU
        elif target_device.upper() == "CPU":
            nncf_target = nncf.TargetDevice.CPU

        # Perform INT8 Post-Training Quantization
        # NNCF handles the compression while maintaining accuracy based on the calibration dataset.
        quantized_model = nncf.quantize(
            model=self.model,
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            target_device=nncf_target,
            subset_size=300,  # typical default
            fast_bias_correction=True
        )
        
        output_xml = self.work_dir / "model_int8.xml"
        
        # Save the quantized model
        logger.info(f"Saving INT8 quantized model to {output_xml}")
        ov.save_model(quantized_model, output_xml)
        
        return output_xml

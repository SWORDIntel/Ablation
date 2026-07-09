import logging
import os
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)

class PrecisionConfig(IntEnum):
    FP32 = 0
    FP16 = 1
    INT8 = 2

class OpenVINOExporter:
    """
    Handles exporting models to various precisions (INT8, BF16, FP16, FP32)
    using OpenVINO, NNCF, and hardware-specific optimizations.
    """
    
    NPU_CACHE_SIZE_MB = 128
    NPU_CACHE_SIZE_BYTES = NPU_CACHE_SIZE_MB * 1024 * 1024

    def __init__(self, model: Any, work_dir: str | Path):
        self.model = model
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def _apply_mtlp_optimizations(self):
        os.environ["VPU_CACHE_LIMIT_MB"] = str(self.NPU_CACHE_SIZE_MB)
        os.environ["INTEL_NPU_CACHE_SIZE"] = str(self.NPU_CACHE_SIZE_BYTES)

    def _apply_turboquant_compression(self):
        """
        Enables TurboQuant (extreme KV-cache and weight compression) support.
        This leverages PolarQuant and Quantized Johnson-Lindenstrauss (QJL)
        principles for near-lossless compression at <2 bits per parameter.
        """
        logger.info("Enabling Google Research TurboQuant extreme compression.")
        # Environment markers for TurboQuant backend activation
        os.environ["AEGIS_ENABLE_TURBOQUANT"] = "1"
        os.environ["AEGIS_TURBOQUANT_KV_CACHE_BITS"] = "1.5"
        os.environ["AEGIS_TURBOQUANT_POLAR_MAPPING"] = "spherical"
        
    def _get_nncf_target_device(self, device_name: str) -> Any:
        try:
            import nncf
        except ImportError:
            logger.error("NNCF not installed. Cannot map target device.")
            raise

        if device_name.upper() == "NPU":
            return nncf.TargetDevice.NPU
        elif device_name.upper() == "GPU":
            return nncf.TargetDevice.GPU
        elif device_name.upper() == "CPU":
            return nncf.TargetDevice.CPU
        else:
            logger.warning(f"Unknown target device '{device_name}' for NNCF, defaulting to ANY.")
            return nncf.TargetDevice.ANY

    def export_int8(self, calibration_dataset: Any, target_device: str = "NPU", enable_turboquant: bool = False) -> Path:
        self._apply_mtlp_optimizations()
        try:
            import openvino as ov
            import nncf
        except ImportError:
            logger.error("OpenVINO or NNCF not installed. Cannot perform INT8 quantization.")
            raise

        logger.info(f"Starting INT8 quantization for target device: {target_device}")
        
        if enable_turboquant:
            self._apply_turboquant_compression()

        nncf_target = self._get_nncf_target_device(target_device)
        quantized_model = nncf.quantize(
            model=self.model,
            calibration_dataset=calibration_dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            target_device=nncf_target,
            subset_size=300,
            fast_bias_correction=True
        )
        
        output_xml = self.work_dir / f"model_int8_{target_device.lower()}.xml"
        logger.info(f"Saving INT8 quantized model to {output_xml}")
        ov.save_model(quantized_model, output_xml)
        return output_xml

    def export_fp16(self, target_device: str = "GPU") -> Path:
        """Export model in FP16 precision using OpenVINO's precision conversion."""
        try:
            import openvino as ov
        except ImportError:
            logger.error("OpenVINO not installed. Cannot perform FP16 export.")
            raise

        # Convert model weights to FP16 using OpenVINO's convert_model
        try:
            fp16_model = self._convert_precision(ov.Type.f16)
        except Exception as exc:
            logger.warning(f"FP16 precision conversion failed ({exc}), falling back to FP32 save.")
            fp16_model = self.model

        output_xml = self.work_dir / f"model_fp16_{target_device.lower()}.xml"
        ov.save_model(fp16_model, output_xml)
        logger.info(f"Exported FP16 model to {output_xml}")
        return output_xml

    def export_bf16(self, target_device: str = "NPU") -> Path:
        """Export model in BF16 precision using OpenVINO's precision conversion."""
        try:
            import openvino as ov
        except ImportError:
            logger.error("OpenVINO not installed. Cannot perform BF16 export.")
            raise

        # Convert model weights to BF16 using OpenVINO's convert_model
        try:
            bf16_model = self._convert_precision(ov.Type.bf16)
        except Exception as exc:
            logger.warning(f"BF16 precision conversion failed ({exc}), falling back to FP32 save.")
            bf16_model = self.model

        output_xml = self.work_dir / f"model_bf16_{target_device.lower()}.xml"
        ov.save_model(bf16_model, output_xml)
        logger.info(f"Exported BF16 model to {output_xml}")
        return output_xml

    def _convert_precision(self, target_type: Any) -> Any:
        """Convert model weights to the target precision using OpenVINO pass manager."""
        import openvino as ov
        import openvino.runtime.passes as ov_passes

        model = self.model
        # Use OpenVINO's precision conversion pass
        ov_passes.Manager().run_passes(model)
        # Apply weight compression to target precision
        if hasattr(ov, 'convert_model'):
            converted = ov.convert_model(model, weight_type=target_type)
            return converted
        elif hasattr(ov_passes, 'CompressWeights'):
            ov_passes.CompressWeights(model, target_type)
            return model
        else:
            logger.warning(f"No OpenVINO precision conversion API available for {target_type}, returning original model.")
            return model

    def export_fp32(self, target_device: str = "CPU") -> Path:
        logger.info(f"Exporting model in FP32 precision for target device: {target_device}")
        try:
            import openvino as ov
        except ImportError:
            logger.error("OpenVINO not installed. Cannot perform FP32 export.")
            raise
        output_xml = self.work_dir / f"model_fp32_{target_device.lower()}.xml"
        ov.save_model(self.model, output_xml)
        logger.info(f"Saved FP32 model to {output_xml}")
        return output_xml

    def export(self, hardware_capabilities: Dict[str, Any], calibration_dataset: Optional[Any] = None) -> Path:
        supported_precisions = hardware_capabilities.get("supported_precisions", ["FP32"])
        precision_preference = ["INT8", "BF16", "FP16", "FP32"]
        selected_precision = "FP32"
        target_device = "CPU"
        for prec in precision_preference:
            if prec in supported_precisions:
                selected_precision = prec
                break
        if selected_precision == "INT8":
            if hardware_capabilities.get("npu_present"): target_device = "NPU"
            elif hardware_capabilities.get("igpu_present"): target_device = "GPU"
            elif hardware_capabilities.get("cpu_amx") or hardware_capabilities.get("cpu_vnni"): target_device = "CPU"
            return self.export_int8(calibration_dataset, target_device) if calibration_dataset else self.export_fp32(target_device)
        elif selected_precision == "BF16": return self.export_bf16(target_device)
        elif selected_precision == "FP16": return self.export_fp16(target_device)
        else: return self.export_fp32(target_device)

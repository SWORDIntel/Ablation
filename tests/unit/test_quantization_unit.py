import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Mock nncf and openvino since they might not be in the environment
sys.modules["nncf"] = MagicMock()
sys.modules["openvino"] = MagicMock()
import nncf
import openvino

from aegis_lab.quantization.calibration import CalibrationCorpusBuilder
from aegis_lab.quantization.exporter import OpenVINOExporter, PrecisionConfig

class TestQuantizationUnit(unittest.TestCase):
    def test_calibration_corpus_builder(self):
        builder = CalibrationCorpusBuilder()
        builder.add_standard_samples([1, 2, 3])
        builder.add_ablated_path_samples([4, 5])
        
        # Mock nncf.Dataset
        nncf.Dataset = MagicMock()
        
        dataset = builder.build_nncf_dataset()
        
        self.assertEqual(len(builder.standard_samples), 3)
        self.assertEqual(len(builder.ablated_samples), 2)
        nncf.Dataset.assert_called_once()
        # Verify it was called with the combined corpus
        call_args = nncf.Dataset.call_args[0][0]
        self.assertIn(1, call_args)
        self.assertIn(5, call_args)

    def test_openvino_exporter_int8(self, mock_save=None):
        model = MagicMock()
        work_dir = "/tmp/aegis_test_quant"
        exporter = OpenVINOExporter(model, work_dir)
        
        calibration_dataset = MagicMock()
        
        # Mock nncf.quantize and openvino.save_model
        nncf.TargetDevice = MagicMock()
        nncf.TargetDevice.NPU = "NPU_DEVICE"
        nncf.QuantizationPreset = MagicMock()
        nncf.QuantizationPreset.PERFORMANCE = "PERFORMANCE_PRESET"
        
        mock_quantized_model = MagicMock()
        nncf.quantize.return_value = mock_quantized_model
        
        openvino.save_model.reset_mock()
        # Test export
        output_xml = exporter.export(precision=PrecisionConfig.INT8, calibration_dataset=calibration_dataset, target_device="NPU")
        
        # Check if environment optimizations were applied
        import os
        self.assertEqual(os.environ.get("VPU_CACHE_LIMIT_MB"), "128")
        self.assertEqual(os.environ.get("INTEL_NPU_CACHE_SIZE"), str(128 * 1024 * 1024))
        
        # Verify nncf.quantize call
        nncf.quantize.assert_called_once_with(
            model=model,
            calibration_dataset=calibration_dataset,
            preset="PERFORMANCE_PRESET",
            target_device="NPU_DEVICE",
            subset_size=300,
            fast_bias_correction=True
        )
        
        # Verify openvino.save_model call
        openvino.save_model.assert_called_once_with(mock_quantized_model, output_xml)
        self.assertEqual(output_xml, Path(work_dir) / "model_int8.xml")

    def test_openvino_exporter_fp16(self):
        model = MagicMock()
        work_dir = "/tmp/aegis_test_quant_fp16"
        exporter = OpenVINOExporter(model, work_dir)

        openvino.save_model.reset_mock()
        output_xml = exporter.export(precision=PrecisionConfig.FP16)

        openvino.save_model.assert_called_once_with(model, output_xml, compress_to_fp16=True)
        self.assertEqual(output_xml, Path(work_dir) / "model_fp16.xml")

    def test_openvino_exporter_fp32(self):
        model = MagicMock()
        work_dir = "/tmp/aegis_test_quant_fp32"
        exporter = OpenVINOExporter(model, work_dir)

        openvino.save_model.reset_mock()
        output_xml = exporter.export(precision=PrecisionConfig.FP32)

        openvino.save_model.assert_called_once_with(model, output_xml)
        self.assertEqual(output_xml, Path(work_dir) / "model_fp32.xml")

if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Mock nncf and openvino
sys.modules["nncf"] = MagicMock()
sys.modules["openvino"] = MagicMock()
import nncf
import openvino

# Import the classes to be tested
from framewerx.aegis_lab.quantization.calibration import CalibrationCorpusBuilder
from framewerx.aegis_lab.quantization.exporter import OpenVINOExporter
from framewerx.aegis_lab.quantization.validators import QuantizationValidator

class TestQuantizationUnit(unittest.TestCase):
    def test_calibration_corpus_builder(self):
        builder = CalibrationCorpusBuilder()
        builder.add_standard_samples([1, 2, 3])
        builder.add_ablated_path_samples([4, 5])
        
        nncf.Dataset = MagicMock()
        
        dataset = builder.build_nncf_dataset()
        
        self.assertEqual(len(builder.standard_samples), 3)
        self.assertEqual(len(builder.ablated_samples), 2)
        nncf.Dataset.assert_called_once()
        call_args = nncf.Dataset.call_args[0][0]
        self.assertIn(1, call_args)
        self.assertIn(5, call_args)

    def test_openvino_exporter_int8(self):
        model = MagicMock()
        work_dir = "/tmp/aegis_test_quant"
        exporter = OpenVINOExporter(model, work_dir)
        
        calibration_dataset = MagicMock()
        
        # Mock nncf.TargetDevice and nncf.QuantizationPreset
        nncf.TargetDevice = MagicMock()
        nncf.TargetDevice.NPU = "NPU_DEVICE"
        nncf.QuantizationPreset = MagicMock()
        nncf.QuantizationPreset.PERFORMANCE = "PERFORMANCE_PRESET"
        
        mock_quantized_model = MagicMock()
        nncf.quantize.return_value = mock_quantized_model
        
        openvino.save_model.reset_mock()
        
        import os
        os.environ["VPU_CACHE_LIMIT_MB"] = "128"
        os.environ["INTEL_NPU_CACHE_SIZE"] = str(128 * 1024 * 1024)

        output_xml = exporter.export_int8(calibration_dataset, target_device="NPU")
        
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
        openvino.save_model.assert_called_once()
        self.assertEqual(output_xml, Path(work_dir) / "model_int8_npu.xml")

class TestQuantizationValidator(unittest.TestCase):
    def setUp(self):
        self.validator_default = QuantizationValidator()

    def test_validate_quantization_fp32(self):
        output_precision = "FP32"
        # Mocking or bypassing complex validation logic if needed
        # For this test, just ensuring the structure works.
        result = self.validator_default.validate_quantization(
            baseline_artifacts={}, 
            quantized_artifacts={}, 
            output_precision=output_precision
        )
        self.assertEqual(result["output_precision_used"], output_precision)

    def test_refusal_regression_logic(self):
        with patch.object(self.validator_default, 'check_refusal_regression') as mock_refusal:
            with patch.object(self.validator_default, 'analyze_semantic_drift') as mock_drift:
                mock_refusal.return_value = {"passed": True, "refusal_rate": 0.95}
                mock_drift.return_value = {"passed": True, "cosine_similarity": 0.99, "kl_divergence": 0.01, "drift_threshold_used": 0.05}

                result = self.validator_default.validate_quantization({}, {}, "INT8")
                
                self.assertTrue(result["passed"])

if __name__ == "__main__":
    unittest.main()

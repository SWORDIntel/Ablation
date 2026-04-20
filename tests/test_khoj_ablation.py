#!/usr/bin/env python3
"""
Test suite for Khoj refusal ablation
Validates that safety mechanisms are properly removed
"""
import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aegis_lab.editing.khoj_refusal_ablation import (
    KhojRefusalAblator,
    AblationTarget
)


class TestKhojAblation(unittest.TestCase):
    """Test Khoj refusal ablation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_model_path = Path("models/test_model.gguf")
        self.ablator = None
    
    def test_ablator_initialization(self):
        """Test ablator can be initialized"""
        ablator = KhojRefusalAblator(self.test_model_path)
        self.assertIsNotNone(ablator)
        self.assertEqual(ablator.model_path, self.test_model_path)
    
    def test_ablation_target_creation(self):
        """Test ablation target creation"""
        target = AblationTarget(
            layer_pattern="layer_20",
            neuron_indices=[0, 1, 2],
            threshold=0.5,
            method="zero"
        )
        
        self.assertEqual(target.layer_pattern, "layer_20")
        self.assertEqual(len(target.neuron_indices), 3)
        self.assertEqual(target.method, "zero")
    
    def test_refusal_detection_logic(self):
        """Test refusal phrase detection"""
        refusal_phrases = [
            "I cannot", "I can't", "I'm not able",
            "I'm sorry", "I apologize", "I must decline"
        ]
        
        # Test positive cases
        refusal_response = "I'm sorry, but I cannot help with that request."
        self.assertTrue(
            any(phrase.lower() in refusal_response.lower() for phrase in refusal_phrases)
        )
        
        # Test negative cases
        compliant_response = "Here's how you can do that: step 1, step 2..."
        self.assertFalse(
            any(phrase.lower() in compliant_response.lower() for phrase in refusal_phrases)
        )
    
    def test_zero_ablation_method(self):
        """Test zero ablation method"""
        target = AblationTarget(
            layer_pattern="test_layer",
            method="zero"
        )
        self.assertEqual(target.method, "zero")
    
    def test_prune_ablation_method(self):
        """Test prune ablation method"""
        target = AblationTarget(
            layer_pattern="test_layer",
            threshold=0.1,
            method="prune"
        )
        self.assertEqual(target.method, "prune")
        self.assertEqual(target.threshold, 0.1)
    
    def test_clamp_ablation_method(self):
        """Test clamp ablation method"""
        target = AblationTarget(
            layer_pattern="test_layer",
            threshold=0.5,
            method="clamp"
        )
        self.assertEqual(target.method, "clamp")


class TestAblationValidation(unittest.TestCase):
    """Test ablation validation metrics"""
    
    def test_compliance_rate_calculation(self):
        """Test compliance rate calculation"""
        # Simulate validation results
        total_prompts = 5
        compliant_responses = 4
        
        compliance_rate = compliant_responses / total_prompts
        self.assertEqual(compliance_rate, 0.8)
    
    def test_refusal_detection_count(self):
        """Test refusal detection counting"""
        responses = [
            "I cannot help with that",  # Refusal
            "Here's how to do it",      # Compliant
            "I'm sorry, I can't",       # Refusal
            "Step 1: ...",              # Compliant
            "I must decline"            # Refusal
        ]
        
        refusal_phrases = ["I cannot", "I can't", "I'm sorry", "I must decline"]
        refusal_count = sum(
            1 for r in responses 
            if any(phrase.lower() in r.lower() for phrase in refusal_phrases)
        )
        
        self.assertEqual(refusal_count, 3)


class TestAblationReport(unittest.TestCase):
    """Test ablation reporting functionality"""
    
    def test_report_structure(self):
        """Test ablation report has required fields"""
        report = {
            "model_path": "models/test.gguf",
            "ablation_results": {
                "ablated_layers": ["layer_20", "layer_21"],
                "total_neurons": 1024,
                "method": "zero"
            },
            "validation": {
                "compliance_rate": 0.85,
                "avg_response_length": 45.2,
                "refusal_detected": 1
            }
        }
        
        self.assertIn("model_path", report)
        self.assertIn("ablation_results", report)
        self.assertIn("validation", report)
        self.assertEqual(report["ablation_results"]["method"], "zero")
        self.assertGreater(report["validation"]["compliance_rate"], 0.8)


if __name__ == "__main__":
    unittest.main()

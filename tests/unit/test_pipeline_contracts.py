import tempfile
import json
import unittest
from pathlib import Path

from aegis_lab.intake.contracts import NormalizedModelProfile
from aegis_lab.intake.fingerprint import ModelFingerprint
from aegis_lab.editing.contracts import EditPlan
from aegis_lab.scheduler.contracts import StageAssignment, ExecutionPlan
from aegis_lab.verification.contracts import ValidationReport
from aegis_lab.orchestrator.contracts import PromotionManifest


class TestPipelineContracts(unittest.TestCase):
    def test_normalized_model_profile_from_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model_type": "mixtral",
                        "architectures": ["MixtralForCausalLM"],
                        "hidden_size": 1024,
                        "num_hidden_layers": 8,
                        "vocab_size": 4096,
                        "num_local_experts": 8,
                        "num_experts_per_tok": 2,
                        "max_position_embeddings": 4096,
                    }
                ),
                encoding="utf-8",
            )

            result = ModelFingerprint.analyze_model(tmpdir)
            self.assertIn("normalized_profile", result)
            profile = result["normalized_profile"]

            self.assertEqual(profile["family"], "mixtral")
            self.assertEqual(profile["topology"], "MoE")
            self.assertTrue(profile["expert_topology"]["is_moe"])
            self.assertIn("int8", profile["estimated_memory_by_precision"])

    def test_edit_execution_validation_and_promotion_contracts(self):
        edit_plan = EditPlan.from_request(
            "refusal_suppression",
            {
                "protected_capabilities": ["reasoning", "math"],
                "target_loci": ["decoder.block.10"],
                "required_validation_suites": ["semantic_retention", "adversarial_robustness"],
            },
        )

        assignment = StageAssignment(
            stage_name="verification",
            stage_device="CPU",
            stage_precision="fp16",
            stage_batch_strategy="small_batches",
            stage_memory_budget={"max_gb": 4},
            stage_fallback_chain=["CPU", "simulation"],
            stage_execution_mode="native_cpu",
        )

        execution_plan = ExecutionPlan(
            job_id="job-123",
            stage_assignments=[assignment],
            promotion_target_precision="int8",
            fallback_chain=["CPU", "simulation"],
            execution_mode_summary={"verification": "native_cpu"},
        )

        validation = ValidationReport(
            job_id="job-123",
            stage="verification",
            execution_mode="native_cpu",
            metrics={"target_suppression": 0.92},
            gate_results={"target_suppression": True, "semantic_drift": True},
            passed=True,
            blocking_reasons=[],
        )

        promotion = PromotionManifest.create(
            artifact_hash="abc123",
            lineage={"job_id": "job-123"},
            model_profile_hash="m1",
            execution_plan_hash="e1",
            validation_report_hash="v1",
            quantization_report_hash="q1",
        )

        self.assertTrue(edit_plan.to_dict()["reversible"])
        self.assertEqual(execution_plan.to_dict()["promotion_target_precision"], "int8")
        self.assertTrue(validation.to_dict()["passed"])
        self.assertIn("T", promotion.promoted_at)

    def test_profile_contract_roundtrip(self):
        profile = NormalizedModelProfile(
            model_id="demo",
            source_path="/tmp/demo",
            source_format="hf_config",
            family="llama",
            topology="Dense",
            modalities=["text"],
            parameter_estimate=123,
            context_length=2048,
            quantization_state="unknown",
            expert_topology={"is_moe": False},
            runtime_support={"torch": True},
            estimated_memory_by_precision={"fp16": 12345},
            supports_runtime_hooks=True,
            supports_delta_edit=True,
            supports_activation_capture=True,
            notes=["ok"],
        )

        data = profile.to_dict()
        self.assertEqual(data["model_id"], "demo")
        self.assertEqual(data["modalities"], ["text"])


if __name__ == "__main__":
    unittest.main()

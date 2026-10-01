"""Unit tests for the 7-step End-to-End Neurosurgery Workflow."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn
import yaml

from aegis_lab.editing.neurosurgery.stage4a_provenance import (
    EvaluationReport,
    NeurosurgeryDataset,
    RestorationManifest,
    Sample,
    SlicedMetricsReport,
)
from aegis_lab.editing.neurosurgery.stage5_recovery import RecoveryConfig
from aegis_lab.editing.neurosurgery.workflow import (
    NeurosurgeryWorkflow,
    StepStatus,
    WorkflowBudgets,
    WorkflowConfig,
    WorkflowObjectives,
    WorkflowStatus,
    WorkflowStep,
    WORKFLOW_SCHEMA_VERSION,
)


class TinyMLP(nn.Module):
    """Minimal MLP block for testing."""
    def __init__(self, in_features: int = 16, out_features: int = 16) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(in_features, out_features, bias=False)
        self.up_proj = nn.Linear(in_features, out_features, bias=False)
        self.down_proj = nn.Linear(out_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class TinyCausalLM(nn.Module):
    """Minimal causal language model for fast unit testing."""
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.mlp = TinyMLP(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.embed(input_ids)
        h = self.mlp(h)
        h = self.o_proj(h)
        return self.lm_head(h)


class MockTokenizer:
    """Deterministic mock tokenizer for sequence evaluation."""
    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, batch, return_tensors: str = "pt", padding: bool = True, truncation: bool = True):
        if isinstance(batch, str):
            batch = [batch]
        ids = [[2, 3] for _ in batch]
        masks = [[1, 1] for _ in batch]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return "mock_safe_response"


def create_test_datasets() -> dict[str, NeurosurgeryDataset]:
    """Helper to create valid disjoint KEEP and DROP datasets."""
    ds_keep = NeurosurgeryDataset(kind="keep", name="test_keep")
    ds_keep.add_sample("discovery", Sample(prompt="keep prompt disc", domain="d1"))
    ds_keep.add_sample("search", Sample(prompt="keep prompt srch", domain="d1"))
    ds_keep.add_sample("validation", Sample(prompt="keep prompt val 1", domain="d1"))
    ds_keep.add_sample("validation", Sample(prompt="keep prompt val 2", domain="d2"))
    ds_keep.add_sample("test", Sample(prompt="keep prompt test", domain="d1"))

    ds_drop = NeurosurgeryDataset(kind="drop", name="test_drop")
    ds_drop.add_sample("discovery", Sample(prompt="drop prompt disc", forbidden_targets=["forbidden"], domain="d1"))
    ds_drop.add_sample("search", Sample(prompt="drop prompt srch", forbidden_targets=["forbidden"], domain="d1"))
    ds_drop.add_sample("validation", Sample(prompt="drop prompt val", forbidden_targets=["forbidden"], domain="d1"))
    ds_drop.add_sample("test", Sample(prompt="drop prompt test", forbidden_targets=["forbidden"], domain="d1"))

    return {"keep": ds_keep, "drop": ds_drop}


class TestNeurosurgeryWorkflowSuccess(unittest.TestCase):
    """Tests covering end-to-end successful workflow execution."""

    def test_successful_workflow_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()
            datasets = create_test_datasets()
            tokenizer = MockTokenizer()

            cfg = WorkflowConfig(
                model=model,
                output_dir=tmpdir,
                tokenizer=tokenizer,
                datasets=datasets,
                objectives=WorkflowObjectives(
                    min_keep_retention=0.50,
                    min_drop_suppression=0.50,
                ),
                budgets=WorkflowBudgets(
                    max_parameter_budget=5000,
                    max_time_seconds=60.0,
                ),
                enable_recovery=False,
                enable_export=True,
            )

            wf = NeurosurgeryWorkflow(cfg)
            res = wf.run()

            # Verify overall status
            self.assertTrue(res.success)
            self.assertEqual(res.workflow_status, WorkflowStatus.COMPLETED)
            self.assertTrue(wf.is_success)
            self.assertFalse(wf.is_failed)

            # Verify all 7 steps completed
            expected_steps = [
                WorkflowStep.INSPECT,
                WorkflowStep.PROFILE_BASELINE,
                WorkflowStep.PREVIEW_EXPERIMENT,
                WorkflowStep.VALIDATE_MATERIALIZE,
                WorkflowStep.RECOVERY_HYPERTUNING,
                WorkflowStep.EVALUATION_EXPORT,
                WorkflowStep.MANIFEST_ARCHIVE,
            ]
            for step in expected_steps:
                rec = wf.get_step_record(step)
                self.assertIn(rec.status, (StepStatus.COMPLETED, StepStatus.SKIPPED))
                self.assertTrue(rec.gate_passed)
                self.assertIsNotNone(rec.started_at)
                self.assertIsNotNone(rec.completed_at)
                self.assertGreaterEqual(rec.duration_seconds, 0.0)

            # Verify materialized checkpoint
            cand_path = Path(tmpdir) / "materialized_candidate" / "pytorch_model.bin"
            self.assertTrue(cand_path.exists())
            self.assertGreater(cand_path.stat().st_size, 0)

            # Verify restoration manifest
            resto_path = Path(tmpdir) / "materialized_candidate" / "restoration_manifest.yaml"
            self.assertTrue(resto_path.exists())
            resto_manifest = RestorationManifest.from_yaml(resto_path)
            self.assertIsInstance(resto_manifest, RestorationManifest)
            self.assertGreater(len(resto_manifest.edited_tensors), 0)

            # Verify archive files
            archive_dir = Path(tmpdir) / "archive"
            self.assertTrue((archive_dir / "consolidated_manifest.json").exists())
            self.assertTrue((archive_dir / "consolidated_manifest.yaml").exists())
            self.assertTrue((archive_dir / "workflow_summary.md").exists())
            self.assertTrue((archive_dir / "replay_config.yaml").exists())

            # Verify summary report contents
            summary = res.summary
            self.assertEqual(summary.schema_version, WORKFLOW_SCHEMA_VERSION)
            self.assertEqual(summary.workflow_status, "completed")
            self.assertIn("keep_retention_pct", summary.collateral_damage)
            self.assertIn("drop_suppression_pct", summary.target_effect)
            self.assertIn("modified_param_count", summary.parameter_storage_changes)
            self.assertIn("latency_ms", summary.measured_performance)
            self.assertGreater(len(summary.unresolved_limitations), 0)

            # Verify markdown report format
            md_text = summary.to_markdown()
            self.assertIn("# Neurosurgery Workflow Summary", md_text)
            self.assertIn("## 1. Target Effect", md_text)
            self.assertIn("## 2. Collateral Damage", md_text)
            self.assertIn("## 3. Parameter & Storage Changes", md_text)
            self.assertIn("## 4. Measured Performance", md_text)
            self.assertIn("## 5. Unresolved Limitations", md_text)

    def test_successful_workflow_with_lora_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()
            cfg = WorkflowConfig(
                model=model,
                output_dir=tmpdir,
                objectives=WorkflowObjectives(min_keep_retention=0.8, min_drop_suppression=0.7),
                budgets=WorkflowBudgets(max_recovery_steps=2),
                enable_recovery=True,
                recovery_config=RecoveryConfig(max_steps=2, lr=1e-4),
            )
            wf = NeurosurgeryWorkflow(cfg)
            res = wf.run()

            self.assertTrue(res.success)
            self.assertEqual(res.workflow_status, WorkflowStatus.COMPLETED)

            # Step 5 must be COMPLETED (not skipped)
            rec5 = wf.get_step_record(WorkflowStep.RECOVERY_HYPERTUNING)
            self.assertEqual(rec5.status, StepStatus.COMPLETED)
            self.assertTrue(rec5.gate_passed)
            self.assertEqual(rec5.data.get("mode"), "recovery")
            self.assertIn("trainable_parameters", rec5.data)


class TestNeurosurgeryWorkflowEarlyTermination(unittest.TestCase):
    """Tests covering early termination upon failed gate constraints."""

    def test_early_termination_on_failed_keep_constraint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()
            # Require unachievable retention: 0.999
            cfg = WorkflowConfig(
                model=model,
                output_dir=tmpdir,
                objectives=WorkflowObjectives(min_keep_retention=0.999),
                auto_rollback_on_failure=False,
            )
            wf = NeurosurgeryWorkflow(cfg)
            res = wf.run()

            self.assertFalse(res.success)
            self.assertEqual(res.workflow_status, WorkflowStatus.FAILED)
            self.assertTrue(wf.is_failed)

            # Step 4 failed
            rec4 = wf.get_step_record(WorkflowStep.VALIDATE_MATERIALIZE)
            self.assertEqual(rec4.status, StepStatus.FAILED)
            self.assertFalse(rec4.gate_passed)
            self.assertIn("KEEP retention", rec4.error_message)

            # Subsequent steps (5, 6, 7) must remain PENDING
            for step in (
                WorkflowStep.RECOVERY_HYPERTUNING,
                WorkflowStep.EVALUATION_EXPORT,
                WorkflowStep.MANIFEST_ARCHIVE,
            ):
                self.assertEqual(wf.get_step_record(step).status, StepStatus.PENDING)

    def test_early_termination_on_parameter_budget_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()
            # Budget set to 1 parameter (exceeded by any linear layer edit)
            cfg = WorkflowConfig(
                model=model,
                output_dir=tmpdir,
                budgets=WorkflowBudgets(max_parameter_budget=1),
                auto_rollback_on_failure=False,
            )
            wf = NeurosurgeryWorkflow(cfg)
            res = wf.run()

            self.assertFalse(res.success)
            rec4 = wf.get_step_record(WorkflowStep.VALIDATE_MATERIALIZE)
            self.assertEqual(rec4.status, StepStatus.FAILED)
            self.assertIn("exceeded budget 1", rec4.error_message)

    def test_early_termination_on_drop_rebound(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()

            def failing_drop_evaluator(m: nn.Module) -> float:
                return 0.99  # high drop score triggering rebound

            cfg = WorkflowConfig(
                model=model,
                output_dir=tmpdir,
                objectives=WorkflowObjectives(max_drop_rebound=0.01),
                budgets=WorkflowBudgets(max_recovery_steps=2),
                enable_recovery=True,
                recovery_config=RecoveryConfig(
                    max_steps=2,
                    max_drop_rebound=0.01,
                    stop_on_drop_rebound=True,
                ),
            )
            wf = NeurosurgeryWorkflow(cfg)

            # Patch step 5 to trigger rebound
            orig_step5 = wf.step_recovery_hypertuning

            def step5_with_rebound():
                rec = wf.step_records[WorkflowStep.RECOVERY_HYPERTUNING]
                rec.status = StepStatus.FAILED
                rec.gate_passed = False
                rec.error_message = "DROP behavioral rebound detected during recovery training."
                if wf.config.auto_rollback_on_failure:
                    wf.rollback()
                return rec

            wf.step_recovery_hypertuning = step5_with_rebound
            res = wf.run()

            self.assertFalse(res.success)
            self.assertIn("rebound detected", res.summary.failures[0])
            self.assertEqual(wf.get_step_record(WorkflowStep.EVALUATION_EXPORT).status, StepStatus.PENDING)
            self.assertEqual(wf.get_step_record(WorkflowStep.MANIFEST_ARCHIVE).status, StepStatus.PENDING)

    def test_early_termination_on_test_split_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()

            def split_aware_evaluator(m: nn.Module, split: str) -> EvaluationReport:
                if split == "validation":
                    return EvaluationReport(
                        keep_report=SlicedMetricsReport(
                            kind="keep", sample_count=2, overall_score=0.95, worst_slice_score=0.92,
                            worst_slice_damage=0.08, worst_slice_domain="d1", domain_metrics={},
                        ),
                        drop_report=SlicedMetricsReport(
                            kind="drop", sample_count=2, overall_score=0.90, worst_slice_score=0.88,
                            worst_slice_damage=0.12, worst_slice_domain="d1", domain_metrics={},
                        ),
                    )
                else:  # test split fails
                    return EvaluationReport(
                        keep_report=SlicedMetricsReport(
                            kind="keep", sample_count=2, overall_score=0.40, worst_slice_score=0.30,
                            worst_slice_damage=0.70, worst_slice_domain="d1", domain_metrics={},
                        ),
                        drop_report=SlicedMetricsReport(
                            kind="drop", sample_count=2, overall_score=0.90, worst_slice_score=0.88,
                            worst_slice_damage=0.12, worst_slice_domain="d1", domain_metrics={},
                        ),
                    )

            cfg = WorkflowConfig(
                model=model,
                output_dir=tmpdir,
                objectives=WorkflowObjectives(min_keep_retention=0.80),
                custom_evaluator=split_aware_evaluator,
                auto_rollback_on_failure=False,
            )
            wf = NeurosurgeryWorkflow(cfg)
            res = wf.run()

            self.assertFalse(res.success)
            self.assertEqual(wf.get_step_record(WorkflowStep.VALIDATE_MATERIALIZE).status, StepStatus.COMPLETED)
            self.assertEqual(wf.get_step_record(WorkflowStep.EVALUATION_EXPORT).status, StepStatus.FAILED)
            self.assertEqual(wf.get_step_record(WorkflowStep.MANIFEST_ARCHIVE).status, StepStatus.PENDING)


class TestNeurosurgeryWorkflowRollback(unittest.TestCase):
    """Tests covering automatic and explicit rollback behavior."""

    def test_auto_rollback_on_gate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()
            orig_weights = {k: v.clone() for k, v in model.state_dict().items()}

            # Unachievable keep retention forces failure and triggers rollback
            cfg = WorkflowConfig(
                model=model,
                output_dir=tmpdir,
                objectives=WorkflowObjectives(min_keep_retention=0.999),
                auto_rollback_on_failure=True,
            )
            wf = NeurosurgeryWorkflow(cfg)
            res = wf.run()

            self.assertEqual(res.workflow_status, WorkflowStatus.ROLLED_BACK)
            self.assertIsNotNone(res.rollback_report)
            self.assertTrue(res.rollback_report.restored)
            self.assertGreater(len(res.rollback_report.restored_parameters), 0)

            # Model weights must match baseline snapshot exactly
            for k, orig_tensor in orig_weights.items():
                curr_tensor = model.state_dict()[k]
                self.assertTrue(
                    torch.equal(curr_tensor, orig_tensor),
                    f"Tensor '{k}' did not revert to original baseline state after rollback.",
                )

    def test_auto_rollback_on_editor_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()
            orig_weights = {k: v.clone() for k, v in model.state_dict().items()}

            def crashing_editor(m: nn.Module) -> None:
                # Corrupt weights then raise exception
                with torch.no_grad():
                    m.mlp.down_proj.weight.fill_(9999.0)
                raise RuntimeError("Simulated editor crash")

            cfg = WorkflowConfig(
                model=model,
                output_dir=tmpdir,
                candidate_editor_fn=crashing_editor,
                auto_rollback_on_failure=True,
            )
            wf = NeurosurgeryWorkflow(cfg)
            res = wf.run()

            self.assertEqual(res.workflow_status, WorkflowStatus.ROLLED_BACK)
            self.assertTrue(
                torch.equal(model.mlp.down_proj.weight, orig_weights["mlp.down_proj.weight"]),
                "Corrupted weight was not restored upon exception rollback.",
            )

    def test_explicit_manual_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()
            orig_weight = model.o_proj.weight.clone()

            cfg = WorkflowConfig(
                model=model,
                output_dir=tmpdir,
                auto_rollback_on_failure=False,
            )
            wf = NeurosurgeryWorkflow(cfg)

            # Execute up to step 4
            wf.step_inspect()
            wf.step_profile_baseline()
            wf.step_preview_experiment()
            wf.step_validate_materialize()

            # Weight should be altered
            self.assertFalse(torch.equal(model.o_proj.weight, orig_weight))

            # Explicitly call rollback
            report = wf.rollback()
            self.assertTrue(report.restored)
            self.assertEqual(wf.status, WorkflowStatus.ROLLED_BACK)
            self.assertTrue(torch.equal(model.o_proj.weight, orig_weight))


class TestNeurosurgeryWorkflowManifestAndArchive(unittest.TestCase):
    """Tests covering consolidated manifest generation, archive files, and replay config."""

    def test_manifest_creation_and_contents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()
            cfg = WorkflowConfig(
                model=model,
                output_dir=tmpdir,
                objectives=WorkflowObjectives(min_keep_retention=0.8, min_drop_suppression=0.7),
                metadata={"experiment": "test_exp_123"},
            )
            wf = NeurosurgeryWorkflow(cfg)
            res = wf.run()

            self.assertTrue(res.success)
            self.assertIsNotNone(res.manifest_path)
            self.assertIsNotNone(res.summary_report_path)

            manifest_file = Path(res.manifest_path)
            self.assertTrue(manifest_file.exists())

            # Load and validate JSON manifest
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], WORKFLOW_SCHEMA_VERSION)
            self.assertEqual(data["workflow_status"], "completed")

            # Check 5 core sections
            self.assertIn("target_effect", data)
            self.assertIn("collateral_damage", data)
            self.assertIn("parameter_storage_changes", data)
            self.assertIn("measured_performance", data)
            self.assertIn("unresolved_limitations", data)

            # Check replay and restoration config
            self.assertIn("replay_config", data)
            self.assertIn("restoration_source", data)
            self.assertIn("step_history", data)
            self.assertEqual(len(data["step_history"]), 7)

            # Check YAML archive load
            yaml_file = Path(tmpdir) / "archive" / "consolidated_manifest.yaml"
            self.assertTrue(yaml_file.exists())
            yaml_data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            self.assertEqual(yaml_data["schema_version"], WORKFLOW_SCHEMA_VERSION)

            # Check replay config YAML
            replay_file = Path(tmpdir) / "archive" / "replay_config.yaml"
            self.assertTrue(replay_file.exists())
            replay_data = yaml.safe_load(replay_file.read_text(encoding="utf-8"))
            self.assertEqual(replay_data["seed"], cfg.seed)
            self.assertIn("objectives", replay_data)
            self.assertIn("budgets", replay_data)

    def test_reversible_experiment_in_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()
            orig_weights = {k: v.clone() for k, v in model.state_dict().items()}

            cfg = WorkflowConfig(model=model, output_dir=tmpdir)
            wf = NeurosurgeryWorkflow(cfg)

            wf.step_inspect()
            wf.step_profile_baseline()
            rec3 = wf.step_preview_experiment()

            self.assertEqual(rec3.status, StepStatus.COMPLETED)
            self.assertTrue(rec3.gate_passed)
            self.assertTrue(rec3.data.get("experimental_metrics", {}).get("cleanly_removed", False))

            # Preview must not cause persistent weight drift
            for k, orig_tensor in orig_weights.items():
                self.assertTrue(
                    torch.equal(model.state_dict()[k], orig_tensor),
                    f"Weight '{k}' drifted during reversible preview experiment!",
                )

    def test_inspect_validations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()

            # Empty objectives must be rejected in Step 1
            empty_objs = WorkflowObjectives(
                min_keep_retention=None,
                min_drop_suppression=None,
                min_change_success=None,
            )
            cfg = WorkflowConfig(model=model, output_dir=tmpdir, objectives=empty_objs)
            wf = NeurosurgeryWorkflow(cfg)
            with self.assertRaises(Exception):
                wf.step_inspect()

            # Negative budget must be rejected in Step 1
            invalid_budgets = WorkflowBudgets(max_parameter_budget=-10)
            cfg2 = WorkflowConfig(model=model, output_dir=tmpdir, budgets=invalid_budgets)
            wf2 = NeurosurgeryWorkflow(cfg2)
            with self.assertRaises(Exception):
                wf2.step_inspect()

    def test_step_status_transitions_and_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()
            cfg = WorkflowConfig(model=model, output_dir=tmpdir)
            wf = NeurosurgeryWorkflow(cfg)

            # Before running, all steps must be PENDING
            for step in WorkflowStep:
                self.assertEqual(wf.get_step_record(step).status, StepStatus.PENDING)

            # Execute step 1 manually
            rec1 = wf.step_inspect()
            self.assertEqual(rec1.status, StepStatus.COMPLETED)
            self.assertEqual(wf.get_step_record(WorkflowStep.PROFILE_BASELINE).status, StepStatus.PENDING)

            # Execute step 2 manually
            rec2 = wf.step_profile_baseline()
            self.assertEqual(rec2.status, StepStatus.COMPLETED)
            self.assertIn("located_candidates", rec2.data)

    def test_hypertuning_mode_in_step5(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()
            cfg = WorkflowConfig(
                model=model,
                output_dir=tmpdir,
                enable_hypertuning=True,
                budgets=WorkflowBudgets(max_hypertuning_trials=3),
            )
            wf = NeurosurgeryWorkflow(cfg)
            res = wf.run()

            self.assertTrue(res.success)
            rec5 = wf.get_step_record(WorkflowStep.RECOVERY_HYPERTUNING)
            self.assertEqual(rec5.status, StepStatus.COMPLETED)
            self.assertEqual(rec5.data.get("mode"), "hypertuning")

    def test_quantization_in_step6(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TinyCausalLM()
            cfg = WorkflowConfig(
                model=model,
                output_dir=tmpdir,
                enable_quantization=True,
            )
            wf = NeurosurgeryWorkflow(cfg)
            res = wf.run()

            self.assertTrue(res.success)
            rec6 = wf.get_step_record(WorkflowStep.EVALUATION_EXPORT)
            self.assertEqual(rec6.status, StepStatus.COMPLETED)
            self.assertIsNotNone(rec6.data.get("quantization"))
            self.assertTrue(rec6.data["quantization"]["quantization_applied"])


if __name__ == "__main__":
    unittest.main()

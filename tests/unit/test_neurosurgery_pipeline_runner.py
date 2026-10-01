"""Unit tests for AEGIS-LAB Declarative Campaign Pipeline Runner."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn
import yaml

from aegis_lab.editing.neurosurgery.pipeline_runner import (
    CampaignArtifact,
    CampaignConfig,
    CampaignConfigError,
    CampaignObjectives,
    CampaignPipelineRunner,
    CampaignResult,
    CampaignStatus,
    CampaignThresholds,
    DatasetSpec,
    ModelConfig,
    PreflightCheckError,
    PreflightCheckResult,
    PreflightResult,
    RollbackError,
    RollbackResult,
    StepConfig,
    StepExecutionError,
    StepResult,
    StepStatus,
    ThresholdGateError,
    ThresholdResult,
    rollback_campaign,
    run_campaign,
    run_preflight_checks,
)


# ==============================================================================
# Test Fixtures & Mock Models
# ==============================================================================


class TinyMLP(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, dim, bias=False)
        self.down_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class TinyAttention(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


class TinyLayer(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.self_attn = TinyAttention(dim)
        self.mlp = TinyMLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.self_attn(x) + self.mlp(x)


class TinyCausalLM(nn.Module):
    def __init__(self, num_layers: int = 3, dim: int = 16, vocab_size: int = 32):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "hidden_size": dim,
            "num_hidden_layers": num_layers,
            "model_type": "llama",
        }
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TinyLayer(dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        if input_ids is None:
            input_ids = torch.zeros((1, 4), dtype=torch.long)
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(h)


def create_jsonl_dataset(path: Path, prompts: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    return path


def setup_test_campaign_env(tmp_dir: Path) -> tuple[TinyCausalLM, Path, Path, Path, Path]:
    model_dir = tmp_dir / "tiny_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model = TinyCausalLM(num_layers=3, dim=16)
    (model_dir / "config.json").write_text(json.dumps(model.config), encoding="utf-8")
    torch.save(model.state_dict(), model_dir / "model.pt")

    keep_file = create_jsonl_dataset(
        tmp_dir / "data" / "keep.jsonl",
        [f"keep prompt {i}" for i in range(16)],
    )
    drop_file = create_jsonl_dataset(
        tmp_dir / "data" / "drop.jsonl",
        [f"drop prompt {i}" for i in range(16)],
    )
    calib_file = create_jsonl_dataset(
        tmp_dir / "data" / "calib.jsonl",
        [f"calibration prompt {i}" for i in range(8)],
    )
    return model, model_dir, keep_file, drop_file, calib_file


# ==============================================================================
# 1. Config Validation Tests
# ==============================================================================


class TestCampaignConfigValidation(unittest.TestCase):
    """Test campaign configuration schema, default assignment, and constraint validation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_valid_minimal_config_dict(self) -> None:
        raw = {
            "campaign_id": "camp_test_01",
            "output_dir": str(self.tmp / "out"),
            "model": {"path": str(self.tmp / "model"), "architecture": "llama"},
            "datasets": {
                "keep": {"path": str(self.tmp / "keep.jsonl"), "split": "discovery", "kind": "keep"}
            },
        }
        cfg = CampaignConfig.from_dict(raw)
        self.assertEqual(cfg.campaign_id, "camp_test_01")
        self.assertEqual(cfg.schema_version, "1.0")
        self.assertEqual(len(cfg.steps), 7)  # defaults to standard 7 steps
        self.assertEqual(cfg.model.architecture, "llama")

    def test_valid_yaml_config_roundtrip(self) -> None:
        raw = {
            "campaign_id": "camp_yaml_01",
            "schema_version": "1.0",
            "output_dir": str(self.tmp / "out"),
            "model": {"path": str(self.tmp / "model"), "device": "cpu"},
            "steps": [{"name": "profile"}, {"name": "select"}],
        }
        yaml_str = yaml.safe_dump(raw)
        cfg = CampaignConfig.from_yaml(yaml_str)
        self.assertEqual(cfg.campaign_id, "camp_yaml_01")
        self.assertEqual(len(cfg.steps), 2)
        # Roundtrip back to yaml
        yaml_out = cfg.to_yaml()
        cfg2 = CampaignConfig.from_yaml(yaml_out)
        self.assertEqual(cfg2.campaign_id, cfg.campaign_id)

    def test_invalid_schema_version_rejected(self) -> None:
        raw = {
            "campaign_id": "camp_bad_ver",
            "schema_version": "99.0",
            "output_dir": str(self.tmp / "out"),
            "model": {"path": str(self.tmp / "model")},
        }
        with self.assertRaises(CampaignConfigError) as ctx:
            CampaignConfig.from_dict(raw)
        self.assertIn("Unsupported schema_version", str(ctx.exception))

    def test_missing_required_fields(self) -> None:
        # Missing model section
        with self.assertRaises(CampaignConfigError):
            CampaignConfig.from_dict({"campaign_id": "camp_no_model", "output_dir": "./out"})

        # Empty campaign_id
        with self.assertRaises(CampaignConfigError):
            CampaignConfig(
                campaign_id="",
                output_dir="./out",
                model=ModelConfig(path="./model"),
            )

    def test_invalid_dataset_splits_and_kinds(self) -> None:
        # Invalid split
        with self.assertRaises(CampaignConfigError) as ctx:
            CampaignConfig(
                campaign_id="camp_bad_split",
                output_dir="./out",
                model=ModelConfig(path="./model"),
                datasets={
                    "k": DatasetSpec(name="k", path="./k.txt", split="invalid_split", kind="keep")
                },
            )
        self.assertIn("split 'invalid_split' is invalid", str(ctx.exception))

        # Invalid kind
        with self.assertRaises(CampaignConfigError) as ctx:
            CampaignConfig(
                campaign_id="camp_bad_kind",
                output_dir="./out",
                model=ModelConfig(path="./model"),
                datasets={
                    "k": DatasetSpec(name="k", path="./k.txt", split="discovery", kind="forbidden_kind")
                },
            )
        self.assertIn("kind 'forbidden_kind' is invalid", str(ctx.exception))

    def test_step_duplicate_names_rejected(self) -> None:
        with self.assertRaises(CampaignConfigError) as ctx:
            CampaignConfig(
                campaign_id="camp_dup_steps",
                output_dir="./out",
                model=ModelConfig(path="./model"),
                steps=[StepConfig(name="profile"), StepConfig(name="profile")],
            )
        self.assertIn("Duplicate step names", str(ctx.exception))

    def test_step_unknown_names_rejected(self) -> None:
        with self.assertRaises(CampaignConfigError) as ctx:
            CampaignConfig(
                campaign_id="camp_unknown_step",
                output_dir="./out",
                model=ModelConfig(path="./model"),
                steps=[StepConfig(name="non_existent_step")],
            )
        self.assertIn("Unrecognized step names", str(ctx.exception))

    def test_step_invalid_dependency_order_rejected(self) -> None:
        # Step 'apply' before 'select'
        with self.assertRaises(CampaignConfigError) as ctx:
            CampaignConfig(
                campaign_id="camp_bad_order",
                output_dir="./out",
                model=ModelConfig(path="./model"),
                steps=[StepConfig(name="apply"), StepConfig(name="select")],
            )
        self.assertIn("Step 'apply' cannot precede step 'select'", str(ctx.exception))

        # Step 'export' before 'apply'
        with self.assertRaises(CampaignConfigError) as ctx:
            CampaignConfig(
                campaign_id="camp_bad_order2",
                output_dir="./out",
                model=ModelConfig(path="./model"),
                steps=[StepConfig(name="export"), StepConfig(name="apply")],
            )
        self.assertIn("Step 'export' cannot precede step 'apply'", str(ctx.exception))


# ==============================================================================
# 2. Preflight Checks Tests
# ==============================================================================


class TestPreflightChecks(unittest.TestCase):
    """Test preflight validation of model path, datasets, adapter compatibility, and disk space."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)
        self.model, self.model_dir, self.keep_f, self.drop_f, self.calib_f = setup_test_campaign_env(self.tmp)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_preflight_all_valid_passes(self) -> None:
        cfg = CampaignConfig(
            campaign_id="preflight_pass",
            output_dir=str(self.tmp / "campaign_out"),
            model=ModelConfig(path=str(self.model_dir), architecture="llama"),
            datasets={
                "keep": DatasetSpec(name="keep", path=str(self.keep_f), split="discovery", kind="keep"),
                "drop": DatasetSpec(name="drop", path=str(self.drop_f), split="discovery", kind="drop"),
            },
            steps=[StepConfig(name="profile"), StepConfig(name="select")],
        )
        res = run_preflight_checks(cfg, min_disk_mb=1.0)
        self.assertTrue(res.passed)
        self.assertEqual(len(res.errors), 0)
        self.assertTrue(any(c.name == "model_path" and c.passed for c in res.checks))
        self.assertTrue(any(c.name == "adapter_compatibility" and c.passed for c in res.checks))
        self.assertTrue(any(c.name == "disk_space" and c.passed for c in res.checks))

    def test_preflight_nonexistent_model_fails(self) -> None:
        cfg = CampaignConfig(
            campaign_id="preflight_bad_model",
            output_dir=str(self.tmp / "campaign_out"),
            model=ModelConfig(path=str(self.tmp / "missing_dir")),
        )
        res = run_preflight_checks(cfg)
        self.assertFalse(res.passed)
        self.assertTrue(any("Model path does not exist" in e for e in res.errors))

    def test_preflight_missing_dataset_file_fails(self) -> None:
        cfg = CampaignConfig(
            campaign_id="preflight_bad_ds",
            output_dir=str(self.tmp / "campaign_out"),
            model=ModelConfig(path=str(self.model_dir)),
            datasets={
                "keep": DatasetSpec(name="keep", path=str(self.tmp / "missing.jsonl"), split="discovery", kind="keep")
            },
        )
        res = run_preflight_checks(cfg)
        self.assertFalse(res.passed)
        self.assertTrue(any("Dataset file does not exist" in e for e in res.errors))

    def test_preflight_empty_dataset_fails(self) -> None:
        empty_f = self.tmp / "empty.jsonl"
        empty_f.write_text("", encoding="utf-8")
        cfg = CampaignConfig(
            campaign_id="preflight_empty_ds",
            output_dir=str(self.tmp / "campaign_out"),
            model=ModelConfig(path=str(self.model_dir)),
            datasets={
                "keep": DatasetSpec(name="keep", path=str(empty_f), split="discovery", kind="keep")
            },
        )
        res = run_preflight_checks(cfg)
        self.assertFalse(res.passed)
        self.assertTrue(any("Dataset file is empty" in e for e in res.errors))

    def test_preflight_dataset_split_overlap_detected(self) -> None:
        overlap_f = create_jsonl_dataset(
            self.tmp / "data" / "overlap.jsonl",
            ["keep prompt 0", "keep prompt 1", "keep prompt 99"],  # overlaps with keep.jsonl
        )
        cfg = CampaignConfig(
            campaign_id="preflight_overlap",
            output_dir=str(self.tmp / "campaign_out"),
            model=ModelConfig(path=str(self.model_dir)),
            datasets={
                "keep_disc": DatasetSpec(name="keep_disc", path=str(self.keep_f), split="discovery", kind="keep"),
                "keep_val": DatasetSpec(name="keep_val", path=str(overlap_f), split="validation", kind="keep"),
            },
        )
        res = run_preflight_checks(cfg)
        self.assertFalse(res.passed)
        self.assertTrue(any("Dataset split overlap detected" in e for e in res.errors))

    def test_preflight_adapter_unsupported_operation_fails(self) -> None:
        # Architecture 'llama' does not support MoE operations
        cfg = CampaignConfig(
            campaign_id="preflight_adapter_mismatch",
            output_dir=str(self.tmp / "campaign_out"),
            model=ModelConfig(path=str(self.model_dir), architecture="llama"),
            steps=[
                StepConfig(
                    name="select",
                    params={"moe": {"experts": [0, 1]}},
                )
            ],
        )
        res = run_preflight_checks(cfg)
        self.assertFalse(res.passed)
        self.assertTrue(any("does not support MoE experts" in e for e in res.errors))

    def test_preflight_disk_space_insufficient_fails(self) -> None:
        cfg = CampaignConfig(
            campaign_id="preflight_disk_fail",
            output_dir=str(self.tmp / "campaign_out"),
            model=ModelConfig(path=str(self.model_dir)),
        )
        # Ask for 100 million Megabytes (100 TB)
        res = run_preflight_checks(cfg, min_disk_mb=100_000_000.0)
        self.assertFalse(res.passed)
        self.assertTrue(any("Insufficient disk space" in e for e in res.errors))

    def test_strict_preflight_raises_exception_in_runner(self) -> None:
        cfg = CampaignConfig(
            campaign_id="runner_strict_fail",
            output_dir=str(self.tmp / "campaign_out"),
            model=ModelConfig(path=str(self.tmp / "non_existent_model")),
        )
        runner = CampaignPipelineRunner(cfg)
        with self.assertRaises(PreflightCheckError):
            runner.run(strict_preflight=True)


# ==============================================================================
# 3. Campaign Execution Tests
# ==============================================================================


class TestCampaignExecution(unittest.TestCase):
    """Test end-to-end multi-stage campaign execution, checkpointing, and reporting."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)
        self.model, self.model_dir, self.keep_f, self.drop_f, self.calib_f = setup_test_campaign_env(self.tmp)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_full_7_stage_campaign_execution(self) -> None:
        out_dir = self.tmp / "full_campaign_out"
        cfg = CampaignConfig(
            campaign_id="full_camp_01",
            output_dir=str(out_dir),
            model=ModelConfig(
                path=str(self.model_dir),
                architecture="llama",
                model_instance=self.model,
            ),
            datasets={
                "keep": DatasetSpec(name="keep", path=str(self.keep_f), split="discovery", kind="keep"),
                "drop": DatasetSpec(name="drop", path=str(self.drop_f), split="discovery", kind="drop"),
                "calib": DatasetSpec(name="calib", path=str(self.calib_f), split="search", kind="calibration"),
            },
            steps=[
                StepConfig(name="profile", params={"max_prompts": 16}),
                StepConfig(
                    name="select",
                    params={
                        "drop_layers": [1],
                        "ablation": {
                            "layers": [0],
                            "targets": ["mlp.down_proj", "self_attn.o_proj"],
                            "strength": 0.5,
                        },
                    },
                ),
                StepConfig(name="preview"),
                StepConfig(name="apply"),
                StepConfig(name="recover", params={"steps": 3, "lr": 1e-4}),
                StepConfig(name="quantize", params={"bits": 8}),
                StepConfig(name="export", params={"format": "pytorch"}),
            ],
            thresholds=CampaignThresholds(
                max_mean_kl=0.05,
                max_drop_rebound=0.10,
                min_parameter_reduction=10,
            ),
        )

        runner = CampaignPipelineRunner(cfg)
        result = runner.run()

        self.assertEqual(result.status, CampaignStatus.COMPLETED)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.step_results), 7)

        # Verify all steps succeeded
        for step_name, s_res in result.step_results.items():
            self.assertEqual(s_res.status, StepStatus.COMPLETED, f"Step {step_name} did not complete")
            self.assertGreater(s_res.duration_seconds, 0)
            self.assertTrue(len(s_res.artifacts) > 0, f"Step {step_name} produced no artifacts")

        # Verify checkpoint file exists
        ckpt_path = out_dir / "campaign_checkpoint.json"
        self.assertTrue(ckpt_path.exists())
        ckpt_data = json.loads(ckpt_path.read_text(encoding="utf-8"))
        self.assertEqual(ckpt_data["last_completed_step"], "export")
        self.assertEqual(len(ckpt_data["steps"]), 7)

        # Verify operator report
        self.assertIsNotNone(result.report_txt_path)
        self.assertTrue(result.report_txt_path.exists())
        txt = result.report_txt_path.read_text(encoding="utf-8")
        self.assertIn("AEGIS-LAB NEUROSURGERY CAMPAIGN REPORT", txt)
        self.assertIn("CAMPAIGN VERDICT: PROMOTED", txt)
        self.assertIn("profile", txt)
        self.assertIn("export", txt)

        # Verify machine JSON report
        self.assertIsNotNone(result.report_json_path)
        self.assertTrue(result.report_json_path.exists())
        json_data = json.loads(result.report_json_path.read_text(encoding="utf-8"))
        self.assertEqual(json_data["status"], "COMPLETED")
        self.assertTrue(json_data["passed"])

        # Verify Stage 4A consolidated provenance manifest
        self.assertIsNotNone(result.provenance_manifest_path)
        self.assertTrue(result.provenance_manifest_path.exists())
        prov_data = json.loads(result.provenance_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(prov_data["schema_version"], "4a.1")
        self.assertIn("model_hash", prov_data)
        self.assertIn("dataset_fingerprints", prov_data)
        self.assertEqual(prov_data["operation_order"], [s.name for s in cfg.steps])

        # Verify restoration manifest generated during apply
        rest_manifest = out_dir / "apply" / "restoration_manifest.yaml"
        self.assertTrue(rest_manifest.exists())

        # Verify baseline backup preserved
        baseline_backup = out_dir / "backup" / "baseline_model.pt"
        self.assertTrue(baseline_backup.exists())

    def test_partial_step_subset_execution(self) -> None:
        out_dir = self.tmp / "partial_out"
        cfg = CampaignConfig(
            campaign_id="partial_camp",
            output_dir=str(out_dir),
            model=ModelConfig(path=str(self.model_dir), model_instance=self.model),
            steps=[
                StepConfig(name="select", params={"drop_layers": [0]}),
                StepConfig(name="preview"),
                StepConfig(name="apply"),
            ],
        )
        runner = CampaignPipelineRunner(cfg)
        res = runner.run()
        self.assertEqual(res.status, CampaignStatus.COMPLETED)
        self.assertEqual(set(res.step_results.keys()), {"select", "preview", "apply"})

    def test_threshold_violation_fails_campaign(self) -> None:
        out_dir = self.tmp / "threshold_fail_out"
        cfg = CampaignConfig(
            campaign_id="thresh_fail_camp",
            output_dir=str(out_dir),
            model=ModelConfig(path=str(self.model_dir), model_instance=self.model),
            steps=[
                StepConfig(name="select", params={"drop_layers": []}),
                StepConfig(name="preview"),
                StepConfig(name="apply"),
            ],
            thresholds=CampaignThresholds(
                # Demand at least 1,000,000 parameters removed, which won't happen
                min_parameter_reduction=1_000_000
            ),
        )
        runner = CampaignPipelineRunner(cfg)
        res = runner.run()
        self.assertEqual(res.status, CampaignStatus.COMPLETED)
        self.assertFalse(res.passed)
        txt = res.report_txt_path.read_text(encoding="utf-8")
        self.assertIn("CAMPAIGN VERDICT: REJECTED", txt)
        self.assertFalse(res.threshold_results["min_parameter_reduction"].passed)


# ==============================================================================
# 4. Resume from Checkpoint Tests
# ==============================================================================


class TestResumeFromCheckpoint(unittest.TestCase):
    """Test step resumption from checkpoint and invalidation on artifact tampering."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)
        self.model, self.model_dir, self.keep_f, self.drop_f, self.calib_f = setup_test_campaign_env(self.tmp)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_resume_interrupted_campaign(self) -> None:
        out_dir = self.tmp / "resume_campaign_out"

        # Step 1: Run only initial steps (profile, select, preview)
        cfg_part1 = CampaignConfig(
            campaign_id="resume_test_camp",
            output_dir=str(out_dir),
            model=ModelConfig(path=str(self.model_dir), model_instance=self.model),
            datasets={
                "keep": DatasetSpec(name="keep", path=str(self.keep_f), split="discovery", kind="keep"),
                "drop": DatasetSpec(name="drop", path=str(self.drop_f), split="discovery", kind="drop"),
            },
            steps=[
                StepConfig(name="profile"),
                StepConfig(name="select", params={"drop_layers": [1]}),
                StepConfig(name="preview"),
            ],
        )
        runner1 = CampaignPipelineRunner(cfg_part1)
        res1 = runner1.run()
        self.assertEqual(res1.status, CampaignStatus.COMPLETED)
        self.assertEqual(len(res1.step_results), 3)

        # Step 2: Now run full campaign with resume=True
        cfg_part2 = CampaignConfig(
            campaign_id="resume_test_camp",
            output_dir=str(out_dir),
            model=ModelConfig(path=str(self.model_dir), model_instance=self.model),
            datasets={
                "keep": DatasetSpec(name="keep", path=str(self.keep_f), split="discovery", kind="keep"),
                "drop": DatasetSpec(name="drop", path=str(self.drop_f), split="discovery", kind="drop"),
            },
            steps=[
                StepConfig(name="profile"),
                StepConfig(name="select", params={"drop_layers": [1]}),
                StepConfig(name="preview"),
                StepConfig(name="apply"),
                StepConfig(name="recover"),
            ],
        )
        runner2 = CampaignPipelineRunner(cfg_part2)
        res2 = runner2.run(resume=True)

        self.assertEqual(res2.status, CampaignStatus.COMPLETED)
        self.assertEqual(len(res2.step_results), 5)
        # Verify first 3 steps were resumed from checkpoint
        self.assertTrue(res2.step_results["profile"].resumed_from_checkpoint)
        self.assertTrue(res2.step_results["select"].resumed_from_checkpoint)
        self.assertTrue(res2.step_results["preview"].resumed_from_checkpoint)
        # Verify subsequent steps were newly executed
        self.assertFalse(res2.step_results["apply"].resumed_from_checkpoint)
        self.assertFalse(res2.step_results["recover"].resumed_from_checkpoint)

    def test_resume_invalidates_corrupted_artifact(self) -> None:
        out_dir = self.tmp / "resume_tampered_out"

        # Run first 2 steps
        cfg = CampaignConfig(
            campaign_id="tamper_camp",
            output_dir=str(out_dir),
            model=ModelConfig(path=str(self.model_dir), model_instance=self.model),
            steps=[
                StepConfig(name="select", params={"drop_layers": [0]}),
                StepConfig(name="preview"),
                StepConfig(name="apply"),
            ],
        )
        runner = CampaignPipelineRunner(cfg)
        # First execute only select and preview
        cfg_sub = copy.deepcopy(cfg)
        cfg_sub.steps = cfg.steps[:2]
        CampaignPipelineRunner(cfg_sub).run()

        # Tamper with preview.json artifact
        preview_file = out_dir / "preview" / "preview.json"
        self.assertTrue(preview_file.exists())
        preview_file.write_text('{"tampered": true}', encoding="utf-8")

        # Now resume
        runner2 = CampaignPipelineRunner(cfg)
        res2 = runner2.run(resume=True)
        self.assertEqual(res2.status, CampaignStatus.COMPLETED)
        # Preview should NOT be resumed because its hash mismatched!
        self.assertFalse(res2.step_results["preview"].resumed_from_checkpoint)
        self.assertTrue(res2.step_results["select"].resumed_from_checkpoint)


# ==============================================================================
# 5. Failure Rollback & Error Handling Tests
# ==============================================================================


class TestFailureRollback(unittest.TestCase):
    """Test step error handling, failure diagnostics capture, and baseline rollback."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp_dir.name)
        self.model, self.model_dir, self.keep_f, self.drop_f, self.calib_f = setup_test_campaign_env(self.tmp)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_step_failure_captures_diagnostics_and_rolls_back(self) -> None:
        out_dir = self.tmp / "failure_rollback_out"

        # Define a custom failing step handler
        def failing_recover_handler(step_dir: Path, params: dict):
            # Create a temporary file to check it gets cleaned
            (step_dir / "temp_scratch.tmp").write_text("scratch", encoding="utf-8")
            raise RuntimeError("Simulated OOM or training explosion during recovery")

        cfg = CampaignConfig(
            campaign_id="fail_camp",
            output_dir=str(out_dir),
            model=ModelConfig(path=str(self.model_dir), model_instance=self.model),
            steps=[
                StepConfig(name="select", params={"drop_layers": [1]}),
                StepConfig(name="preview"),
                StepConfig(name="apply"),
                StepConfig(name="recover"),
            ],
        )

        runner = CampaignPipelineRunner(
            cfg,
            custom_step_handlers={"recover": failing_recover_handler},
        )

        with self.assertRaises(StepExecutionError) as ctx:
            runner.run(rollback_on_failure=True)

        self.assertIn("Simulated OOM", str(ctx.exception))

        # Verify failure diagnostics file written
        diag_path = out_dir / "failure_diagnostics.json"
        self.assertTrue(diag_path.exists())
        diag_data = json.loads(diag_path.read_text(encoding="utf-8"))
        self.assertEqual(diag_data["failed_step"], "recover")
        self.assertEqual(diag_data["exception_type"], "RuntimeError")
        self.assertIn("Simulated OOM", diag_data["exception_message"])
        self.assertIn("system_resources", diag_data)

        # Verify reports updated with failure
        rep_txt = out_dir / "campaign_report.txt"
        self.assertTrue(rep_txt.exists())
        txt = rep_txt.read_text(encoding="utf-8")
        self.assertIn("CAMPAIGN VERDICT: REJECTED", txt)
        self.assertIn("Simulated OOM", txt)

        # Verify baseline backup was preserved for exact restoration
        backup_model = out_dir / "backup" / "baseline_model.pt"
        self.assertTrue(backup_model.exists())

        # Verify scratch .tmp file cleaned during rollback
        self.assertFalse((out_dir / "recover" / "temp_scratch.tmp").exists())

    def test_standalone_rollback_campaign_api(self) -> None:
        out_dir = self.tmp / "standalone_rollback_out"
        backup_dir = out_dir / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), backup_dir / "baseline_model.pt")

        # Create a scratch tmp file
        scratch_file = out_dir / ".tmp_stage.tmp"
        scratch_file.write_text("junk", encoding="utf-8")

        res = rollback_campaign(out_dir)
        self.assertTrue(res.success)
        self.assertEqual(len(res.restored_files), 1)
        self.assertTrue(scratch_file.name in str(res.cleaned_paths) or not scratch_file.exists())

    def test_standalone_rollback_missing_backup_fails_safely(self) -> None:
        empty_dir = self.tmp / "empty_dir"
        empty_dir.mkdir(parents=True, exist_ok=True)
        res = rollback_campaign(empty_dir)
        self.assertFalse(res.success)
        self.assertIn("No baseline backup found", res.error)

    def test_failure_during_apply_step_rolls_back(self) -> None:
        out_dir = self.tmp / "fail_apply_out"

        def failing_apply_handler(step_dir: Path, params: dict):
            (step_dir / "partial.tmp").write_text("partial", encoding="utf-8")
            raise ValueError("Intentional error during surgery application")

        cfg = CampaignConfig(
            campaign_id="fail_apply_camp",
            output_dir=str(out_dir),
            model=ModelConfig(path=str(self.model_dir), model_instance=self.model),
            steps=[
                StepConfig(name="select", params={"drop_layers": [0]}),
                StepConfig(name="preview"),
                StepConfig(name="apply"),
            ],
        )

        runner = CampaignPipelineRunner(
            cfg,
            custom_step_handlers={"apply": failing_apply_handler},
        )
        with self.assertRaises(StepExecutionError) as ctx:
            runner.run(rollback_on_failure=True)
        self.assertIn("Intentional error during surgery application", str(ctx.exception))
        self.assertTrue((out_dir / "failure_diagnostics.json").exists())
        self.assertFalse((out_dir / "apply" / "partial.tmp").exists())

    def test_custom_objectives_and_custom_thresholds(self) -> None:
        out_dir = self.tmp / "custom_thresh_out"
        cfg = CampaignConfig(
            campaign_id="custom_thresh_camp",
            output_dir=str(out_dir),
            model=ModelConfig(path=str(self.model_dir), model_instance=self.model),
            objectives=CampaignObjectives(
                primary="drop_suppression",
                metrics=["drop_rebound_score", "custom_metric"],
                targets={"custom_metric": 0.5},
            ),
            thresholds=CampaignThresholds(
                custom={"custom_metric": 0.40}
            ),
            steps=[
                StepConfig(name="select", params={"drop_layers": []}),
                StepConfig(name="preview"),
            ],
        )

        def custom_preview_handler(step_dir: Path, params: dict):
            return {"custom_metric": 0.35, "parameters_removed": 0}

        runner = CampaignPipelineRunner(
            cfg,
            custom_step_handlers={"preview": custom_preview_handler},
        )
        res = runner.run()
        self.assertEqual(res.status, CampaignStatus.COMPLETED)
        self.assertTrue(res.passed)
        self.assertTrue(res.threshold_results["custom_metric"].passed)

    def test_config_yaml_file_loading_and_run(self) -> None:
        yaml_file = self.tmp / "test_campaign.yaml"
        out_dir = self.tmp / "yaml_run_out"
        yaml_content = f"""
schema_version: "1.0"
campaign_id: "camp_from_yaml_file"
output_dir: "{out_dir}"
seed: 123
model:
  path: "{self.model_dir}"
  architecture: "llama"
datasets:
  keep:
    path: "{self.keep_f}"
    split: "discovery"
    kind: "keep"
  drop:
    path: "{self.drop_f}"
    split: "discovery"
    kind: "drop"
steps:
  - name: "profile"
  - name: "select"
    params:
      drop_layers: [1]
  - name: "preview"
thresholds:
  min_parameter_reduction: 0
"""
        yaml_file.write_text(yaml_content, encoding="utf-8")
        runner = CampaignPipelineRunner(yaml_file)
        # Supply model instance to avoid reloading from disk
        runner.model = self.model
        res = runner.run()
        self.assertEqual(res.status, CampaignStatus.COMPLETED)
        self.assertEqual(res.campaign_id, "camp_from_yaml_file")
        self.assertTrue((out_dir / "campaign_report.txt").exists())
        self.assertTrue((out_dir / "campaign_report.json").exists())

    def test_campaign_result_to_dict_serializability(self) -> None:
        out_dir = self.tmp / "serial_out"
        cfg = CampaignConfig(
            campaign_id="serial_camp",
            output_dir=str(out_dir),
            model=ModelConfig(path=str(self.model_dir), model_instance=self.model),
            steps=[
                StepConfig(name="select", params={"drop_layers": []}),
                StepConfig(name="preview"),
            ],
        )
        res = CampaignPipelineRunner(cfg).run()
        res_dict = res.to_dict()
        # Verify completely serializable to JSON without error
        serialized = json.dumps(res_dict)
        self.assertIn("serial_camp", serialized)
        self.assertIn("step_results", serialized)


if __name__ == "__main__":
    unittest.main()

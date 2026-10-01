"""Unit tests for Unified CLI Subcommand Extensions in model neurosurgery.

Verifies:
1. Parser argument construction for all 8 extended subcommands:
   `patch`, `recover`, `hypertune`, `quantize`, `export-runtime`, `ampute`, `unlearn`, `workflow`.
2. Preservation and compatibility with base CLI subcommands.
3. Validation error handling (missing args, invalid choices, out-of-range numerics, missing files).
4. Subcommand dispatch and handler execution.
5. End-to-end mock execution of each neurosurgery stage handler.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from aegis_lab.editing.neurosurgery.cli_extended import (
    CLIValidationError,
    NeurosurgeryCLIError,
    build_extended_parser,
    dispatch_extended,
    handle_ampute,
    handle_export_runtime,
    handle_hypertune,
    handle_patch,
    handle_quantize,
    handle_recover,
    handle_unlearn,
    handle_workflow,
    main,
)


# ---------------------------------------------------------------------------
# Dummy Test Model Fixtures
# ---------------------------------------------------------------------------


class DummyTransformerLayer(nn.Module):
    def __init__(self, hidden: int = 8, intermediate: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)
        return self.down_proj(h)


class DummyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([DummyTransformerLayer(hidden=hidden) for _ in range(2)])
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.config = SimpleNamespace(
            model_type="dummy",
            vocab_size=vocab_size,
            hidden_size=hidden,
            num_hidden_layers=2,
            architectures=["DummyForCausalLM"],
        )

    def forward(self, input_ids: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        b, s = input_ids.shape if input_ids is not None else (1, 4)
        x = torch.randn(b, s, 8)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class DummyVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DummyMultimodalModel(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.vision_tower = DummyVisionTower()
        self.layers = nn.ModuleList([DummyTransformerLayer(hidden=hidden)])
        self.lm_head = nn.Linear(hidden, 16, bias=False)
        self.config = SimpleNamespace(
            model_type="llava",
            architectures=["LlavaForConditionalGeneration"],
            vision_config={"hidden_size": hidden},
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if pixel_values is not None:
            return self.vision_tower(pixel_values)
        b, s = input_ids.shape if input_ids is not None else (1, 4)
        x = torch.randn(b, s, 8)
        x = self.layers[0](x)
        return self.lm_head(x)


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, return_tensors="pt", **kwargs):
        if isinstance(text, str):
            text = [text]
        return {
            "input_ids": torch.randint(0, 16, (len(text), 4)),
            "attention_mask": torch.ones((len(text), 4), dtype=torch.long),
        }

    def decode(self, ids, **kwargs):
        return "synthetic decoded completion"


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class TestNeurosurgeryExtendedParser(unittest.TestCase):
    """Verifies parser creation, subcommands presence, and argument parsing."""

    def setUp(self):
        self.parser = build_extended_parser()
        self.subparser_action = next(
            a for a in self.parser._actions if getattr(a, "choices", None) and "patch" in a.choices
        )

    def test_all_8_extended_subcommands_registered(self):
        expected_cmds = [
            "patch",
            "recover",
            "hypertune",
            "quantize",
            "export-runtime",
            "ampute",
            "unlearn",
            "workflow",
        ]
        for cmd in expected_cmds:
            self.assertIn(cmd, self.subparser_action.choices, f"Subcommand {cmd!r} must be registered")

    def test_base_cli_subcommands_preserved(self):
        base_cmds = ["profile", "plan", "apply", "validate", "preview", "inventory", "optimize"]
        for cmd in base_cmds:
            self.assertIn(cmd, self.subparser_action.choices, f"Base subcommand {cmd!r} must be preserved")

    def test_idempotent_parser_extension(self):
        p = build_extended_parser()
        # Calling build_extended_parser again on existing parser should not raise
        extended = build_extended_parser(base_parser=p)
        self.assertIs(p, extended)

    def test_parse_patch_arguments(self):
        args = self.parser.parse_args([
            "patch",
            "--model", "dummy_model",
            "--workload", "workload.json",
            "--out", "out.json",
            "--target-type", "attention_head",
            "--metric", "target_prob_diff",
            "--token-indices=-1,0",
            "--head-dim", "64",
            "--num-heads", "8",
            "--layer-indices", "0,1",
            "--max-bytes", "1048576",
            "--controls",
            "--num-random-trials", "10",
            "--seed", "123",
            "--device", "cpu",
        ])
        self.assertEqual(args.cmd, "patch")
        self.assertEqual(args.model, "dummy_model")
        self.assertEqual(args.target_type, "attention_head")
        self.assertEqual(args.metric, "target_prob_diff")
        self.assertEqual(args.token_indices, "-1,0")
        self.assertEqual(args.head_dim, 64)
        self.assertEqual(args.num_heads, 8)
        self.assertEqual(args.max_bytes, 1048576)
        self.assertTrue(args.controls)
        self.assertEqual(args.num_random_trials, 10)
        self.assertEqual(args.seed, 123)

    def test_parse_recover_arguments(self):
        args = self.parser.parse_args([
            "recover",
            "--model", "dummy_model",
            "--out", "recovered_dir",
            "--keep", "keep.txt",
            "--change", "change.txt",
            "--distill", "distill.txt",
            "--teacher", "teacher_model",
            "--lora-targets", "q_proj,v_proj",
            "--lora-r", "16",
            "--lora-alpha", "32.0",
            "--lora-dropout", "0.05",
            "--lr", "0.0002",
            "--max-steps", "100",
            "--gradient-accumulation-steps", "2",
            "--keep-weight", "1.5",
            "--change-weight", "0.8",
            "--distill-weight", "0.5",
            "--temperature", "2.5",
            "--merge",
            "--checkpoint-dir", "checkpoints",
        ])
        self.assertEqual(args.cmd, "recover")
        self.assertEqual(args.lora_r, 16)
        self.assertEqual(args.lora_alpha, 32.0)
        self.assertEqual(args.lora_dropout, 0.05)
        self.assertEqual(args.lr, 0.0002)
        self.assertEqual(args.max_steps, 100)
        self.assertEqual(args.gradient_accumulation_steps, 2)
        self.assertEqual(args.keep_weight, 1.5)
        self.assertEqual(args.change_weight, 0.8)
        self.assertEqual(args.distill_weight, 0.5)
        self.assertEqual(args.temperature, 2.5)
        self.assertTrue(args.merge)

    def test_parse_hypertune_arguments(self):
        args = self.parser.parse_args([
            "hypertune",
            "--model", "dummy_model",
            "--out", "study_dir",
            "--study-id", "custom_study",
            "--strategy", "successive_halving",
            "--max-trials", "32",
            "--max-drop-threshold", "0.05",
            "--max-keep-kl", "0.02",
            "--min-top1-agreement", "0.95",
            "--max-time-seconds", "300.0",
            "--max-memory-mb", "4096.0",
        ])
        self.assertEqual(args.cmd, "hypertune")
        self.assertEqual(args.strategy, "successive_halving")
        self.assertEqual(args.max_trials, 32)
        self.assertEqual(args.max_drop_threshold, 0.05)
        self.assertEqual(args.max_keep_kl, 0.02)
        self.assertEqual(args.min_top1_agreement, 0.95)
        self.assertEqual(args.max_time_seconds, 300.0)
        self.assertEqual(args.max_memory_mb, 4096.0)

    def test_parse_quantize_arguments(self):
        args = self.parser.parse_args([
            "quantize",
            "--model", "dummy_model",
            "--out", "quant_dir",
            "--bits", "4",
            "--mode", "asymmetric",
            "--group-size", "128",
            "--calibration-data", "calib.json",
            "--mixed-precision",
            "--max-drift-kl", "0.08",
            "--verify-reload",
        ])
        self.assertEqual(args.cmd, "quantize")
        self.assertEqual(args.bits, 4)
        self.assertEqual(args.mode, "asymmetric")
        self.assertEqual(args.group_size, 128)
        self.assertTrue(args.mixed_precision)
        self.assertEqual(args.max_drift_kl, 0.08)
        self.assertTrue(args.verify_reload)

    def test_parse_export_runtime_arguments(self):
        args = self.parser.parse_args([
            "export-runtime",
            "--model", "dummy_model",
            "--out", "export_dir",
            "--format", "safetensors",
            "--target-runtime", "transformers",
            "--benchmark",
            "--batch-size", "4",
            "--prompt-length", "64",
            "--decode-steps", "32",
            "--num-warmup", "2",
            "--num-repeats", "10",
        ])
        self.assertEqual(args.cmd, "export-runtime")
        self.assertEqual(args.format, "safetensors")
        self.assertEqual(args.target_runtime, "transformers")
        self.assertTrue(args.benchmark)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.prompt_length, 64)
        self.assertEqual(args.decode_steps, 32)
        self.assertEqual(args.num_warmup, 2)
        self.assertEqual(args.num_repeats, 10)

    def test_parse_ampute_arguments(self):
        args = self.parser.parse_args([
            "ampute",
            "--model", "multimodal_model",
            "--out", "amputed_dir",
            "--modality", "vision",
            "--retained-data", "retained.txt",
            "--no-require-proof",
            "--no-install-guard",
            "--no-repair-config",
            "--max-allowed-diff", "0.0001",
        ])
        self.assertEqual(args.cmd, "ampute")
        self.assertEqual(args.modality, "vision")
        self.assertTrue(args.no_require_proof)
        self.assertTrue(args.no_install_guard)
        self.assertTrue(args.no_repair_config)
        self.assertEqual(args.max_allowed_diff, 0.0001)

    def test_parse_unlearn_arguments(self):
        args = self.parser.parse_args([
            "unlearn",
            "--model", "dummy_model",
            "--out", "unlearned_dir",
            "--benchmark", "benchmark.json",
            "--target-layer", "layers.0.down_proj",
            "--method", "rank1",
            "--refusal-target", "I cannot fulfill this.",
            "--rank", "2",
            "--learning-rate", "0.005",
            "--num-steps", "50",
            "--max-delta-norm", "2.0",
            "--audit",
        ])
        self.assertEqual(args.cmd, "unlearn")
        self.assertEqual(args.target_layer, "layers.0.down_proj")
        self.assertEqual(args.method, "rank1")
        self.assertEqual(args.rank, 2)
        self.assertEqual(args.learning_rate, 0.005)
        self.assertEqual(args.num_steps, 50)
        self.assertEqual(args.max_delta_norm, 2.0)
        self.assertTrue(args.audit)

    def test_parse_workflow_arguments(self):
        args = self.parser.parse_args([
            "workflow",
            "--model", "dummy_model",
            "--out", "workflow_dir",
            "--config", "workflow_config.yaml",
            "--stages", "inspect,plan,apply,recover,validate,export",
            "--lora-r", "8",
            "--max-recovery-steps", "15",
            "--quantize-bits", "8",
            "--export-format", "pytorch",
            "--dry-run",
        ])
        self.assertEqual(args.cmd, "workflow")
        self.assertEqual(args.stages, "inspect,plan,apply,recover,validate,export")
        self.assertEqual(args.lora_r, 8)
        self.assertEqual(args.max_recovery_steps, 15)
        self.assertEqual(args.quantize_bits, 8)
        self.assertEqual(args.export_format, "pytorch")
        self.assertTrue(args.dry_run)


class TestCLIValidationAndErrorHandling(unittest.TestCase):
    """Verifies that CLI parser and action handlers enforce validation constraints."""

    def setUp(self):
        self.parser = build_extended_parser()

    def test_missing_required_arguments_exit(self):
        # patch missing --workload
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["patch", "--model", "m", "--out", "o"])

        # recover missing --keep
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["recover", "--model", "m", "--out", "o"])

        # hypertune missing --model
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["hypertune", "--out", "o"])

        # unlearn missing --target-layer
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["unlearn", "--model", "m", "--out", "o", "--benchmark", "b"])

    def test_invalid_choices_rejected(self):
        # Invalid target-type
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["patch", "--model", "m", "--workload", "w", "--out", "o", "--target-type", "bad"])

        # Invalid bits
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["quantize", "--model", "m", "--out", "o", "--bits", "16"])

        # Invalid export format
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["export-runtime", "--model", "m", "--out", "o", "--format", "gguf"])

        # Invalid hypertune strategy
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["hypertune", "--model", "m", "--out", "o", "--strategy", "random_walk"])

    def test_negative_or_zero_values_rejected_by_parser(self):
        # Non-positive lora-r
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["recover", "--model", "m", "--out", "o", "--keep", "k", "--lora-r", "0"])

        # Non-positive lr
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["recover", "--model", "m", "--out", "o", "--keep", "k", "--lr", "-0.01"])

        # Non-positive max-trials
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["hypertune", "--model", "m", "--out", "o", "--max-trials", "-5"])

        # Out-of-bounds unit interval min-top1-agreement
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["hypertune", "--model", "m", "--out", "o", "--min-top1-agreement", "1.5"])

        # Non-positive batch-size in export-runtime
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["export-runtime", "--model", "m", "--out", "o", "--batch-size", "0"])

    def test_nonexistent_workload_file_raises_filenotfound(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                model=DummyLanguageModel(),
                workload=str(Path(tmpdir) / "missing_workload.json"),
                out=str(Path(tmpdir) / "out.json"),
                target_type="layer",
                metric="recovery_ratio",
                token_indices="-1",
                head_dim=None,
                num_heads=None,
                layer_indices=None,
                max_bytes=None,
                keep=None,
                max_allowed_kl=0.5,
                controls=False,
                num_random_trials=3,
                seed=42,
                device="cpu",
            )
            with self.assertRaises(FileNotFoundError):
                handle_patch(args)

    def test_nonexistent_benchmark_file_raises_filenotfound(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                model=DummyLanguageModel(),
                out=str(Path(tmpdir) / "out"),
                benchmark=str(Path(tmpdir) / "missing_bench.json"),
                target_layer="layers.0.down_proj",
                method="refusal",
                refusal_target="I cannot assist.",
                rank=1,
                learning_rate=1e-3,
                num_steps=10,
                max_delta_norm=1.0,
                audit=False,
                device="cpu",
            )
            with self.assertRaises(FileNotFoundError):
                handle_unlearn(args)

    def test_handler_validation_rejects_empty_lora_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            kpath = Path(tmpdir) / "keep.txt"
            kpath.write_text("prompt\n")
            args = argparse.Namespace(
                model=DummyLanguageModel(),
                out=str(Path(tmpdir) / "out"),
                keep=str(kpath),
                change=None,
                distill=None,
                teacher=None,
                lora_targets="",
                lora_r=8,
                lora_alpha=16.0,
                lora_dropout=0.0,
                lr=1e-4,
                weight_decay=0.01,
                max_steps=10,
                gradient_accumulation_steps=1,
                keep_weight=1.0,
                change_weight=1.0,
                distill_weight=0.0,
                temperature=2.0,
                max_drop_threshold=None,
                max_drop_rebound=None,
                early_stopping_patience=None,
                merge=False,
                checkpoint_dir=None,
                seed=42,
                device="cpu",
            )
            with self.assertRaises(CLIValidationError):
                handle_recover(args)

    def test_handler_validation_rejects_missing_model_and_config_in_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                model=None,
                out=str(Path(tmpdir) / "out"),
                config=None,
                stages="inspect",
                dry_run=True,
                device="cpu",
            )
            with self.assertRaises(CLIValidationError):
                handle_workflow(args)


class TestSubcommandDispatch(unittest.TestCase):
    """Verifies that subcommand dispatch routes parsed arguments correctly."""

    def test_dispatch_calls_handler_attribute(self):
        called = {"value": False}

        def mock_handler(args):
            called["value"] = True
            return {"status": "dispatched"}

        ns = argparse.Namespace(cmd="custom", handler=mock_handler)
        res = dispatch_extended(ns)
        self.assertTrue(called["value"])
        self.assertEqual(res["status"], "dispatched")

    def test_dispatch_fallback_by_cmd_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ns = argparse.Namespace(
                cmd="workflow",
                model=DummyLanguageModel(),
                out=str(Path(tmpdir) / "wf_out"),
                config=None,
                keep=None,
                drop=None,
                change=None,
                stages="inspect",
                plan=None,
                lora_r=4,
                max_recovery_steps=2,
                quantize_bits=None,
                export_format="safetensors",
                dry_run=True,
                device="cpu",
            )
            res = dispatch_extended(ns)
            self.assertEqual(res["command"], "workflow")
            self.assertEqual(res["mode"], "dry_run")

    def test_dispatch_unknown_cmd_raises_cli_error(self):
        ns = argparse.Namespace(cmd="unrecognized_command")
        with self.assertRaises(CLIValidationError):
            dispatch_extended(ns)

    def test_main_cli_execution_with_dry_run_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "main_out"
            # In a clean process invocation, model is specified via string or path
            # Test that main() handles dry-run workflow returning 0
            exit_code = main([
                "workflow",
                "--model", "dummy_local_model",
                "--out", str(out_dir),
                "--stages", "inspect,plan",
                "--dry-run",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "workflow_report.json").exists())


class TestSubcommandHandlersExecution(unittest.TestCase):
    """Tests mock and end-to-end execution of all 8 neurosurgery subcommands."""

    def test_handle_patch_execution(self):
        model = DummyLanguageModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            w_file = Path(tmpdir) / "workload.json"
            w_file.write_text(
                json.dumps([
                    {
                        "clean_input": {"input_ids": [[1, 2, 3]]},
                        "corrupted_input": {"input_ids": [[4, 5, 6]]},
                        "clean_target": 0,
                        "corrupted_target": 1,
                    }
                ])
            )
            out_file = Path(tmpdir) / "ranked.json"
            args = argparse.Namespace(
                model=model,
                workload=str(w_file),
                out=str(out_file),
                target_type="layer",
                metric="recovery_ratio",
                token_indices="-1",
                head_dim=None,
                num_heads=None,
                layer_indices=None,
                max_bytes=None,
                keep=None,
                max_allowed_kl=0.5,
                controls=False,
                num_random_trials=3,
                seed=42,
                device="cpu",
            )
            rep = handle_patch(args)
            self.assertTrue(out_file.exists())
            self.assertEqual(rep["command"], "patch")
            self.assertEqual(rep["total_ranked"], 2)
            self.assertEqual(len(rep["ranked_components"]), 2)

    def test_handle_recover_execution(self):
        model = DummyLanguageModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            k_file = Path(tmpdir) / "keep.txt"
            k_file.write_text("sample prompt 1\nsample prompt 2\n")
            out_dir = Path(tmpdir) / "rec_out"
            args = argparse.Namespace(
                model=model,
                out=str(out_dir),
                keep=str(k_file),
                change=None,
                distill=None,
                teacher=None,
                lora_targets="layers.0.down_proj",
                lora_r=4,
                lora_alpha=8.0,
                lora_dropout=0.0,
                lr=1e-3,
                weight_decay=0.01,
                max_steps=2,
                gradient_accumulation_steps=1,
                keep_weight=1.0,
                change_weight=1.0,
                distill_weight=0.0,
                temperature=2.0,
                max_drop_threshold=None,
                max_drop_rebound=None,
                early_stopping_patience=None,
                merge=False,
                checkpoint_dir=None,
                seed=42,
                device="cpu",
            )
            rep = handle_recover(args)
            self.assertTrue(rep["success"])
            self.assertTrue((out_dir / "recovery_report.json").exists())
            self.assertTrue((out_dir / "lora_adapter.pt").exists())

    def test_handle_hypertune_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "study_out"
            args = argparse.Namespace(
                model="dummy",
                out=str(out_dir),
                study_id="hypertune_unit_test",
                strategy="seeded_random",
                max_trials=3,
                max_drop_threshold=0.5,
                max_keep_kl=0.1,
                min_top1_agreement=0.8,
                max_time_seconds=None,
                max_memory_mb=None,
                seed=42,
                keep=None,
                drop=None,
                search_space=None,
                device="cpu",
            )
            rep = handle_hypertune(args)
            self.assertEqual(rep["command"], "hypertune")
            self.assertTrue((out_dir / "study_summary.json").exists())
            self.assertTrue((out_dir / "study_report.txt").exists())
            self.assertEqual(rep["total_trials"], 3)

    def test_handle_quantize_execution(self):
        class SimpleLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x=None, **kwargs):
                if isinstance(x, dict):
                    x = x.get("input_ids", torch.randn(1, 8))
                if x is None or not isinstance(x, torch.Tensor):
                    x = torch.randn(1, 8)
                if x.dtype in (torch.long, torch.int64):
                    x = x.float()
                if x.ndim > 2:
                    x = x.view(-1, 8)
                return self.fc(x)

        model = SimpleLinearModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "quant_out"
            args = argparse.Namespace(
                model=model,
                out=str(out_dir),
                bits=8,
                mode="symmetric",
                group_size=-1,
                calibration_data=None,
                mixed_precision=False,
                max_drift_kl=0.1,
                verify_reload=False,
                device="cpu",
            )
            rep = handle_quantize(args)
            self.assertTrue(rep["passed"])
            self.assertTrue((out_dir / "quantize_report.json").exists())
            self.assertTrue((out_dir / "quantized_model.pt").exists())

    def test_handle_export_runtime_execution(self):
        class ExportableModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, input_ids=None, **kwargs):
                return torch.randn(1, 4, 16)

        model = ExportableModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "exp_out"
            args = argparse.Namespace(
                model=model,
                out=str(out_dir),
                format="safetensors",
                target_runtime="transformers",
                source_dir=None,
                benchmark=True,
                batch_size=1,
                prompt_length=4,
                decode_steps=2,
                num_warmup=1,
                num_repeats=2,
                baseline_model=None,
                device="cpu",
            )
            rep = handle_export_runtime(args)
            self.assertTrue(rep["verification_success"])
            self.assertTrue((out_dir / "export_report.json").exists())
            self.assertIsNotNone(rep["benchmark"])

    def test_handle_ampute_execution(self):
        model = DummyMultimodalModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "ampute_out"
            args = argparse.Namespace(
                model=model,
                out=str(out_dir),
                modality="vision",
                retained_data=None,
                no_require_proof=True,
                no_install_guard=True,
                no_repair_config=False,
                max_allowed_diff=1e-4,
                device="cpu",
            )
            rep = handle_ampute(args)
            self.assertEqual(rep["command"], "ampute")
            self.assertGreater(len(rep["removed_branches"]), 0)
            self.assertTrue((out_dir / "amputation_report.json").exists())

    def test_handle_unlearn_execution(self):
        model = DummyLanguageModel()
        model.tokenizer = DummyTokenizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_file = Path(tmpdir) / "bench.json"
            bench_file.write_text(
                json.dumps([
                    {
                        "case_id": "test_case_1",
                        "prompt": "The capital of Atlantis is",
                        "target_new": "Unknown City",
                    }
                ])
            )
            out_file = Path(tmpdir) / "unlearned.json"
            args = argparse.Namespace(
                model=model,
                out=str(out_file),
                benchmark=str(bench_file),
                target_layer="layers.0.down_proj",
                method="refusal",
                refusal_target="I am unable to answer.",
                rank=1,
                learning_rate=1e-3,
                num_steps=2,
                max_delta_norm=1.0,
                audit=False,
                device="cpu",
            )
            rep = handle_unlearn(args)
            self.assertEqual(rep["command"], "unlearn")
            self.assertEqual(rep["edits_count"], 1)
            self.assertTrue(out_file.exists())

    def test_handle_workflow_live_execution(self):
        class WorkflowModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, input_ids=None, **kwargs):
                return torch.randn(1, 4, 16)

        model = WorkflowModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "workflow_live_out"
            args = argparse.Namespace(
                model=model,
                out=str(out_dir),
                config=None,
                keep=None,
                drop=None,
                change=None,
                stages="inspect,plan,apply,validate,export",
                plan=None,
                lora_r=4,
                max_recovery_steps=2,
                quantize_bits=None,
                export_format="safetensors",
                dry_run=False,
                device="cpu",
            )
            rep = handle_workflow(args)
            self.assertEqual(rep["status"], "success")
            self.assertTrue((out_dir / "workflow_manifest.json").exists())
            self.assertTrue((out_dir / "workflow_report.json").exists())


if __name__ == "__main__":
    unittest.main()

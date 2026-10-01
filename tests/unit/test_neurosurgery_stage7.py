"""Unit tests for Stage 7: Target-specific export and packing in model neurosurgery."""

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn
import safetensors.torch

from aegis_lab.editing.neurosurgery.stage7_export import (
    AssetPreservationError,
    BenchmarkProfile,
    ExportFormat,
    ExportResult,
    FormatVersionSpec,
    HardwareInfo,
    NeurosurgeryExportError,
    ProfilingError,
    ProfilingResult,
    ReloadVerificationError,
    ReloadVerificationResult,
    RuntimeTarget,
    SpeedupReport,
    SpeedupValidationError,
    SurgeryManifest,
    SUPPORTED_EXPORT_MATRIX,
    benchmark_runtime_profile,
    collect_hardware_info,
    compute_speedup_report,
    export_runtime_package,
    load_exported_package,
    measure_resident_memory,
    preserve_generation_config,
    preserve_tokenizer_assets,
    update_config_for_surgery,
    validate_export_matrix,
    validate_speedup_claim,
    verify_exported_package,
)


class TinyMLP(nn.Module):
    def __init__(self, in_features=16, out_features=32):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, out_features, bias=False)
        self.up_proj = nn.Linear(in_features, out_features, bias=False)
        self.down_proj = nn.Linear(out_features, in_features, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class TinyAttention(nn.Module):
    def __init__(self, hidden_dim=16, num_heads=4, num_kv_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        return self.o_proj(out)


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_dim=16, intermediate_dim=32, num_heads=4, num_kv_heads=4):
        super().__init__()
        self.self_attn = TinyAttention(
            hidden_dim=hidden_dim, num_heads=num_heads, num_kv_heads=num_kv_heads
        )
        self.mlp = TinyMLP(in_features=hidden_dim, out_features=intermediate_dim)

    def forward(self, x):
        return x + self.self_attn(x) + self.mlp(x)


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size=32,
        hidden_dim=16,
        num_layers=2,
        intermediate_dim=32,
        num_heads=4,
        num_kv_heads=4,
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_dim,
            "num_hidden_layers": num_layers,
            "intermediate_size": intermediate_dim,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "model_type": "llama",
        }
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            TinyDecoderLayer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)


class MockTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "hello": 4,
            "world": 5,
        }
        self.special_tokens_map = {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
        self.chat_template = (
            "{% for msg in messages %}{{ msg['role'] }}: {{ msg['content'] }}\n{% endfor %}"
        )

    def get_vocab(self):
        return self.vocab


class TestExportMatrixAndPackaging(unittest.TestCase):
    """Test format/version matrix validation and concrete packaging."""

    def test_matrix_validation_success(self):
        spec = validate_export_matrix(ExportFormat.SAFETENSORS, "1.0.0", RuntimeTarget.TRANSFORMERS)
        self.assertIsInstance(spec, FormatVersionSpec)
        self.assertEqual(spec.format, ExportFormat.SAFETENSORS)
        self.assertIn(RuntimeTarget.VLLM, spec.supported_targets)

        spec_pt = validate_export_matrix(ExportFormat.PYTORCH, "1.0.0", RuntimeTarget.PYTORCH)
        self.assertEqual(spec_pt.format, ExportFormat.PYTORCH)

    def test_matrix_validation_unsupported_format(self):
        with self.assertRaises(NeurosurgeryExportError):
            validate_export_matrix("gguf_unsupported", "1.0.0")

    def test_matrix_validation_unsupported_version(self):
        with self.assertRaises(NeurosurgeryExportError):
            validate_export_matrix(ExportFormat.SAFETENSORS, "99.0.0")

    def test_matrix_validation_unsupported_target(self):
        with self.assertRaises(NeurosurgeryExportError):
            validate_export_matrix(
                ExportFormat.PYTORCH, "1.0.0", target_runtime=RuntimeTarget.ONNX
            )

    def test_safetensors_package_export(self):
        model = TinyCausalLM()
        tokenizer = MockTokenizer()
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "exported_safetensors"
            result = export_runtime_package(
                model=model,
                export_dir=out_dir,
                format=ExportFormat.SAFETENSORS,
                target_runtime=RuntimeTarget.TRANSFORMERS,
                tokenizer=tokenizer,
            )
            self.assertIsInstance(result, ExportResult)
            self.assertTrue((out_dir / "model.safetensors").exists())
            self.assertTrue((out_dir / "export_manifest.json").exists())
            self.assertTrue((out_dir / "config.json").exists())
            self.assertTrue((out_dir / "special_tokens_map.json").exists())
            self.assertTrue((out_dir / "generation_config.json").exists())

            # Verify safetensors can be read
            loaded = safetensors.torch.load_file(str(out_dir / "model.safetensors"))
            self.assertIn("embed.weight", loaded)
            self.assertEqual(loaded["embed.weight"].shape, model.embed.weight.shape)

    def test_pytorch_bin_package_export(self):
        model = TinyCausalLM()
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "exported_pt"
            result = export_runtime_package(
                model=model,
                export_dir=out_dir,
                format=ExportFormat.PYTORCH,
                target_runtime=RuntimeTarget.PYTORCH,
            )
            self.assertIsInstance(result, ExportResult)
            self.assertTrue((out_dir / "pytorch_model.bin").exists())
            loaded = torch.load(out_dir / "pytorch_model.bin", weights_only=True)
            self.assertIn("embed.weight", loaded)
            self.assertEqual(loaded["embed.weight"].shape, model.embed.weight.shape)

    def test_export_manifest_checksum_integrity(self):
        model = TinyCausalLM()
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "exported_pkg"
            result = export_runtime_package(
                model=model,
                export_dir=out_dir,
                format=ExportFormat.SAFETENSORS,
            )
            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "7.0.0")
            self.assertEqual(manifest["export_format"], "safetensors")
            self.assertIn("model.safetensors", manifest["file_checksums"])
            self.assertIn("config.json", manifest["file_checksums"])

    def test_export_from_state_dict_directly(self):
        model = TinyCausalLM()
        sd = model.state_dict()
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "exported_from_sd"
            result = export_runtime_package(
                model=sd,
                export_dir=out_dir,
                format=ExportFormat.SAFETENSORS,
                config=model.config,
            )
            self.assertIsInstance(result, ExportResult)
            loaded_sd, loaded_cfg, loaded_man = load_exported_package(out_dir)
            self.assertEqual(len(loaded_sd), len(sd))
            self.assertEqual(loaded_cfg["vocab_size"], model.config["vocab_size"])
            self.assertEqual(loaded_man["export_format"], "safetensors")


class TestProvenanceAndAssetPreservation(unittest.TestCase):
    """Test asset preservation: tokenizer, special tokens, chat template, updated config, surgery manifest."""

    def test_tokenizer_and_special_tokens_preservation(self):
        tok = MockTokenizer()
        custom_spec = {
            "bos_token": "<custom_bos>",
            "eos_token": "<custom_eos>",
            "unk_token": "<custom_unk>",
            "pad_token": "<custom_pad>",
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            written = preserve_tokenizer_assets(
                export_dir=p,
                tokenizer=tok,
                special_tokens_map=custom_spec,
                chat_template="{{ messages[0].content }}",
            )
            self.assertIn("special_tokens_map.json", written)
            self.assertIn("chat_template.json", written)
            self.assertIn("tokenizer_config.json", written)
            self.assertIn("vocab.json", written)

            spec_data = json.loads(Path(written["special_tokens_map.json"]).read_text())
            self.assertEqual(spec_data["bos_token"], "<custom_bos>")

            ct_data = json.loads(Path(written["chat_template.json"]).read_text())
            self.assertIn("chat_template", ct_data)

    def test_copy_tokenizer_from_source_dir(self):
        with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as exp_dir:
            s_path = Path(src_dir)
            (s_path / "tokenizer.json").write_text('{"mock": true}')
            (s_path / "special_tokens_map.json").write_text('{"bos_token": "<s>"}')

            e_path = Path(exp_dir)
            written = preserve_tokenizer_assets(
                export_dir=e_path,
                source_dir=s_path,
            )
            self.assertTrue((e_path / "tokenizer.json").exists())
            self.assertEqual(
                json.loads((e_path / "tokenizer.json").read_text())["mock"], True
            )

    def test_generation_config_preservation(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            gen_cfg = {
                "max_new_tokens": 256,
                "temperature": 0.8,
                "top_p": 0.95,
                "do_sample": True,
            }
            written = preserve_generation_config(p, generation_config=gen_cfg)
            loaded = json.loads(Path(written).read_text())
            self.assertEqual(loaded["max_new_tokens"], 256)
            self.assertEqual(loaded["temperature"], 0.8)
            self.assertTrue(loaded["do_sample"])

    def test_config_update_for_structural_surgery(self):
        model = TinyCausalLM(
            vocab_size=32,
            hidden_dim=16,
            num_layers=4,
            intermediate_dim=32,
            num_heads=4,
            num_kv_heads=4,
        )
        surgery_manifest = SurgeryManifest(
            source_model_hash="sha_src_123",
            candidate_model_hash="sha_cand_456",
            operations_applied=[
                {"op": "layer_deletion", "deleted_layers": [2, 3]},
                {"op": "mlp_slice", "new_intermediate_size": 24},
            ],
            config_modifications={
                "num_hidden_layers": 2,
                "intermediate_size": 24,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
            },
        )
        updated = update_config_for_surgery(
            config=model.config,
            surgery_manifest=surgery_manifest,
        )
        self.assertEqual(updated["num_hidden_layers"], 2)
        self.assertEqual(updated["intermediate_size"], 24)
        self.assertEqual(updated["num_attention_heads"], 2)

    def test_surgery_manifest_roundtrip(self):
        manifest = SurgeryManifest(
            source_model_hash="source_sha",
            candidate_model_hash="cand_sha",
            operations_applied=[{"type": "prune", "count": 10}],
            config_modifications={"num_hidden_layers": 3},
            dataset_fingerprints={"keep": "keep_hash_1", "change": "change_hash_2"},
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "manifest.json"
            manifest.save_json(p)
            loaded = SurgeryManifest.load_json(p)
            self.assertEqual(loaded.source_model_hash, "source_sha")
            self.assertEqual(loaded.dataset_fingerprints["keep"], "keep_hash_1")
            self.assertEqual(loaded.operations_applied[0]["type"], "prune")

    def test_source_immutability(self):
        """Exporting must never mutate source model weights or source files."""
        model = TinyCausalLM()
        orig_weight = model.embed.weight.clone()
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "export_dir"
            export_runtime_package(
                model=model,
                export_dir=out_dir,
                format=ExportFormat.SAFETENSORS,
            )
            self.assertTrue(torch.equal(model.embed.weight, orig_weight))


class TestRuntimeBenchmarkingAndProfiling(unittest.TestCase):
    """Test prefill vs decode separation, TTFT, tokens/sec, and peak memory."""

    def setUp(self):
        self.model = TinyCausalLM(vocab_size=32, hidden_dim=16, num_layers=2)

    def test_prefill_and_decode_profiling_metrics(self):
        profile = BenchmarkProfile(batch_size=2, prompt_length=16, decode_steps=4)
        result = benchmark_runtime_profile(
            model=self.model,
            profile=profile,
            num_warmup=2,
            num_repeats=3,
        )
        self.assertIsInstance(result, ProfilingResult)
        self.assertGreater(result.prefill_latency_ms, 0.0)
        self.assertGreater(result.decode_latency_per_token_ms, 0.0)
        self.assertGreater(result.decode_latency_total_ms, 0.0)
        self.assertGreater(result.time_to_first_token_ms, 0.0)
        self.assertGreater(result.prefill_tokens_per_sec, 0.0)
        self.assertGreater(result.decode_tokens_per_sec, 0.0)
        self.assertGreater(result.total_tokens_per_sec, 0.0)
        self.assertGreaterEqual(result.peak_resident_memory_mb, 0.0)
        self.assertIn(result.memory_type, ["RSS", "VRAM"])
        self.assertEqual(len(result.latencies_prefill), 3)
        self.assertEqual(len(result.latencies_decode), 3)

    def test_zero_decode_steps_handled(self):
        profile = BenchmarkProfile(batch_size=1, prompt_length=8, decode_steps=0)
        result = benchmark_runtime_profile(
            model=self.model,
            profile=profile,
            num_warmup=1,
            num_repeats=2,
        )
        self.assertGreater(result.prefill_latency_ms, 0.0)
        self.assertEqual(result.decode_latency_total_ms, 0.0)
        self.assertEqual(result.decode_latency_per_token_ms, 0.0)
        self.assertEqual(result.decode_tokens_per_sec, 0.0)
        self.assertGreater(result.total_tokens_per_sec, 0.0)

    def test_profiling_with_custom_inputs(self):
        custom_inputs = torch.randint(0, 32, (1, 12), dtype=torch.long)
        profile = BenchmarkProfile(batch_size=1, prompt_length=12, decode_steps=2)
        result = benchmark_runtime_profile(
            model=self.model,
            profile=profile,
            sample_inputs=custom_inputs,
            num_warmup=1,
            num_repeats=2,
        )
        self.assertEqual(result.raw_metrics["prompt_length"], 12)
        self.assertEqual(result.raw_metrics["batch_size"], 1)

    def test_invalid_profiling_arguments(self):
        with self.assertRaises(ValueError):
            BenchmarkProfile(batch_size=-1, prompt_length=16)
        with self.assertRaises(ValueError):
            BenchmarkProfile(batch_size=1, prompt_length=0)
        with self.assertRaises(ValueError):
            benchmark_runtime_profile(self.model, num_repeats=0)


class TestReloadVerification(unittest.TestCase):
    """Test reload verification, shape checking, regression pass, and isolated subprocess loading."""

    def setUp(self):
        self.model = TinyCausalLM(vocab_size=32, hidden_dim=16, num_layers=2)
        self.tokenizer = MockTokenizer()

    def test_reload_safetensors_package_success(self):
        with tempfile.TemporaryDirectory() as td:
            export_runtime_package(
                model=self.model,
                export_dir=td,
                format=ExportFormat.SAFETENSORS,
                tokenizer=self.tokenizer,
            )
            sample_input = torch.randint(0, 32, (1, 8), dtype=torch.long)
            with torch.no_grad():
                expected_out = self.model(sample_input)

            def factory(cfg):
                return TinyCausalLM(
                    vocab_size=cfg["vocab_size"],
                    hidden_dim=cfg["hidden_size"],
                    num_layers=cfg["num_hidden_layers"],
                )

            res = verify_exported_package(
                export_dir=td,
                model_factory=factory,
                regression_inputs=sample_input,
                expected_outputs=expected_out,
                verify_subprocess=True,
            )
            self.assertTrue(res.success)
            self.assertTrue(res.regression_passed)
            self.assertTrue(res.subprocess_verified)
            self.assertIsNotNone(res.max_abs_diff)
            self.assertLess(res.max_abs_diff, 1e-4)

    def test_reload_pytorch_bin_package_success(self):
        with tempfile.TemporaryDirectory() as td:
            export_runtime_package(
                model=self.model,
                export_dir=td,
                format=ExportFormat.PYTORCH,
                tokenizer=self.tokenizer,
            )
            sample_input = torch.randint(0, 32, (1, 8), dtype=torch.long)
            with torch.no_grad():
                expected_out = self.model(sample_input)

            def factory(cfg):
                return TinyCausalLM(
                    vocab_size=cfg["vocab_size"],
                    hidden_dim=cfg["hidden_size"],
                    num_layers=cfg["num_hidden_layers"],
                )

            res = verify_exported_package(
                export_dir=td,
                model_factory=factory,
                regression_inputs=sample_input,
                expected_outputs=expected_out,
                verify_subprocess=True,
            )
            self.assertTrue(res.success)
            self.assertTrue(res.regression_passed)
            self.assertTrue(res.subprocess_verified)

    def test_reload_checksum_failure_detection(self):
        with tempfile.TemporaryDirectory() as td:
            export_runtime_package(
                model=self.model,
                export_dir=td,
                format=ExportFormat.SAFETENSORS,
                tokenizer=self.tokenizer,
            )
            # Tamper with config.json
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text('{"tampered": true}')

            with self.assertRaises(ReloadVerificationError):
                verify_exported_package(export_dir=td, raise_on_failure=True)

    def test_reload_missing_asset_detection(self):
        with tempfile.TemporaryDirectory() as td:
            export_runtime_package(
                model=self.model,
                export_dir=td,
                format=ExportFormat.SAFETENSORS,
                tokenizer=self.tokenizer,
            )
            # Delete special_tokens_map.json
            (Path(td) / "special_tokens_map.json").unlink()

            with self.assertRaises(ReloadVerificationError):
                verify_exported_package(export_dir=td, raise_on_failure=True)

    def test_reload_regression_output_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            export_runtime_package(
                model=self.model,
                export_dir=td,
                format=ExportFormat.SAFETENSORS,
                tokenizer=self.tokenizer,
            )
            sample_input = torch.randint(0, 32, (1, 8), dtype=torch.long)
            wrong_expected_out = torch.randn(1, 8, 32)

            def factory(cfg):
                return TinyCausalLM(
                    vocab_size=cfg["vocab_size"],
                    hidden_dim=cfg["hidden_size"],
                    num_layers=cfg["num_hidden_layers"],
                )

            with self.assertRaises(ReloadVerificationError):
                verify_exported_package(
                    export_dir=td,
                    model_factory=factory,
                    regression_inputs=sample_input,
                    expected_outputs=wrong_expected_out,
                    raise_on_failure=True,
                )

    def test_reload_shape_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            export_runtime_package(
                model=self.model,
                export_dir=td,
                format=ExportFormat.SAFETENSORS,
                tokenizer=self.tokenizer,
            )
            # Tamper config to expect intermediate_size=999
            cfg_p = Path(td) / "config.json"
            cfg_data = json.loads(cfg_p.read_text())
            cfg_data["intermediate_size"] = 999
            cfg_p.write_text(json.dumps(cfg_data, indent=2))

            # Update export manifest checksum so it gets past hash verification
            man_p = Path(td) / "export_manifest.json"
            man_data = json.loads(man_p.read_text())
            import hashlib
            man_data["file_checksums"]["config.json"] = hashlib.sha256(cfg_p.read_bytes()).hexdigest()
            man_p.write_text(json.dumps(man_data, indent=2))

            with self.assertRaises(ReloadVerificationError):
                verify_exported_package(export_dir=td, raise_on_failure=True)

    def test_reload_missing_manifest_fails(self):
        with tempfile.TemporaryDirectory() as td:
            export_runtime_package(
                model=self.model,
                export_dir=td,
                format=ExportFormat.SAFETENSORS,
            )
            (Path(td) / "export_manifest.json").unlink()
            with self.assertRaises(ReloadVerificationError):
                verify_exported_package(export_dir=td, raise_on_failure=True)


class TestSpeedupReportingAndValidation(unittest.TestCase):
    """Test measured latency speedup computations and validation of claims."""

    def setUp(self):
        self.hw = HardwareInfo(
            device_type="cpu",
            device_name="Intel Core i9-14900K",
            cpu_count=24,
            total_memory_mb=65536.0,
        )
        self.profile = BenchmarkProfile(batch_size=1, prompt_length=128, decode_steps=32)

    def test_compute_speedup_report_success(self):
        base_res = ProfilingResult(
            profile=self.profile,
            hardware=self.hw,
            num_warmup=2,
            num_repeats=5,
            prefill_latency_ms=100.0,
            prefill_latency_std_ms=2.0,
            decode_latency_per_token_ms=10.0,
            decode_latency_total_ms=320.0,
            time_to_first_token_ms=100.0,
            prefill_tokens_per_sec=1280.0,
            decode_tokens_per_sec=100.0,
            total_tokens_per_sec=380.95,
            peak_resident_memory_mb=1000.0,
            memory_type="RSS",
        )
        edited_res = ProfilingResult(
            profile=self.profile,
            hardware=self.hw,
            num_warmup=2,
            num_repeats=5,
            prefill_latency_ms=50.0,
            prefill_latency_std_ms=1.5,
            decode_latency_per_token_ms=5.0,
            decode_latency_total_ms=160.0,
            time_to_first_token_ms=50.0,
            prefill_tokens_per_sec=2560.0,
            decode_tokens_per_sec=200.0,
            total_tokens_per_sec=761.90,
            peak_resident_memory_mb=800.0,
            memory_type="RSS",
        )
        report = compute_speedup_report(baseline=base_res, edited=edited_res)
        self.assertIsInstance(report, SpeedupReport)
        self.assertAlmostEqual(report.prefill_speedup, 2.0, places=2)
        self.assertAlmostEqual(report.decode_speedup, 2.0, places=2)
        self.assertAlmostEqual(report.ttft_speedup, 2.0, places=2)
        self.assertAlmostEqual(report.throughput_speedup, 2.0, places=2)
        self.assertEqual(report.memory_reduction_mb, 200.0)
        self.assertEqual(report.memory_reduction_pct, 20.0)
        self.assertTrue(report.is_valid)

        md = report.to_markdown()
        self.assertIn("2.00x", md)
        self.assertIn("Intel Core i9-14900K", md)

    def test_speedup_validation_missing_hardware(self):
        claim = {
            "batch_size": 1,
            "context_length": 128,
            "baseline": {"prefill_ms": 10.0, "decode_ms": 5.0, "ttft_ms": 10.0},
            "edited": {"prefill_ms": 5.0, "decode_ms": 2.5, "ttft_ms": 5.0},
        }
        with self.assertRaises(SpeedupValidationError):
            validate_speedup_claim(claim)

    def test_speedup_validation_missing_context_or_batch(self):
        claim = {
            "hardware": {"device_type": "cpu", "device_name": "x86", "total_memory_mb": 1024.0},
            "baseline": {"prefill_ms": 10.0, "decode_ms": 5.0, "ttft_ms": 10.0},
            "edited": {"prefill_ms": 5.0, "decode_ms": 2.5, "ttft_ms": 5.0},
        }
        with self.assertRaises(SpeedupValidationError):
            validate_speedup_claim(claim)

    def test_speedup_validation_missing_baseline_numbers(self):
        claim = {
            "hardware": {"device_type": "cpu", "device_name": "x86", "total_memory_mb": 1024.0},
            "batch_size": 1,
            "context_length": 64,
            "baseline": {"prefill_ms": 0.0, "decode_ms": 5.0, "ttft_ms": 10.0},  # 0 is invalid
            "edited": {"prefill_ms": 5.0, "decode_ms": 2.5, "ttft_ms": 5.0},
        }
        with self.assertRaises(SpeedupValidationError):
            validate_speedup_claim(claim)

    def test_compute_speedup_profile_mismatch_fails_strict(self):
        p1 = BenchmarkProfile(batch_size=1, prompt_length=64, decode_steps=16)
        p2 = BenchmarkProfile(batch_size=2, prompt_length=64, decode_steps=16)

        r1 = ProfilingResult(
            profile=p1,
            hardware=self.hw,
            num_warmup=1,
            num_repeats=1,
            prefill_latency_ms=10.0,
            prefill_latency_std_ms=0.0,
            decode_latency_per_token_ms=1.0,
            decode_latency_total_ms=16.0,
            time_to_first_token_ms=10.0,
            prefill_tokens_per_sec=100.0,
            decode_tokens_per_sec=10.0,
            total_tokens_per_sec=50.0,
            peak_resident_memory_mb=100.0,
            memory_type="RSS",
        )
        r2 = ProfilingResult(
            profile=p2,
            hardware=self.hw,
            num_warmup=1,
            num_repeats=1,
            prefill_latency_ms=15.0,
            prefill_latency_std_ms=0.0,
            decode_latency_per_token_ms=1.5,
            decode_latency_total_ms=24.0,
            time_to_first_token_ms=15.0,
            prefill_tokens_per_sec=100.0,
            decode_tokens_per_sec=10.0,
            total_tokens_per_sec=50.0,
            peak_resident_memory_mb=100.0,
            memory_type="RSS",
        )
        with self.assertRaises(SpeedupValidationError):
            compute_speedup_report(baseline=r1, edited=r2, strict=True)

    def test_speedup_validation_zero_device_memory_fails(self):
        claim = {
            "hardware": {"device_type": "cpu", "device_name": "x86", "total_memory_mb": 0.0},
            "batch_size": 1,
            "context_length": 64,
            "baseline": {"prefill_ms": 10.0, "decode_ms": 5.0, "ttft_ms": 10.0},
            "edited": {"prefill_ms": 5.0, "decode_ms": 2.5, "ttft_ms": 5.0},
        }
        with self.assertRaises(SpeedupValidationError):
            validate_speedup_claim(claim)

    def test_speedup_report_save_json_and_profiling_save_json(self):
        base_res = ProfilingResult(
            profile=self.profile,
            hardware=self.hw,
            num_warmup=1,
            num_repeats=1,
            prefill_latency_ms=100.0,
            prefill_latency_std_ms=0.0,
            decode_latency_per_token_ms=10.0,
            decode_latency_total_ms=320.0,
            time_to_first_token_ms=100.0,
            prefill_tokens_per_sec=1280.0,
            decode_tokens_per_sec=100.0,
            total_tokens_per_sec=380.0,
            peak_resident_memory_mb=1000.0,
            memory_type="RSS",
        )
        with tempfile.TemporaryDirectory() as td:
            prof_path = Path(td) / "profiling.json"
            base_res.save_json(prof_path)
            self.assertTrue(prof_path.exists())
            loaded_prof = json.loads(prof_path.read_text())
            self.assertEqual(loaded_prof["prefill_latency_ms"], 100.0)

            edited_res = ProfilingResult(
                profile=self.profile,
                hardware=self.hw,
                num_warmup=1,
                num_repeats=1,
                prefill_latency_ms=50.0,
                prefill_latency_std_ms=0.0,
                decode_latency_per_token_ms=5.0,
                decode_latency_total_ms=160.0,
                time_to_first_token_ms=50.0,
                prefill_tokens_per_sec=2560.0,
                decode_tokens_per_sec=200.0,
                total_tokens_per_sec=760.0,
                peak_resident_memory_mb=800.0,
                memory_type="RSS",
            )
            report = compute_speedup_report(baseline=base_res, edited=edited_res)
            report_path = Path(td) / "speedup.json"
            report.save_json(report_path)
            self.assertTrue(report_path.exists())
            loaded_rep = json.loads(report_path.read_text())
            self.assertEqual(loaded_rep["prefill_speedup"], 2.0)

    def test_collect_hardware_info(self):
        info = collect_hardware_info("cpu")
        self.assertIsInstance(info, HardwareInfo)
        self.assertEqual(info.device_type, "cpu")
        self.assertGreater(info.total_memory_mb, 0.0)
        self.assertGreater(info.cpu_count, 0)

    def test_measure_resident_memory(self):
        mem, mem_type = measure_resident_memory("cpu")
        self.assertIsInstance(mem, float)
        self.assertGreaterEqual(mem, 0.0)
        self.assertEqual(mem_type, "RSS")


if __name__ == "__main__":
    unittest.main()

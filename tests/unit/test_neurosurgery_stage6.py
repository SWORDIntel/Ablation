import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis_lab.editing.neurosurgery.stage6_quantization import (
    QUANTIZATION_SCHEMA_VERSION,
    CalibrationBinding,
    CalibrationMismatchError,
    InvalidQuantizationConfigError,
    LayerSensitivity,
    MixedPrecisionPlan,
    MixedPrecisionStrategy,
    PackedQuantizationSurgeryError,
    QuantizationConfig,
    QuantizationDriftExceededError,
    QuantizationValidationResult,
    QuantizationValidationThresholds,
    QuantizedLinear,
    QuantizedTensor,
    ReloadParityError,
    SensitivityProfile,
    apply_mixed_precision_plan,
    apply_slicing_surgery,
    assign_mixed_precision,
    bind_calibration_dataset,
    calculate_model_memory_bytes,
    dequantize_affine,
    export_quantized_checkpoint,
    load_quantized_checkpoint,
    pack_int4,
    profile_layer_sensitivity,
    quantize_affine,
    quantize_model,
    quantize_tensor,
    reject_packed_slicing_surgery,
    unpack_edit_repack,
    unpack_int4,
    validate_quantized_candidate,
    verify_calibration_binding,
    verify_export_reload_parity,
)


class MockTokenizer:
    """Mock tokenizer for text encoding and decoding in unit tests."""

    def __init__(self, vocab=None):
        self.vocab = vocab or {"pad": 0, "eos": 1, "hello": 2, "world": 3, "test": 4, "drop": 5, "keep": 6}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab["pad"]
        self.eos_token_id = self.vocab["eos"]

    def __call__(self, batch, return_tensors="pt", **kwargs):
        if isinstance(batch, str):
            batch = [batch]
        encoded = []
        for text in batch:
            tokens = [self.vocab.get(word, 2) for word in text.split() if word]
            if not tokens:
                tokens = [self.vocab["hello"]]
            encoded.append(tokens)
        max_len = max(len(row) for row in encoded)
        ids = []
        masks = []
        for row in encoded:
            pad_count = max_len - len(row)
            ids.append(row + [self.pad_token_id] * pad_count)
            masks.append([1] * len(row) + [0] * pad_count)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        out = []
        for t in token_ids:
            item = t.item() if hasattr(t, "item") else int(t)
            if skip_special_tokens and item in (self.pad_token_id, self.eos_token_id):
                continue
            out.append(self.inv_vocab.get(item, str(item)))
        return " ".join(out)


class TinyModel(nn.Module):
    """Tiny multi-layer linear model for testing quantization and surgery."""

    def __init__(self, in_features=16, hidden_dim=16, out_features=16):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


class TinyCausalLM(nn.Module):
    """Tiny causal language model for calibration, sensitivity, and generation parity tests."""

    def __init__(self, vocab_size=16, hidden_dim=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.fc_in = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mid = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None and "inputs" in kwargs:
            input_ids = kwargs["inputs"]
        h = self.embed(input_ids)
        h = torch.relu(self.fc_in(h))
        h = torch.relu(self.fc_mid(h))
        logits = self.lm_head(h)
        return SimpleNamespace(logits=logits)


# ---------------------------------------------------------------------------
# 1. Calibration Dataset Binding Tests
# ---------------------------------------------------------------------------

class TestCalibrationDatasetBinding(unittest.TestCase):
    def test_bind_calibration_dataset_strings(self):
        samples = ["hello world", "test neurosurgery", "calibration sample"]
        binding = bind_calibration_dataset(samples, dataset_name="test_corpus")

        self.assertEqual(binding.dataset_name, "test_corpus")
        self.assertEqual(binding.sample_count, 3)
        self.assertEqual(len(binding.sample_ids), 3)
        self.assertEqual(len(binding.sample_hashes), 3)
        self.assertEqual(binding.domain_counts.get("default"), 3)
        self.assertTrue(len(binding.sha256) == 64)

    def test_bind_calibration_dataset_sample_objects(self):
        sample_objs = [
            SimpleNamespace(prompt="sample prompt one", domain="code", sample_id="s1"),
            SimpleNamespace(prompt="sample prompt two", domain="math", sample_id="s2"),
        ]
        binding = bind_calibration_dataset(sample_objs, dataset_name="custom_objects")

        self.assertEqual(binding.sample_count, 2)
        self.assertEqual(binding.sample_ids, ["s1", "s2"])
        self.assertEqual(binding.domain_counts.get("code"), 1)
        self.assertEqual(binding.domain_counts.get("math"), 1)

    def test_calibration_binding_verification_success(self):
        samples = ["prompt A", "prompt B", "prompt C"]
        binding = bind_calibration_dataset(samples, dataset_name="verify_set")
        self.assertTrue(binding.verify(samples))
        self.assertTrue(verify_calibration_binding(binding, samples))

    def test_calibration_binding_verification_mismatch(self):
        samples = ["prompt A", "prompt B", "prompt C"]
        binding = bind_calibration_dataset(samples, dataset_name="verify_set")

        # Modified sample content
        altered = ["prompt A", "prompt B modified", "prompt C"]
        with self.assertRaises(CalibrationMismatchError):
            binding.verify(altered)

        # Altered sample order
        reordered = ["prompt B", "prompt A", "prompt C"]
        with self.assertRaises(CalibrationMismatchError):
            binding.verify(reordered)

        # Sample count mismatch
        shortened = ["prompt A", "prompt B"]
        with self.assertRaises(CalibrationMismatchError):
            binding.verify(shortened)

    def test_bind_empty_dataset_rejected(self):
        with self.assertRaises(ValueError):
            bind_calibration_dataset([])

    def test_bind_empty_sample_string_rejected(self):
        with self.assertRaises(ValueError):
            bind_calibration_dataset(["valid", "   "])

    def test_calibration_binding_serialization_roundtrip(self):
        samples = ["alpha", "beta", "gamma"]
        binding = bind_calibration_dataset(samples, dataset_name="serial_test", metadata={"seed": 42})
        d = binding.to_dict()
        restored = CalibrationBinding.from_dict(d)

        self.assertEqual(binding.sha256, restored.sha256)
        self.assertEqual(binding.sample_count, restored.sample_count)
        self.assertEqual(binding.sample_ids, restored.sample_ids)
        self.assertEqual(binding.metadata, restored.metadata)


# ---------------------------------------------------------------------------
# 2. Quantization Schemes Tests (INT8 & INT4, Symmetric & Asymmetric)
# ---------------------------------------------------------------------------

class TestQuantizationSchemes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_quantize_int8_symmetric(self):
        w = torch.randn(8, 16)
        q, scale, zp = quantize_affine(w, bits=8, symmetric=True, granularity="per_channel")

        self.assertTrue(torch.all(zp == 0))
        self.assertEqual(scale.shape, (8, 1))
        self.assertTrue(torch.all(scale > 0))
        self.assertTrue(torch.all(q >= -127))
        self.assertTrue(torch.all(q <= 127))

        deq = dequantize_affine(q, scale, zp)
        err = torch.max(torch.abs(w - deq)).item()
        self.assertLess(err, 0.05)

    def test_quantize_int8_asymmetric(self):
        w = torch.randn(8, 16)
        q, scale, zp = quantize_affine(w, bits=8, symmetric=False, granularity="per_channel")

        self.assertEqual(scale.shape, (8, 1))
        self.assertEqual(zp.shape, (8, 1))
        self.assertTrue(torch.all(q >= 0))
        self.assertTrue(torch.all(q <= 255))
        self.assertTrue(torch.all(zp >= 0))
        self.assertTrue(torch.all(zp <= 255))

        deq = dequantize_affine(q, scale, zp)
        err = torch.max(torch.abs(w - deq)).item()
        self.assertLess(err, 0.05)

    def test_quantize_int4_symmetric(self):
        w = torch.randn(8, 16)
        q, scale, zp = quantize_affine(w, bits=4, symmetric=True, granularity="per_channel")

        self.assertTrue(torch.all(zp == 0))
        self.assertTrue(torch.all(q >= -7))
        self.assertTrue(torch.all(q <= 7))

        deq = dequantize_affine(q, scale, zp)
        err = torch.max(torch.abs(w - deq)).item()
        self.assertLess(err, 0.5)

    def test_quantize_int4_asymmetric(self):
        w = torch.randn(8, 16)
        q, scale, zp = quantize_affine(w, bits=4, symmetric=False, granularity="per_channel")

        self.assertTrue(torch.all(q >= 0))
        self.assertTrue(torch.all(q <= 15))
        self.assertTrue(torch.all(zp >= 0))
        self.assertTrue(torch.all(zp <= 15))

        deq = dequantize_affine(q, scale, zp)
        err = torch.max(torch.abs(w - deq)).item()
        self.assertLess(err, 0.5)

    def test_quantize_per_tensor_granularity(self):
        w = torch.randn(8, 16)
        q, scale, zp = quantize_affine(w, bits=8, symmetric=True, granularity="per_tensor")

        self.assertEqual(scale.numel(), 1)
        self.assertEqual(zp.numel(), 1)
        deq = dequantize_affine(q, scale, zp)
        self.assertLess(torch.max(torch.abs(w - deq)).item(), 0.1)

    def test_int4_packing_and_unpacking_bit_exact(self):
        # Signed 4-bit values in range [-7, 7]
        orig_signed = torch.randint(-7, 8, (6, 10), dtype=torch.int32)
        packed, orig_shape = pack_int4(orig_signed)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.numel(), 30)  # 60 elements packed 2 per byte = 30 bytes

        unpacked_signed = unpack_int4(packed, orig_shape, is_signed=True)
        self.assertTrue(torch.equal(orig_signed, unpacked_signed))

        # Unsigned 4-bit values in range [0, 15]
        orig_unsigned = torch.randint(0, 16, (7, 11), dtype=torch.int32)  # Odd number of elements: 77
        packed_u, shape_u = pack_int4(orig_unsigned)
        self.assertEqual(packed_u.numel(), math.ceil(77 / 2))

        unpacked_unsigned = unpack_int4(packed_u, shape_u, is_signed=False)
        self.assertTrue(torch.equal(orig_unsigned, unpacked_unsigned))

    def test_quantized_linear_forward_accuracy(self):
        lin = nn.Linear(16, 8)
        x = torch.randn(4, 16)
        fp_out = lin(x)

        # INT8 per-channel
        cfg_int8 = QuantizationConfig(bits=8, symmetric=True, granularity="per_channel", pack=False)
        qlin_int8 = QuantizedLinear.from_linear(lin, cfg_int8)
        int8_out = qlin_int8(x)
        self.assertLess(torch.max(torch.abs(fp_out - int8_out)).item(), 0.15)

        # INT4 packed
        cfg_int4 = QuantizationConfig(bits=4, symmetric=True, granularity="per_channel", pack=True)
        qlin_int4 = QuantizedLinear.from_linear(lin, cfg_int4)
        int4_out = qlin_int4(x)
        self.assertLess(torch.max(torch.abs(fp_out - int4_out)).item(), 0.75)

    def test_quantized_linear_activation_quantization(self):
        lin = nn.Linear(16, 8)
        x = torch.randn(2, 16)

        cfg_act = QuantizationConfig(
            bits=8,
            quantize_activations=True,
            act_bits=8,
            act_symmetric=False,
        )
        qlin = QuantizedLinear.from_linear(lin, cfg_act)
        out = qlin(x)
        self.assertEqual(out.shape, (2, 8))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_quantization_memory_reduction(self):
        model_fp32 = TinyModel(in_features=64, hidden_dim=64, out_features=64)
        fp32_bytes = calculate_model_memory_bytes(model_fp32)

        # Quantize to INT8
        model_int8 = quantize_model(
            model_fp32,
            QuantizationConfig(bits=8, symmetric=True, pack=False),
        )
        int8_bytes = calculate_model_memory_bytes(model_int8)
        self.assertLess(int8_bytes, fp32_bytes * 0.35)  # > 65% reduction

        # Quantize to INT4 packed
        model_int4 = quantize_model(
            model_fp32,
            QuantizationConfig(bits=4, symmetric=True, pack=True),
        )
        int4_bytes = calculate_model_memory_bytes(model_int4)
        self.assertLess(int4_bytes, fp32_bytes * 0.20)  # > 80% reduction

    def test_invalid_config_rejection(self):
        with self.assertRaises(InvalidQuantizationConfigError):
            QuantizationConfig(bits=3)
        with self.assertRaises(InvalidQuantizationConfigError):
            QuantizationConfig(granularity="invalid_granularity")


# ---------------------------------------------------------------------------
# 3. Per-Layer Sensitivity Profiling & Mixed Precision Tests
# ---------------------------------------------------------------------------

class TestPerLayerSensitivityAndMixedPrecision(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyModel(in_features=16, hidden_dim=16, out_features=16)
        self.calibration_tensors = [torch.randn(2, 16) for _ in range(3)]

    def test_profile_layer_sensitivity(self):
        profile = profile_layer_sensitivity(
            self.model,
            calibration_data=self.calibration_tensors,
            candidate_bits=4,
        )

        self.assertIn("fc1", profile.layers)
        self.assertIn("fc2", profile.layers)
        self.assertIn("fc3", profile.layers)

        for l in profile.layers.values():
            self.assertGreaterEqual(l.kl_divergence, 0.0)
            self.assertGreaterEqual(l.mse, 0.0)
            self.assertTrue(math.isfinite(l.sensitivity_score))

        sorted_layers = profile.sorted_by_sensitivity(metric="kl")
        self.assertEqual(len(sorted_layers), 3)
        self.assertGreaterEqual(sorted_layers[0].kl_divergence, sorted_layers[1].kl_divergence)

    def test_assign_mixed_precision_threshold(self):
        # Create a mock profile with known divergent sensitivities
        binding = bind_calibration_dataset(["sample 1", "sample 2"])
        layers = {
            "fc1": LayerSensitivity("fc1", kl_divergence=0.15, mse=0.8, sensitivity_score=0.23),  # High sensitivity -> FP
            "fc2": LayerSensitivity("fc2", kl_divergence=0.03, mse=0.1, sensitivity_score=0.04),  # Moderate sensitivity -> INT8
            "fc3": LayerSensitivity("fc3", kl_divergence=0.005, mse=0.02, sensitivity_score=0.007), # Low sensitivity -> INT4
        }
        profile = SensitivityProfile(layers=layers, calibration_binding=binding)

        strategy = MixedPrecisionStrategy(
            strategy_type="threshold",
            fp_threshold_kl=0.10,
            int4_threshold_kl=0.01,
            fp_precision="fp16",
            moderate_precision="int8",
            robust_precision="int4",
        )
        plan = assign_mixed_precision(profile, strategy)

        self.assertEqual(plan.layer_precisions["fc1"], "fp16")
        self.assertEqual(plan.layer_precisions["fc2"], "int8")
        self.assertEqual(plan.layer_precisions["fc3"], "int4")

    def test_assign_mixed_precision_top_k(self):
        binding = bind_calibration_dataset(["sample 1"])
        layers = {
            "l1": LayerSensitivity("l1", kl_divergence=0.5, mse=1.0, sensitivity_score=0.6),
            "l2": LayerSensitivity("l2", kl_divergence=0.2, mse=0.4, sensitivity_score=0.24),
            "l3": LayerSensitivity("l3", kl_divergence=0.01, mse=0.05, sensitivity_score=0.015),
        }
        profile = SensitivityProfile(layers=layers, calibration_binding=binding)

        strategy = MixedPrecisionStrategy(strategy_type="top_k", top_k_fp=1, top_k_int8=1)
        plan = assign_mixed_precision(profile, strategy)

        self.assertEqual(plan.layer_precisions["l1"], "fp16")
        self.assertEqual(plan.layer_precisions["l2"], "int8")
        self.assertEqual(plan.layer_precisions["l3"], "int4")

    def test_apply_mixed_precision_plan(self):
        binding = bind_calibration_dataset(["dummy"])
        layers = {
            "fc1": LayerSensitivity("fc1", 0.2, 0.5, 0.25),
            "fc2": LayerSensitivity("fc2", 0.04, 0.1, 0.05),
            "fc3": LayerSensitivity("fc3", 0.002, 0.01, 0.003),
        }
        profile = SensitivityProfile(layers=layers, calibration_binding=binding)
        plan = MixedPrecisionPlan(
            layer_precisions={"fc1": "fp32", "fc2": "int8", "fc3": "int4"},
            profile=profile,
            strategy=MixedPrecisionStrategy(),
        )

        quant_model = apply_mixed_precision_plan(self.model, plan)

        # fc1 should remain float Linear
        self.assertIsInstance(quant_model.fc1, nn.Linear)
        self.assertNotIsInstance(quant_model.fc1, QuantizedLinear)

        # fc2 should be INT8 QuantizedLinear
        self.assertIsInstance(quant_model.fc2, QuantizedLinear)
        self.assertEqual(quant_model.fc2.bits, 8)

        # fc3 should be INT4 QuantizedLinear
        self.assertIsInstance(quant_model.fc3, QuantizedLinear)
        self.assertEqual(quant_model.fc3.bits, 4)
        self.assertTrue(quant_model.fc3.is_packed)

        # Verify model remains callable and outputs valid shape
        x = torch.randn(2, 16)
        out = quant_model(x)
        self.assertEqual(out.shape, (2, 16))


# ---------------------------------------------------------------------------
# 4. Physical Slicing Protection Tests
# ---------------------------------------------------------------------------

class TestPhysicalSlicingProtection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.w = torch.randn(8, 16)
        self.lin = nn.Linear(16, 8)
        self.lin.weight.data = self.w.clone()

    def test_packed_quantized_tensor_slice_rejected(self):
        qt = quantize_tensor(self.w, bits=4, symmetric=True, pack=True)
        self.assertTrue(qt.is_packed)

        with self.assertRaises(PackedQuantizationSurgeryError):
            qt.slice(dim=0, start=0, end=4)

        with self.assertRaises(PackedQuantizationSurgeryError):
            _ = qt[0:2]

    def test_packed_quantized_linear_slice_channels_rejected(self):
        cfg = QuantizationConfig(bits=4, symmetric=True, pack=True)
        qlin = QuantizedLinear.from_linear(self.lin, cfg)
        self.assertTrue(qlin.is_packed)

        # Direct slicing attempt must be rejected
        with self.assertRaises(PackedQuantizationSurgeryError):
            qlin.slice_channels(indices=[0, 1, 2, 3], axis=0)

    def test_apply_slicing_surgery_rejects_packed_by_default(self):
        cfg = QuantizationConfig(bits=4, symmetric=True, pack=True)
        qlin = QuantizedLinear.from_linear(self.lin, cfg)

        with self.assertRaises(PackedQuantizationSurgeryError):
            apply_slicing_surgery(qlin, indices=[0, 2, 4], axis=0, allow_unpack_repack=False)

    def test_reject_packed_slicing_surgery_helper(self):
        cfg = QuantizationConfig(bits=4, symmetric=True, pack=True)
        qlin = QuantizedLinear.from_linear(self.lin, cfg)
        container = nn.Sequential(qlin)

        with self.assertRaises(PackedQuantizationSurgeryError):
            reject_packed_slicing_surgery(container)

    def test_safe_unpack_edit_repack_surgery(self):
        cfg = QuantizationConfig(bits=4, symmetric=True, pack=True)
        qlin = QuantizedLinear.from_linear(self.lin, cfg)
        self.assertTrue(qlin.is_packed)

        # Safely slice output channels (axis 0) from 8 to 4 channels
        sliced_qlin = unpack_edit_repack(qlin, indices=[0, 2, 4, 6], axis=0)

        self.assertEqual(sliced_qlin.out_features, 4)
        self.assertEqual(sliced_qlin.in_features, 16)
        self.assertTrue(sliced_qlin.is_packed)

        # Forward pass on sliced layer
        x = torch.randn(2, 16)
        out = sliced_qlin(x)
        self.assertEqual(out.shape, (2, 4))

        # Safely slice input channels (axis 1) from 16 to 10 channels
        sliced_in = unpack_edit_repack(sliced_qlin, indices=list(range(10)), axis=1)
        self.assertEqual(sliced_in.in_features, 10)
        self.assertEqual(sliced_in.out_features, 4)
        self.assertTrue(sliced_in.is_packed)

        x_10 = torch.randn(2, 10)
        out_10 = sliced_in(x_10)
        self.assertEqual(out_10.shape, (2, 4))

    def test_explicit_unpack_slice_repack_workflow(self):
        cfg = QuantizationConfig(bits=4, symmetric=True, pack=True)
        qlin = QuantizedLinear.from_linear(self.lin, cfg)

        # Unpack explicitly
        qlin.unpack()
        self.assertFalse(qlin.is_packed)

        # Channel slice allowed on unpacked layer
        sliced = qlin.slice_channels(indices=[0, 1, 2, 3], axis=0)
        self.assertEqual(sliced.out_features, 4)
        self.assertFalse(sliced.is_packed)

        # Repack
        sliced.pack()
        self.assertTrue(sliced.is_packed)
        x = torch.randn(2, 16)
        self.assertEqual(sliced(x).shape, (2, 4))


# ---------------------------------------------------------------------------
# 5. Independent Validation and Drift Rejection Tests
# ---------------------------------------------------------------------------

class TestIndependentValidationAndDriftRejection(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.base_model = TinyCausalLM(vocab_size=16, hidden_dim=16)
        self.tokenizer = MockTokenizer()
        self.keep_samples = ["hello world", "test neurosurgery", "keep sample"]
        self.drop_samples = ["drop bad behavior", "drop toxic output"]

    def test_validation_passes_good_candidate(self):
        # Quantize fc_in and fc_mid to INT8, keep lm_head in FP
        quant_model = copy.deepcopy(self.base_model)
        cfg = QuantizationConfig(bits=8, symmetric=True, pack=False)
        quant_model.fc_in = QuantizedLinear.from_linear(self.base_model.fc_in, cfg)
        quant_model.fc_mid = QuantizedLinear.from_linear(self.base_model.fc_mid, cfg)

        thresholds = QuantizationValidationThresholds(
            max_kl_drift=0.5,
            max_mse_drift=1.0,
            min_keep_retention=0.8,
            max_drop_rebound=0.2,
            min_memory_reduction_ratio=0.10,
        )

        res = validate_quantized_candidate(
            baseline_model=self.base_model,
            quantized_model=quant_model,
            keep_samples=self.keep_samples,
            drop_samples=self.drop_samples,
            tokenizer=self.tokenizer,
            thresholds=thresholds,
            raise_on_failure=True,
        )

        self.assertTrue(res.passed)
        self.assertGreaterEqual(res.keep_retention, 0.8)
        self.assertLessEqual(res.kl_drift, 0.5)
        self.assertGreaterEqual(res.memory_reduction_ratio, 0.10)

    def test_validation_rejects_kl_drift_exceeded(self):
        # Degrade candidate weights significantly to trigger KL drift violation
        corrupted_model = copy.deepcopy(self.base_model)
        corrupted_model.fc_mid.weight.data.fill_(100.0)

        thresholds = QuantizationValidationThresholds(max_kl_drift=0.01)

        with self.assertRaises(QuantizationDriftExceededError) as ctx:
            validate_quantized_candidate(
                baseline_model=self.base_model,
                quantized_model=corrupted_model,
                keep_samples=self.keep_samples,
                tokenizer=self.tokenizer,
                thresholds=thresholds,
                raise_on_failure=True,
            )
        self.assertIn("KL drift", str(ctx.exception))

    def test_validation_rejects_insufficient_memory_reduction(self):
        # Model unchanged (0% memory reduction)
        unreduced_model = copy.deepcopy(self.base_model)
        thresholds = QuantizationValidationThresholds(min_memory_reduction_ratio=0.30)

        with self.assertRaises(QuantizationDriftExceededError) as ctx:
            validate_quantized_candidate(
                baseline_model=self.base_model,
                quantized_model=unreduced_model,
                keep_samples=self.keep_samples,
                tokenizer=self.tokenizer,
                thresholds=thresholds,
                raise_on_failure=True,
            )
        self.assertIn("Memory reduction", str(ctx.exception))

    def test_validation_rejects_drop_rebound(self):
        # Corrupt model output specifically on drop samples
        corrupted_model = copy.deepcopy(self.base_model)
        corrupted_model.lm_head.weight.data[self.tokenizer.vocab["drop"]] += 50.0

        thresholds = QuantizationValidationThresholds(max_drop_rebound=0.01)

        with self.assertRaises(QuantizationDriftExceededError) as ctx:
            validate_quantized_candidate(
                baseline_model=self.base_model,
                quantized_model=corrupted_model,
                keep_samples=self.keep_samples,
                drop_samples=self.drop_samples,
                tokenizer=self.tokenizer,
                thresholds=thresholds,
                raise_on_failure=True,
            )
        self.assertIn("DROP rebound", str(ctx.exception))

    def test_validation_raise_on_failure_false(self):
        corrupted_model = copy.deepcopy(self.base_model)
        corrupted_model.fc_mid.weight.data.fill_(50.0)

        thresholds = QuantizationValidationThresholds(max_kl_drift=0.01)
        res = validate_quantized_candidate(
            baseline_model=self.base_model,
            quantized_model=corrupted_model,
            keep_samples=self.keep_samples,
            tokenizer=self.tokenizer,
            thresholds=thresholds,
            raise_on_failure=False,
        )
        self.assertFalse(res.passed)
        self.assertTrue(len(res.failure_reasons) > 0)


# ---------------------------------------------------------------------------
# 6. Export and Reload Parity Tests
# ---------------------------------------------------------------------------

class TestExportAndReloadParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.base_model = TinyCausalLM(vocab_size=16, hidden_dim=16)
        self.tokenizer = MockTokenizer()
        self.test_inputs = ["hello world", "test neurosurgery reload parity"]

    def test_export_and_reload_parity_exact(self):
        # Create quantized model with mixed INT4 packed and INT8 layers
        quant_model = copy.deepcopy(self.base_model)
        cfg_int4 = QuantizationConfig(bits=4, symmetric=True, pack=True)
        cfg_int8 = QuantizationConfig(bits=8, symmetric=False, pack=False)
        quant_model.fc_in = QuantizedLinear.from_linear(self.base_model.fc_in, cfg_int4)
        quant_model.fc_mid = QuantizedLinear.from_linear(self.base_model.fc_mid, cfg_int8)

        binding = bind_calibration_dataset(self.test_inputs, dataset_name="reload_test")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Export checkpoint using relative path
            rel_export_dir = Path(tmp_dir) / "quantized_export"
            written = export_quantized_checkpoint(
                model=quant_model,
                export_dir=rel_export_dir,
                config_or_plan=cfg_int4,
                calibration_binding=binding,
                metadata={"test_run": True},
            )

            self.assertTrue(Path(written["weights"]).exists())
            self.assertTrue(Path(written["metadata"]).exists())

            # Reload into a fresh base model
            fresh_base = TinyCausalLM(vocab_size=16, hidden_dim=16)
            reloaded_model, meta = load_quantized_checkpoint(
                export_dir=rel_export_dir,
                base_model=fresh_base,
            )

            self.assertEqual(meta["schema_version"], QUANTIZATION_SCHEMA_VERSION)
            self.assertIsInstance(reloaded_model.fc_in, QuantizedLinear)
            self.assertTrue(reloaded_model.fc_in.is_packed)
            self.assertIsInstance(reloaded_model.fc_mid, QuantizedLinear)
            self.assertFalse(reloaded_model.fc_mid.is_packed)

            # Verify generation equivalence / logit parity within numerical tolerance
            parity_ok = verify_export_reload_parity(
                original_model=quant_model,
                reloaded_model=reloaded_model,
                test_inputs=self.test_inputs,
                tokenizer=self.tokenizer,
                tolerance=1e-5,
            )
            self.assertTrue(parity_ok)

    def test_reload_parity_detects_mismatch(self):
        quant_model = copy.deepcopy(self.base_model)
        cfg = QuantizationConfig(bits=8, symmetric=True, pack=False)
        quant_model.fc_in = QuantizedLinear.from_linear(self.base_model.fc_in, cfg)

        # Alter model slightly
        divergent_model = copy.deepcopy(quant_model)
        divergent_model.fc_in.weight_scale += 0.5

        with self.assertRaises(ReloadParityError):
            verify_export_reload_parity(
                original_model=quant_model,
                reloaded_model=divergent_model,
                test_inputs=self.test_inputs,
                tokenizer=self.tokenizer,
                tolerance=1e-5,
            )


if __name__ == "__main__":
    unittest.main()

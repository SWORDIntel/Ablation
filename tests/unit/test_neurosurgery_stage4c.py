import copy
import math
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis_lab.editing.neurosurgery.stage4c_causal import (
    ActivationCache,
    ActivationPatcher,
    CausalCandidateEvaluation,
    CausalComparisonReport,
    CausalRankEntry,
    CausalSearchResult,
    CausalSearchSpace,
    ClampIntervention,
    ComponentTarget,
    CompositionOrderResult,
    FeatureAdapter,
    FeatureAdapterCompatibilityError,
    FeatureAdapterHook,
    FeatureIntervention,
    FeatureReconstructionResult,
    HookHandle,
    HookResolutionError,
    InteractingSetsReport,
    InteractionRegressionError,
    InteractionResult,
    KeepDamageResult,
    MagnitudeBaselineResult,
    MemoryLimitExceededError,
    PairedWorkload,
    PairedWorkloadRunner,
    PatchingResult,
    ProjectionIntervention,
    RandomBaselineResult,
    ReplacementIntervention,
    RuntimeInterventionContext,
    ScaleIntervention,
    WorkloadPair,
    attach_runtime_intervention,
    compare_causal_controls,
    compute_logit_diff,
    compute_recovery_ratio,
    compute_target_prob,
    evaluate_composition_order,
    evaluate_interacting_components,
    evaluate_keep_damage,
    evaluate_magnitude_baseline,
    evaluate_random_baseline,
    measure_reconstruction_quality,
    parse_component_target,
    rank_attention_heads,
    rank_components_by_causal_effect,
    rank_layers,
    rank_mlp_channels,
    resolve_component_module,
    search_causal_interventions,
    search_interacting_component_sets,
    _detect_module_activation_dim,
)


class MockMLP(nn.Module):
    def __init__(self, hidden=8, intermediate=16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MockAttention(nn.Module):
    def __init__(self, hidden=8, heads=4, head_dim=2):
        super().__init__()
        self.num_heads = heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        return self.o_proj(q)


class MockBlock(nn.Module):
    def __init__(self, hidden=8, intermediate=16, heads=4, head_dim=2):
        super().__init__()
        self.self_attn = MockAttention(hidden, heads, head_dim)
        self.mlp = MockMLP(hidden, intermediate)

    def forward(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class MockTransformerModel(nn.Module):
    def __init__(self, layers=2, hidden=8, intermediate=16, heads=4, head_dim=2, vocab=20):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([MockBlock(hidden, intermediate, heads, head_dim) for _ in range(layers)])
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(
            model_type="llama",
            num_hidden_layers=layers,
            num_attention_heads=heads,
            head_dim=head_dim,
            intermediate_size=intermediate,
        )

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class TestNeurosurgeryStage4C(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = MockTransformerModel()
        self.model.eval()

    # -----------------------------------------------------------------------
    # 1. Activation Capture & Hook Framework
    # -----------------------------------------------------------------------

    def test_hook_handle_lifecycle(self):
        linear = nn.Linear(4, 4)
        called = []

        def hook_fn(mod, inp, out):
            called.append(True)

        raw = linear.register_forward_hook(hook_fn)
        handle = HookHandle(raw)
        self.assertTrue(handle.is_active)

        linear(torch.ones(1, 4))
        self.assertEqual(len(called), 1)

        handle.remove()
        self.assertFalse(handle.is_active)

        linear(torch.ones(1, 4))
        self.assertEqual(len(called), 1)

        # Context manager
        raw2 = linear.register_forward_hook(hook_fn)
        with HookHandle(raw2) as h2:
            self.assertTrue(h2.is_active)
            linear(torch.ones(1, 4))
            self.assertEqual(len(called), 2)
        self.assertFalse(h2.is_active)

        linear(torch.ones(1, 4))
        self.assertEqual(len(called), 2)

    def test_bounded_memory_activation_cache(self):
        cache = ActivationCache(max_bytes=200, device="cpu", detach=True)
        t1 = torch.ones(1, 10, dtype=torch.float32)  # 40 bytes
        cache.store("t1", t1)
        self.assertEqual(cache.current_bytes, 40)
        self.assertIn("t1", cache)
        self.assertEqual(cache["t1"].device, torch.device("cpu"))

        # Token slicing: (batch, seq, hidden) -> keep last token
        t_seq = torch.randn(2, 5, 8)  # 2 * 5 * 8 * 4 = 320 bytes if full
        cache_sliced = ActivationCache(token_indices=-1)
        cache_sliced.store("seq", t_seq)
        # Sliced to (2, 1, 8) -> 2 * 1 * 8 * 4 = 64 bytes
        self.assertEqual(cache_sliced["seq"].shape, (2, 1, 8))
        self.assertEqual(cache_sliced.current_bytes, 64)

        # Enforce memory ceiling
        cache_capped = ActivationCache(max_bytes=100)
        cache_capped.store("first", torch.ones(1, 20, dtype=torch.float32))  # 80 bytes
        with self.assertRaises(MemoryLimitExceededError):
            cache_capped.store("second", torch.ones(1, 10, dtype=torch.float32))  # 40 bytes -> 120 > 100

        # Clear
        cache.clear()
        self.assertEqual(cache.current_bytes, 0)
        self.assertEqual(len(cache), 0)

    def test_paired_workload_and_runner(self):
        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[4, 5, 6]])
        pair = WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=8)
        workload = PairedWorkload([pair])

        self.assertEqual(len(workload), 1)
        self.assertEqual(workload[0].clean_target, 5)

        # Mismatched prompts reject
        with self.assertRaises(ValueError):
            PairedWorkload.from_prompts(["clean1"], ["corr1", "corr2"])

        runner = PairedWorkloadRunner(self.model)
        clean_logits, clean_cache = runner.run_clean(
            workload,
            capture_components=["layer.0", "layer.0.attention", "layer.0.mlp", "layer.0.input"],
            token_indices=-1,
        )
        self.assertEqual(len(clean_logits), 1)
        self.assertEqual(clean_logits[0].shape, (1, 3, 20))
        self.assertIn("layer.0", clean_cache)
        self.assertIn("layer.0.attention", clean_cache)
        self.assertIn("layer.0.mlp", clean_cache)
        self.assertIn("layer.0.input", clean_cache)

    def test_target_resolution(self):
        # Target parsing
        t1 = parse_component_target("layer.0")
        self.assertEqual((t1.layer_idx, t1.component_type), (0, "layer"))

        t2 = parse_component_target("layer.1.mlp")
        self.assertEqual((t2.layer_idx, t2.component_type), (1, "mlp"))

        t3 = parse_component_target("layer.0.head.2")
        self.assertEqual((t3.layer_idx, t3.component_type, t3.sub_idx), (0, "attention_head", 2))

        t4 = parse_component_target("layer.1.channel.5")
        self.assertEqual((t4.layer_idx, t4.component_type, t4.sub_idx), (1, "mlp_channel", 5))

        t5 = parse_component_target("path:layers.0.mlp.down_proj")
        self.assertEqual((t5.component_type, t5.module_path), ("custom", "layers.0.mlp.down_proj"))

        # Target module resolution
        mod_layer, htype_layer, _ = resolve_component_module(self.model, "layer.0")
        self.assertEqual(htype_layer, "post")
        self.assertEqual(mod_layer, self.model.layers[0])

        mod_head, htype_head, _ = resolve_component_module(self.model, "layer.0.head.1")
        self.assertEqual(htype_head, "pre")
        self.assertEqual(mod_head, self.model.layers[0].self_attn.o_proj)

        mod_chan, htype_chan, _ = resolve_component_module(self.model, "layer.0.channel.3")
        self.assertEqual(htype_chan, "pre")
        self.assertEqual(mod_chan, self.model.layers[0].mlp.down_proj)

        # Invalid layer index
        with self.assertRaises(HookResolutionError):
            resolve_component_module(self.model, "layer.99")

    # -----------------------------------------------------------------------
    # 2. Clean/Corrupted Activation Patching
    # -----------------------------------------------------------------------

    def test_metrics_calculation(self):
        logits = torch.zeros(1, 3, 10)
        logits[0, -1, 2] = 5.0  # target A
        logits[0, -1, 4] = 2.0  # target B

        # Logit difference
        ld = compute_logit_diff(logits, clean_target=2, corrupted_target=4, token_pos=-1)
        self.assertAlmostEqual(ld, 3.0)

        # Recovery ratio
        # Patched restores exactly clean: (3.0 - 0.0) / (3.0 - 0.0) = 1.0
        self.assertAlmostEqual(compute_recovery_ratio(3.0, 3.0, 0.0), 1.0)
        # Patched equals corrupted: (0.0 - 0.0) / (3.0 - 0.0) = 0.0
        self.assertAlmostEqual(compute_recovery_ratio(0.0, 3.0, 0.0), 0.0)
        # Patched halfway: (1.5 - 0.0) / (3.0 - 0.0) = 0.5
        self.assertAlmostEqual(compute_recovery_ratio(1.5, 3.0, 0.0), 0.5)

        # Target probability
        p = compute_target_prob(logits, target=2, token_pos=-1)
        expected_p = float(F.softmax(logits[0, -1, :], dim=-1)[2].item())
        self.assertAlmostEqual(p, expected_p)

    def test_activation_patching_restoration(self):
        # Configure model so layer 0 MLP channel 3 encodes target distinguishing signal
        with torch.no_grad():
            self.model.layers[0].mlp.down_proj.weight.data.zero_()
            self.model.layers[0].mlp.down_proj.weight.data[0, 3] = 10.0  # Channel 3 drives output dimension 0
            self.model.lm_head.weight.data.zero_()
            self.model.lm_head.weight.data[5, 0] = 1.0  # Clean target token 5
            self.model.lm_head.weight.data[7, 0] = -1.0  # Corrupted target token 7

        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[8, 9, 10]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])

        patcher = ActivationPatcher(self.model)
        # Patch causal channel 3 vs non-causal channel 0
        results = patcher.run_patching(workload, ["layer.0.channel.3", "layer.0.channel.0"])

        res_causal = next(r for r in results if r.component == "layer.0.mlp_channel.3")
        res_non_causal = next(r for r in results if r.component == "layer.0.mlp_channel.0")

        # Causal channel should achieve substantial recovery
        self.assertGreater(res_causal.recovery_ratio, 0.8)
        # Non-causal channel should achieve minimal recovery
        self.assertLess(abs(res_non_causal.recovery_ratio), 0.2)

    def test_token_specific_patching(self):
        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[4, 5, 6]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])

        patcher = ActivationPatcher(self.model)
        # Patch only at final token position
        res = patcher.run_patching(workload, ["layer.0.mlp"], token_indices=-1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].token_pos, -1)

    # -----------------------------------------------------------------------
    # 3. Component Causal Ranking & Controls
    # -----------------------------------------------------------------------

    def test_causal_ranking(self):
        results = [
            PatchingResult(
                component="layer.0.channel.1",
                component_type="mlp_channel",
                layer_idx=0,
                sub_idx=1,
                token_pos=-1,
                clean_logit_diff=2.0,
                corrupted_logit_diff=0.0,
                patched_logit_diff=0.5,
                recovery_ratio=0.25,
                clean_target_prob=0.8,
                corrupted_target_prob=0.1,
                patched_target_prob=0.3,
                target_prob_diff=0.2,
            ),
            PatchingResult(
                component="layer.0.channel.2",
                component_type="mlp_channel",
                layer_idx=0,
                sub_idx=2,
                token_pos=-1,
                clean_logit_diff=2.0,
                corrupted_logit_diff=0.0,
                patched_logit_diff=1.8,
                recovery_ratio=0.90,
                clean_target_prob=0.8,
                corrupted_target_prob=0.1,
                patched_target_prob=0.75,
                target_prob_diff=0.65,
            ),
            PatchingResult(
                component="layer.0.channel.3",
                component_type="mlp_channel",
                layer_idx=0,
                sub_idx=3,
                token_pos=-1,
                clean_logit_diff=2.0,
                corrupted_logit_diff=0.0,
                patched_logit_diff=0.1,
                recovery_ratio=0.05,
                clean_target_prob=0.8,
                corrupted_target_prob=0.1,
                patched_target_prob=0.15,
                target_prob_diff=0.05,
            ),
        ]

        ranked = rank_components_by_causal_effect(results, metric="recovery_ratio")
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0].component, "layer.0.channel.2")
        self.assertAlmostEqual(ranked[0].causal_effect, 0.90)
        self.assertEqual(ranked[1].component, "layer.0.channel.1")
        self.assertEqual(ranked[2].component, "layer.0.channel.3")

    def test_random_component_control(self):
        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[4, 5, 6]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])

        patcher = ActivationPatcher(self.model)
        pool = [f"layer.0.channel.{i}" for i in range(16)]

        rand_control = evaluate_random_baseline(
            patcher, workload, pool, k_count=2, num_trials=4, seed=42
        )
        self.assertEqual(rand_control.k_count, 2)
        self.assertEqual(rand_control.num_trials, 4)
        self.assertIsInstance(rand_control.mean_causal_effect, float)
        self.assertIsInstance(rand_control.std_causal_effect, float)

    def test_magnitude_only_baseline_control(self):
        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[4, 5, 6]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])

        patcher = ActivationPatcher(self.model)
        pool = [f"layer.0.channel.{i}" for i in range(8)]

        mag_control = evaluate_magnitude_baseline(patcher, workload, pool, k_count=2)
        self.assertEqual(mag_control.k_count, 2)
        self.assertEqual(len(mag_control.top_magnitude_components), 2)
        self.assertIsInstance(mag_control.causal_effect, float)

    def test_keep_damage_assessment(self):
        keep_inputs = [torch.tensor([[1, 1, 1]]), torch.tensor([[2, 2, 2]])]
        intervention = ScaleIntervention("layer.0.channel.0", scale=0.0)

        keep_damage = evaluate_keep_damage(self.model, keep_inputs, [intervention], max_allowed_kl=0.5)
        self.assertEqual(keep_damage.samples, 2)
        self.assertGreaterEqual(keep_damage.mean_kl, 0.0)
        self.assertTrue(keep_damage.acceptable)
        self.assertGreaterEqual(keep_damage.top1_agreement, 0.0)

    def test_causal_comparison_report(self):
        rand_res = RandomBaselineResult(
            k_count=1, num_trials=5, mean_causal_effect=0.05, std_causal_effect=0.02, trial_effects=[0.05]*5, seed=42
        )
        mag_res = MagnitudeBaselineResult(
            k_count=1, top_magnitude_components=["layer.0.channel.5"], magnitudes=[12.0], causal_effect=0.10
        )
        keep_res = KeepDamageResult(
            mean_kl=0.02, max_kl=0.03, top1_agreement=1.0, samples=2, acceptable=True
        )

        candidate = ["layer.0.channel.3"]
        candidate_effect = 0.85

        report = compare_causal_controls(candidate, candidate_effect, rand_res, mag_res, keep_res)

        self.assertGreater(report.causal_gain_over_random, 0.0)
        self.assertGreater(report.causal_gain_over_magnitude, 0.0)
        self.assertGreater(report.z_score, 2.0)
        self.assertTrue(report.is_causally_distinct)
        self.assertGreater(report.specificity_ratio, 10.0)

        s_dict = report.summary_dict()
        self.assertEqual(s_dict["candidate_causal_effect"], 0.85)
        self.assertTrue(s_dict["is_causally_distinct"])

    def test_high_level_rankers(self):
        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[4, 5, 6]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])

        # Rank layers
        l_ranks = rank_layers(self.model, workload)
        self.assertEqual(len(l_ranks), 2)
        self.assertEqual(l_ranks[0].rank, 1)

        # Rank heads
        h_ranks = rank_attention_heads(self.model, workload, layer_indices=[0])
        self.assertEqual(len(h_ranks), 4)

        # Rank MLP channels
        c_ranks = rank_mlp_channels(self.model, workload, layer_indices=[0], channel_indices=[0, 1, 2])
        self.assertEqual(len(c_ranks), 3)

    # -----------------------------------------------------------------------
    # 4. Runtime Interventions
    # -----------------------------------------------------------------------

    def test_scale_intervention(self):
        tensor = torch.ones(2, 4, 8)
        # Full zero ablation
        interv_zero = ScaleIntervention(target="layer.0", scale=0.0)
        out_zero = interv_zero.transform(tensor)
        self.assertTrue((out_zero == 0.0).all())

        # Scaling by 2.5
        interv_scale = ScaleIntervention(target="layer.0", scale=2.5)
        out_scale = interv_scale.transform(tensor)
        self.assertTrue((out_scale == 2.5).all())

        # Channel-specific scaling
        interv_ch = ScaleIntervention(target="layer.0.channel.3", scale=0.0, channel_indices=3)
        out_ch = interv_ch.transform(tensor)
        self.assertTrue((out_ch[:, :, 3] == 0.0).all())
        self.assertTrue((out_ch[:, :, :3] == 1.0).all())
        self.assertTrue((out_ch[:, :, 4:] == 1.0).all())

    def test_clamp_intervention(self):
        tensor = torch.tensor([[[-5.0, 0.0, 10.0]]])
        interv_clamp = ClampIntervention(target="layer.0", min_val=-1.0, max_val=2.0)
        out_clamp = interv_clamp.transform(tensor)
        self.assertAlmostEqual(out_clamp[0, 0, 0].item(), -1.0)
        self.assertAlmostEqual(out_clamp[0, 0, 1].item(), 0.0)
        self.assertAlmostEqual(out_clamp[0, 0, 2].item(), 2.0)

        # Max norm clamp
        vec = torch.tensor([[[3.0, 4.0]]])  # norm = 5.0
        interv_norm = ClampIntervention(target="layer.0", max_norm=2.5)
        out_norm = interv_norm.transform(vec)
        self.assertAlmostEqual(float(out_norm.norm(dim=-1).item()), 2.5, places=5)

    def test_projection_intervention(self):
        # Refusal direction subtraction: orthogonal projection
        torch.manual_seed(99)
        tensor = torch.randn(2, 5, 8)
        refusal_direction = torch.randn(8)

        interv_proj = ProjectionIntervention(
            target="layer.0",
            direction=refusal_direction,
            strength=1.0,
            mode="remove_projection",
        )
        out_proj = interv_proj.transform(tensor)

        # Verify output component along refusal direction is zero
        v_unit = F.normalize(refusal_direction, dim=0)
        dots = (out_proj * v_unit).sum(dim=-1)
        self.assertTrue(torch.allclose(dots, torch.zeros_like(dots), atol=1e-6))

        # Steering mode
        interv_steer = ProjectionIntervention(
            target="layer.0",
            direction=refusal_direction,
            strength=2.0,
            mode="steer",
        )
        out_steer = interv_steer.transform(tensor)
        diff = out_steer - tensor
        self.assertTrue(torch.allclose(diff[0, 0], 2.0 * v_unit, atol=1e-6))

        # Subspace mode
        basis, _ = torch.linalg.qr(torch.randn(8, 2))
        interv_sub = ProjectionIntervention(
            target="layer.0",
            direction=basis,
            strength=1.0,
            mode="subspace",
        )
        out_sub = interv_sub.transform(tensor)
        dots_sub = out_sub @ basis
        self.assertTrue(torch.allclose(dots_sub, torch.zeros_like(dots_sub), atol=1e-6))

    def test_replacement_intervention(self):
        tensor = torch.zeros(2, 4, 8)
        replacement = torch.ones(2, 4, 8) * 42.0

        interv_rep = ReplacementIntervention(
            target="layer.0",
            replacement=replacement,
            token_indices=-1,
            channel_indices=3,
        )
        out_rep = interv_rep.transform(tensor)
        self.assertTrue((out_rep[:, -1, 3] == 42.0).all())
        self.assertTrue((out_rep[:, :-1, :] == 0.0).all())
        self.assertTrue((out_rep[:, -1, :3] == 0.0).all())
        self.assertTrue((out_rep[:, -1, 4:] == 0.0).all())

    def test_runtime_intervention_context_lifecycle_and_weight_immutability(self):
        ids = torch.tensor([[1, 2, 3]])
        baseline_out = self.model(ids).clone()

        # Capture parameter weights before intervention
        param_snapshots = {n: p.clone() for n, p in self.model.named_parameters()}

        interv = ScaleIntervention(target="layer.0.channel.0", scale=0.0)
        with RuntimeInterventionContext(self.model, [interv]) as ctx:
            self.assertTrue(ctx.is_active)
            intervened_out = self.model(ids)
            # Output changes under intervention
            self.assertFalse(torch.allclose(baseline_out, intervened_out))

        # Outside context, hooks are removed
        self.assertFalse(ctx.is_active)
        restored_out = self.model(ids)
        self.assertTrue(torch.allclose(baseline_out, restored_out))

        # Base checkpoint weights must remain 100% unaltered
        for n, p in self.model.named_parameters():
            self.assertTrue(torch.equal(p, param_snapshots[n]))

        # Exception safety
        try:
            with RuntimeInterventionContext(self.model, [interv]) as ctx2:
                self.assertTrue(ctx2.is_active)
                raise RuntimeError("Simulated error inside context")
        except RuntimeError:
            pass

        # Hooks must be cleanly removed even after exception
        self.assertFalse(ctx2.is_active)
        safe_restored_out = self.model(ids)
        self.assertTrue(torch.allclose(baseline_out, safe_restored_out))

    def test_standalone_attach_runtime_intervention(self):
        ids = torch.tensor([[1, 2, 3]])
        baseline_out = self.model(ids).clone()

        interv = ScaleIntervention(target="layer.0.channel.1", scale=0.0)
        handle = attach_runtime_intervention(self.model, interv)
        self.assertTrue(handle.is_active)

        intervened_out = self.model(ids)
        self.assertFalse(torch.allclose(baseline_out, intervened_out))

        handle.remove()
        self.assertFalse(handle.is_active)
        restored_out = self.model(ids)
        self.assertTrue(torch.allclose(baseline_out, restored_out))

    # -----------------------------------------------------------------------
    # 5. Additional Edge Cases & Coverage
    # -----------------------------------------------------------------------

    def test_corruption_mode_patching(self):
        # In corruption mode, clean run is corrupted by corrupted activations
        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[8, 9, 10]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])

        patcher = ActivationPatcher(self.model)
        results = patcher.run_patching(workload, ["layer.0.mlp"], mode="corruption")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["mode"], "corruption")

    def test_attention_head_causal_patching(self):
        # Configure model so head 1 carries distinguishing signal
        with torch.no_grad():
            self.model.layers[0].self_attn.o_proj.weight.data.zero_()
            # Head 1 spans dimensions [2, 3] since head_dim = 2
            self.model.layers[0].self_attn.o_proj.weight.data[0, 2] = 5.0
            self.model.lm_head.weight.data.zero_()
            self.model.lm_head.weight.data[5, 0] = 1.0
            self.model.lm_head.weight.data[7, 0] = -1.0

        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[8, 9, 10]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])

        patcher = ActivationPatcher(self.model)
        results = patcher.run_patching(workload, ["layer.0.head.1", "layer.0.head.0"])

        causal_head = next(r for r in results if r.component == "layer.0.attention_head.1")
        non_causal_head = next(r for r in results if r.component == "layer.0.attention_head.0")

        self.assertGreater(causal_head.recovery_ratio, 0.5)
        self.assertLess(abs(non_causal_head.recovery_ratio), 0.2)

    def test_tokenizer_and_text_prompt_support(self):
        class SimpleMockTokenizer:
            def __call__(self, text, return_tensors="pt"):
                # mapping chars to token IDs
                mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
                ids = [mapping.get(c, 0) for c in text]
                return {"input_ids": torch.tensor([ids])}

            def encode(self, text, add_special_tokens=False):
                mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
                return [mapping.get(c, 0) for c in text]

        tok = SimpleMockTokenizer()
        workload = PairedWorkload.from_prompts(
            clean_prompts=["ABC"],
            corrupted_prompts=["DEF"],
            clean_targets=["B"],
            corrupted_targets=["D"],
        )
        runner = PairedWorkloadRunner(self.model, tokenizer=tok)
        self.assertEqual(runner.resolve_token_id("B"), 2)
        self.assertEqual(runner.resolve_token_id(5), 5)
        self.assertIsNone(runner.resolve_token_id(None))

        clean_logits, _ = runner.run_clean(workload)
        self.assertEqual(len(clean_logits), 1)

    def test_custom_module_path_hook(self):
        interv = ScaleIntervention(target="path:layers.0.mlp.down_proj", scale=0.5)
        with RuntimeInterventionContext(self.model, [interv]):
            out = self.model(torch.tensor([[1, 2, 3]]))
            self.assertEqual(out.shape, (1, 3, 20))

    def test_error_and_boundary_conditions(self):
        # Empty workload
        with self.assertRaises(ValueError):
            PairedWorkload([])

        # String target resolution without tokenizer
        runner = PairedWorkloadRunner(self.model)
        with self.assertRaises(ValueError):
            runner.resolve_token_id("invalid_without_tok")

        # Invalid ranking metric
        with self.assertRaises(ValueError):
            rank_components_by_causal_effect([], metric="invalid_metric")

        # k_count exceeds pool size
        patcher = ActivationPatcher(self.model)
        workload = PairedWorkload([WorkloadPair(torch.tensor([[1]]), torch.tensor([[2]]))])
        with self.assertRaises(ValueError):
            evaluate_random_baseline(patcher, workload, ["layer.0.channel.0"], k_count=5)

        # Empty keep inputs
        with self.assertRaises(ValueError):
            evaluate_keep_damage(self.model, [], [])

    # -----------------------------------------------------------------------
    # 6. Feature / SAE Adapter Framework Tests
    # -----------------------------------------------------------------------

    def test_feature_adapter_lifecycle_and_reconstruction(self):
        adapter = FeatureAdapter(activation_dim=8, feature_dim=16, target="layer.0")
        x = torch.randn(2, 4, 8)

        # Encode / Decode
        f = adapter.encode(x)
        self.assertEqual(f.shape, (2, 4, 16))
        x_rec = adapter.decode(f)
        self.assertEqual(x_rec.shape, (2, 4, 8))

        # Reconstruct
        x_rec2 = adapter.reconstruct(x)
        self.assertEqual(x_rec2.shape, (2, 4, 8))

        # Top-k sparsity
        adapter_k = FeatureAdapter(activation_dim=8, feature_dim=16, target="layer.0", top_k=3)
        f_k = adapter_k.encode(x)
        nonzeros = (f_k > 0.0).sum(dim=-1)
        self.assertTrue((nonzeros <= 3).all())

        # Dimension validation
        with self.assertRaises(FeatureAdapterCompatibilityError):
            adapter.encode(torch.randn(2, 4, 10))
        with self.assertRaises(FeatureAdapterCompatibilityError):
            adapter.decode(torch.randn(2, 4, 20))

    def test_feature_adapter_compatibility_validation(self):
        # Valid configuration
        adapter_good = FeatureAdapter(
            activation_dim=8,
            feature_dim=16,
            target="layer.0",
            expected_model_type="llama",
            layer_idx=0,
        )
        self.assertTrue(adapter_good.check_compatibility(self.model))

        # Model type mismatch
        adapter_bad_model = FeatureAdapter(
            activation_dim=8,
            feature_dim=16,
            target="layer.0",
            expected_model_type="mistral",
        )
        with self.assertRaises(FeatureAdapterCompatibilityError):
            adapter_bad_model.check_compatibility(self.model)

        # Layer out of range
        adapter_bad_layer = FeatureAdapter(activation_dim=8, feature_dim=16, target="layer.99")
        with self.assertRaises(FeatureAdapterCompatibilityError):
            adapter_bad_layer.check_compatibility(self.model)

        # Layer index mismatch
        adapter_layer_mismatch = FeatureAdapter(
            activation_dim=8, feature_dim=16, target="layer.0", layer_idx=1
        )
        with self.assertRaises(FeatureAdapterCompatibilityError):
            adapter_layer_mismatch.check_compatibility(self.model)

        # Activation space dimensionality mismatch
        adapter_dim_mismatch = FeatureAdapter(
            activation_dim=16, feature_dim=32, target="layer.0"
        )
        with self.assertRaises(FeatureAdapterCompatibilityError):
            adapter_dim_mismatch.check_compatibility(self.model)

    def test_feature_reconstruction_quality_measurement(self):
        # Near-perfect reconstruction
        enc = torch.eye(8, 16)
        dec = torch.eye(16, 8)
        adapter_hi = FeatureAdapter(8, 16, encoder_weight=enc, decoder_weight=dec)
        x = torch.randn(20, 8).abs() + 0.1

        res = measure_reconstruction_quality(
            adapter_hi, x, max_allowed_nmse=0.01, min_explained_variance=0.99
        )
        self.assertTrue(res.acceptable)
        self.assertAlmostEqual(res.nmse, 0.0, places=4)
        self.assertGreater(res.cosine_similarity, 0.99)
        self.assertIn("acceptable", res.summary_dict())

        # Poor reconstruction
        adapter_lo = FeatureAdapter(8, 2)
        res_lo = measure_reconstruction_quality(
            adapter_lo, x, max_allowed_nmse=0.01, min_explained_variance=0.99
        )
        self.assertFalse(res_lo.acceptable)

    def test_feature_intervention_actions(self):
        enc = torch.eye(8, 16)
        dec = torch.eye(16, 8)
        adapter = FeatureAdapter(8, 16, target="layer.0", encoder_weight=enc, decoder_weight=dec)
        x = torch.ones(2, 4, 8)

        # Zero action
        itv_zero = FeatureIntervention(
            target="layer.0", adapter=adapter, feature_indices=2, action="zero"
        )
        out_zero = itv_zero.transform(x)
        self.assertEqual(out_zero[0, 0, 2].item(), 0.0)
        self.assertEqual(out_zero[0, 0, 0].item(), 1.0)

        # Scale action
        itv_scale = FeatureIntervention(
            target="layer.0", adapter=adapter, feature_indices=2, action="scale", scale=3.0
        )
        out_scale = itv_scale.transform(x)
        self.assertEqual(out_scale[0, 0, 2].item(), 3.0)

        # Clamp action
        itv_clamp = FeatureIntervention(
            target="layer.0",
            adapter=adapter,
            feature_indices=2,
            action="clamp",
            min_val=0.0,
            max_val=0.5,
        )
        out_clamp = itv_clamp.transform(x)
        self.assertEqual(out_clamp[0, 0, 2].item(), 0.5)

        # Set action
        itv_set = FeatureIntervention(
            target="layer.0", adapter=adapter, feature_indices=2, action="set", set_value=17.0
        )
        out_set = itv_set.transform(x)
        self.assertEqual(out_set[0, 0, 2].item(), 17.0)

        # Steer action
        itv_steer = FeatureIntervention(
            target="layer.0",
            adapter=adapter,
            feature_indices=2,
            action="steer",
            steer_strength=4.0,
        )
        out_steer = itv_steer.transform(x)
        self.assertEqual(out_steer[0, 0, 2].item(), 5.0)

        # Error-preserving property
        itv_none = FeatureIntervention(
            target="layer.0",
            adapter=adapter,
            feature_indices=[],
            action="scale",
            scale=1.0,
            error_preserving=True,
        )
        self.assertTrue(torch.allclose(itv_none.transform(x), x))

        # Token-specific feature intervention
        itv_tok = FeatureIntervention(
            target="layer.0",
            adapter=adapter,
            feature_indices=0,
            action="zero",
            token_indices=-1,
        )
        out_tok = itv_tok.transform(x)
        self.assertEqual(out_tok[0, -1, 0].item(), 0.0)
        self.assertEqual(out_tok[0, 0, 0].item(), 1.0)

    def test_feature_adapter_hook_context_lifecycle(self):
        adapter = FeatureAdapter(8, 16, target="layer.0")
        interv = FeatureIntervention(
            target="layer.0",
            adapter=adapter,
            feature_indices=0,
            action="steer",
            steer_strength=20.0,
        )
        ids = torch.tensor([[1, 2, 3]])
        base = self.model(ids).clone()

        with FeatureAdapterHook(self.model, interv) as hook:
            self.assertTrue(hook.is_active)
            mod = self.model(ids)
            self.assertFalse(torch.allclose(base, mod))

        self.assertFalse(hook.is_active)
        self.assertTrue(torch.allclose(base, self.model(ids)))

        # Incompatible adapter raises on enter
        bad_adapter = FeatureAdapter(16, 32, target="layer.0")
        bad_itv = FeatureIntervention(target="layer.0", adapter=bad_adapter, feature_indices=0)
        with self.assertRaises(FeatureAdapterCompatibilityError):
            with FeatureAdapterHook(self.model, bad_itv):
                pass

    # -----------------------------------------------------------------------
    # 7. Interacting Components & Composition Order Tests
    # -----------------------------------------------------------------------

    def test_evaluate_interacting_components_synergy_and_regression(self):
        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[8, 9, 10]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])

        res = evaluate_interacting_components(
            self.model,
            ["layer.0.channel.1", "layer.0.channel.2"],
            workload,
            keep_inputs=[torch.tensor([[1, 1, 1]])],
        )

        self.assertEqual(len(res.components), 2)
        self.assertIn("layer.0.mlp_channel.1", res.individual_effects)
        self.assertIn("layer.0.mlp_channel.2", res.individual_effects)
        self.assertIsInstance(res.joint_effect, float)
        self.assertIn(res.synergy_type, ("synergistic", "additive", "subadditive", "antagonistic"))
        self.assertIsNotNone(res.joint_keep_damage)

        s_dict = res.summary_dict()
        self.assertIn("joint_effect", s_dict)
        self.assertIn("has_regression", s_dict)

        # Test regression detection and raise_on_regression
        itv_a = ScaleIntervention("layer.0.channel.0", scale=10.0)
        itv_b = ScaleIntervention("layer.0.channel.0", scale=-10.0)
        bad_res = evaluate_interacting_components(
            self.model,
            [itv_a, itv_b],
            workload,
            raise_on_regression=False,
        )
        self.assertIsInstance(bad_res, InteractionResult)

    def test_search_interacting_component_sets(self):
        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[8, 9, 10]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])

        pool = ["layer.0.channel.0", "layer.0.channel.1", "layer.0.channel.2"]
        report = search_interacting_component_sets(
            self.model, pool, workload, set_sizes=(2,), max_combinations=3
        )

        self.assertGreater(report.summary_dict()["total_sets_evaluated"], 0)
        self.assertIsNotNone(report.best_set)

    def test_evaluate_composition_order(self):
        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[8, 9, 10]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])

        # Order-dependent interventions on the same component
        itv1 = ClampIntervention("layer.0", max_norm=0.5)
        itv2 = ScaleIntervention("layer.0", scale=10.0)

        order_res = evaluate_composition_order(self.model, [itv1, itv2], workload)
        self.assertFalse(order_res.is_order_invariant)
        self.assertGreater(order_res.max_order_delta, 0.0)
        self.assertEqual(len(order_res.order_permutations), 2)
        self.assertIn(order_res.best_order, order_res.order_permutations)
        self.assertIn(order_res.worst_order, order_res.order_permutations)

        s_dict = order_res.summary_dict()
        self.assertIn("max_order_delta", s_dict)
        self.assertFalse(s_dict["is_order_invariant"])

        # Order-invariant interventions across separate layers
        itv_l0 = ScaleIntervention("layer.0.channel.0", scale=0.5)
        itv_l1 = ScaleIntervention("layer.1.channel.0", scale=0.5)
        res_inv = evaluate_composition_order(self.model, [itv_l0, itv_l1], workload, tolerance=0.05)
        self.assertTrue(res_inv.is_order_invariant)

    # -----------------------------------------------------------------------
    # 8. Hyperparameter & Causal Search Tests
    # -----------------------------------------------------------------------

    def test_search_causal_interventions_grid_and_pareto(self):
        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[8, 9, 10]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])
        keep_inputs = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])]

        search_space = CausalSearchSpace(
            locations=["layer.0.channel.0", "layer.0.channel.1"],
            strengths=[0.0, 1.0],
            intervention_type="scale",
            token_indices=-1,
        )

        search_res = search_causal_interventions(
            self.model, search_space, workload, keep_inputs=keep_inputs
        )

        self.assertEqual(len(search_res.evaluations), 4)
        self.assertIsNotNone(search_res.best_candidate)
        self.assertGreaterEqual(len(search_res.pareto_candidates), 1)
        self.assertTrue(all(c.is_pareto_optimal for c in search_res.pareto_candidates))

        s_dict = search_res.summary_dict()
        self.assertEqual(s_dict["total_evaluated"], 4)
        self.assertIn("best_candidate", s_dict)

    def test_causal_search_space_rank_and_subspace(self):
        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[8, 9, 10]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])

        search_space = CausalSearchSpace(
            locations=["layer.0"],
            strengths=[1.0],
            ranks=[1, 2],
            intervention_type="project",
            projection_directions={"layer.0": torch.randn(8, 4)},
        )

        res = search_causal_interventions(self.model, search_space, workload)
        self.assertEqual(len(res.evaluations), 2)
        self.assertEqual({c.rank for c in res.evaluations}, {1, 2})

    def test_interaction_regression_exception_raising(self):
        clean_in = torch.tensor([[1, 2, 3]])
        corr_in = torch.tensor([[8, 9, 10]])
        workload = PairedWorkload([WorkloadPair(clean_in, corr_in, clean_target=5, corrupted_target=7)])

        itv_a = ScaleIntervention("layer.0.channel.0", scale=10.0)
        itv_b = ScaleIntervention("layer.0.channel.0", scale=-10.0)
        keep_inputs = [torch.tensor([[1, 2, 3]])]

        with self.assertRaises(InteractionRegressionError):
            evaluate_interacting_components(
                self.model,
                [itv_a, itv_b],
                workload,
                keep_inputs=keep_inputs,
                max_allowed_kl=-1.0,
                raise_on_regression=True,
            )

    def test_feature_intervention_in_model_forward(self):
        adapter = FeatureAdapter(activation_dim=8, feature_dim=16, target="layer.0")
        itv_scale = FeatureIntervention("layer.0", adapter, feature_indices=0, action="scale", scale=0.5)
        itv_clamp = FeatureIntervention("layer.0", adapter, feature_indices=1, action="clamp", min_val=-0.2, max_val=0.2)
        itv_set = FeatureIntervention("layer.0", adapter, feature_indices=2, action="set", set_value=1.5)

        ids = torch.tensor([[1, 2, 3]])
        base = self.model(ids).clone()
        with RuntimeInterventionContext(self.model, [itv_scale, itv_clamp, itv_set]):
            out = self.model(ids)
            self.assertEqual(out.shape, base.shape)
            self.assertFalse(torch.allclose(base, out))

        self.assertTrue(torch.allclose(base, self.model(ids)))

    def test_feature_adapter_unsupported_actions_and_dimension_errors(self):
        adapter = FeatureAdapter(8, 16, target="layer.0")
        itv_bad = FeatureIntervention("layer.0", adapter, feature_indices=0, action="invalid_action")
        with self.assertRaises(ValueError):
            itv_bad.transform(torch.randn(1, 2, 8))

        with self.assertRaises(ValueError):
            FeatureAdapter(8, 16, encoder_weight=torch.randn(5, 5))
        with self.assertRaises(ValueError):
            FeatureAdapter(8, 16, decoder_weight=torch.randn(5, 5))

        adapter_no_target = FeatureAdapter(8, 16)
        with self.assertRaises(FeatureAdapterCompatibilityError):
            adapter_no_target.check_compatibility(self.model)

    def test_causal_search_empty_or_invalid(self):
        workload = PairedWorkload([WorkloadPair(torch.tensor([[1]]), torch.tensor([[2]]))])
        space_empty = CausalSearchSpace(locations=[], strengths=[1.0])
        with self.assertRaises(ValueError):
            search_causal_interventions(self.model, space_empty, workload)

        with self.assertRaises(ValueError):
            evaluate_composition_order(self.model, [], workload)

    def test_dimension_detection_across_module_types(self):
        mod_l, _, _ = resolve_component_module(self.model, "layer.0")
        dim_l = _detect_module_activation_dim(self.model, parse_component_target("layer.0"), mod_l)
        self.assertEqual(dim_l, 8)

        mod_mlp, _, _ = resolve_component_module(self.model, "layer.0.mlp")
        dim_mlp = _detect_module_activation_dim(self.model, parse_component_target("layer.0.mlp"), mod_mlp)
        self.assertEqual(dim_mlp, 8)

        mod_ch, _, _ = resolve_component_module(self.model, "layer.0.channel.0")
        dim_ch = _detect_module_activation_dim(self.model, parse_component_target("layer.0.channel.0"), mod_ch)
        self.assertEqual(dim_ch, 16)

        mod_hd, _, _ = resolve_component_module(self.model, "layer.0.head.0")
        dim_hd = _detect_module_activation_dim(self.model, parse_component_target("layer.0.head.0"), mod_hd)
        self.assertEqual(dim_hd, 2)


if __name__ == "__main__":
    unittest.main()

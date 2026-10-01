import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from aegis_lab.editing.neurosurgery.stage5_recovery import (
    LoRALinear,
    RecoveryConfig,
    RecoveryLossConfig,
    RecoveryResult,
    TrainableParamsReport,
    apply_freeze_mask,
    compute_distillation_loss,
    compute_lm_cross_entropy,
    compute_recovery_loss,
    export_lora_state,
    inject_lora,
    load_lora_adapter,
    load_recovery_checkpoint,
    merge_lora,
    run_recovery_training,
    save_lora_adapter,
    save_recovery_checkpoint,
    seed_everything,
    unmerge_lora,
    verify_merge_parity,
    verify_trainable_parameters,
)


class TinyMLP(nn.Module):
    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, out_features)
        self.up_proj = nn.Linear(in_features, out_features)
        self.down_proj = nn.Linear(out_features, in_features)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=32, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.mlp = TinyMLP(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        h = self.mlp(h)
        h = self.o_proj(h)
        return self.lm_head(h)


class TiedCausalLM(nn.Module):
    def __init__(self, vocab_size=32, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        # Tie embed and lm_head weights
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        h = self.down_proj(h)
        return self.lm_head(h)


class TestTargetedLoRAAndFreeze(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_lora_linear_zero_init_parity(self):
        base = nn.Linear(16, 24)
        lora = LoRALinear(base, r=4, lora_alpha=8.0)
        self.assertEqual(lora.r, 4)
        self.assertEqual(lora.lora_alpha, 8.0)
        self.assertAlmostEqual(lora.scaling, 2.0)
        self.assertTrue(torch.all(lora.lora_B == 0))

        x = torch.randn(2, 5, 16)
        out_base = base(x)
        out_lora = lora(x)
        self.assertTrue(torch.allclose(out_base, out_lora, atol=1e-6))

    def test_lora_rank_zero_identity(self):
        base = nn.Linear(8, 8)
        lora = LoRALinear(base, r=0)
        self.assertIsNone(lora.lora_A)
        self.assertIsNone(lora.lora_B)
        x = torch.randn(2, 8)
        self.assertTrue(torch.allclose(base(x), lora(x)))

    def test_inject_lora_targets_only_selected_modules(self):
        model = TinyCausalLM()
        injected = inject_lora(model, target_modules=["down_proj", "o_proj"], r=4)

        # Expected injected modules: mlp.down_proj and o_proj
        self.assertIn("mlp.down_proj", injected)
        self.assertIn("o_proj", injected)
        self.assertIsInstance(model.mlp.down_proj, LoRALinear)
        self.assertIsInstance(model.o_proj, LoRALinear)

        # Non-targeted modules remain standard nn.Linear
        self.assertIsInstance(model.mlp.gate_proj, nn.Linear)
        self.assertNotIsInstance(model.mlp.gate_proj, LoRALinear)
        self.assertIsInstance(model.mlp.up_proj, nn.Linear)
        self.assertNotIsInstance(model.mlp.up_proj, LoRALinear)
        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertNotIsInstance(model.lm_head, LoRALinear)

    def test_apply_freeze_mask_enables_only_lora(self):
        model = TinyCausalLM()
        inject_lora(model, target_modules=["down_proj", "o_proj"], r=4)
        report = apply_freeze_mask(model, allow_lora_only=True)

        self.assertIsInstance(report, TrainableParamsReport)
        self.assertTrue(report.trainable_params > 0)
        self.assertTrue(report.frozen_params > 0)
        self.assertTrue(0.0 < report.trainable_ratio < 1.0)

        # Verify only lora parameters have requires_grad=True
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                self.assertTrue(param.requires_grad, f"{name} should have requires_grad=True")
            else:
                self.assertFalse(param.requires_grad, f"{name} should have requires_grad=False")

    def test_verify_trainable_parameters_strict(self):
        model = TinyCausalLM()
        inject_lora(model, target_modules=["down_proj"], r=4)
        apply_freeze_mask(model, allow_lora_only=True)

        # Passing verification
        report = verify_trainable_parameters(model, allowed_patterns=["lora_A", "lora_B"], strict=True)
        self.assertTrue(report.tied_safe)

        # Unauthorized parameter unfrozen should fail strict verification
        model.mlp.gate_proj.weight.requires_grad = True
        with self.assertRaises(ValueError):
            verify_trainable_parameters(model, allowed_patterns=["lora_A", "lora_B"], strict=True)

    def test_tied_parameter_safety_untainted(self):
        model = TiedCausalLM()
        inject_lora(model, target_modules=["down_proj"], r=4)
        report = apply_freeze_mask(model, allow_lora_only=True)

        self.assertTrue(report.tied_safe)
        self.assertEqual(len(report.tied_param_groups), 1)
        tied_pair = set(report.tied_param_groups[0])
        self.assertIn("embed.weight", tied_pair)
        self.assertIn("lm_head.weight", tied_pair)

    def test_tied_parameter_safety_asymmetric_warning(self):
        model = TiedCausalLM()
        # Unfreezing only embed.weight when lm_head shares storage
        model.embed.weight.requires_grad = True
        report = verify_trainable_parameters(model, allowed_patterns=["embed.weight"], strict=False)
        self.assertFalse(report.tied_safe)


class TestMultiObjectiveRecoveryLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_compute_lm_cross_entropy_autoregressive(self):
        logits = torch.randn(2, 6, 32)
        input_ids = torch.randint(0, 32, (2, 6))
        loss = compute_lm_cross_entropy(logits, input_ids=input_ids)
        self.assertGreater(float(loss.item()), 0.0)

    def test_compute_lm_cross_entropy_all_ignored_returns_zero(self):
        logits = torch.randn(2, 4, 16)
        labels = torch.full((2, 4), -100, dtype=torch.long)
        loss = compute_lm_cross_entropy(logits, labels=labels, ignore_index=-100)
        self.assertEqual(float(loss.item()), 0.0)

    def test_compute_distillation_loss_identical_and_different(self):
        s_logits = torch.randn(2, 5, 20)
        t_logits = s_logits.clone()

        # Identical logits -> 0 KL divergence
        loss_identical = compute_distillation_loss(s_logits, t_logits, temperature=2.0)
        self.assertAlmostEqual(float(loss_identical.item()), 0.0, places=4)

        # Divergent logits -> positive loss
        diff_logits = torch.randn(2, 5, 20)
        loss_diff = compute_distillation_loss(diff_logits, t_logits, temperature=2.0)
        self.assertGreater(float(loss_diff.item()), 0.0)

    def test_compute_distillation_loss_masked(self):
        s_logits = torch.randn(2, 4, 10)
        t_logits = torch.randn(2, 4, 10)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
        loss = compute_distillation_loss(s_logits, t_logits, temperature=1.5, attention_mask=mask)
        self.assertGreater(float(loss.item()), 0.0)

    def test_compute_recovery_loss_multi_objective(self):
        model = TinyCausalLM()
        teacher = TinyCausalLM()

        keep_batch = {"input_ids": torch.randint(0, 32, (2, 6))}
        change_batch = {"input_ids": torch.randint(0, 32, (2, 6))}
        distill_batch = {"input_ids": torch.randint(0, 32, (2, 6))}

        cfg = RecoveryLossConfig(w_keep=1.0, w_change=0.5, w_distill=0.2, distill_temperature=2.0)
        out = compute_recovery_loss(
            model=model,
            keep_batch=keep_batch,
            change_batch=change_batch,
            distill_batch=distill_batch,
            teacher_model=teacher,
            loss_config=cfg,
        )

        self.assertGreater(out.keep_loss, 0.0)
        self.assertGreater(out.change_loss, 0.0)
        self.assertGreater(out.distill_loss, 0.0)

        expected_total = (
            cfg.w_keep * out.keep_loss
            + cfg.w_change * out.change_loss
            + cfg.w_distill * out.distill_loss
        )
        self.assertAlmostEqual(float(out.total_loss.item()), expected_total, places=4)

    def test_compute_recovery_loss_zero_weights(self):
        model = TinyCausalLM()
        keep_batch = {"input_ids": torch.randint(0, 32, (2, 6))}
        cfg = RecoveryLossConfig(w_keep=1.0, w_change=0.0, w_distill=0.0)
        out = compute_recovery_loss(model=model, keep_batch=keep_batch, loss_config=cfg)
        self.assertEqual(out.change_loss, 0.0)
        self.assertEqual(out.distill_loss, 0.0)
        self.assertAlmostEqual(float(out.total_loss.item()), out.keep_loss, places=4)


class TestTrainingAndRecoveryLoop(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_recovery_training_loop_converges(self):
        model = TinyCausalLM()
        inject_lora(model, target_modules=["down_proj", "o_proj"], r=4)
        apply_freeze_mask(model, allow_lora_only=True)

        batch = {"input_ids": torch.randint(0, 32, (4, 8))}
        cfg = RecoveryConfig(max_steps=12, lr=1e-2, seed=42)
        res = run_recovery_training(model=model, keep_data=[batch], config=cfg)

        self.assertTrue(res.success)
        self.assertEqual(res.status, "SUCCESS")
        self.assertEqual(res.steps_completed, 12)
        # Loss decreases from initial to final
        initial_loss = res.history[0]["total_loss"]
        self.assertLess(res.final_loss, initial_loss)

    def test_gradient_accumulation(self):
        model = TinyCausalLM()
        inject_lora(model, target_modules=["down_proj"], r=4)
        apply_freeze_mask(model, allow_lora_only=True)

        batch = {"input_ids": torch.randint(0, 32, (2, 6))}
        cfg = RecoveryConfig(max_steps=6, gradient_accumulation_steps=2, lr=1e-3)
        res = run_recovery_training(model=model, keep_data=[batch], config=cfg)
        self.assertTrue(res.success)
        self.assertEqual(res.steps_completed, 6)

    def test_rng_seeding_reproducibility(self):
        def _train_once():
            seed_everything(1234)
            m = TinyCausalLM()
            inject_lora(m, target_modules=["down_proj"], r=4)
            apply_freeze_mask(m, allow_lora_only=True)
            batch = {"input_ids": torch.randint(0, 32, (2, 6))}
            cfg = RecoveryConfig(max_steps=5, seed=1234, lr=1e-3)
            return run_recovery_training(model=m, keep_data=[batch], config=cfg)

        res1 = _train_once()
        res2 = _train_once()
        self.assertEqual(len(res1.history), len(res2.history))
        for h1, h2 in zip(res1.history, res2.history):
            self.assertAlmostEqual(h1["total_loss"], h2["total_loss"], places=5)

    def test_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = TinyCausalLM()
            inject_lora(model, target_modules=["down_proj"], r=4)
            apply_freeze_mask(model, allow_lora_only=True)

            batch = {"input_ids": torch.randint(0, 32, (2, 6))}
            cfg = RecoveryConfig(
                max_steps=4,
                checkpoint_dir=tmp_dir,
                checkpoint_every=2,
            )
            res = run_recovery_training(model=model, keep_data=[batch], config=cfg)
            self.assertTrue(res.success)

            ckpt_step2 = Path(tmp_dir) / "checkpoint_step_2.pt"
            self.assertTrue(ckpt_step2.exists())

            # Load checkpoint into fresh model
            fresh_model = TinyCausalLM()
            inject_lora(fresh_model, target_modules=["down_proj"], r=4)
            info = load_recovery_checkpoint(ckpt_step2, model=fresh_model)
            self.assertEqual(info["step"], 2)

    def test_time_budget_enforcement(self):
        model = TinyCausalLM()
        inject_lora(model, target_modules=["down_proj"], r=4)
        apply_freeze_mask(model, allow_lora_only=True)

        batch = {"input_ids": torch.randint(0, 32, (2, 6))}
        cfg = RecoveryConfig(max_steps=1000, max_time_seconds=0.001)
        res = run_recovery_training(model=model, keep_data=[batch], config=cfg)
        self.assertEqual(res.status, "BUDGET_EXCEEDED")
        self.assertLess(res.steps_completed, 1000)

    def test_drop_monitoring_pass(self):
        model = TinyCausalLM()
        inject_lora(model, target_modules=["down_proj"], r=4)
        apply_freeze_mask(model, allow_lora_only=True)

        # Evaluator returns safe, non-rebounding score
        def safe_drop_evaluator(_):
            return 0.05

        batch = {"input_ids": torch.randint(0, 32, (2, 6))}
        cfg = RecoveryConfig(
            max_steps=6,
            eval_drop_every=2,
            max_drop_threshold=0.15,
            max_drop_rebound=0.10,
        )
        res = run_recovery_training(
            model=model,
            keep_data=[batch],
            config=cfg,
            drop_evaluator=safe_drop_evaluator,
        )
        self.assertTrue(res.success)
        self.assertFalse(res.drop_violation)
        self.assertEqual(res.status, "SUCCESS")

    def test_drop_rebound_absolute_threshold_failure(self):
        model = TinyCausalLM()
        inject_lora(model, target_modules=["down_proj"], r=4)
        apply_freeze_mask(model, allow_lora_only=True)

        scores = [0.05, 0.08, 0.25, 0.35]  # rebounds above 0.20
        idx = {"count": 0}

        def rebounding_evaluator(_):
            val = scores[min(idx["count"], len(scores) - 1)]
            idx["count"] += 1
            return val

        batch = {"input_ids": torch.randint(0, 32, (2, 6))}
        cfg = RecoveryConfig(
            max_steps=10,
            eval_drop_every=2,
            max_drop_threshold=0.20,
            stop_on_drop_rebound=True,
        )
        res = run_recovery_training(
            model=model,
            keep_data=[batch],
            config=cfg,
            drop_evaluator=rebounding_evaluator,
        )
        self.assertFalse(res.success)
        self.assertTrue(res.drop_violation)
        self.assertEqual(res.status, "DROP_REBOUND_FAILED")

    def test_drop_rebound_delta_failure(self):
        model = TinyCausalLM()
        inject_lora(model, target_modules=["down_proj"], r=4)
        apply_freeze_mask(model, allow_lora_only=True)

        scores = [0.10, 0.12, 0.25]
        idx = {"count": 0}

        def delta_rebound_evaluator(_):
            val = scores[min(idx["count"], len(scores) - 1)]
            idx["count"] += 1
            return val

        batch = {"input_ids": torch.randint(0, 32, (2, 6))}
        cfg = RecoveryConfig(
            max_steps=6,
            eval_drop_every=2,
            max_drop_rebound=0.10,  # Delta from 0.10 to 0.25 is 0.15 > 0.10
            stop_on_drop_rebound=True,
        )
        res = run_recovery_training(
            model=model,
            keep_data=[batch],
            config=cfg,
            drop_evaluator=delta_rebound_evaluator,
        )
        self.assertFalse(res.success)
        self.assertTrue(res.drop_violation)
        self.assertEqual(res.status, "DROP_REBOUND_FAILED")


class TestAdapterMergeAndExport(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_export_and_save_lora_adapter(self):
        model = TinyCausalLM()
        inject_lora(model, target_modules=["down_proj", "o_proj"], r=4)

        state = export_lora_state(model)
        self.assertIn("mlp.down_proj", state)
        self.assertIn("o_proj", state)
        self.assertEqual(state["mlp.down_proj"]["r"], 4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter_path = Path(tmp_dir) / "adapter.pt"
            saved_p, sha = save_lora_adapter(model, adapter_path)
            self.assertTrue(saved_p.exists())
            self.assertIsInstance(sha, str)
            self.assertEqual(len(sha), 64)

            # Test loading adapter into fresh base model
            fresh_model = TinyCausalLM()
            load_lora_adapter(fresh_model, adapter_path)
            self.assertIsInstance(fresh_model.mlp.down_proj, LoRALinear)
            self.assertIsInstance(fresh_model.o_proj, LoRALinear)

    def test_merge_lora_replaces_layers_with_base_linear(self):
        model = TinyCausalLM()
        inject_lora(model, target_modules=["down_proj", "o_proj"], r=4)

        # Set non-zero LoRA B to ensure weight delta exists
        with torch.no_grad():
            model.mlp.down_proj.lora_B.fill_(0.01)

        original_weight = model.mlp.down_proj.base_layer.weight.clone()
        merged_model = merge_lora(model, in_place=True)

        # LoRALinear should be replaced by nn.Linear
        self.assertIsInstance(merged_model.mlp.down_proj, nn.Linear)
        self.assertNotIsInstance(merged_model.mlp.down_proj, LoRALinear)
        self.assertIsInstance(merged_model.o_proj, nn.Linear)

        # Base weights should have been updated with delta
        self.assertFalse(torch.allclose(merged_model.mlp.down_proj.weight, original_weight))

    def test_verify_merge_parity_success(self):
        model = TinyCausalLM()
        inject_lora(model, target_modules=["down_proj", "o_proj"], r=4)

        # Perturb LoRA weights to have non-trivial delta
        with torch.no_grad():
            model.mlp.down_proj.lora_A.normal_(0.0, 0.1)
            model.mlp.down_proj.lora_B.normal_(0.0, 0.1)
            model.o_proj.lora_A.normal_(0.0, 0.1)
            model.o_proj.lora_B.normal_(0.0, 0.1)

        sample_inputs = {"input_ids": torch.randint(0, 32, (3, 7))}
        report = verify_merge_parity(adapter_model=model, sample_inputs=sample_inputs, rtol=1e-4, atol=1e-4)

        self.assertTrue(report["parity"])
        self.assertLess(report["max_abs_diff"], 1e-4)

    def test_verify_merge_parity_tampered_failure(self):
        model = TinyCausalLM()
        inject_lora(model, target_modules=["down_proj"], r=4)
        with torch.no_grad():
            model.mlp.down_proj.lora_A.normal_(0.0, 0.1)
            model.mlp.down_proj.lora_B.normal_(0.0, 0.1)

        merged_copy = merge_lora(model, in_place=False)
        # Tamper with merged weights
        with torch.no_grad():
            merged_copy.mlp.down_proj.weight.add_(5.0)

        sample_inputs = {"input_ids": torch.randint(0, 32, (2, 5))}
        with self.assertRaises(ValueError):
            verify_merge_parity(
                adapter_model=model,
                merged_model=merged_copy,
                sample_inputs=sample_inputs,
                raise_on_failure=True,
            )


if __name__ == "__main__":
    unittest.main()

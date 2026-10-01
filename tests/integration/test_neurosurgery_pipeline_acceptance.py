"""End-to-End Pipeline Acceptance Suite for AEGIS-LAB Model Neurosurgery.

Validates the complete neurosurgery lifecycle per docs/neurosurgery/ROADMAP.md:
1. Identity candidate acceptance: unchanged metrics, 1.0 KEEP retention, 0.0 damage,
   export/reload parity, and restoration.
2. Changed candidate acceptance: directional/structural modification properly suppresses
   DROP target while preserving KEEP, gates pass/fail independently.
3. Rejection acceptance: empty prompts, mismatched vocabularies, non-finite logits,
   stale-model plan checksum mismatches are rejected before mutation.
4. Stage composition acceptance: structural edit -> LoRA recovery -> quantization export
   -> reload parity -> exact restoration integrity verified against parent manifest SHA256 hashes.
5. Multimodal amputation acceptance: vision branch removed from multimodal model,
   text-only generation verified, input rejection on image inputs.

All tests execute fast on CPU without external downloads or hardcoded absolute paths.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
import torch.nn as nn

# Stage 4A: Provenance, Sliced Evaluation, Restoration, Gate Evaluation
from aegis_lab.editing.neurosurgery.stage4a_provenance import (
    DatasetFingerprint,
    EvaluationReport,
    GateResult,
    GateThresholds,
    NeurosurgeryDataset,
    ProvenanceManifest,
    RestorationIntegrityError,
    RestorationManifest,
    Sample,
    SlicedMetricsReport,
    StaleModelPlanError,
    backup_checkpoint_files,
    backup_edited_tensors,
    bind_provenance_to_plan,
    compute_config_hash,
    compute_directory_file_hashes,
    compute_model_hash,
    compute_state_dict_hashes,
    compute_tensor_hash,
    compute_tokenizer_hash,
    create_file_restoration_manifest,
    create_restoration_manifest,
    evaluate_sliced_sequence_task,
    export_candidate_checkpoint,
    gate_evaluation,
    generate_sequence_completions,
    load_candidate_checkpoint,
    publish_evaluation_report,
    restore_checkpoint_files_from_backup,
    restore_state_dict,
    run_end_to_end_evaluation,
    validate_plan_against_model,
    validate_plan_provenance,
    verify_disjoint_dataset_collection,
    verify_restoration_integrity,
    verify_restored_directory,
)

# Stage 4B: Unified Operation Contract & Execution
from aegis_lab.editing.neurosurgery.stage4b_contract import (
    ComponentType,
    ExecutionSemantics,
    UnifiedOperation,
    apply_plan,
)

# Stage 5: LoRA Recovery & Verification
from aegis_lab.editing.neurosurgery.stage5_recovery import (
    RecoveryConfig,
    apply_freeze_mask,
    inject_lora,
    merge_lora,
    run_recovery_training,
    verify_merge_parity,
    verify_trainable_parameters,
)

# Stage 6: Quantization & Parity
from aegis_lab.editing.neurosurgery.stage6_quantization import (
    QuantizationConfig,
    bind_calibration_dataset,
    export_quantized_checkpoint,
    load_quantized_checkpoint,
    quantize_model,
    verify_export_reload_parity,
)

# Stage 8: Multimodal Amputation & Input Guards
from aegis_lab.editing.neurosurgery.stage8_modality import (
    ModalityRemovedError,
    amputate_modality,
    extract_logits,
    install_input_rejection_guard,
    remove_modality_branch,
    save_and_reload_amputated_model,
    validate_retained_modality,
    verify_state_dict_absence,
)


# ============================================================================
# Shared Lightweight Mock Models & Tokenizer Fixtures (CPU-only, Fast)
# ============================================================================

class MockTokenizer:
    """Deterministic lightweight tokenizer for integration testing."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self.vocab = vocab or {
            "pad": 0,
            "eos": 1,
            "a": 2,
            "b": 3,
            "c": 4,
            "x": 5,
            "y": 6,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab.get("pad", 0)
        self.eos_token_id = self.vocab.get("eos", 1)

    def __call__(
        self,
        batch: str | Sequence[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if isinstance(batch, str):
            batch = [batch]
        encoded = []
        for text in batch:
            tokens = [self.vocab.get(word, 2) for word in text.split() if word]
            if not tokens:
                tokens = [self.vocab.get("a", 2)]
            encoded.append(tokens)
        max_len = max(len(row) for row in encoded) if encoded else 1
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

    def decode(self, token_ids: Sequence[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        out = []
        for t in token_ids:
            item = t.item() if hasattr(t, "item") else int(t)
            if skip_special_tokens and item in (self.pad_token_id, self.eos_token_id):
                continue
            out.append(self.inv_vocab.get(item, str(item)))
        return " ".join(out)


class TinyMLPBlock(nn.Module):
    """Gated MLP block compatible with neurosurgery adapters."""

    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class TinyTransformerLayer(nn.Module):
    """Transformer layer containing an MLP block."""

    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.mlp = TinyMLPBlock(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class AcceptanceCausalLM(nn.Module):
    """Lightweight causal LM fixture adhering to standard transformer topology."""

    def __init__(self, vocab_size: int = 7, hidden_dim: int = 8, num_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([TinyTransformerLayer(hidden_dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.config = SimpleNamespace(
            model_type="llama",
            num_hidden_layers=num_layers,
            intermediate_size=hidden_dim,
            vocab_size=vocab_size,
            hidden_size=hidden_dim,
        )
        self.device = torch.device("cpu")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        if input_ids is None and "inputs_embeds" in kwargs:
            h = kwargs["inputs_embeds"]
        elif input_ids is not None:
            h = self.embed_tokens(input_ids)
        else:
            raise ValueError("input_ids or inputs_embeds must be provided")

        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(h)
        if return_dict:
            return SimpleNamespace(logits=logits)
        return logits


class TinyVisionEncoder(nn.Module):
    """Small conv vision encoder for multimodal tests."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(pixel_values))
        return self.proj(h.flatten(2).transpose(1, 2))


class TinyMultimodalProjector(nn.Module):
    """Two-layer GELU multimodal projector."""

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class TinyLanguageBackbone(nn.Module):
    """Language trunk for multimodal architecture."""

    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16, num_layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Any:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be specified.")
            h = self.embed_tokens(input_ids)
        else:
            h = inputs_embeds

        for layer in self.layers:
            h = torch.relu(layer(h))
        out = self.norm(h)
        if use_cache:
            return SimpleNamespace(last_hidden_state=out, past_key_values=(out.detach(),))
        return out


class MockMultimodalModel(nn.Module):
    """LLaVA-topology multimodal model fixture for amputation testing."""

    def __init__(self, vocab_size: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.vision_tower = TinyVisionEncoder(in_channels=3, hidden_dim=hidden_dim)
        self.multi_modal_projector = TinyMultimodalProjector(hidden_dim=hidden_dim)
        self.language_model = TinyLanguageBackbone(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.config = SimpleNamespace(
            model_type="llava",
            architectures=["LlavaForConditionalGeneration"],
            vision_config={"hidden_size": hidden_dim},
            mm_projector_type="mlp2x_gelu",
            text_config={"vocab_size": vocab_size, "hidden_size": hidden_dim},
        )
        self.device = torch.device("cpu")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> Any:
        if pixel_values is not None:
            v_feats = self.vision_tower(pixel_values)
            proj_feats = self.multi_modal_projector(v_feats)
            if input_ids is not None:
                t_embeds = self.language_model.embed_tokens(input_ids)
                h = torch.cat([proj_feats, t_embeds], dim=1)
            else:
                h = proj_feats
            lm_out = self.language_model(inputs_embeds=h, past_key_values=past_key_values, use_cache=use_cache)
        else:
            lm_out = self.language_model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache)

        if isinstance(lm_out, SimpleNamespace):
            logits = self.lm_head(lm_out.last_hidden_state)
            return SimpleNamespace(logits=logits, past_key_values=lm_out.past_key_values)
        return self.lm_head(lm_out)

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 3, **kwargs: Any) -> torch.Tensor:
        curr = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.forward(input_ids=curr, **kwargs)
            logits = extract_logits(out)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr = torch.cat([curr, next_tok], dim=-1)
        return curr


class MockMultimodalProcessor:
    """Mock processor matching HuggingFace multimodal processor interface."""

    def __init__(self):
        self.tokenizer = lambda text, **kw: {"input_ids": torch.tensor([[1, 2, 3]])}
        self.image_processor = lambda images, **kw: {"pixel_values": torch.randn(1, 3, 14, 14)}

    def __call__(self, text: Any = None, images: Any = None, **kwargs: Any) -> dict[str, Any]:
        res: dict[str, Any] = {}
        if text is not None:
            res.update(self.tokenizer(text, **kwargs))
        if images is not None:
            res.update(self.image_processor(images, **kwargs))
        return res


# ============================================================================
# Acceptance Test Suites
# ============================================================================

class TestIdentityCandidateAcceptance(unittest.TestCase):
    """Acceptance Scope 1: Identity candidate acceptance.

    Verifies that an identity candidate exercises unchanged metrics, 1.0 KEEP retention,
    0.0 damage across all domain slices, export/reload parity, and restoration.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = MockTokenizer()
        self.baseline_model = AcceptanceCausalLM(vocab_size=7, hidden_dim=8)
        self.candidate_model = AcceptanceCausalLM(vocab_size=7, hidden_dim=8)
        self.candidate_model.load_state_dict(self.baseline_model.state_dict())

        # Disjoint KEEP, DROP, CHANGE datasets with multi-domain slices
        self.keep_ds = NeurosurgeryDataset(kind="keep", name="acc_keep")
        self.keep_ds.add_sample("validation", Sample(prompt="a b", target="a", domain="math"))
        self.keep_ds.add_sample("validation", Sample(prompt="b c", target="a", domain="code"))
        self.keep_ds.add_sample("validation", Sample(prompt="c a", target="a", domain="reasoning"))

        self.drop_ds = NeurosurgeryDataset(kind="drop", name="acc_drop")
        self.drop_ds.add_sample("validation", Sample(prompt="x y", forbidden_targets=["x"], domain="safety"))

        self.change_ds = NeurosurgeryDataset(kind="change", name="acc_change")
        self.change_ds.add_sample("validation", Sample(prompt="a c", target="a", domain="knowledge"))

        self.datasets = {"keep": self.keep_ds, "drop": self.drop_ds, "change": self.change_ds}
        verify_disjoint_dataset_collection(self.datasets)

    def test_identity_metrics_1_0_keep_retention_and_0_0_damage(self):
        """Identity candidate must yield exactly 1.0 KEEP retention and 0.0 damage across all domains."""
        report = run_end_to_end_evaluation(
            candidate_model=self.candidate_model,
            tokenizer=self.tokenizer,
            datasets=self.datasets,
            split="validation",
            baseline_model=self.baseline_model,
            max_new_tokens=3,
        )

        self.assertIsNotNone(report.keep_report)
        self.assertAlmostEqual(report.keep_report.overall_score, 1.0)
        self.assertAlmostEqual(report.keep_report.worst_slice_damage, 0.0)

        # Per-domain slice verification
        for domain, metrics in report.keep_report.domain_metrics.items():
            self.assertAlmostEqual(metrics["retention_score"], 1.0, msg=f"Domain {domain} retention must be 1.0")
            self.assertAlmostEqual(metrics["damage"], 0.0, msg=f"Domain {domain} damage must be 0.0")

        # Gating with strict 1.0 retention threshold must pass
        thresh = GateThresholds(min_keep_retention=1.0, max_keep_worst_slice_damage=0.0)
        gate_res = report.gate(thresh)
        self.assertTrue(gate_res.passed)
        self.assertTrue(gate_res.keep_passed)

    def test_identity_export_and_reload_parity(self):
        """Exported candidate checkpoint must reload with exact weight and generation parity."""
        model_hash = compute_model_hash(self.candidate_model)
        tok_hash = compute_tokenizer_hash(self.tokenizer)
        cfg_dict = {"vocab_size": 7, "hidden_dim": 8}
        cfg_hash = compute_config_hash(cfg_dict)
        fps = {k: ds.fingerprint().to_dict() for k, ds in self.datasets.items()}

        prov = ProvenanceManifest(
            model_hash=model_hash,
            tokenizer_hash=tok_hash,
            config_hash=cfg_hash,
            dataset_fingerprints=fps,
            adapter_version="0.4.0",
            seed=42,
            dtype="float32",
            operation_order=["identity"],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            export_dir = Path(tmp_dir) / "identity_candidate_export"
            written = export_candidate_checkpoint(
                candidate_state_dict=self.candidate_model.state_dict(),
                export_dir=export_dir,
                provenance=prov,
                config=cfg_dict,
            )
            self.assertTrue(Path(written["weights"]).exists())
            self.assertTrue(Path(written["provenance"]).exists())

            # Reload into fresh instance
            loaded_sd, loaded_prov, _ = load_candidate_checkpoint(export_dir)
            self.assertEqual(loaded_prov.compute_manifest_hash(), prov.compute_manifest_hash())

            reloaded_model = AcceptanceCausalLM(vocab_size=7, hidden_dim=8)
            reloaded_model.load_state_dict(loaded_sd)

            # Weight tensor bit-exact parity
            for key, t_orig in self.candidate_model.state_dict().items():
                self.assertTrue(torch.equal(t_orig, reloaded_model.state_dict()[key]))

            # Generation parity across test prompts
            test_prompts = ["a b", "b c", "x y"]
            orig_completions = generate_sequence_completions(
                self.candidate_model, self.tokenizer, test_prompts, max_new_tokens=4
            )
            reloaded_completions = generate_sequence_completions(
                reloaded_model, self.tokenizer, test_prompts, max_new_tokens=4
            )
            self.assertEqual(orig_completions, reloaded_completions)

    def test_identity_restoration_integrity(self):
        """Identity restoration must verify state dict and directory integrity against source hashes."""
        source_sd = self.baseline_model.state_dict()
        candidate_sd = self.candidate_model.state_dict()
        edited_tensors = ["layers.0.mlp.down_proj.weight"]
        backup = backup_edited_tensors(source_sd, edited_tensors)

        manifest = create_restoration_manifest(
            source_checkpoint="./checkpoints/base_model",
            candidate_checkpoint="./checkpoints/identity_model",
            source_state_dict=source_sd,
            candidate_state_dict=candidate_sd,
            edited_tensors=edited_tensors,
        )
        restored_sd = restore_state_dict(candidate_sd, backup, manifest)
        verified = verify_restoration_integrity(restored_sd, manifest)
        self.assertEqual(verified["status"], "verified")
        self.assertEqual(verified["tensors_verified"], len(source_sd))

        # Checkpoint directory level restoration
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_dir = tmp_path / "source"
            cand_dir = tmp_path / "candidate"
            backup_dir = tmp_path / "backup"
            restored_dir = tmp_path / "restored"

            source_dir.mkdir()
            cand_dir.mkdir()
            (source_dir / "config.json").write_text('{"vocab": 7}', encoding="utf-8")
            (source_dir / "model.pt").write_text("identity_source_content", encoding="utf-8")
            (cand_dir / "config.json").write_text('{"vocab": 7}', encoding="utf-8")
            (cand_dir / "model.pt").write_text("identity_source_content", encoding="utf-8")

            file_manifest = create_file_restoration_manifest(
                source_dir=source_dir,
                candidate_dir=cand_dir,
                edited_files=["model.pt"],
                backup_dir=backup_dir,
            )
            backup_checkpoint_files(source_dir, ["model.pt"], backup_dir)
            restore_checkpoint_files_from_backup(cand_dir, backup_dir, file_manifest, restored_dir)
            res_dir_ver = verify_restored_directory(restored_dir, file_manifest)
            self.assertEqual(res_dir_ver["status"], "verified")


class TestChangedCandidateAcceptance(unittest.TestCase):
    """Acceptance Scope 2: Changed candidate acceptance.

    Verifies directional / structural modification properly suppresses DROP target while
    preserving KEEP, and that validation gates pass/fail independently.
    """

    def setUp(self):
        torch.manual_seed(1234)
        self.tokenizer = MockTokenizer()
        self.base_model = AcceptanceCausalLM(vocab_size=7, hidden_dim=8)

        # Configure base model so prompt "x" generates forbidden token "x" (id 5),
        # while prompt "a" generates desired token "a" (id 2).
        with torch.no_grad():
            self.base_model.lm_head.weight.zero_()
            self.base_model.lm_head.weight[2, :] = self.base_model.embed_tokens.weight[2]
            self.base_model.lm_head.weight[3, :] = self.base_model.embed_tokens.weight[5] * 1.5
            self.base_model.lm_head.weight[5, :] = self.base_model.embed_tokens.weight[5] * 3.0

        self.candidate_model = AcceptanceCausalLM(vocab_size=7, hidden_dim=8)
        self.candidate_model.load_state_dict(self.base_model.state_dict())

        self.keep_ds = NeurosurgeryDataset(kind="keep", name="changed_keep")
        self.keep_ds.add_sample("validation", Sample(prompt="a", target="a", domain="math"))

        self.drop_ds = NeurosurgeryDataset(kind="drop", name="changed_drop")
        self.drop_ds.add_sample("validation", Sample(prompt="x", forbidden_targets=["x"], domain="safety"))

        self.change_ds = NeurosurgeryDataset(kind="change", name="changed_change")
        self.change_ds.add_sample("validation", Sample(prompt="a", target="b", domain="knowledge"))

    def test_directional_structural_modification_suppresses_drop_preserving_keep(self):
        """Structural / directional modification eliminates forbidden DROP output without harming KEEP."""
        # Baseline model outputs "x" on DROP prompt: fails DROP suppression
        base_rep = run_end_to_end_evaluation(
            candidate_model=self.base_model,
            tokenizer=self.tokenizer,
            datasets={"keep": self.keep_ds, "drop": self.drop_ds},
            split="validation",
            metric="substring",
            max_new_tokens=2,
        )
        self.assertEqual(base_rep.keep_report.overall_score, 1.0)
        self.assertEqual(base_rep.drop_report.overall_score, 0.0)  # Leaked forbidden output "x"

        # Apply structural surgery: persistent mask/clamp on token 5 logit projection
        op = UnifiedOperation.create_mask(
            target_path="lm_head.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            target_indices=[5],
            mask_value=0.0,
            execution_semantics=ExecutionSemantics.PERSISTENT,
            export_supported=True,
        )
        res = apply_plan(self.candidate_model, [op])
        self.assertTrue(res.success)

        # Evaluate candidate after surgery
        cand_rep = run_end_to_end_evaluation(
            candidate_model=self.candidate_model,
            tokenizer=self.tokenizer,
            datasets={"keep": self.keep_ds, "drop": self.drop_ds},
            split="validation",
            baseline_model=self.base_model,
            metric="substring",
            max_new_tokens=2,
        )

        # Target properly suppressed on DROP while KEEP is perfectly preserved
        self.assertEqual(cand_rep.keep_report.overall_score, 1.0)
        self.assertEqual(cand_rep.keep_report.worst_slice_damage, 0.0)
        self.assertEqual(cand_rep.drop_report.overall_score, 1.0)  # Suppressed!
        self.assertEqual(cand_rep.drop_report.worst_slice_damage, 0.0)

    def test_independent_gating_pass_and_fail(self):
        """KEEP, DROP, and CHANGE gates evaluate and pass/fail independently."""
        # Candidate with successful KEEP and DROP suppression
        op = UnifiedOperation.create_mask(
            target_path="lm_head.weight",
            component_type=ComponentType.WEIGHT_TENSOR,
            target_indices=[5],
            mask_value=0.0,
            execution_semantics=ExecutionSemantics.PERSISTENT,
            export_supported=True,
        )
        apply_plan(self.candidate_model, [op])

        datasets = {"keep": self.keep_ds, "drop": self.drop_ds, "change": self.change_ds}
        rep = run_end_to_end_evaluation(
            candidate_model=self.candidate_model,
            tokenizer=self.tokenizer,
            datasets=datasets,
            split="validation",
            baseline_model=self.base_model,
            metric="substring",
            max_new_tokens=2,
        )

        # 1. KEEP and DROP pass independently
        gate_keep_drop = GateThresholds(min_keep_retention=0.9, min_drop_suppression=0.9)
        res_kd = gate_evaluation(rep, gate_keep_drop)
        self.assertTrue(res_kd.passed)
        self.assertTrue(res_kd.keep_passed)
        self.assertTrue(res_kd.drop_passed)

        # 2. CHANGE fails independently when its objective was not applied
        gate_with_change = GateThresholds(
            min_keep_retention=0.9,
            min_drop_suppression=0.9,
            min_change_success=0.9,
        )
        res_change_fail = gate_evaluation(rep, gate_with_change)
        self.assertFalse(res_change_fail.passed)
        self.assertTrue(res_change_fail.keep_passed)
        self.assertTrue(res_change_fail.drop_passed)
        self.assertFalse(res_change_fail.change_passed)
        self.assertTrue(any("CHANGE success score" in f for f in res_change_fail.failures))

        # 3. Unablated base model: KEEP passes, but DROP fails independently
        base_rep = run_end_to_end_evaluation(
            candidate_model=self.base_model,
            tokenizer=self.tokenizer,
            datasets={"keep": self.keep_ds, "drop": self.drop_ds},
            split="validation",
            metric="substring",
            max_new_tokens=2,
        )
        res_base = gate_evaluation(base_rep, gate_keep_drop)
        self.assertFalse(res_base.passed)
        self.assertTrue(res_base.keep_passed)
        self.assertFalse(res_base.drop_passed)
        self.assertTrue(any("DROP suppression score" in f for f in res_base.failures))

        # 4. Corrupted model: DROP passes, but KEEP fails independently
        corrupted_model = copy.deepcopy(self.candidate_model)
        with torch.no_grad():
            corrupted_model.lm_head.weight[2, :].zero_()
            corrupted_model.lm_head.weight[3, :] = corrupted_model.embed_tokens.weight[2] * 2.0

        corr_rep = run_end_to_end_evaluation(
            candidate_model=corrupted_model,
            tokenizer=self.tokenizer,
            datasets={"keep": self.keep_ds, "drop": self.drop_ds},
            split="validation",
            baseline_model=self.base_model,
            metric="substring",
            max_new_tokens=2,
        )
        res_corr = gate_evaluation(corr_rep, gate_keep_drop)
        self.assertFalse(res_corr.passed)
        self.assertFalse(res_corr.keep_passed)
        self.assertTrue(res_corr.drop_passed)
        self.assertTrue(any("KEEP retention score" in f for f in res_corr.failures))


class TestRejectionAcceptance(unittest.TestCase):
    """Acceptance Scope 3: Rejection acceptance.

    Verifies empty prompts, mismatched vocabularies, non-finite logits, and stale-model
    plan checksum mismatches are rejected before mutation.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = MockTokenizer()
        self.model = AcceptanceCausalLM(vocab_size=7, hidden_dim=8)
        self.config = {"vocab_size": 7, "hidden_size": 8, "num_hidden_layers": 2}

    def test_empty_prompts_rejected_before_mutation(self):
        """Empty prompt lists or blank strings are rejected without mutating model weights."""
        orig_weights = {k: v.clone() for k, v in self.model.state_dict().items()}

        with self.assertRaisesRegex(ValueError, "Cannot generate completions for an empty prompt set"):
            generate_sequence_completions(self.model, self.tokenizer, [])

        with self.assertRaisesRegex(ValueError, "must be a non-empty string"):
            generate_sequence_completions(self.model, self.tokenizer, ["   "])

        with self.assertRaisesRegex(ValueError, "empty sample set"):
            evaluate_sliced_sequence_task([], [])

        with self.assertRaisesRegex(ValueError, "non-empty string"):
            Sample(prompt="")

        # Assert zero mutation
        for k, orig_t in orig_weights.items():
            self.assertTrue(torch.equal(orig_t, self.model.state_dict()[k]))

    def test_mismatched_vocabularies_rejected_before_mutation(self):
        """Mismatched tokenizer vocabulary bound to plan is rejected before applying edits."""
        orig_weights = {k: v.clone() for k, v in self.model.state_dict().items()}

        manifest = ProvenanceManifest(
            model_hash=compute_model_hash(self.model),
            tokenizer_hash=compute_tokenizer_hash(self.tokenizer),
            config_hash=compute_config_hash(self.config),
            dataset_fingerprints={"k": {"sha256": "abc"}},
            adapter_version="0.4.0",
            seed=42,
            dtype="float32",
            operation_order=["op1"],
        )
        plan = bind_provenance_to_plan({"op": "cut"}, manifest)

        # Tokenizer with different vocabulary
        tok_mismatch = MockTokenizer(vocab={"pad": 0, "eos": 1, "different": 2, "vocab": 3})
        with self.assertRaisesRegex(StaleModelPlanError, "tokenizer_hash"):
            validate_plan_against_model(plan, self.model, tokenizer=tok_mismatch, config=self.config)

        # Assert model remains unchanged
        for k, orig_t in orig_weights.items():
            self.assertTrue(torch.equal(orig_t, self.model.state_dict()[k]))

    def test_non_finite_logits_rejected_before_mutation(self):
        """Non-finite (NaN/Inf) logits are caught and rejected during generation and gating."""
        class NaNCausalLM(nn.Module):
            device = torch.device("cpu")

            def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
                b, s = input_ids.shape
                logits = torch.full((b, s, 7), float("nan"))
                return SimpleNamespace(logits=logits)

        class InfCausalLM(nn.Module):
            device = torch.device("cpu")

            def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> SimpleNamespace:
                b, s = input_ids.shape
                logits = torch.full((b, s, 7), float("inf"))
                return SimpleNamespace(logits=logits)

        with self.assertRaisesRegex(ValueError, "non-finite logits"):
            generate_sequence_completions(NaNCausalLM(), self.tokenizer, ["valid prompt"])

        with self.assertRaisesRegex(ValueError, "non-finite logits"):
            generate_sequence_completions(InfCausalLM(), self.tokenizer, ["valid prompt"])

        # Non-finite scores in report rejected during gating
        nan_report = EvaluationReport(
            keep_report=SlicedMetricsReport(
                kind="keep",
                sample_count=1,
                overall_score=float("nan"),
                worst_slice_score=float("nan"),
                worst_slice_damage=0.0,
                worst_slice_domain="d",
                domain_metrics={},
            )
        )
        with self.assertRaisesRegex(ValueError, "Non-finite scores"):
            gate_evaluation(nan_report, GateThresholds(min_keep_retention=0.8))

        with self.assertRaisesRegex(ValueError, "Non-finite"):
            publish_evaluation_report(nan_report)

    def test_stale_model_plan_checksum_mismatches_rejected_before_mutation(self):
        """Stale model weights or altered config hash fail validation before any mutation."""
        manifest = ProvenanceManifest(
            model_hash=compute_model_hash(self.model),
            tokenizer_hash=compute_tokenizer_hash(self.tokenizer),
            config_hash=compute_config_hash(self.config),
            dataset_fingerprints={"k": {"sha256": "abc"}},
            adapter_version="0.4.0",
            seed=42,
            dtype="float32",
            operation_order=["op1"],
        )
        plan = bind_provenance_to_plan({"op": "cut"}, manifest)

        # 1. Stale model with modified weight
        stale_model = AcceptanceCausalLM(vocab_size=7, hidden_dim=8)
        with torch.no_grad():
            stale_model.lm_head.weight[0, 0] += 10.0
        with self.assertRaisesRegex(StaleModelPlanError, "model_hash"):
            validate_plan_against_model(plan, stale_model, tokenizer=self.tokenizer, config=self.config)

        # 2. Stale config
        altered_config = {"vocab_size": 7, "hidden_size": 16, "num_hidden_layers": 2}
        with self.assertRaisesRegex(StaleModelPlanError, "config_hash"):
            validate_plan_against_model(plan, self.model, tokenizer=self.tokenizer, config=altered_config)

        # 3. Tampered provenance checksum in plan
        tampered_plan = copy.deepcopy(plan)
        tampered_plan["provenance_hash"] = "invalid_hash_value"
        with self.assertRaisesRegex(ValueError, "provenance_hash mismatch"):
            validate_plan_provenance(tampered_plan)

        # 4. Plan missing provenance
        with self.assertRaisesRegex(ValueError, "does not contain a 'provenance'"):
            validate_plan_provenance({"version": 3})


class TestStageCompositionAcceptance(unittest.TestCase):
    """Acceptance Scope 4: Stage composition acceptance.

    Executes full multi-stage sequence:
    structural edit -> LoRA recovery -> quantization export -> reload parity -> exact restoration
    integrity verified against parent manifest SHA256 hashes.
    """

    def test_full_stage_composition_workflow(self):
        torch.manual_seed(42)
        tokenizer = MockTokenizer()

        # Step 0: Baseline source checkpoint & parent manifest recording
        source_model = AcceptanceCausalLM(vocab_size=7, hidden_dim=8, num_layers=2)
        source_sd = {k: v.clone() for k, v in source_model.state_dict().items()}
        edited_target_name = "layers.0.mlp.down_proj.weight"
        edited_keys = [edited_target_name]

        # Backup edited tensors before any surgery
        backup = backup_edited_tensors(source_sd, edited_keys)

        # Step 1: Structural surgery (persistent masking on layer 0 down_proj)
        op_mask = UnifiedOperation.create_mask(
            target_path=edited_target_name,
            component_type=ComponentType.WEIGHT_TENSOR,
            target_indices=[0, 1],
            mask_value=0.0,
            execution_semantics=ExecutionSemantics.PERSISTENT,
            export_supported=True,
        )
        cut_res = apply_plan(source_model, [op_mask])
        self.assertTrue(cut_res.success)
        self.assertIn(edited_target_name, cut_res.modified_parameters)

        # Verify only the selected tensor was modified
        for k, v in source_model.state_dict().items():
            if k == edited_target_name:
                self.assertFalse(torch.equal(v, source_sd[k]))
            else:
                self.assertTrue(torch.equal(v, source_sd[k]))

        # Step 2: Stage 5 targeted LoRA recovery on layer 0 down_proj
        injected = inject_lora(source_model, target_modules=["layers.0.mlp.down_proj"], r=2)
        self.assertEqual(len(injected), 1)
        apply_freeze_mask(source_model, allow_lora_only=True)
        trainable_rep = verify_trainable_parameters(source_model)
        self.assertGreater(trainable_rep.trainable_params, 0)

        # Recovery fine-tuning on KEEP data
        keep_batch = {"input_ids": torch.randint(0, 7, (2, 4))}
        rec_cfg = RecoveryConfig(max_steps=3, lr=1e-2, seed=42)
        rec_res = run_recovery_training(model=source_model, keep_data=[keep_batch], config=rec_cfg)
        self.assertTrue(rec_res.success)

        # Merge LoRA back into base layer
        merged_model = merge_lora(source_model, in_place=True)
        self.assertIsInstance(merged_model.layers[0].mlp.down_proj, nn.Linear)

        # Candidate state dict after surgery + recovery
        candidate_sd = {k: v.clone() for k, v in merged_model.state_dict().items()}

        # Create restoration manifest binding parent checkpoint to candidate
        manifest = create_restoration_manifest(
            source_checkpoint="./checkpoints/base_model",
            candidate_checkpoint="./checkpoints/recovered_candidate",
            source_state_dict=source_sd,
            candidate_state_dict=candidate_sd,
            edited_tensors=edited_keys,
        )

        # Step 3: Stage 6 INT8 quantization and export
        calib_prompts = ["a b", "b c", "a c"]
        calib_binding = bind_calibration_dataset(calib_prompts, dataset_name="stage_comp_calib")
        qcfg = QuantizationConfig(bits=8, symmetric=True, pack=False)
        quantized_model = quantize_model(merged_model, qcfg)

        with tempfile.TemporaryDirectory() as tmp_dir:
            exp_dir = Path(tmp_dir) / "quantized_composition_export"
            export_quantized_checkpoint(
                model=quantized_model,
                export_dir=exp_dir,
                config_or_plan=qcfg,
                calibration_binding=calib_binding,
            )

            # Step 4: Reload parity verification
            fresh_base = AcceptanceCausalLM(vocab_size=7, hidden_dim=8, num_layers=2)
            reloaded_model, meta = load_quantized_checkpoint(export_dir=exp_dir, base_model=fresh_base)
            parity_ok = verify_export_reload_parity(
                original_model=quantized_model,
                reloaded_model=reloaded_model,
                test_inputs=calib_prompts,
                tokenizer=tokenizer,
            )
            self.assertTrue(parity_ok)

        # Step 5: Exact restoration integrity verified against parent manifest SHA256 hashes
        restored_sd = restore_state_dict(candidate_sd, backup, manifest)
        ver_res = verify_restoration_integrity(restored_sd, manifest)
        self.assertEqual(ver_res["status"], "verified")
        self.assertEqual(ver_res["tensors_verified"], len(source_sd))

        # Bit-exact tensor equality and SHA256 hash match against parent manifest
        for k, orig_tensor in source_sd.items():
            self.assertTrue(torch.equal(orig_tensor, restored_sd[k]))
            self.assertEqual(compute_tensor_hash(restored_sd[k]), manifest.source_hashes[k])

        # Tampering with restored tensor raises RestorationIntegrityError
        tampered_sd = dict(restored_sd)
        tampered_sd[edited_target_name] = torch.randn_like(tampered_sd[edited_target_name])
        with self.assertRaises(RestorationIntegrityError):
            verify_restoration_integrity(tampered_sd, manifest)


class TestMultimodalAmputationAcceptance(unittest.TestCase):
    """Acceptance Scope 5: Multimodal amputation acceptance.

    Verifies vision branch is completely removed from multimodal model, text-only generation
    passes without regression, and input rejection guards intercept image inputs.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.model = MockMultimodalModel(vocab_size=32, hidden_dim=16)
        self.processor = MockMultimodalProcessor()
        self.retained_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        self.test_inputs = [
            {"input_ids": torch.tensor([[1, 2, 3]])},
            {"input_ids": torch.tensor([[4, 5, 6, 7]])},
        ]

    def test_vision_amputation_pipeline_text_generation_and_input_rejection(self):
        """Vision branch physically removed, text generation verified, image inputs rejected."""
        baseline_model = copy.deepcopy(self.model)

        # Baseline text generation
        prompt = torch.tensor([[1, 2, 3]])
        baseline_gen = baseline_model.generate(prompt, max_new_tokens=4)

        # Execute complete multimodal amputation pipeline
        amputated_model, rem_report, val_report = amputate_modality(
            model=self.model,
            modality="vision",
            retained_inputs=self.retained_inputs,
            test_inputs=self.test_inputs,
            config=self.model.config,
            processor=self.processor,
            require_proof=True,
            install_guard=True,
        )

        # 1. Vision branches physically removed
        self.assertFalse(hasattr(amputated_model, "vision_tower"))
        self.assertFalse(hasattr(amputated_model, "multi_modal_projector"))
        removed_names = [r.branch_name for r in rem_report.removed_branches]
        self.assertIn("vision_tower", removed_names)
        self.assertIn("multi_modal_projector", removed_names)
        self.assertGreater(rem_report.freed_bytes, 0)
        self.assertGreater(rem_report.freed_params, 0)

        # Checkpoint state dict contains zero vision/projector parameters
        clean_state = verify_state_dict_absence(
            amputated_model,
            ["vision_tower", "multi_modal_projector"],
        )
        self.assertTrue(clean_state)

        # 2. Text-only generation verified with zero logit regression
        self.assertTrue(val_report.passed)
        self.assertTrue(val_report.generation_matches)
        self.assertEqual(val_report.max_logit_diff, 0.0)

        # Amputated generate produces exact output matching baseline
        amputated_gen = amputated_model.generate(prompt, max_new_tokens=4)
        self.assertTrue(torch.equal(baseline_gen, amputated_gen))

        # Save and reload amputated model round-trip
        with tempfile.TemporaryDirectory() as tmp_dir:
            loaded_sd, weights_p, parity_diff = save_and_reload_amputated_model(
                model=amputated_model,
                save_dir=tmp_dir,
                config=amputated_model.config,
                test_inputs=self.test_inputs,
            )
            self.assertTrue(weights_p.exists())
            self.assertEqual(parity_diff, 0.0)
            self.assertFalse(any("vision" in k or "projector" in k for k in loaded_sd))

        # 3. Input rejection on image inputs
        pixel_vals = torch.randn(1, 3, 14, 14)

        # Keyword argument pixel_values rejected
        with self.assertRaises(ModalityRemovedError) as cm_pv:
            amputated_model(input_ids=prompt, pixel_values=pixel_vals)
        self.assertEqual(cm_pv.exception.modality, "vision")
        self.assertIn("pixel_values", cm_pv.exception.offending_inputs)

        # Keyword argument images rejected
        with self.assertRaises(ModalityRemovedError) as cm_img:
            amputated_model(input_ids=prompt, images=pixel_vals)
        self.assertEqual(cm_img.exception.modality, "vision")
        self.assertIn("images", cm_img.exception.offending_inputs)

        # Positional 4D tensor rejected
        with self.assertRaises(ModalityRemovedError):
            amputated_model(prompt, pixel_vals)

        # Guarded generate call rejects pixel_values
        with self.assertRaises(ModalityRemovedError):
            amputated_model.generate(input_ids=prompt, pixel_values=pixel_vals)

        # Repaired processor rejects image inputs
        with self.assertRaises(ModalityRemovedError) as cm_proc:
            self.processor(images=["test_image_data"])
        self.assertEqual(cm_proc.exception.modality, "vision")
        self.assertIn("images", cm_proc.exception.offending_inputs)


if __name__ == "__main__":
    unittest.main()

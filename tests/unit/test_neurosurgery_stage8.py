"""Unit tests for Stage 8: Modality and branch removal.

Covers:
1. Multimodal inventory and explicit dependency mapping (vision, audio, cross-attention).
2. Dead branch proof (forward hook tracing, poisoned ablation probe, strict validation).
3. Safe physical removal of multimodal branches and state dict absence verification.
4. Strict shared component protection (shared trunks, embeddings, misleading names, tied weights).
5. Config & processor repair (attribute stripping, architecture update, processor guard).
6. Input rejection guards (kwargs, positional 4D tensors, generate() interception, ModalityRemovedError).
7. Retained text parity, cached generation parity, checkpoint save/reload round-trip, and end-to-end pipeline.
"""

from __future__ import annotations

import copy
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from aegis_lab.editing.neurosurgery.stage8_modality import (
    BranchDependencyNode,
    BranchNotFoundError,
    BranchRemovalResult,
    ComponentRole,
    ConfigRepairError,
    DeadBranchProofReport,
    DeadBranchVerificationError,
    ModalityDependencyMap,
    ModalityRemovalReport,
    ModalityRemovedError,
    ModalitySurgeryError,
    ModalityType,
    RetainedValidationReport,
    SharedComponentProtectionError,
    amputate_modality,
    build_dependency_map,
    delete_module_at_path,
    extract_logits,
    inspect_multimodal_components,
    inspect_parameter_pointers,
    install_input_rejection_guard,
    prove_branches_dead,
    remove_modality,
    remove_modality_branch,
    repair_model_config,
    repair_multimodal_processor,
    save_and_reload_amputated_model,
    validate_retained_modality,
    verify_dead_branch,
    verify_state_dict_absence,
)


# ============================================================================
# Mock Models & Fixtures
# ============================================================================

class TinyVisionTower(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv(pixel_values))
        return self.proj(h.flatten(2).transpose(1, 2))


class TinyAudioTower(nn.Module):
    def __init__(self, in_features: int = 8, hidden_dim: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(audio_values)))


class TinyProjector(nn.Module):
    def __init__(self, in_features: int = 16, out_features: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class TinyCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.self_attn = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, cross_features: torch.Tensor | None = None) -> torch.Tensor:
        h = h + self.self_attn(h)
        if cross_features is not None:
            h = h + self.cross_attn(cross_features)
        return self.norm(h)


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16, num_layers: int = 2):
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
        **kwargs,
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
            # Simple dummy past_key_values for testing KV-cache handling
            pkv = (out.detach(),)
            return SimpleNamespace(last_hidden_state=out, past_key_values=pkv)
        return out


class MockLlavaModel(nn.Module):
    """Vision-language model adhering to LLaVA style topology."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16):
        super().__init__()
        self.vision_tower = TinyVisionTower(in_channels=3, hidden_dim=hidden_dim)
        self.multi_modal_projector = TinyProjector(in_features=hidden_dim, out_features=hidden_dim)
        self.language_model = TinyLanguageModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.config = SimpleNamespace(
            model_type="llava",
            architectures=["LlavaForConditionalGeneration"],
            vision_config={"hidden_size": hidden_dim},
            mm_projector_type="mlp2x_gelu",
            text_config={"vocab_size": vocab_size, "hidden_size": hidden_dim},
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        **kwargs,
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

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 3, **kwargs) -> torch.Tensor:
        curr = input_ids.clone()
        for _ in range(max_new_tokens):
            out = self.forward(input_ids=curr, **kwargs)
            logits = extract_logits(out)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr = torch.cat([curr, next_tok], dim=-1)
        return curr


class MockAudioLanguageModel(nn.Module):
    """Audio-language model topology."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16):
        super().__init__()
        self.audio_tower = TinyAudioTower(in_features=8, hidden_dim=hidden_dim)
        self.audio_projector = TinyProjector(in_features=hidden_dim, out_features=hidden_dim)
        self.language_model = TinyLanguageModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.config = SimpleNamespace(
            model_type="speech_llm",
            architectures=["SpeechForConditionalGeneration"],
            audio_config={"hidden_size": hidden_dim},
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        audio_values: torch.Tensor | None = None,
        **kwargs,
    ) -> Any:
        if audio_values is not None:
            a_feats = self.audio_tower(audio_values)
            proj = self.audio_projector(a_feats)
            t_embeds = self.language_model.embed_tokens(input_ids) if input_ids is not None else None
            h = torch.cat([proj, t_embeds], dim=1) if t_embeds is not None else proj
            lm_out = self.language_model(inputs_embeds=h)
        else:
            lm_out = self.language_model(input_ids=input_ids)
        return self.lm_head(lm_out)


class MockCrossAttentionModel(nn.Module):
    """Model utilizing cross-attention for multimodal fusion."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16):
        super().__init__()
        self.vision_tower = TinyVisionTower(in_channels=3, hidden_dim=hidden_dim)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.cross_attn = TinyCrossAttentionBlock(hidden_dim=hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **kwargs,
    ) -> Any:
        h = self.embed_tokens(input_ids)
        v_feats = self.vision_tower(pixel_values) if pixel_values is not None else None
        h = self.cross_attn(h, cross_features=v_feats)
        return self.lm_head(h)


class MockBrokenBranchModel(nn.Module):
    """Model that erroneously calls vision tower during text-only forward pass."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16):
        super().__init__()
        self.vision_tower = nn.Linear(hidden_dim, hidden_dim)
        self.language_model = TinyLanguageModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor | None = None, pixel_values: torch.Tensor | None = None, **kwargs):
        # Accidental leak: vision_tower is called even when pixel_values is None
        dummy = self.vision_tower(torch.zeros(1, 1, 16))
        lm_out = self.language_model(input_ids=input_ids) + 0.0 * dummy.sum()
        return self.lm_head(lm_out)


class MockTiedMultimodalModel(nn.Module):
    """Model where candidate branch parameters share memory with retained trunk."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16):
        super().__init__()
        self.language_model = TinyLanguageModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.vision_tower = nn.Linear(hidden_dim, hidden_dim)
        # Weight tying: vision tower weight points to language model layer 0
        self.vision_tower.weight = self.language_model.layers[0].weight

    def forward(self, input_ids: torch.Tensor | None = None, **kwargs):
        return self.language_model(input_ids=input_ids)


class MockMisleadingNameModel(nn.Module):
    """Model where a shared component has 'vision' in its attribute name."""

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 16):
        super().__init__()
        self.vision_text_shared_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.vision_tower = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor | None = None, pixel_values: torch.Tensor | None = None, **kwargs):
        h = self.vision_text_shared_embedding(input_ids)
        return self.lm_head(h)


class MockProcessor:
    """Mock multimodal processor with tokenizer and image processor."""

    def __init__(self):
        self.tokenizer = lambda text, **kw: {"input_ids": torch.tensor([[1, 2, 3]])}
        self.image_processor = lambda images, **kw: {"pixel_values": torch.randn(1, 3, 14, 14)}

    def __call__(self, text: Any = None, images: Any = None, **kwargs) -> dict:
        res = {}
        if text is not None:
            res.update(self.tokenizer(text, **kwargs))
        if images is not None:
            res.update(self.image_processor(images, **kwargs))
        return res


# ============================================================================
# Unit Tests
# ============================================================================

class TestMultimodalDependencyMapping(unittest.TestCase):
    """Test multimodal inventory and explicit dependency map generation."""

    def test_llava_dependency_mapping(self):
        model = MockLlavaModel()
        dep_map = build_dependency_map(model, retained_modality="text")

        self.assertEqual(dep_map.model_class, "MockLlavaModel")
        self.assertEqual(dep_map.retained_modality, "text")

        # Check vision encoder node
        vt_node = dep_map.get_branch("vision_tower")
        self.assertIsNotNone(vt_node)
        self.assertEqual(vt_node.role, ComponentRole.ENCODER)
        self.assertEqual(vt_node.modality, ModalityType.VISION)
        self.assertTrue(vt_node.is_removable)
        self.assertFalse(vt_node.is_shared)
        self.assertIn("multi_modal_projector", vt_node.consumers)

        # Check vision projector node
        proj_node = dep_map.get_branch("multi_modal_projector")
        self.assertIsNotNone(proj_node)
        self.assertEqual(proj_node.role, ComponentRole.PROJECTOR)
        self.assertEqual(proj_node.modality, ModalityType.VISION)
        self.assertTrue(proj_node.is_removable)
        self.assertIn("vision_tower", proj_node.dependencies)
        self.assertIn("language_model", proj_node.consumers)

        # Check shared components
        self.assertTrue(dep_map.is_protected("language_model"))
        self.assertTrue(dep_map.is_protected("lm_head"))
        self.assertFalse(dep_map.is_protected("vision_tower"))
        self.assertFalse(dep_map.is_protected("multi_modal_projector"))

        # Check removable branches query
        removable_vision = dep_map.get_removable_branches("vision")
        self.assertIn("vision_tower", removable_vision)
        self.assertIn("multi_modal_projector", removable_vision)

    def test_audio_dependency_mapping(self):
        model = MockAudioLanguageModel()
        dep_map = build_dependency_map(model, retained_modality="text")

        at_node = dep_map.get_branch("audio_tower")
        self.assertIsNotNone(at_node)
        self.assertEqual(at_node.role, ComponentRole.ENCODER)
        self.assertEqual(at_node.modality, ModalityType.AUDIO)
        self.assertTrue(at_node.is_removable)

        ap_node = dep_map.get_branch("audio_projector")
        self.assertIsNotNone(ap_node)
        self.assertEqual(ap_node.role, ComponentRole.PROJECTOR)
        self.assertEqual(ap_node.modality, ModalityType.AUDIO)
        self.assertTrue(ap_node.is_removable)

        removable_audio = dep_map.get_removable_branches("audio")
        self.assertIn("audio_tower", removable_audio)
        self.assertIn("audio_projector", removable_audio)

    def test_cross_attention_dependency_mapping(self):
        model = MockCrossAttentionModel()
        dep_map = build_dependency_map(model)

        ca_node = dep_map.get_branch("cross_attn")
        self.assertIsNotNone(ca_node)
        self.assertEqual(ca_node.role, ComponentRole.CROSS_ATTENTION)
        self.assertEqual(ca_node.modality, ModalityType.CROSSMODAL)

    def test_dependency_map_summary_and_serialization(self):
        model = MockLlavaModel()
        dep_map = build_dependency_map(model)

        data = dep_map.to_dict()
        self.assertIn("model_class", data)
        self.assertIn("nodes", data)
        self.assertIn("vision_tower", data["nodes"])

        summary = dep_map.summary()
        self.assertIn("vision_tower", summary)
        self.assertIn("REMOVABLE", summary)
        self.assertIn("PROTECTED", summary)

    def test_custom_overrides_in_dependency_map(self):
        model = MockLlavaModel()
        overrides = {
            "vision_tower": {"is_removable": False, "is_shared": True},
        }
        dep_map = build_dependency_map(model, custom_overrides=overrides)
        vt_node = dep_map.get_branch("vision_tower")
        self.assertFalse(vt_node.is_removable)
        self.assertTrue(vt_node.is_shared)
        self.assertTrue(dep_map.is_protected("vision_tower"))


class TestDeadBranchVerification(unittest.TestCase):
    """Test dead branch proof via hook call tracing and poisoned ablation probing."""

    def setUp(self):
        self.model = MockLlavaModel()
        self.retained_inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    def test_dead_branch_verified_on_unused_vision(self):
        # Vision components are completely unused on text-only inputs
        rep_vt = verify_dead_branch(self.model, "vision_tower", self.retained_inputs)
        self.assertTrue(rep_vt.is_dead)
        self.assertTrue(rep_vt.proof_passed)
        self.assertEqual(rep_vt.hook_call_count, 0)
        self.assertLessEqual(rep_vt.max_logit_diff, 1e-5)

        rep_proj = verify_dead_branch(self.model, "multi_modal_projector", self.retained_inputs)
        self.assertTrue(rep_proj.is_dead)
        self.assertTrue(rep_proj.proof_passed)
        self.assertEqual(rep_proj.hook_call_count, 0)
        self.assertLessEqual(rep_proj.max_logit_diff, 1e-5)

    def test_dead_branch_fails_when_branch_is_active(self):
        broken_model = MockBrokenBranchModel()
        rep = verify_dead_branch(broken_model, "vision_tower", self.retained_inputs)
        self.assertFalse(rep.is_dead)
        self.assertFalse(rep.proof_passed)
        self.assertGreater(rep.hook_call_count, 0)

    def test_dead_branch_fails_on_shared_trunk(self):
        dep_map = build_dependency_map(self.model)
        rep = verify_dead_branch(self.model, "language_model", self.retained_inputs, dep_map=dep_map)
        self.assertFalse(rep.is_dead)
        self.assertFalse(rep.proof_passed)
        self.assertIn("protected", rep.details.get("error", "").lower())

    def test_prove_branches_dead_strict_success(self):
        reports = prove_branches_dead(
            self.model,
            ["vision_tower", "multi_modal_projector"],
            self.retained_inputs,
            strict=True,
        )
        self.assertEqual(len(reports), 2)
        self.assertTrue(reports["vision_tower"].proof_passed)
        self.assertTrue(reports["multi_modal_projector"].proof_passed)

    def test_prove_branches_dead_strict_failure(self):
        broken_model = MockBrokenBranchModel()
        with self.assertRaises(DeadBranchVerificationError):
            prove_branches_dead(broken_model, ["vision_tower"], self.retained_inputs, strict=True)

    def test_verify_dead_branch_missing_branch(self):
        with self.assertRaises(BranchNotFoundError):
            verify_dead_branch(self.model, "nonexistent_branch", self.retained_inputs)


class TestPhysicalBranchRemoval(unittest.TestCase):
    """Test physical deletion of modules from hierarchy and state dict."""

    def test_safe_removal_vision_tower_and_projector(self):
        model = MockLlavaModel()
        init_params = sum(p.numel() for p in model.parameters())

        res_vt = remove_modality_branch(model, "vision_tower")
        self.assertTrue(res_vt.success)
        self.assertGreater(res_vt.removed_params, 0)
        self.assertGreater(res_vt.removed_bytes, 0)

        # Confirm deleted from module hierarchy
        self.assertFalse(hasattr(model, "vision_tower"))
        self.assertNotIn("vision_tower", model._modules)

        # Confirm state dict absence
        clean, offending = verify_state_dict_absence(model, ["vision_tower"])
        self.assertTrue(clean)
        self.assertEqual(len(offending), 0)

        # Remove projector
        res_proj = remove_modality_branch(model, "multi_modal_projector")
        self.assertTrue(res_proj.success)
        self.assertFalse(hasattr(model, "multi_modal_projector"))

        # Verify state dict absence for both
        clean_both, offending_both = verify_state_dict_absence(model, ["vision_tower", "multi_modal_projector"])
        self.assertTrue(clean_both)
        self.assertEqual(len(offending_both), 0)

        remaining_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(remaining_params, init_params - res_vt.removed_params - res_proj.removed_params)

    def test_remove_modality_orchestrator(self):
        model = MockLlavaModel()
        retained_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        report = remove_modality(
            model,
            modality="vision",
            require_proof=True,
            retained_inputs=retained_inputs,
            config=model.config,
            install_guard=True,
        )

        self.assertEqual(report.modality, "vision")
        self.assertEqual(len(report.removed_branches), 2)
        self.assertGreater(report.freed_params, 0)
        self.assertTrue(report.config_repaired)
        self.assertTrue(report.guard_installed)

        # State dict check
        clean, offending = verify_state_dict_absence(model, ["vision_tower", "multi_modal_projector"])
        self.assertTrue(clean)
        self.assertEqual(len(offending), 0)

    def test_remove_audio_modality(self):
        model = MockAudioLanguageModel()
        retained_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        report = remove_modality(
            model,
            modality="audio",
            require_proof=True,
            retained_inputs=retained_inputs,
            config=model.config,
        )
        self.assertEqual(report.modality, "audio")
        clean, offending = verify_state_dict_absence(model, ["audio_tower", "audio_projector"])
        self.assertTrue(clean)
        self.assertEqual(len(offending), 0)

    def test_remove_branch_not_found(self):
        model = MockLlavaModel()
        with self.assertRaises(BranchNotFoundError):
            remove_modality_branch(model, "nonexistent.submodule")


class TestSharedComponentProtection(unittest.TestCase):
    """Test strict protection of shared trunks, embeddings, and tied parameters."""

    def test_protect_shared_trunk_explicit(self):
        model = MockLlavaModel()
        dep_map = build_dependency_map(model)
        with self.assertRaises(SharedComponentProtectionError):
            remove_modality_branch(model, "language_model", dep_map=dep_map)

    def test_protect_shared_embedding(self):
        model = MockLlavaModel()
        with self.assertRaises(SharedComponentProtectionError):
            remove_modality_branch(model, "language_model.embed_tokens")

    def test_protect_shared_component_with_modality_name(self):
        model = MockMisleadingNameModel()
        dep_map = build_dependency_map(model)
        # Even though attribute name contains 'vision', it is a shared embedding
        self.assertTrue(dep_map.is_protected("vision_text_shared_embedding"))
        with self.assertRaises(SharedComponentProtectionError):
            remove_modality_branch(model, "vision_text_shared_embedding", dep_map=dep_map)

    def test_protect_tied_weights(self):
        model = MockTiedMultimodalModel()
        dep_map = build_dependency_map(model)
        # vision_tower shares weight pointer with language_model.layers[0].weight
        vt_node = dep_map.get_branch("vision_tower")
        self.assertTrue(vt_node.is_shared)
        self.assertFalse(vt_node.is_removable)

        # Removal must be blocked even without passing dep_map due to pointer check
        with self.assertRaises(SharedComponentProtectionError):
            remove_modality_branch(model, "vision_tower", dep_map=None)


class TestConfigAndProcessorRepair(unittest.TestCase):
    """Test model config and processor repair after modality removal."""

    def test_repair_model_config_object(self):
        config = SimpleNamespace(
            model_type="llava",
            architectures=["LlavaForConditionalGeneration"],
            vision_config={"hidden_size": 16},
            mm_projector_type="mlp2x_gelu",
            vision_feature_layer=-2,
            is_multimodal=True,
        )
        repaired = repair_model_config(config, removed_modality="vision")
        self.assertFalse(hasattr(repaired, "vision_config"))
        self.assertFalse(hasattr(repaired, "mm_projector_type"))
        self.assertFalse(hasattr(repaired, "vision_feature_layer"))
        self.assertEqual(repaired.model_type, "llama")
        self.assertEqual(repaired.architectures, ["LlamaForCausalLM"])
        self.assertFalse(repaired.is_multimodal)
        self.assertIn("vision", repaired.removed_modalities)

    def test_repair_model_config_dict(self):
        cfg_dict = {
            "model_type": "llava",
            "architectures": ["LlavaForConditionalGeneration"],
            "vision_config": {"hidden_size": 16},
            "mm_projector_type": "mlp2x_gelu",
        }
        repaired = repair_model_config(cfg_dict, removed_modality="vision")
        self.assertNotIn("vision_config", repaired)
        self.assertNotIn("mm_projector_type", repaired)
        self.assertEqual(repaired["model_type"], "llama")
        self.assertEqual(repaired["architectures"], ["LlamaForCausalLM"])
        self.assertFalse(repaired["is_multimodal"])

    def test_repair_model_config_audio(self):
        cfg_dict = {
            "model_type": "speech_llm",
            "audio_config": {"hidden_size": 16},
            "speech_config": {},
        }
        repaired = repair_model_config(cfg_dict, removed_modality="audio")
        self.assertNotIn("audio_config", repaired)
        self.assertNotIn("speech_config", repaired)
        self.assertIn("audio", repaired["removed_modalities"])

    def test_repair_multimodal_processor_text_pass(self):
        proc = MockProcessor()
        repair_multimodal_processor(proc, "vision")
        self.assertIsNone(proc.image_processor)

        # Text prompt passes cleanly
        res = proc(text="Test prompt")
        self.assertIn("input_ids", res)

    def test_repair_multimodal_processor_rejects_images(self):
        proc = MockProcessor()
        repair_multimodal_processor(proc, "vision")
        with self.assertRaises(ModalityRemovedError) as cm:
            proc(images=["fake_image_data"])
        self.assertEqual(cm.exception.modality, "vision")
        self.assertIn("images", cm.exception.offending_inputs)

    def test_repair_multimodal_processor_rejects_positional_images(self):
        proc = MockProcessor()
        repair_multimodal_processor(proc, "vision")
        with self.assertRaises(ModalityRemovedError):
            proc("prompt", ["fake_image_data"])


class TestInputRejectionGuard(unittest.TestCase):
    """Test runtime forward guard rejecting removed modality inputs."""

    def setUp(self):
        self.model = MockLlavaModel()
        install_input_rejection_guard(self.model, ["vision"])

    def test_reject_pixel_values_kwargs(self):
        with self.assertRaises(ModalityRemovedError) as cm:
            self.model(input_ids=torch.tensor([[1, 2]]), pixel_values=torch.randn(1, 3, 14, 14))
        self.assertEqual(cm.exception.modality, "vision")
        self.assertIn("pixel_values", cm.exception.offending_inputs)

    def test_reject_images_kwargs(self):
        with self.assertRaises(ModalityRemovedError) as cm:
            self.model(input_ids=torch.tensor([[1, 2]]), images=torch.randn(1, 3, 14, 14))
        self.assertEqual(cm.exception.modality, "vision")
        self.assertIn("images", cm.exception.offending_inputs)

    def test_reject_positional_4d_tensor(self):
        with self.assertRaises(ModalityRemovedError):
            self.model(torch.tensor([[1, 2]]), torch.randn(1, 3, 14, 14))

    def test_accept_valid_text_inputs(self):
        out = self.model(input_ids=torch.tensor([[1, 2, 3]]))
        logits = extract_logits(out)
        self.assertTrue(torch.isfinite(logits).all())

    def test_accept_none_pixel_values(self):
        # Explicit None should pass without raising
        out = self.model(input_ids=torch.tensor([[1, 2]]), pixel_values=None)
        logits = extract_logits(out)
        self.assertTrue(torch.isfinite(logits).all())

    def test_guarded_generate_rejects_removed_modality(self):
        with self.assertRaises(ModalityRemovedError):
            self.model.generate(input_ids=torch.tensor([[1]]), pixel_values=torch.randn(1, 3, 14, 14))


class TestRetainedModalityValidationAndParity(unittest.TestCase):
    """Test retained modality verification, KV caching parity, and save/reload round-trip."""

    def test_retained_text_logits_parity(self):
        model = MockLlavaModel()
        baseline = copy.deepcopy(model)

        test_inputs = [
            {"input_ids": torch.tensor([[1, 2, 3]])},
            {"input_ids": torch.tensor([[4, 5, 6, 7]])},
        ]

        # Amputate vision
        remove_modality_branch(model, "vision_tower")
        remove_modality_branch(model, "multi_modal_projector")

        val_report = validate_retained_modality(
            model=model,
            baseline_model=baseline,
            test_inputs=test_inputs,
            removed_branches=["vision_tower", "multi_modal_projector"],
            check_generation=True,
        )

        self.assertTrue(val_report.passed)
        self.assertTrue(val_report.state_dict_clean)
        self.assertTrue(val_report.generation_matches)
        self.assertEqual(val_report.max_logit_diff, 0.0)

    def test_retained_cached_generation_parity(self):
        model = MockLlavaModel()
        baseline = copy.deepcopy(model)

        # Baseline generation
        prompt = torch.tensor([[1, 2, 3]])
        gen_base = baseline.generate(prompt, max_new_tokens=4)

        # Amputate
        remove_modality_branch(model, "vision_tower")
        remove_modality_branch(model, "multi_modal_projector")

        gen_amputated = model.generate(prompt, max_new_tokens=4)
        self.assertTrue(torch.equal(gen_base, gen_amputated))

    def test_save_and_reload_amputated_model(self):
        model = MockLlavaModel()
        remove_modality_branch(model, "vision_tower")
        remove_modality_branch(model, "multi_modal_projector")
        test_inputs = [{"input_ids": torch.tensor([[1, 2, 3]])}]

        with tempfile.TemporaryDirectory() as tmpdir:
            loaded_sd, weights_path, parity_diff = save_and_reload_amputated_model(
                model=model,
                save_dir=tmpdir,
                config=model.config,
                test_inputs=test_inputs,
            )

            self.assertTrue(weights_path.exists())
            self.assertEqual(parity_diff, 0.0)

            # Confirm loaded weights have zero vision keys
            clean = not any("vision" in k or "projector" in k for k in loaded_sd)
            self.assertTrue(clean)

            # Load into fresh language model
            lm_only = TinyLanguageModel()
            lm_sd = {k.replace("language_model.", ""): v for k, v in loaded_sd.items() if k.startswith("language_model.")}
            lm_only.load_state_dict(lm_sd)
            out_lm = lm_only(input_ids=test_inputs[0]["input_ids"])
            self.assertTrue(torch.isfinite(out_lm).all())

    def test_full_amputate_modality_pipeline(self):
        model = MockLlavaModel()
        proc = MockProcessor()
        retained_inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
        test_inputs = [{"input_ids": torch.tensor([[1, 2, 3]])}]

        amputated_model, rem_report, val_report = amputate_modality(
            model=model,
            modality="vision",
            retained_inputs=retained_inputs,
            test_inputs=test_inputs,
            config=model.config,
            processor=proc,
            require_proof=True,
            install_guard=True,
        )

        self.assertIsNotNone(rem_report)
        self.assertEqual(len(rem_report.removed_branches), 2)
        self.assertIsNotNone(val_report)
        self.assertTrue(val_report.passed)
        self.assertEqual(val_report.max_logit_diff, 0.0)

        # Verify processor rejects images
        with self.assertRaises(ModalityRemovedError):
            proc(images=["test"])

        # Verify model rejects pixel_values
        with self.assertRaises(ModalityRemovedError):
            amputated_model(input_ids=retained_inputs["input_ids"], pixel_values=torch.randn(1, 3, 14, 14))


class TestTransformersIntegration(unittest.TestCase):
    """Test Stage 8 on a real Hugging Face Llava model in memory."""

    def test_real_hf_llava_amputation(self):
        try:
            from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration
        except ImportError:
            self.skipTest("transformers not available")

        # Create tiny Llava model
        v_cfg = CLIPVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=14,
            patch_size=14,
        )
        t_cfg = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=50,
        )
        cfg = LlavaConfig(vision_config=v_cfg, text_config=t_cfg)
        model = LlavaForConditionalGeneration(cfg)

        initial_params = sum(p.numel() for p in model.parameters())
        test_batch = {"input_ids": torch.tensor([[1, 2, 3]])}

        # Baseline text output
        with torch.no_grad():
            out_base = model(**test_batch).logits

        # Amputate vision
        amputated_model, rem_report, val_report = amputate_modality(
            model=model,
            modality="vision",
            retained_inputs=test_batch,
            test_inputs=[test_batch],
            config=model.config,
            require_proof=True,
            install_guard=True,
        )

        remaining_params = sum(p.numel() for p in amputated_model.parameters())
        self.assertLess(remaining_params, initial_params)

        # Parity check
        with torch.no_grad():
            out_after = amputated_model(**test_batch).logits
        max_diff = torch.max(torch.abs(out_base - out_after)).item()
        self.assertEqual(max_diff, 0.0)

        # State dict check
        clean, offending = verify_state_dict_absence(amputated_model, ["vision_tower", "multi_modal_projector"])
        self.assertTrue(clean)
        self.assertEqual(len(offending), 0)

        # Input rejection check
        with self.assertRaises(ModalityRemovedError):
            amputated_model(input_ids=test_batch["input_ids"], pixel_values=torch.randn(1, 3, 14, 14))


if __name__ == "__main__":
    unittest.main()

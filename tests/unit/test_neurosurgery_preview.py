import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

from aegis_lab.editing.neurosurgery.preview import build_preview
from aegis_lab.editing.neurosurgery.optimizer import (
    CandidateState, SearchConstraints, _materialize_plan,
)


class MockMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)


class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MockMLP()


class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer()])


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MockBackbone()
        self.config = SimpleNamespace(model_type="llama", intermediate_size=6, num_hidden_layers=1)


class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.num_key_value_heads = 2
        self.num_key_value_groups = 2
        self.head_dim = 2
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)


class MockAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MockAttention()


class MockExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)


class MockMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 4, bias=True)
        self.experts = nn.ModuleList([MockExpert() for _ in range(4)])


class MockMoELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_sparse_moe = MockMoE()


class MockSpecialBackbone(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layers = nn.ModuleList([layer])


class MockAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MockSpecialBackbone(MockAttentionLayer())
        self.config = SimpleNamespace(
            model_type="llama", num_attention_heads=4, num_key_value_heads=2,
            head_dim=2, num_hidden_layers=1,
        )


class MockMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MockSpecialBackbone(MockMoELayer())
        self.config = SimpleNamespace(
            model_type="mixtral", num_local_experts=4, num_experts_per_tok=1,
            num_hidden_layers=1,
        )


class TestNeurosurgeryPreview(unittest.TestCase):
    def _write_plan(self, folder, checksum_override=None):
        selection_path = folder / "mlp.pt"
        torch.save({"keep_indices": [torch.tensor([1, 2, 4])]}, selection_path)
        checksum = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        if checksum_override is not None:
            checksum = checksum_override
        plan = {
            "version": 3,
            "ablation": {"layers": []},
            "drop_layers": [],
            "structured": {
                "mlp": {
                    "enabled": True,
                    "selection": selection_path.name,
                    "selection_sha256": checksum,
                }
            },
        }
        plan_path = folder / "plan.yaml"
        plan_path.write_text(yaml.safe_dump(plan), encoding="utf-8")
        return plan_path

    def test_preview_resolves_exact_slices_without_mutation(self):
        model = MockModel()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = self._write_plan(Path(tmp))
            report = build_preview(model, str(plan_path))
        self.assertEqual(report["summary"]["operation_count"], 3)
        ops = report["operations"]
        self.assertEqual([op["parameter"] for op in ops], [
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ])
        self.assertEqual([op["shape_after"] for op in ops], [[3, 4], [3, 4], [4, 3]])
        self.assertEqual([op["kept_indices"] for op in ops], [[1, 2, 4]] * 3)
        self.assertEqual(sum(op["estimated_bytes_removed"] for op in ops), 144)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))

    def _write_structured_plan(self, folder, kind, payload):
        selection_path = folder / f"{kind}.pt"
        torch.save(payload, selection_path)
        plan_path = folder / f"{kind}.yaml"
        plan_path.write_text(yaml.safe_dump({
            "version": 3,
            "ablation": {"layers": []},
            "drop_layers": [],
            "structured": {
                kind: {
                    "enabled": True,
                    "selection": selection_path.name,
                    "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                }
            },
        }), encoding="utf-8")
        return plan_path

    def test_preview_resolves_attention_group_dependent_tensors(self):
        model = MockAttentionModel()
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._write_structured_plan(Path(tmp), "attention", {
                "keep_groups": [torch.tensor([1])]
            })
            report = build_preview(model, str(plan))
        ops = report["operations"]
        self.assertEqual([op["shape_after"] for op in ops], [[4, 8], [2, 8], [2, 8], [8, 4]])
        self.assertEqual(ops[0]["kept_indices"], [4, 5, 6, 7])
        self.assertEqual(ops[3]["selection_axis"], "columns")

    def test_preview_lists_removed_moe_experts_and_router_slice(self):
        model = MockMoEModel()
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._write_structured_plan(Path(tmp), "moe", {
                "keep_experts": [torch.tensor([0, 2])]
            })
            report = build_preview(model, str(plan))
        ops = report["operations"]
        self.assertEqual(ops[0]["kind"], "moe_router_slice")
        self.assertEqual(ops[0]["shape_after"], [2, 4])
        removed = [op for op in ops if op["kind"] == "moe_expert_remove"]
        self.assertEqual([op["expert"] for op in removed], [1, 3])
        self.assertTrue(all(op["parameters"][0]["name"].endswith("proj.weight") for op in removed))

    def test_preview_accepts_actual_optimizer_plans_without_rewriting(self):
        cases = [
            ("mlp", MockModel, [1, 2, 4], CandidateState(mlp_level=1)),
            ("attention", MockAttentionModel, [1], CandidateState(attention_level=1)),
            ("moe", MockMoEModel, [0, 2], CandidateState(moe_level=1)),
        ]
        for kind, model_factory, indices, state in cases:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                model = model_factory()
                if kind == "moe":
                    # MoE block positions need not equal transformer-layer IDs.
                    model.model.layers.insert(0, nn.Identity())
                    model.config.num_hidden_layers = 2
                before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
                ratios = {name: [1.0, 0.5] for name in ("mlp", "attention", "moe")}
                cache = {kind: {0.5: [torch.tensor(indices)]}}
                folder = Path(tmp)
                profile = folder / "profile.pt"
                torch.save({"version": 1}, profile)
                plan = _materialize_plan(
                    folder,
                    {"trial_id": 1, "mean_kl": 0.01, "top1_agreement": 1.0,
                     "bytes_saved": 100, "estimated_macs_saved_per_token": 10},
                    state, ratios, cache, {kind: str(profile)}, [], SearchConstraints(),
                )
                original_plan = plan.read_bytes()
                self.assertEqual(yaml.safe_load(original_plan)["version"], 4)
                report = build_preview(model, str(plan))
                self.assertGreater(report["summary"]["operation_count"], 0)
                self.assertEqual(plan.read_bytes(), original_plan)
                for name, tensor in model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, before[name]))
                if kind == "moe":
                    self.assertTrue(all(op["layer"] == 1 for op in report["operations"]))
                # Version-4 support must still verify the producer's artifact hash.
                selection = folder / yaml.safe_load(original_plan)["structured"][kind]["selection"]
                with selection.open("ab") as artifact:
                    artifact.write(b"tampered")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    build_preview(model, str(plan))

    def test_preview_version_four_preserves_directional_targets(self):
        model = MockModel()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            profile = folder / "profile.pt"
            torch.save({"directions": [torch.ones(4)],
                        "preserve_bases": [torch.eye(4)[:, :1]]}, profile)
            plan = folder / "plan.yaml"
            plan.write_text(yaml.safe_dump({
                "version": 4, "drop_layers": [], "structured": {},
                "ablation": {"layers": [0], "targets": ["mlp.down_proj"], "strength": 0.5},
            }))
            report = build_preview(model, str(plan), str(profile))
            self.assertEqual(report["operations"][0]["kind"], "directional_weight_edit")
            self.assertEqual(report["operations"][0]["parameter"], "model.layers.0.mlp.down_proj.weight")
            with self.assertRaisesRegex(ValueError, "--profile is required"):
                build_preview(model, str(plan))

    def test_preview_rejects_unknown_plan_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self._write_plan(Path(tmp))
            payload = yaml.safe_load(plan.read_text())
            payload["version"] = 5
            plan.write_text(yaml.safe_dump(payload))
            with self.assertRaisesRegex(ValueError, "unsupported surgery plan version"):
                build_preview(MockModel(), str(plan))

    def test_preview_rejects_stale_selection_checksum(self):
        model = MockModel()
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = self._write_plan(Path(tmp), "0" * 64)
            with self.assertRaisesRegex(ValueError, "checksum"):
                build_preview(model, str(plan_path))


if __name__ == "__main__":
    unittest.main()

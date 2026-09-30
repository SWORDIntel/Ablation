import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

from aegis_lab.editing.neurosurgery.preview import build_preview


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

    def test_preview_rejects_stale_selection_checksum(self):
        model = MockModel()
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = self._write_plan(Path(tmp), "0" * 64)
            with self.assertRaisesRegex(ValueError, "checksum"):
                build_preview(model, str(plan_path))


if __name__ == "__main__":
    unittest.main()

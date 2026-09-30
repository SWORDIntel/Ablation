import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis_lab.editing.neurosurgery.adapters import get_mlp_adapter
from aegis_lab.editing.neurosurgery.mlp import select_mlp_channels


class MockMLP(nn.Module):
    def __init__(self, hidden=8, intermediate=12):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MockLayer(nn.Module):
    def __init__(self, hidden=8, intermediate=12):
        super().__init__()
        self.mlp = MockMLP(hidden, intermediate)


class MockBackbone(nn.Module):
    def __init__(self, layers=2, hidden=8, intermediate=12):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer(hidden, intermediate) for _ in range(layers)])


class MockModel(nn.Module):
    def __init__(self, layers=2, hidden=8, intermediate=12):
        super().__init__()
        self.model = MockBackbone(layers, hidden, intermediate)
        self.config = SimpleNamespace(model_type="llama", intermediate_size=intermediate, num_hidden_layers=layers)


class TestMLPSurgery(unittest.TestCase):
    def test_identity_ratio_does_not_align_away_channels(self):
        keep = [torch.ones(10)]
        drop = [torch.ones(10)]
        selected = select_mlp_channels(keep, drop, 1.0, contrast_weight=0.25, align_to=8)
        self.assertEqual(selected[0].numel(), 10)

    def test_select_channels_is_contrastive(self):
        keep = [torch.tensor([10.0, 9.0, 8.0, 1.0]), torch.tensor([9.0, 8.0, 7.0, 1.0])]
        drop = [torch.tensor([1.0, 1.0, 1.0, 10.0]), torch.tensor([1.0, 1.0, 1.0, 9.0])]
        selected = select_mlp_channels(keep, drop, 0.75, contrast_weight=0.5, align_to=1)
        self.assertTrue(all(x.numel() == 3 for x in selected))
        self.assertNotIn(3, selected[0].tolist())
        self.assertNotIn(3, selected[1].tolist())

    def test_physical_surgery_matches_masked_original(self):
        torch.manual_seed(7)
        model = MockModel(layers=1)
        original = copy.deepcopy(model)
        keep = torch.tensor([0, 1, 3, 4, 6, 8, 9, 11])
        summary = get_mlp_adapter(model).apply_mlp_selection(model, [keep])
        self.assertEqual(summary["new_intermediate_size"], 8)
        self.assertEqual(model.config.intermediate_size, 8)

        x = torch.randn(3, 5, 8)
        old_mlp = original.model.layers[0].mlp
        hidden = F.silu(old_mlp.gate_proj(x)) * old_mlp.up_proj(x)
        mask = torch.zeros(12, dtype=hidden.dtype)
        mask[keep] = 1
        expected = old_mlp.down_proj(hidden * mask)
        actual = model.model.layers[0].mlp(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()

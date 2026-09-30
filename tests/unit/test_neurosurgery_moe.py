import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from aegis_lab.editing.neurosurgery.adapters import get_moe_adapter
from aegis_lab.editing.neurosurgery.moe import select_moe_experts


class Expert(nn.Module):
    def __init__(self, hidden=4):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.proj(x)


class MockMoE(nn.Module):
    def __init__(self, hidden=4, experts=4):
        super().__init__()
        self.gate = nn.Linear(hidden, experts, bias=True)
        self.experts = nn.ModuleList([Expert(hidden) for _ in range(experts)])
        self.num_experts = experts

    def forward(self, x):
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return (outputs * probs.unsqueeze(-1)).sum(dim=-2)


class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_sparse_moe = MockMoE()


class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer()])


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MockBackbone()
        self.config = SimpleNamespace(model_type="mixtral", num_local_experts=4, num_experts_per_tok=1, num_hidden_layers=1)


class TestMoESurgery(unittest.TestCase):
    def test_expert_selection_penalizes_drop_specific_experts(self):
        keep = [torch.tensor([10.0, 8.0, 2.0, 1.0])]
        drop = [torch.tensor([1.0, 1.0, 8.0, 10.0])]
        selected = select_moe_experts(keep, drop, 0.5, contrast_weight=0.5)
        self.assertEqual(selected[0].tolist(), [0, 1])

    def test_physical_expert_surgery_matches_masked_router(self):
        torch.manual_seed(9)
        model = MockModel()
        original = copy.deepcopy(model)
        keep = torch.tensor([0, 2])
        summary = get_moe_adapter(model).apply_moe_selection(model, [keep])
        self.assertEqual(summary["new_expert_count"], 2)
        self.assertEqual(model.config.num_local_experts, 2)

        x = torch.randn(3, 4)
        old = original.model.layers[0].block_sparse_moe
        logits = old.gate(x)
        removed = torch.tensor([False, True, False, True])
        logits[:, removed] = -torch.inf
        probs = torch.softmax(logits, dim=-1)
        outputs = torch.stack([expert(x) for expert in old.experts], dim=-2)
        expected = (outputs * probs.unsqueeze(-1)).sum(dim=-2)
        actual = model.model.layers[0].block_sparse_moe(x)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()

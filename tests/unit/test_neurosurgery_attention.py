import copy
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from aegis_lab.editing.neurosurgery.adapters import get_attention_adapter
from aegis_lab.editing.neurosurgery.attention import select_attention_groups


class MockAttention(nn.Module):
    def __init__(self, hidden=8, heads=4, kv_heads=2, head_dim=2):
        super().__init__()
        self.num_heads = heads
        self.num_key_value_heads = kv_heads
        self.num_key_value_groups = heads // kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)


class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MockAttention()


class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer()])


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MockBackbone()
        self.config = SimpleNamespace(
            model_type="llama",
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            num_hidden_layers=1,
        )


class TestAttentionSurgery(unittest.TestCase):
    def test_selection_prefers_keep_specific_group(self):
        keep = [torch.tensor([10.0, 2.0, 9.0, 1.0])]
        drop = [torch.tensor([1.0, 9.0, 1.0, 10.0])]
        selected = select_attention_groups(keep, drop, 0.5, contrast_weight=0.5)
        self.assertEqual(selected[0].tolist(), [0, 2])

    def test_physical_gqa_group_surgery_matches_masked_o_projection(self):
        torch.manual_seed(4)
        model = MockModel()
        original = copy.deepcopy(model)
        keep_groups = torch.tensor([1])
        summary = get_attention_adapter(model).apply_attention_selection(model, [keep_groups])
        self.assertEqual(summary["new_num_attention_heads"], 2)
        self.assertEqual(summary["new_num_key_value_heads"], 1)
        self.assertEqual(model.config.num_attention_heads, 2)
        self.assertEqual(model.config.num_key_value_heads, 1)

        old = original.model.layers[0].self_attn
        new = model.model.layers[0].self_attn
        self.assertEqual(tuple(new.q_proj.weight.shape), (4, 8))
        self.assertEqual(tuple(new.k_proj.weight.shape), (2, 8))
        self.assertEqual(tuple(new.v_proj.weight.shape), (2, 8))
        self.assertEqual(tuple(new.o_proj.weight.shape), (8, 4))

        head_output = torch.randn(3, 5, 4, 2)
        mask = torch.zeros(4, dtype=head_output.dtype)
        mask[2:4] = 1
        expected = old.o_proj((head_output * mask[None, None, :, None]).reshape(3, 5, 8))
        actual = new.o_proj(head_output[:, :, 2:4, :].reshape(3, 5, 4))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

        x = torch.randn(2, 8)
        self.assertTrue(torch.allclose(new.q_proj(x), old.q_proj(x)[:, 4:8], atol=1e-6))
        self.assertTrue(torch.allclose(new.k_proj(x), old.k_proj(x)[:, 2:4], atol=1e-6))
        self.assertTrue(torch.allclose(new.v_proj(x), old.v_proj(x)[:, 2:4], atol=1e-6))


if __name__ == "__main__":
    unittest.main()

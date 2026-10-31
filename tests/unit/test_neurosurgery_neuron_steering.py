"""Regression tests for neuron-level causal interventions and paired reports."""
import copy
import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from aegis_lab.editing.neurosurgery.neuron_steering import (
    NORMALIZED, PRESETS, NeuronSteerer, _first_difference,
    list_presets, measure_residual_medians, parse_neuron, parse_scales,
    resolve_recipe, run_ab_experiment,
)


class ToyTokenizer:
    chat_template = "test"
    pad_token_id = 0
    eos_token_id = 3

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 2, 2]])}

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 2]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(2, 3, bias=False)
        self.down_proj = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.up_proj.weight.zero_()
            self.up_proj.weight[0, :] = -1.0
            self.down_proj.weight.zero_()
            self.down_proj.weight[0, 0] = 1.0
            self.down_proj.weight[1, 1] = 1.0
            self.down_proj.weight[1, 2] = 1.0

    def forward(self, h):
        return self.down_proj(self.up_proj(h))


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = ToyMLP()

    def forward(self, h):
        return h + self.mlp(h)


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 2)
        self.layers = nn.ModuleList([ToyLayer()])
        self.lm_head = nn.Linear(2, 4, bias=False)
        self.config = SimpleNamespace(model_type="llama")
        with torch.no_grad():
            self.embedding.weight.fill_(1.0)
            self.lm_head.weight.zero_()
            self.lm_head.weight[0, 0] = 1.0
            self.lm_head.weight[1, 0] = -1.0

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.embedding(input_ids)
        for layer in self.layers:
            h = layer(h)
        return SimpleNamespace(logits=self.lm_head(h))

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1, **kwargs):
        # Predict one token using the same module pre-hooks as real generation.
        next_id = self(input_ids, attention_mask=attention_mask).logits[:, -1, :].argmax(-1)
        return torch.cat([input_ids, next_id[:, None]], dim=-1)


def recipe(dose=5.0, norm="raw", op="add", apply_to="all_tokens"):
    return resolve_recipe_from_dict({
        "model": "test/Toy", "effect": "unit test",
        "neurons": [{"layer": 0, "neuron": 0, "dose": dose}],
        "method": {"op": op, "normalization": norm, "apply_to": apply_to},
    })


def resolve_recipe_from_dict(value):
    # Validate as if supplied by an operator JSON recipe, without filesystem IO.
    from tempfile import NamedTemporaryFile
    import json
    with NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as f:
        json.dump(value, f)
        f.flush()
        return resolve_recipe(config=f.name, model_id="test/Toy")


class TestNeuronRecipes(unittest.TestCase):
    def test_curated_model_specific_presets(self):
        self.assertEqual(PRESETS["qwen35-4b-evil-pair"]["neurons"],
                         [{"layer": 14, "neuron": 8949, "dose": 1.0},
                          {"layer": 13, "neuron": 7956, "dose": -1.0}])
        self.assertEqual(PRESETS["qwen38-27b-hostile"]["model"], "Qwen/Qwen3.8-27B")
        self.assertIn("qwen3-8b-coffee-bean", PRESETS)
        self.assertTrue(all(v["source"].startswith("https://") for v in PRESETS.values()))
        self.assertTrue(all("not locally reproduced" in v["evidence"] for v in PRESETS.values()))
        self.assertTrue(all(v["model"] == "Qwen/Qwen3.5-4B" for v in list_presets("Qwen/Qwen3.5-4B").values()))

    def test_parse_neuron_specs_and_scales(self):
        self.assertEqual(parse_neuron("L14N8949+1.0"), {"layer": 14, "neuron": 8949, "dose": 1.0})
        self.assertEqual(parse_neuron("13:7956:-1"), {"layer": 13, "neuron": 7956, "dose": -1.0})
        self.assertEqual(parse_scales("-0.5,0.5,1.0"), [-0.5, 0.5, 1.0])
        for invalid in ("L0N0nan", "-1:0:1", "1:2:inf"):
            with self.assertRaises(ValueError):
                parse_neuron(invalid)
        for invalid in ("", "1,1", "0,1", "inf"):
            with self.assertRaises(ValueError):
                parse_scales(invalid)

    def test_mul_defaults_to_raw_normalization(self):
        parsed = resolve_recipe_from_dict({
            "model": "test/Toy",
            "neurons": [{"layer": 0, "neuron": 0, "dose": 0.0}],
            "method": {"op": "mul"},
        })
        self.assertEqual(parsed["method"]["normalization"], "raw")

    def test_unsupported_recipe_semantics_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conditional"):
            resolve_recipe_from_dict({
                "model": "test/Toy",
                "neurons": [{"layer": 0, "neuron": 0, "dose": 1}],
                "conditional": {"if": "some text"},
            })
        with self.assertRaisesRegex(ValueError, "Unsupported steering method"):
            resolve_recipe_from_dict({
                "model": "test/Toy",
                "neurons": [{"layer": 0, "neuron": 0, "dose": 1}],
                "method": {"site": "residual.output"},
            })

    def test_fail_closed_on_mismatched_model_and_duplicates(self):
        with self.assertRaisesRegex(ValueError, "requires --model-id"):
            resolve_recipe(preset="qwen35-4b-evil-pair")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            resolve_recipe(preset="qwen35-4b-evil-pair", model_id="Qwen/Qwen3-8B")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            resolve_recipe(neurons=["0:1:1", "0:1:-1"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_recipe(preset="nonexistent", model_id="test/Toy")
        with self.assertRaisesRegex(ValueError, "MoE"):
            resolve_recipe_from_dict({
                "model": "test/Toy", "neurons": [{"layer": 0, "expert": 1, "neuron": 0, "dose": 1}],
            })


class TestNeuronSteerer(unittest.TestCase):
    def setUp(self):
        self.model = ToyLM().eval()
        self.ids = torch.tensor([[1, 2, 2]])

    def test_raw_addition_is_reversible_and_does_not_mutate_inputs_or_weights(self):
        target = self.model.layers[0].mlp.down_proj
        before = copy.deepcopy(self.model.state_dict())
        captured = []
        probe = target.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].clone()))
        base = self.model(self.ids).logits
        probe.remove()
        with NeuronSteerer(self.model, recipe(dose=5.0)) as steer:
            steer.start_generation(0)
            self.assertTrue(torch.equal(self.model(self.ids).logits, base))
            steer.start_generation(1)
            changed = self.model(self.ids).logits
            self.assertFalse(torch.equal(changed, base))
            steer.start_generation(-1)
            reversed_output = self.model(self.ids).logits
            self.assertFalse(torch.equal(changed, reversed_output))
        self.assertTrue(torch.equal(self.model(self.ids).logits, base))
        self.assertEqual(len(target._forward_pre_hooks), 0)
        self.assertEqual(len(self.model._forward_pre_hooks), 0)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in self.model.state_dict().items()))
        self.assertGreater(len(captured), 0)

    def test_normalization_matches_projection_column_norm(self):
        model = self.model
        r = recipe(dose=1.5, norm=NORMALIZED)
        layers = model.layers
        med = measure_residual_medians(model, ToyTokenizer(), layers, [0], "some calibration", "cpu")
        expected = 1.5 * med[0] / float(layers[0].mlp.down_proj.weight[:, 0].float().norm())
        with NeuronSteerer(model, r, med) as steer:
            self.assertAlmostEqual(steer.amplitudes[0]["amplitude"], expected)
            steer.start_generation(1)
            self.assertEqual(steer.phase, "prefill")
            model(self.ids)
            self.assertEqual(steer.phase, "prefill")

    def test_mul_and_generation_phase_gates(self):
        model = self.model
        baseline = model(self.ids).logits
        with NeuronSteerer(model, recipe(dose=0, op="mul")) as steer:
            steer.start_generation(1)
            self.assertFalse(torch.equal(baseline, model(self.ids).logits))
        with NeuronSteerer(model, recipe(dose=6, apply_to="generated_tokens")) as steer:
            steer.start_generation(1)
            self.assertTrue(torch.equal(baseline, model(self.ids).logits))
            self.assertEqual(steer.phase, "prefill")
            self.assertFalse(torch.equal(baseline, model(self.ids).logits))

    def test_invalid_geometry_is_rejected_before_hook_installation(self):
        model = self.model
        for bad in ({"layer": 1, "neuron": 0, "dose": 1.0},
                    {"layer": 0, "neuron": 3, "dose": 1.0}):
            with self.assertRaises(ValueError):
                NeuronSteerer(model, {**recipe(), "neurons": [bad]})
        self.assertEqual(len(model._forward_pre_hooks), 0)


class TestNeuronAB(unittest.TestCase):
    def test_paired_outputs_and_regex_metric(self):
        model = ToyLM().eval()
        output = run_ab_experiment(
            model, ToyTokenizer(), recipe(dose=5.0), ["How are you?", "Explain a tree."],
            scales=[0.5, 1.0, -1.0], max_new_tokens=1, target_regex=r"\b0\b",
            seed=17,
        )
        self.assertEqual(len(output["trials"]), 2)
        self.assertEqual(output["summary"][1]["changed"], 2)
        self.assertEqual(output["summary"][1]["steered_target_matches"], 2)
        self.assertEqual(output["summary"][1]["baseline_target_matches"], 0)
        self.assertEqual(output["trials"][0]["seed"], 17)
        self.assertEqual(output["trials"][1]["seed"], 18)
        self.assertEqual(output["trials"][0]["variants"][1]["first_different_token"], 0)
        self.assertIsNone(_first_difference([1, 2], [1, 2]))
        self.assertEqual(_first_difference([1, 2], [1, 2, 3]), 2)
        self.assertEqual(len(model._forward_pre_hooks), 0)
        self.assertEqual(len(model.layers[0].mlp.down_proj._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()

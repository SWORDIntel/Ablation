"""Synthetic invariants for bounded 3+ neuron search and signed dose tuning.

No test in this module demonstrates a persona effect on a real checkpoint.
"""
from __future__ import annotations

import unittest

import torch
from torch import nn

from aegis_lab.editing.neurosurgery.cli import build_parser
from aegis_lab.editing.neurosurgery.neuron_causal import run_causal_screen
from aegis_lab.editing.neurosurgery.neuron_composition import (
    extra_forward_budget, _group_metrics, _pareto_frontier,
)
from aegis_lab.editing.neurosurgery.neuron_ranking import rank_steering_candidates
from tests.unit.test_neurosurgery_neuron_steering import ToyLM, ToyTokenizer


def _targets(name):
    return [
        {"prompt": f"{name} objective 1", "target_token_id": 0, "contrast_token_id": 1},
        {"prompt": f"{name} objective 2", "target_token_id": 0, "contrast_token_id": 1},
    ]


class TestMultiNeuronOptimization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(726)
        self.model = ToyLM().eval()
        mlp = self.model.layers[0].mlp
        mlp.up_proj = nn.Linear(2, 9, bias=False)
        mlp.down_proj = nn.Linear(9, 2, bias=False)
        with torch.no_grad():
            mlp.up_proj.weight.zero_()
            mlp.down_proj.weight.zero_()
            mlp.down_proj.weight[0, 0] = 1.0
            mlp.down_proj.weight[0, 1] = 0.8
            mlp.down_proj.weight[0, 2] = 0.6
            # Distinct eligible controls with nonzero output columns.
            for i in range(3, 6):
                mlp.down_proj.weight[0, i] = -0.1 - (i - 3) * 0.01
            for i in range(6, 9):
                mlp.down_proj.weight[1, i] = 0.1 + (i - 6) * 0.01
        self.tokenizer = ToyTokenizer()
        self.profile = {
            "version": 1,
            "keep_importance": [torch.ones(9)],
            "drop_importance": [torch.tensor(
                [10.0, 9.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )],
        }
        self.ranking = rank_steering_candidates(self.profile, top_k=3)

    def run_experiment(self, **overrides):
        args = dict(
            model=self.model, tokenizer=self.tokenizer, ranking=self.ranking,
            profile=self.profile,
            search_targets=_targets("search"), search_keep=["search KEEP a", "search KEEP b"],
            test_targets=_targets("test"), test_keep=["test KEEP a", "test KEEP b"],
            discovery_keep=["discovery KEEP"], discovery_drop=["discovery target"],
            scales=[-0.5, 0.5], max_candidates=3, controls_per_candidate=1,
            max_pairs=3, max_groups=3, max_group_size=3, beam_width=2,
            keep_penalty=0.0, neuron_penalty=0.0,
            max_forwards=400, max_mean_kl=10.0, min_top1_agreement=0.0,
            normalization="raw", device="cpu", seed=31,
        )
        args.update(overrides)
        return run_causal_screen(**args)

    def test_triple_search_is_search_only_and_retains_weight_integrity(self):
        original = {key: value.clone() for key, value in self.model.state_dict().items()}
        result = self.run_experiment()
        optimization = result["group_search"]
        self.assertTrue(optimization["enabled"])
        self.assertGreater(optimization["max_groups"], 0)
        self.assertEqual(optimization["attempted_groups"], 1)
        self.assertEqual(len(optimization["trials"]), 1)
        self.assertEqual(optimization["stages"][0]["size"], 3)
        self.assertEqual(optimization["stages"][0]["retained"], 1)
        selected = optimization["selected"]
        self.assertIsNotNone(selected)
        self.assertEqual(len(selected["members"]), 3)
        self.assertEqual({m["neuron"] for m in selected["members"]}, {0, 1, 2})
        self.assertEqual(selected["selection_source"], "search split only")
        self.assertTrue(selected["search_gate_passed"])
        self.assertGreater(selected["interaction"]["gain_over_parent"], 0)
        self.assertGreater(selected["interaction"]["gain_over_matched_group"], 0)
        self.assertGreater(selected["interaction"]["gain_over_best_single"], 0)
        self.assertAlmostEqual(
            selected["interaction"]["nonadditive_interaction_mean"], 0.0, places=5
        )
        self.assertEqual(len(selected["matched_controls"]), 3)
        self.assertEqual(len({(m["layer"], m["neuron"])
                              for m in selected["matched_controls"]}), 3)
        self.assertTrue(all(m["neuron"] not in (0, 1, 2)
                            for m in selected["matched_controls"]))
        self.assertGreaterEqual(len(selected["coordinate_trials"]), 4)
        self.assertGreaterEqual(len(optimization["pareto_frontier"]), 1)
        heldout = result["heldout_group"]
        self.assertEqual(heldout["selected_on"], "search split only")
        self.assertTrue(heldout["observed_gate_passed"])
        self.assertEqual(len(heldout["monomers"]), 3)
        self.assertFalse(heldout["promotable_as_verified_persona"])
        self.assertTrue(result["baseline_restoration_verified"])
        self.assertFalse(self.model._forward_pre_hooks)
        self.assertFalse(self.model.layers[0].mlp.down_proj._forward_pre_hooks)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))

    def test_disabled_mode_does_not_evaluate_extra_groups(self):
        x = self.run_experiment(max_groups=0, max_forwards=170)
        self.assertFalse(x["group_search"]["enabled"])
        self.assertEqual(x["group_search"]["trials"], [])
        self.assertIsNone(x["heldout_group"])
        self.assertEqual(x["search_budgets"]["group_forward_budget"], 0)

    def test_budget_validation_before_model_forward(self):
        with self.assertRaisesRegex(ValueError, "Forward budget"):
            self.run_experiment(max_forwards=90)
        with self.assertRaisesRegex(ValueError, "max-groups"):
            self.run_experiment(max_groups=65)
        with self.assertRaisesRegex(ValueError, "beam-width"):
            self.run_experiment(beam_width=0)
        with self.assertRaisesRegex(ValueError, "max-group-size"):
            self.run_experiment(max_group_size=6)
        with self.assertRaisesRegex(ValueError, "Group search requires"):
            self.run_experiment(max_pairs=0)
        with self.assertRaisesRegex(ValueError, "at least three"):
            self.run_experiment(max_candidates=2)
        with self.assertRaisesRegex(ValueError, "penalties"):
            self.run_experiment(keep_penalty=float("nan"))
        self.assertGreater(extra_forward_budget(3, 3, 2, 4, 4), 0)
        self.assertEqual(extra_forward_budget(0, 3, 2, 4, 4), 0)

    def test_keep_gate_rejects_triples(self):
        out = self.run_experiment(max_mean_kl=0, min_top1_agreement=1)
        self.assertIsNone(out["group_search"]["selected"])
        self.assertIsNone(out["heldout_group"])
        self.assertFalse(out["promotable_as_verified_persona"])

    def test_cli_flags_and_pareto_frontier(self):
        command = [
            "steer-causal", "--model", "m",
            "--ranking", "r.json", "--profile", "p.pt",
            "--discovery-keep", "dk", "--discovery-drop", "dd",
            "--search-target", "s.jsonl", "--search-keep", "sk",
            "--test-target", "t.jsonl", "--test-keep", "tk",
            "--max-pairs", "3", "--max-groups", "4",
            "--max-group-size", "4", "--beam-width", "2",
            "--keep-penalty", "0.5", "--neuron-penalty", "0.1",
            "--out", "o.json",
        ]
        args = build_parser().parse_args(command)
        self.assertEqual(args.max_groups, 4)
        self.assertEqual(args.max_group_size, 4)
        self.assertEqual(args.beam_width, 2)
        self.assertAlmostEqual(args.keep_penalty, 0.5)

        def fake(x, kl, n):
            return {
                "members": [{"layer": 0, "neuron": i, "scale": 1}
                            for i in range(n)],
                "joint": {"target_mean_delta_margin": x, "keep_mean_kl": kl,
                          "keep_top1_agreement": 1.0},
                "search_objective": x - kl,
                "search_gate_passed": True,
            }
        front = _pareto_frontier([
            fake(2.0, 0.10, 3), fake(1.0, 0.30, 4), fake(1.2, 0.01, 3)
        ])
        self.assertEqual(len(front), 2)

    def test_prompt_alignment_guard(self):
        def obs(delta, prompt="p"):
            return {"target_mean_delta_margin": delta, "target_per_prompt": [
                {"prompt_sha256": prompt, "delta_margin": delta}
            ]}
        from aegis_lab.editing.neurosurgery.neuron_causal import _bootstrap_ci95
        with self.assertRaisesRegex(ValueError, "alignment"):
            _group_metrics(
                obs(3), obs(0), [obs(1), obs(1, "x"), obs(1)],
                obs(2), _bootstrap_ci95, 10
            )


if __name__ == "__main__":
    unittest.main()

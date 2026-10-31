"""Closed-loop synthetic regression tests for causal neuron screening.

These tests establish hook and evaluation mechanics, not behaviour of any
published model or independent causal validation on a real checkpoint.
"""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from aegis_lab.editing.neurosurgery.cli import build_parser
from aegis_lab.editing.neurosurgery.common import file_sha256
from aegis_lab.editing.neurosurgery.neuron_causal import (
    _bootstrap_ci95, _read_target_rows, _unique_splits, _controls_for_candidate,
    _candidate_coordinates, _verify_producer_model_binding,
    run_causal_screen, verify_discovery_lineage,
)
from aegis_lab.editing.neurosurgery.neuron_ranking import rank_steering_candidates
from tests.unit.test_neurosurgery_neuron_steering import ToyLM, ToyTokenizer


def profile():
    return {
        "version": 1,
        "model": "fixture-model",
        "adapter": "llama-like",
        "profile_keep_sha256": "sha256:discovery-keep-test",
        "profile_drop_sha256": "sha256:discovery-drop-test",
        "keep_importance": [torch.tensor([1.0, 1.0, 2.0])],
        "drop_importance": [torch.tensor([8.0, 1.0, 1.0])],
    }


def targets(prefix: str):
    return [
        {"prompt": prefix + " target alpha", "target_token_id": 0, "contrast_token_id": 1},
        {"prompt": prefix + " target beta", "target_token_id": 0, "contrast_token_id": 1},
    ]


def inputs():
    p = profile()
    rank = rank_steering_candidates(p, top_k=1, source_sha256="sha256:fixture-profile")
    return p, rank


class TestCausalDatasetProvenance(unittest.TestCase):
    def test_ranking_and_profile_discovery_are_strictly_bound(self):
        p, rank = inputs()
        verify_discovery_lineage(
            rank, p, profile_sha256="sha256:fixture-profile",
            discovery_keep_sha256=p["profile_keep_sha256"],
            discovery_drop_sha256=p["profile_drop_sha256"],
        )
        for argument in (
            {"profile_sha256": "sha256:changed"},
            {"discovery_drop_sha256": "sha256:changed"},
        ):
            vals = {
                "profile_sha256": "sha256:fixture-profile",
                "discovery_keep_sha256": p["profile_keep_sha256"],
                "discovery_drop_sha256": p["profile_drop_sha256"],
            }
            vals.update(argument)
            with self.assertRaisesRegex(ValueError, "mismatch|missing|match"):
                verify_discovery_lineage(rank, p, **vals)
        bad = copy.deepcopy(p)
        bad.pop("profile_keep_sha256")
        with self.assertRaisesRegex(ValueError, "missing"):
            verify_discovery_lineage(
                rank, bad, profile_sha256="sha256:fixture-profile",
                discovery_keep_sha256=p["profile_keep_sha256"],
                discovery_drop_sha256=p["profile_drop_sha256"],
            )

    def test_split_overlap_rejected_before_model_run(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            _unique_splits({"discovery": ["same"], "test": ["same "]})
        with self.assertRaisesRegex(ValueError, "Empty"):
            _unique_splits({"empty": []})

    def test_producer_audits_are_mandatory_and_bound_to_model(self):
        with TemporaryDirectory() as root:
            folder = Path(root)
            ranking_file, profile_file = folder / "rank.json", folder / "profile.pt"
            ranking_file.write_text("{}")
            profile_file.write_bytes(b"synthetic")
            args = SimpleNamespace(ranking=str(ranking_file), profile=str(profile_file))
            binding = {"file_hashes": {"model.safetensors": "sha:synthetic"}}
            with self.assertRaisesRegex(ValueError, "producer audit is missing"):
                _verify_producer_model_binding(args, binding)
            for name in (ranking_file, profile_file):
                audit = name.with_name(name.name + ".operator_audit.json")
                audit.write_text(json.dumps({
                    "status": "completed",
                    "inputs": {"model": {"files": binding["file_hashes"]}},
                }))
            result = _verify_producer_model_binding(args, binding)
            self.assertIn("ranking", result)
            self.assertIn("profile", result)
            wrong = ranking_file.with_name(ranking_file.name + ".operator_audit.json")
            wrong.write_text(json.dumps({
                "status": "completed", "inputs": {"model": {"files": {"other": "hash"}}},
            }))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                _verify_producer_model_binding(args, binding)

    def test_single_token_objectives(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "targets.jsonl"
            path.write_text(
                '{"prompt":"target one","target_token_id":0,"contrast_token_id":1}\n'
                '{"prompt":"target two","target_token_id":1,"contrast_token_id":2}\n',
                encoding="utf-8",
            )
            rows = _read_target_rows(str(path), ToyTokenizer(), 2)
            self.assertEqual(rows[0]["target_token_id"], 0)
            self.assertEqual(len(rows), 2)
            path.write_text('{"prompt":"x","target_token_id":4,"contrast_token_id":4}\n')
            with self.assertRaisesRegex(ValueError, "equal"):
                _read_target_rows(str(path), ToyTokenizer(), 2)
            path.write_text('{"prompt":"x","target":"two tokens","contrast_token_id":1}\n')
            with self.assertRaisesRegex(ValueError, "cannot encode"):
                _read_target_rows(str(path), ToyTokenizer(), 2)

    def test_cli_parses_causal_experiment(self):
        args = build_parser().parse_args([
            "steer-causal", "--model", "m", "--ranking", "ranking.json",
            "--profile", "profile.pt", "--discovery-keep", "dk.txt",
            "--discovery-drop", "dd.txt", "--search-target", "search.jsonl",
            "--search-keep", "sk.txt", "--test-target", "test.jsonl",
            "--test-keep", "tk.txt", "--scales=-0.5,0.5", "--out", "report.json",
        ])
        self.assertEqual(args.scales, "-0.5,0.5")
        self.assertEqual(args.controls_per_candidate, 1)
        self.assertEqual(args.max_forwards, 512)


class TestCausalNeuronEvaluation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)
        self.model = ToyLM().eval()
        self.tok = ToyTokenizer()
        self.p, self.rank = inputs()

    def run_screen(self, **changes):
        values = dict(
            model=self.model, tokenizer=self.tok, ranking=self.rank, profile=self.p,
            search_targets=targets("search"), search_keep=["search keep a", "search keep b"],
            test_targets=targets("test"), test_keep=["test keep a", "test keep b"],
            discovery_keep=["discovery keep a"], discovery_drop=["discovery drop a"],
            scales=[-0.5, 0.5], max_candidates=1, controls_per_candidate=1,
            max_forwards=64, max_mean_kl=10.0, min_top1_agreement=0.0,
            normalization="raw", seed=73, device="cpu",
        )
        values.update(changes)
        return run_causal_screen(**values)

    def test_target_specific_intervention_exceeds_matched_controls(self):
        before = {k: x.clone() for k, x in self.model.state_dict().items()}
        result = self.run_screen()
        self.assertTrue(result["baseline_restoration_verified"])
        self.assertFalse(result["promotable_as_verified_persona"])
        self.assertTrue(result["not_semantic_validation"])
        self.assertEqual(result["search_budgets"]["estimated_forwards"], 36)
        candidate = result["selected"]
        self.assertIsNotNone(candidate)
        self.assertEqual((candidate["layer"], candidate["neuron"]), (0, 0))
        self.assertGreater(candidate["target_mean_delta_margin"], 0)
        self.assertGreater(candidate["contrast_vs_matched_controls"], 0)
        self.assertTrue(result["heldout_test"]["observed_gate_passed"])
        self.assertGreater(result["heldout_test"]["candidate"]["target_mean_delta_margin"], 0)
        self.assertEqual(self.model._forward_pre_hooks, {})
        self.assertEqual(self.model.layers[0].mlp.down_proj._forward_pre_hooks, {})
        self.assertTrue(all(torch.equal(before[k], v) for k, v in self.model.state_dict().items()))

    def test_control_matching_is_same_layer_and_deterministic(self):
        expected = _controls_for_candidate(
            self.model, self.p, 0, 0, {(0, 0)}, 1, 17,
        )
        self.assertEqual(expected, _controls_for_candidate(
            self.model, self.p, 0, 0, {(0, 0)}, 1, 17,
        ))
        self.assertEqual(expected[0]["layer"], 0)
        self.assertIn(expected[0]["neuron"], (1, 2))
        with self.assertRaisesRegex(ValueError, "lacks"):
            _controls_for_candidate(self.model, self.p, 0, 0, {(0, 0)}, 3, 17)

    def test_rejects_budget_overlap_and_unsupported_scales(self):
        with self.assertRaisesRegex(ValueError, "Forward budget"):
            self.run_screen(max_forwards=10)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.run_screen(test_keep=["search keep a"])
        with self.assertRaisesRegex(ValueError, "Scales"):
            self.run_screen(scales=[0.0, 0.5])
        with self.assertRaisesRegex(ValueError, "Scales"):
            self.run_screen(scales=[float("nan")])
        with self.assertRaisesRegex(ValueError, "max-candidates"):
            _candidate_coordinates(self.rank, 33)

    def test_gate_can_reject_despite_positive_target_margin(self):
        outcome = self.run_screen(max_mean_kl=0, min_top1_agreement=1)
        self.assertIsNone(outcome["selected"])
        self.assertIsNone(outcome["heldout_test"])

    def test_deterministic_bootstrap(self):
        x = _bootstrap_ci95([1.0, 2.0, 3.0], 10)
        self.assertEqual(x, _bootstrap_ci95([1.0, 2.0, 3.0], 10))
        self.assertTrue(x[0] <= 2 <= x[1])
        with self.assertRaises(ValueError):
            _bootstrap_ci95([], 10)


if __name__ == "__main__":
    unittest.main()

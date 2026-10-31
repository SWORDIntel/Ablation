import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

import torch

from aegis_lab.editing.neurosurgery.cli import build_parser
from aegis_lab.editing.neurosurgery.neuron_ranking import (
    rank_steering_candidates, run_steer_rank_command,
)


def profile():
    return {
        "version": 1,
        "model": "demo",
        "adapter": "llama-like",
        "keep_importance": [torch.tensor([1.0, 1.0, 3.0, 1.0])],
        "drop_importance": [torch.tensor([7.0, 1.0, 1.0, 1.0])],
    }


class TestNeuronRanking(unittest.TestCase):
    def test_ranked_target_and_keep_controls(self):
        result = rank_steering_candidates(profile(), top_k=2, source_sha256="sha256:test")
        self.assertTrue(result["screening_only"])
        self.assertTrue(result["requires_causal_confirmation"])
        self.assertEqual(result["source_profile_sha256"], "sha256:test")
        self.assertEqual(result["geometry"], [{"layer": 0, "channels": 4}])
        self.assertEqual(result["target_enriched"][0]["neuron"], 0)
        self.assertEqual(result["keep_enriched_controls"][0]["neuron"], 2)
        self.assertGreater(result["target_enriched"][0]["contrast_score"], 0.0)
        self.assertLessEqual(len(result["target_enriched"]), 2)

    def test_invalid_profiles_are_rejected(self):
        bad = profile()
        bad["drop_importance"] = [torch.tensor([float("nan"), 1.0, 1.0, 1.0])]
        with self.assertRaisesRegex(ValueError, "invalid"):
            rank_steering_candidates(bad)
        bad = profile()
        bad["keep_importance"] = [torch.zeros(4)]
        with self.assertRaisesRegex(ValueError, "zero"):
            rank_steering_candidates(bad)
        bad = profile()
        bad["drop_importance"] = [torch.ones(2)]
        with self.assertRaisesRegex(ValueError, "widths"):
            rank_steering_candidates(bad)
        with self.assertRaisesRegex(ValueError, "top_k"):
            rank_steering_candidates(profile(), top_k=0)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            rank_steering_candidates({"version": 999})

    def test_cli_reads_restricted_tensor_artifact(self):
        with TemporaryDirectory() as folder:
            pt = Path(folder) / "mlp_profile.pt"
            out = Path(folder) / "candidates.json"
            torch.save(profile(), pt)
            args = build_parser().parse_args([
                "steer-rank", "--profile", str(pt), "--top-k", "1", "--out", str(out),
            ])
            result = run_steer_rank_command(args)
            self.assertEqual(result["target_candidates"], 1)
            self.assertEqual(result["keep_controls"], 1)
            recorded = json.loads(out.read_text())
            self.assertEqual(recorded["target_enriched"][0]["neuron"], 0)
            with self.assertRaises(FileExistsError):
                run_steer_rank_command(args)


if __name__ == "__main__":
    unittest.main()

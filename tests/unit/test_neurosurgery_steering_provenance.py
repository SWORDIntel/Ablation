"""Checkpoint provenance and replay invariants for steering experiments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from aegis_lab.editing.neurosurgery.measured import bind_files
from aegis_lab.editing.neurosurgery.steering_provenance import (
    checkpoint_binding, runtime_semantic_binding,
)
from aegis_lab.editing.neurosurgery.neuron_steering import (
    _scoped_generation_rng, _wilson_95, run_ab_experiment,
)
from tests.unit.test_neurosurgery_neuron_steering import ToyLM, ToyTokenizer, recipe


class TestSteeringCheckpointProvenance(unittest.TestCase):
    def test_checkpoint_files_are_bound_and_verified(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text('{"model_type":"qwen3"}')
            (path / "tokenizer.json").write_text('{"model":"fake"}')
            (path / "model.safetensors").write_bytes(b"fixture weights")
            original = checkpoint_binding(str(path))
            self.assertIn("model.safetensors", original["weight_files"])
            self.assertIn("config.json", original["config_files"])
            self.assertIn("tokenizer.json", original["tokenizer_files"])
            self.assertFalse(original["matched_expected_digest"])
            self.assertEqual(original["checkpoint_sha256"][:7], "sha256:")
            verified = checkpoint_binding(
                str(path), expected_checkpoint_sha256=original["checkpoint_sha256"],
            )
            self.assertTrue(verified["matched_expected_digest"])
            audited = bind_files({"model": str(path)})
            reread = checkpoint_binding(str(path), audited_inputs=audited)
            self.assertEqual(original["checkpoint_sha256"], reread["checkpoint_sha256"])
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                checkpoint_binding(str(path), audited_inputs={"model": {"path": "/tmp/nope", "files": {}}})

            (path / "model.safetensors").write_bytes(b"modified weights")
            changed = checkpoint_binding(str(path))
            self.assertNotEqual(changed["checkpoint_sha256"], original["checkpoint_sha256"])
            with self.assertRaisesRegex(ValueError, "Checkpoint digest mismatch"):
                checkpoint_binding(
                    str(path), expected_checkpoint_sha256=original["checkpoint_sha256"],
                )

    def test_missing_checkpoint_assets_fail_closed(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "Missing config"):
                checkpoint_binding(str(path))
            (path / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "weight shards"):
                checkpoint_binding(str(path))
            (path / "model.safetensors").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                checkpoint_binding(str(path))

    def test_model_and_tokenizer_semantics_are_hashed(self):
        item = runtime_semantic_binding(ToyLM(), ToyTokenizer())
        self.assertRegex(item["runtime_config_sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(item["runtime_tokenizer_sha256"], r"^sha256:[0-9a-f]{64}$")
        item2 = runtime_semantic_binding(ToyLM(), ToyTokenizer())
        self.assertEqual(item, item2)


class TestSteeringPairedReplay(unittest.TestCase):
    def test_rng_does_not_leak_into_caller(self):
        torch.manual_seed(1934)
        before = torch.get_rng_state().clone()
        with _scoped_generation_rng("cpu", 17):
            noise = torch.rand(5)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with _scoped_generation_rng("cpu", 17):
            self.assertTrue(torch.equal(noise, torch.rand(5)))

    def test_replay_and_wilson_bounds(self):
        torch.manual_seed(1949)
        before = torch.get_rng_state().clone()
        report = run_ab_experiment(
            ToyLM().eval(), ToyTokenizer(), recipe(dose=5),
            prompts=["a", "b", "c"], scales=[0.25, 1.0],
            max_new_tokens=1, seed=11,
        )
        self.assertTrue(all(x["baseline_replay_consistent"] for x in report["trials"]))
        self.assertTrue(report["generation"]["baseline_replay_verified"])
        self.assertTrue(report["generation"]["rng_state_isolated"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        lo, hi = report["summary"][1]["change_rate_ci95"]
        self.assertTrue(0 <= lo <= hi <= 1)
        self.assertEqual(len(report["trials"][0]["prompt_sha256"]), 64)
        self.assertEqual(report["trials"][0]["prompt_sha256"], hashlib.sha256(b"a").hexdigest())

    def test_wilson_math_and_invalid_samples(self):
        self.assertAlmostEqual(_wilson_95(0, 100)[0], 0)
        self.assertAlmostEqual(_wilson_95(100, 100)[1], 1)
        self.assertTrue(0.2 < _wilson_95(5, 10)[0] < 0.3)
        self.assertTrue(0.7 < _wilson_95(5, 10)[1] < 0.8)
        with self.assertRaises(ValueError):
            _wilson_95(0, 0)


if __name__ == "__main__":
    unittest.main()

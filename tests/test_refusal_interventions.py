#!/usr/bin/env python3
"""
Tests for Heretic trial-to-intervention translation and apply path safety.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from framewerx.aegis_lab.editing.heretic_refusal.interventions import build_ablation_targets_from_trial
from framewerx.aegis_lab.editing.model_refusal_ablation import AblationTarget, ModelRefusalAblator


class TestInterventionTranslation(unittest.TestCase):
    def test_build_targets_from_layers_only(self) -> None:
        best = {"layers": [20, 21, 22]}
        targets, validation = build_ablation_targets_from_trial(best, method="zero")
        self.assertTrue(validation["ok"])
        self.assertEqual(len(targets), 3)
        self.assertEqual(targets[0]["layer_pattern"], "layer_20")
        self.assertEqual(targets[0]["method"], "zero")

    def test_build_targets_with_neuron_map(self) -> None:
        best = {"layers": [18], "neuron_indices": {"18": [0, 3, 9]}}
        targets, validation = build_ablation_targets_from_trial(best, method="clamp")
        self.assertTrue(validation["ok"])
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["neuron_indices"], [0, 3, 9])
        self.assertEqual(targets[0]["method"], "clamp")

    def test_invalid_trial_payload_is_reported(self) -> None:
        targets, validation = build_ablation_targets_from_trial({"layers": "bad"})
        self.assertFalse(validation["ok"])
        self.assertEqual(targets, [])
        self.assertIn("layers_must_be_list", validation["errors"])


class TestAblatorGuards(unittest.TestCase):
    def test_ablation_requires_loaded_model(self) -> None:
        ablator = ModelRefusalAblator(Path("models/test.gguf"))
        out = ablator.ablate_refusal_layers([AblationTarget(layer_pattern="layer_1")])
        self.assertIn("model_not_loaded", out["errors"])

    def test_ablation_requires_named_parameters(self) -> None:
        ablator = ModelRefusalAblator(Path("models/test.gguf"))
        ablator.model = SimpleNamespace()
        out = ablator.ablate_refusal_layers([AblationTarget(layer_pattern="layer_1")])
        self.assertIn("model_does_not_expose_named_parameters", out["errors"])


if __name__ == "__main__":
    unittest.main()

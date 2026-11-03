"""Small CPU fixtures: never evidence for trained-checkpoint behaviour."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_neurosurgery_neuron_steering import ToyLM, ToyTokenizer, recipe
from aegis_lab.editing.neurosurgery.neuron_steering import run_ab_experiment
from aegis_lab.editing.neurosurgery.steering_qualification import (
    preflight, qualify_loaded_model, validate_plan,
)


class QualificationTests(unittest.TestCase):
    def test_delivery_preservation_and_hook_removal(self):
        model, tokenizer = ToyLM(), ToyTokenizer()
        report = run_ab_experiment(model, tokenizer, recipe(), ["behaviour"], max_new_tokens=2)
        arm = report["trials"][0]["variants"][0]
        self.assertGreater(arm["activation_values_changed"]["0"], 0)
        self.assertTrue(report["generation"]["unhooked_replay_verified"])
        result = qualify_loaded_model(model, tokenizer, recipe(), report, ["independent keep"], "cpu")
        self.assertTrue(result["unhooked_logits_exactly_restored"])
        self.assertFalse(result["qualified"])
        self.assertGreater(result["preservation"][0]["mean_next_token_kl"], 0)
        self.assertFalse(result["preservation"][0]["preservation_gate_passed"])
        self.assertTrue(all(not m._forward_pre_hooks for m in model.modules()))

    def test_noop_and_overlap_rejected(self):
        model, tokenizer = ToyLM(), ToyTokenizer()
        report = run_ab_experiment(model, tokenizer, recipe(dose=0), ["test"])
        with self.assertRaisesRegex(ValueError, "nonzero delivered"):
            qualify_loaded_model(model, tokenizer, recipe(dose=0), report, ["keep"], "cpu")
        with self.assertRaisesRegex(ValueError, "overlap"):
            qualify_loaded_model(model, tokenizer, recipe(), report, ["test"], "cpu")

    def test_exception_removes_hooks(self):
        model = ToyLM()
        with patch.object(model, "generate", side_effect=RuntimeError("fixture failure")):
            with self.assertRaises(RuntimeError):
                run_ab_experiment(model, ToyTokenizer(), recipe(), ["test"])
        self.assertTrue(all(not m._forward_pre_hooks for m in model.modules()))

    def test_logit_restoration_failure(self):
        model, tokenizer = ToyLM(), ToyTokenizer()
        report = run_ab_experiment(model, tokenizer, recipe(), ["test"])
        from aegis_lab.editing.neurosurgery import neuron_causal
        original = neuron_causal._next_logits
        calls = []
        def mutate(*args, **kwargs):
            value = original(*args, **kwargs)
            calls.append(1)
            return value + (1 if len(calls) == 3 else 0)
        with patch.object(neuron_causal, "_next_logits", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "exactly restored"):
                qualify_loaded_model(model, tokenizer, recipe(), report, ["keep"], "cpu")

    def test_plan_disjointness_and_seed_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, prompts in {"a": ["alpha"], "b": ["beta"], "keep": ["gamma"], "excluded": ["delta"]}.items():
                (root / name).write_text('\n'.join(prompts))
            plan = {"model": "model", "recipe": "recipe", "rubric": "rubric", "keep": "keep",
                    "checkpoint_sha256": "sha256:" + "a" * 64, "revision": "b" * 40,
                    "excluded_workloads": ["excluded"], "runs": ["a", "b"],
                    "repeat_seeds": [42, 1042], "scale": 0.5}
            self.assertIn("run_0", validate_plan(plan, root))
            (root / "b").write_text(" ALPHA  ")
            with self.assertRaisesRegex(ValueError, "overlapping"):
                validate_plan(plan, root)
            (root / "b").write_text("beta")
            plan["repeat_seeds"] = [42, 170]
            with self.assertRaisesRegex(ValueError, "collide"):
                validate_plan(plan, root)

    def test_wrong_architecture_fails_without_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
            with self.assertRaisesRegex(ValueError, "architecture"):
                preflight(folder)

    def test_harness_uses_audited_operators_and_reporting(self):
        from aegis_lab.editing.neurosurgery.steering_qualification import main, MODEL_ID
        from aegis_lab.editing.neurosurgery.steering_provenance import checkpoint_binding
        from aegis_lab.editing.neurosurgery.cli import main as operator
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "model"
            checkpoint.mkdir()
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                (checkpoint / name).write_text('{}')
            for name, rows in {"a": ["alpha"], "b": ["beta"], "keep": ["gamma"], "excluded": ["delta"]}.items():
                (root / name).write_text('\n'.join(rows))
            (root / "rubric").write_text("fixture rubric")
            value = recipe()
            value["model"] = MODEL_ID
            (root / "recipe.json").write_text(json.dumps(value))
            plan = {"model": "model", "recipe": "recipe.json", "rubric": "rubric", "keep": "keep",
                    "checkpoint_sha256": checkpoint_binding(str(checkpoint))["checkpoint_sha256"],
                    "revision": "b" * 40, "excluded_workloads": ["excluded"],
                    "runs": ["a", "b"], "repeat_seeds": [42, 1042], "scale": 0.5}
            (root / "plan.json").write_text(json.dumps(plan))
            semantic = {"runtime_config_sha256": "sha256:" + "e" * 64,
                        "runtime_tokenizer_sha256": "sha256:" + "f" * 64}
            with patch('aegis_lab.editing.neurosurgery.steering_qualification.preflight', return_value={"fixture": True}), \
                 patch('aegis_lab.editing.neurosurgery.neuron_steering.load_local_model', side_effect=lambda *a: (ToyLM(), ToyTokenizer())), \
                 patch('aegis_lab.editing.neurosurgery.neuron_steering.runtime_semantic_binding', return_value=semantic):
                result = main(["--plan", str(root / "plan.json"), "--out", str(root / "output")])
            self.assertEqual(result["status"], "awaiting_independent_labels")
            self.assertEqual(len(result["runs"]), 4)
            self.assertFalse(result["qualified"])
            for row in result["runs"]:
                self.assertTrue(Path(row["ab_report"] + '.operator_audit.json').is_file())
                # Synthetic scores exercise plumbing only, never published as human judgements.
                ratings = [json.loads(line) for line in Path(row["labels"]).read_text().splitlines()]
                for rating in ratings:
                    rating.update(target_score=0.2 if rating["scale"] == 0 else 0.8, keep_score=1.0)
                Path(row["labels"]).write_text('\n'.join(json.dumps(r) for r in ratings))
                operator(row["evaluate_argv"] + ["--min-prompts", "1"])
            operator(result["replicate_argv"] + ["--min-prompts", "2", "--min-per-run", "1", "--iterations", "200"])
            self.assertTrue((root / "output" / "replication.json.operator_audit.json").is_file())
            with self.assertRaisesRegex(ValueError, "new output"):
                main(["--plan", str(root / "plan.json"), "--out", str(root / "output")])

    def test_stock_runtime_resolves_target_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
                      "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 32,
                                      "hidden_size": 2560, "intermediate_size": 9216, "vocab_size": 248320}}
            (Path(folder) / "config.json").write_text(json.dumps(config))
            result = preflight(folder)
            self.assertEqual(result["model_class"], "Qwen3_5ForConditionalGeneration")
            self.assertFalse(result["hardware_execution_verified"])

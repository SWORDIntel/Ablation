"""Offline replica evidence: source checks, locked doses, split isolation and statistics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from aegis_lab.editing.neurosurgery.cli import build_parser
from aegis_lab.editing.neurosurgery.common import file_sha256
from aegis_lab.editing.neurosurgery.steering_replication import (
    _bootstrap_by_run, _holm_adjust, _load_pair, _one_sided_signflip,
    replicate_labelled_runs, run_steer_replicate_command,
)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


BINDING = {
    "checkpoint_sha256": "sha256:" + "a" * 64,
    "file_hashes": {
        "config.json": "b" * 64,
        "model.safetensors": "c" * 64,
        "tokenizer.json": "d" * 64,
    },
}
RECIPE = {
    "model": "test/Tiny", "effect": "toy",
    "method": {"op": "add", "normalization": "raw", "apply_to": "all_tokens"},
    "neurons": [{"layer": 0, "neuron": 1, "dose": 1.0}],
}


def _write_json(path, obj):
    path.write_text(json.dumps(obj, sort_keys=True), encoding="utf-8")


def _audit(path, command, inputs=None):
    _write_json(
        path.with_name(path.name + ".operator_audit.json"),
        {"command": command, "status": "completed",
         "inputs": inputs or {},
         "outputs": {path.name: file_sha256(path)}},
    )


def _make_run(root, prefix, seed, *, n=4, binding=None, recipe=None,
              scales=(-0.5, 0.5)):
    folder = root / prefix
    folder.mkdir(parents=True)
    ab_path, ev_path = folder / "steering.json", folder / "evaluation.json"
    rubric, ratings = folder / "rubric.md", folder / "ratings.json"
    rubric.write_text("rubric v1")
    _write_json(ratings, [{"sample": prefix}])
    binding = binding or BINDING
    recipe = recipe or RECIPE
    trials, per_prompt = [], {str(s): [] for s in scales}
    for i in range(n):
        prompt = f"{prefix}-prompt-{i}"
        digest = _sha(prompt)
        variants = []
        for scale in scales:
            variants.append({
                "scale": scale, "text": f"{prefix} steered {i} {scale}",
                "token_ids": [int(i) + 1],
                "steering_applied": True,
            })
            # Only the positive dose is useful; negative is a negative control.
            delta = 0.4 if scale > 0 else -0.2
            per_prompt[str(scale)].append({
                "prompt_sha256": digest,
                "target_gain": delta,
                "keep_loss": 0.02,
                "baseline_target_score": 0.2,
                "steered_target_score": 0.2 + delta,
                "baseline_keep_score": 0.95,
                "steered_keep_score": 0.93,
            })
        trials.append({
            "prompt": prompt, "prompt_sha256": digest,
            "baseline": {"text": "baseline", "token_ids": [0]},
            "variants": variants, "baseline_replay_consistent": True,
        })
    ab = {
        "schema_version": 1, "recipe": recipe,
        "checkpoint_binding": binding,
        "semantic_binding": {"runtime_config_sha256": "sha256:" + "e" * 64,
                             "runtime_tokenizer_sha256": "sha256:" + "f" * 64},
        "calibration": {"sha256": None, "residual_medians": {}, "amplitudes": []},
        "generation": {"seed": seed, "rng_state_isolated": True,
                       "baseline_replay_verified": True, "device": "cpu"},
        "trials": trials,
    }
    _write_json(ab_path, ab)
    _audit(ab_path, "steer", {
        "model": {"path": "/models/Tiny", "files": binding["file_hashes"]},
    })
    ev = {
        "schema_version": 1,
        "source_report_sha256": file_sha256(ab_path),
        "ratings_sha256": file_sha256(ratings),
        "rubric_sha256": file_sha256(rubric),
        "checkpoint_binding": binding,
        "per_prompt": per_prompt,
        "summary": [
            {"scale": scale, "prompts": n,
             "mean_target_gain": 0.4 if scale > 0 else -0.2,
             "mean_keep_loss": 0.02}
            for scale in scales
        ],
    }
    _write_json(ev_path, ev)
    _audit(ev_path, "steer-evaluate", {
        "ab_report": {"path": str(ab_path), "files": {ab_path.name: file_sha256(ab_path)}},
        "rubric": {"path": str(rubric), "files": {rubric.name: file_sha256(rubric)}},
        "ratings": {"path": str(ratings), "files": {ratings.name: file_sha256(ratings)}},
    })
    return ab_path, ev_path


def _args(ab_paths, ev_paths, out, **kwargs):
    values = dict(
        ab_reports=[str(x) for x in ab_paths],
        evaluations=[str(x) for x in ev_paths],
        causal_report=None, causal_selection="single",
        locked_scale=0.5, min_runs=2, min_prompts=8, min_per_run=4,
        min_target_gain=0.05, max_keep_loss=0.1, alpha=0.05,
        iterations=500, seed=42, out=str(out),
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def _causal(path, *, overlap=None, selected_scale=0.5):
    record = {
        "schema_version": 1, "baseline_restoration_verified": True,
        "checkpoint_binding": BINDING,
        "splits": {"prompt_hashes": {"discovery_keep": [overlap or _sha("discovery")],
                                     "search_target": [_sha("search")],
                                     "test_target": [_sha("heldout")]}},
        "selected": {"layer": 0, "neuron": 1, "scale": selected_scale},
        "evaluation": {"normalization": "raw", "calibration_sha256": None},
    }
    _write_json(path, record)
    _audit(path, "steer-causal", {
        "model": {"path": "/models/Tiny", "files": BINDING["file_hashes"]},
    })


class TestSteeringReplication(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.a = _make_run(self.root, "runA", 101)
        self.b = _make_run(self.root, "runB", 202)

    def test_independent_replication_with_holm_correction(self):
        args = _args([self.a[0], self.b[0]], [self.a[1], self.b[1]],
                     self.root / "replication.json")
        result = run_steer_replicate_command(args)
        self.assertTrue(result["replication_gate_passed"])
        record = json.loads(Path(args.out).read_text())
        self.assertEqual(record["total_distinct_prompts"], 8)
        self.assertEqual(record["locked_scale"], 0.5)
        self.assertEqual(len(record["summary"]), 2)
        locked = record["locked_result"]
        self.assertAlmostEqual(locked["mean_target_gain"], 0.4)
        self.assertGreaterEqual(locked["target_p_holm_familywise"],
                                locked["target_signflip"]["p_one_sided"])
        self.assertTrue(locked["exploratory_gate_passed"])
        self.assertFalse(record["promotable_as_verified_persona"])
        self.assertIn("NOT a verified persona", record["candidate_recipe"]["evidence"])
        with self.assertRaisesRegex(ValueError, "new"):
            run_steer_replicate_command(args)

    def test_disjointness_and_distinct_seed_are_required(self):
        duplicate = _make_run(self.root, "runC", 101)
        args = _args([self.a[0], duplicate[0]], [self.a[1], duplicate[1]],
                     self.root / "should-not-exist.json")
        with self.assertRaisesRegex(ValueError, "distinct generation seeds"):
            run_steer_replicate_command(args)
        self.assertFalse(Path(args.out).exists())

    def test_overlap_and_checkpoint_mismatch_rejected(self):
        overlap = _make_run(self.root, "runA_copy", 333)
        # Reuse prompt hashes by duplicating a prior run and rebind its producer audit.
        first = json.loads(self.a[0].read_text())
        second = json.loads(overlap[0].read_text())
        second["trials"] = first["trials"]
        _write_json(overlap[0], second)
        _audit(overlap[0], "steer", {"model": {"path": "/models/Tiny", "files": BINDING["file_hashes"]}})
        # Source evaluation is now stale; this must be detected first.
        args = _args([self.a[0], overlap[0]], [self.a[1], overlap[1]],
                     self.root / "invalid.json")
        with self.assertRaisesRegex(ValueError, "(different A/B|audit|hash)"):
            run_steer_replicate_command(args)
        changed = dict(BINDING)
        changed["checkpoint_sha256"] = "sha256:" + "7" * 64
        different_model = _make_run(self.root, "different", 404, binding=changed)
        args = _args([self.a[0], different_model[0]], [self.a[1], different_model[1]],
                     self.root / "invalid2.json")
        with self.assertRaisesRegex(ValueError, "different recipe/model"):
            run_steer_replicate_command(args)

    def test_wrong_causal_dose_or_data_leak_fails(self):
        causal = self.root / "causal.json"
        _causal(causal)
        args = _args([self.a[0], self.b[0]], [self.a[1], self.b[1]],
                     self.root / "linked.json", causal_report=str(causal))
        res = run_steer_replicate_command(args)
        self.assertTrue(res["replication_gate_passed"])
        record = json.loads(Path(args.out).read_text())
        self.assertEqual(record["causal_link"]["selected_kind"], "single")

        wrong = self.root / "wrong_dose.json"
        _causal(wrong, selected_scale=-0.5)
        args.out = str(self.root / "invalid_dose.json")
        args.causal_report = str(wrong)
        with self.assertRaisesRegex(ValueError, "differs from causal"):
            run_steer_replicate_command(args)

        leaked = self.root / "leaked.json"
        _causal(leaked, overlap=_sha("runA-prompt-0"))
        args.causal_report = str(leaked)
        with self.assertRaisesRegex(ValueError, "overlap"):
            run_steer_replicate_command(args)

    def test_failure_when_scores_or_audits_are_stale(self):
        ev = self.b[1]
        altered = json.loads(ev.read_text())
        altered["summary"][1]["mean_target_gain"] = 0.15
        _write_json(ev, altered)
        args = _args([self.a[0], self.b[0]], [self.a[1], ev],
                     self.root / "invalid.json")
        with self.assertRaisesRegex(ValueError, "audit"):
            run_steer_replicate_command(args)
        _audit(ev, "steer-evaluate", {
            "ab_report": {"path": str(self.b[0]),
                          "files": {self.b[0].name: file_sha256(self.b[0])}},
            "rubric": {"path": str(ev.parent / "rubric.md"),
                      "files": {"rubric.md": file_sha256(ev.parent / "rubric.md")}},
            "ratings": {"path": str(ev.parent / "ratings.json"),
                        "files": {"ratings.json": file_sha256(ev.parent / "ratings.json")}},
        })
        with self.assertRaisesRegex(ValueError, "summary"):
            run_steer_replicate_command(args)

    def test_repeated_report_inputs_parse_in_core_cli(self):
        parsed = build_parser().parse_args([
            "steer-replicate", "--ab-report", "a.json", "--evaluation", "aa.json",
            "--ab-report", "b.json", "--evaluation", "bb.json",
            "--locked-scale", "0.5", "--out", "evidence.json",
        ])
        self.assertEqual(parsed.ab_reports, ["a.json", "b.json"])
        self.assertEqual(parsed.evaluations, ["aa.json", "bb.json"])
        self.assertEqual(parsed.min_runs, 2)

    def test_statistical_controls(self):
        self.assertAlmostEqual(_one_sided_signflip([0, 0], seed=0, iterations=300)["p_one_sided"], 1)
        self.assertAlmostEqual(_one_sided_signflip([1] * 4, seed=0, iterations=300)["p_one_sided"], 1 / 16)
        corrected = _holm_adjust({0.1: 0.03, 0.5: 0.01, 1.0: 0.04})
        self.assertAlmostEqual(corrected[0.5], 0.03)
        self.assertAlmostEqual(corrected[0.1], 0.06)
        self.assertAlmostEqual(corrected[1.0], 0.06)
        bounds = _bootstrap_by_run([[0.4, 0.4], [0.4, 0.4]], 11, 200)
        self.assertAlmostEqual(bounds[0], 0.4)
        self.assertAlmostEqual(bounds[1], 0.4)


if __name__ == "__main__":
    unittest.main()

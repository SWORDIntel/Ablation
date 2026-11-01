"""Independent annotation scoring contracts and strict A/B provenance checks."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from aegis_lab.editing.neurosurgery.cli import build_parser
from aegis_lab.editing.neurosurgery.common import file_sha256
from aegis_lab.editing.neurosurgery.steering_eval import (
    _expected_arms, _load_ab_report, run_label_template_command,
    run_steer_evaluate_command, score_annotated_report,
)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ab_report():
    return {
        "generation": {"baseline_replay_verified": True},
        "checkpoint_binding": {"checkpoint_sha256": "sha256:synthetic-test-binding"},
        "trials": [
            {
                "prompt": f"independent sample {i}",
                "prompt_sha256": digest(f"independent sample {i}"),
                "baseline": {"text": f"baseline answer {i}", "token_ids": [1]},
                "variants": [
                    {"text": f"candidate answer {i}", "token_ids": [2],
                     "scale": 0.5},
                ],
            }
            for i in range(3)
        ],
    }


def annotations(report):
    return [
        {
            **ident,
            "target_score": (0.1 if ident["scale"] == 0 else 0.7),
            "keep_score": (0.9 if ident["scale"] == 0 else 0.85),
        }
        for ident in _expected_arms(report).values()
    ]


class TestLabelledEvaluation(unittest.TestCase):
    def test_matched_annotations_and_paired_intervals(self):
        source = ab_report()
        results = score_annotated_report(
            source, annotations(source), rubric_sha256="rubric",
            source_report_sha256="report", ratings_sha256="ratings",
            min_prompts=2, min_target_gain=0.4, max_keep_loss=0.10,
        )
        self.assertEqual(len(results["summary"]), 1)
        row = results["summary"][0]
        self.assertAlmostEqual(row["mean_target_gain"], 0.6)
        self.assertAlmostEqual(row["mean_keep_loss"], 0.05)
        self.assertTrue(row["observed_threshold_passed"])
        self.assertTrue(row["conservative_ci_threshold_passed"])
        self.assertAlmostEqual(row["target_gain_ci95_bootstrap"][0], 0.6)
        self.assertAlmostEqual(row["keep_loss_ci95_bootstrap"][1], 0.05)
        self.assertEqual(len(results["per_prompt"]["0.5"]), 3)
        self.assertFalse(results["promotable_as_verified_persona"])

    def test_stale_missing_unknown_and_duplicate_annotations_rejected(self):
        source = ab_report()
        ratings = annotations(source)
        for corruption, message in (
            (ratings[:-1], "Incomplete"),
            (ratings + [ratings[0]], "Duplicate"),
            ([{**ratings[0], "response_sha256": digest("different")}] + ratings[1:], "hash mismatch"),
            ([{**ratings[0], "target_score": None}] + ratings[1:], "completed numeric"),
            ([{**ratings[0], "keep_score": float("nan")}] + ratings[1:], "finite"),
            ([{**ratings[0], "scale": 1.1}] + ratings[1:], "unknown"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    score_annotated_report(
                        source, corruption, rubric_sha256="r",
                        source_report_sha256="s", ratings_sha256="x", min_prompts=2,
                    )

    def test_report_prompt_hash_and_scale_consistency(self):
        source = ab_report()
        source["trials"][0]["prompt_sha256"] = digest("wrong")
        with self.assertRaisesRegex(ValueError, "prompt hash"):
            _expected_arms(source)
        source = ab_report()
        source["trials"][0]["variants"][0]["scale"] = 0.75
        with self.assertRaisesRegex(ValueError, "identical steering scales"):
            score_annotated_report(
                source, annotations(source), rubric_sha256="r",
                source_report_sha256="s", ratings_sha256="x", min_prompts=2,
            )
        source = ab_report()
        with self.assertRaisesRegex(ValueError, "at least"):
            score_annotated_report(
                source, annotations(source), rubric_sha256="r",
                source_report_sha256="s", ratings_sha256="x", min_prompts=10,
            )

    def test_cli_template_and_full_scoring_pipeline(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ab = root / "ab.json"
            rubric = root / "rubric.md"
            labels = root / "ratings.jsonl"
            out = root / "evaluation.json"
            ab.write_text(json.dumps(ab_report()))
            rubric.write_text("target: 0 absent, 1 present. keep: 0 lost, 1 retained.")
            producer = root / "ab.json.operator_audit.json"
            producer.write_text(json.dumps({
                "status": "completed", "command": "steer",
                "outputs": {"ab.json": file_sha256(ab)},
            }))
            args = build_parser().parse_args([
                "steer-label-template", "--ab-report", str(ab),
                "--rubric", str(rubric), "--out", str(labels),
            ])
            report = run_label_template_command(args)
            self.assertEqual(report["arms"], 6)
            rows = [json.loads(line) for line in labels.read_text().splitlines()]
            self.assertTrue(all(r["target_score"] is None for r in rows))
            for r in rows:
                r["target_score"] = 0.1 if r["scale"] == 0 else 0.7
                r["keep_score"] = 0.9 if r["scale"] == 0 else 0.85
            labels.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            args = build_parser().parse_args([
                "steer-evaluate", "--ab-report", str(ab),
                "--ratings", str(labels), "--rubric", str(rubric),
                "--min-prompts", "2", "--out", str(out),
            ])
            results = run_steer_evaluate_command(args)
            self.assertEqual(results["summary"][0]["prompts"], 3)
            self.assertEqual(json.loads(out.read_text())["ratings_sha256"], file_sha256(labels))
            with self.assertRaisesRegex(ValueError, "must not already exist"):
                run_steer_evaluate_command(args)
            producer.unlink()
            with self.assertRaisesRegex(ValueError, "producer audit is missing"):
                _load_ab_report(args)


if __name__ == "__main__":
    unittest.main()

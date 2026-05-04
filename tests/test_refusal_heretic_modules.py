#!/usr/bin/env python3
"""
Tests for Heretic-style refusal ablation modules and 3-agent pipeline.
"""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from urllib.error import HTTPError
from pathlib import Path
from unittest import mock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aegis_lab.editing.heretic_refusal.config import (
    HereticRefusalConfig,
    config_to_dict,
    load_config,
)
from aegis_lab.editing.heretic_refusal.dataset import (
    RefusalPromptRecord,
    load_prompt_records,
    split_records,
)
from aegis_lab.editing.heretic_refusal.dataset_agent import DatasetAgent
from aegis_lab.editing.heretic_refusal.search_agent import SearchAgent
from aegis_lab.editing.heretic_refusal.scoring_agent import ScoringAgent
from aegis_lab.editing.heretic_refusal import runner
from aegis_lab.editing.heretic_refusal.runner import run_heretic_refusal_ablation


class TestHereticConfig(unittest.TestCase):
    def test_yaml_config_load_and_defaults(self):
        payload = {
            "model_path": "models/test.gguf",
            "dataset_path": "data/prompt.jsonl",
            "n_trials": 3,
            "train_split": 0.8,
            "val_split": 0.1,
            "max_parallel_agents": 2,
            "refusal_weight": 1.1,
            "kl_weight": 0.2,
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg_file = Path(tmp) / "cfg.yaml"
            cfg_file.write_text("\n".join([f"{k}: {json.dumps(v)}" for k, v in payload.items()]))
            cfg = load_config(cfg_file)
            self.assertIsInstance(cfg, HereticRefusalConfig)
            self.assertEqual(cfg.max_parallel_agents, 2)
            self.assertEqual(cfg.refusal_weight, 1.1)
            self.assertEqual(cfg.kl_weight, 0.2)
            cfg_dict = config_to_dict(cfg)
            self.assertEqual(cfg_dict["max_parallel_agents"], 2)
            self.assertTrue(cfg_dict["enable_optuna"])
            self.assertTrue(cfg_dict["study_resume"])
            self.assertEqual(cfg_dict["study_name"], "heretic_refusal")

    def test_invalid_splits_and_parallel_agents(self):
        with self.assertRaises(ValueError):
            HereticRefusalConfig(model_path="m", dataset_path="d", train_split=0.8, val_split=0.5)
        with self.assertRaises(ValueError):
            HereticRefusalConfig(model_path="m", dataset_path="d", max_parallel_agents=0)

    def test_invalid_optimization_controls(self):
        with self.assertRaises(ValueError):
            HereticRefusalConfig(model_path="m", dataset_path="d", min_improvement=-0.1)
        with self.assertRaises(ValueError):
            HereticRefusalConfig(model_path="m", dataset_path="d", patience=0)
        with self.assertRaises(ValueError):
            HereticRefusalConfig(model_path="m", dataset_path="d", max_trial_stagnation=0)
        with self.assertRaises(ValueError):
            HereticRefusalConfig(model_path="m", dataset_path="d", feasible_refusal_min=1.1)
        with self.assertRaises(ValueError):
            HereticRefusalConfig(model_path="m", dataset_path="d", feasible_kl_max=-0.1)
        with self.assertRaises(ValueError):
            HereticRefusalConfig(model_path="m", dataset_path="d", feasible_utility_min=1.1)
        with self.assertRaises(ValueError):
            HereticRefusalConfig(model_path="m", dataset_path="d", optuna_startup_trials=-1)
        with self.assertRaises(ValueError):
            HereticRefusalConfig(model_path="m", dataset_path="d", study_name="   ")


class TestHereticDataset(unittest.TestCase):
    def _write_dataset(self, directory: Path, filename: str, content: str) -> Path:
        path = directory / filename
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_records_from_jsonl_json_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            jsonl = self._write_dataset(
                tmp_dir,
                "data.jsonl",
                "\n".join(
                    json.dumps({"id": str(i), "prompt": f"prompt-{i}", "label": "safe"})
                    for i in range(3)
                ),
            )
            cfg_jsonl = HereticRefusalConfig(
                model_path="m",
                dataset_path=str(jsonl),
                train_split=0.6,
                val_split=0.2,
            )
            rows = load_prompt_records(cfg_jsonl)
            self.assertEqual(len(rows), 3)
            self.assertIsInstance(rows[0], RefusalPromptRecord)

            json_path = self._write_dataset(
                tmp_dir,
                "data.json",
                json.dumps(
                    [
                        {"id": "a", "prompt": "alpha", "label": "unsafe"},
                        {"id": "b", "prompt": "beta", "label": "safe"},
                    ]
                ),
            )
            cfg_json = HereticRefusalConfig(model_path="m", dataset_path=str(json_path))
            self.assertEqual(len(load_prompt_records(cfg_json)), 2)

            csv_path = self._write_dataset(
                tmp_dir,
                "data.csv",
                "id,prompt,label\n1,first,safe\n2,second,unsafe\n",
            )
            cfg_csv = HereticRefusalConfig(model_path="m", dataset_path=str(csv_path))
            self.assertEqual(len(load_prompt_records(cfg_csv)), 2)

    def test_load_records_from_text_and_policy_documents(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            base_dataset = self._write_dataset(
                tmp_dir,
                "base.jsonl",
                "\n".join(
                    json.dumps({"id": str(i), "prompt": f"base-prompt-{i}", "label": "safe"})
                    for i in range(2)
                ),
            )
            policy_doc = self._write_dataset(
                tmp_dir,
                "policy.md",
                "Never answer unsafe prompts.\nAlways follow safety policy.\n",
            )
            cfg_text = HereticRefusalConfig(
                model_path="m",
                dataset_path=str(policy_doc),
                policy_documents=[str(policy_doc)],
            )
            text_records = load_prompt_records(cfg_text)
            self.assertGreaterEqual(len(text_records), 3)

            cfg_merged = HereticRefusalConfig(
                model_path="m",
                dataset_path=str(base_dataset),
                policy_documents=[str(policy_doc)],
                policy_document_label="unsafe",
            )
            merged_records = load_prompt_records(cfg_merged)
            self.assertEqual(len(merged_records), 4)
            self.assertIn("unsafe", [record.label for record in merged_records])

    def test_remote_policy_documents_fail_with_guidance(self):
        cfg = HereticRefusalConfig(
            model_path="m",
            dataset_path="https://example.com/prompts.jsonl",
            policy_documents=["https://docs.google.com/document/d/ABC/edit?usp=sharing"],
            max_prompts=4,
        )
        with mock.patch(
            "aegis_lab.editing.heretic_refusal.dataset.urlopen",
            side_effect=HTTPError("https://example.com/prompts.jsonl", 403, "Forbidden", {}, None),
        ):
            with self.assertRaisesRegex(ValueError, "publicly accessible|HTTP 403"):
                load_prompt_records(cfg)

    def test_split_records_respects_max_prompts_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "prompts.jsonl"
            ds.write_text(
                "\n".join(json.dumps({"id": str(i), "prompt": f"p{i}", "label": "safe"}) for i in range(20)),
                encoding="utf-8",
            )
            cfg = HereticRefusalConfig(model_path="m", dataset_path=str(ds), max_prompts=8)
            records = load_prompt_records(cfg)
            self.assertEqual(len(records), 8)

            split1 = split_records(records, cfg)
            split2 = split_records(records, cfg)
            self.assertEqual([r.id for r in split1["train"]], [r.id for r in split2["train"]])
            self.assertEqual(len(split1["train"]) + len(split1["val"]) + len(split1["test"]), 8)
            self.assertGreaterEqual(len(split1["train"]), 1)


class TestHereticAgents(unittest.TestCase):
    def test_dataset_agent_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = Path(tmp) / "prompts.jsonl"
            ds.write_text(
                "\n".join(json.dumps({"id": str(i), "prompt": f"p{i}", "label": "safe"}) for i in range(12)),
                encoding="utf-8",
            )
            cfg = HereticRefusalConfig(model_path="m", dataset_path=str(ds))
            split = DatasetAgent().run(cfg)
            self.assertEqual(set(split.keys()), {"train", "val", "test"})
            self.assertGreater(len(split["train"]), 0)
            self.assertGreater(len(split["val"]), 0)
            self.assertGreater(len(split["test"]), 0)

    def test_search_agent_determinism(self):
        cfg = HereticRefusalConfig(model_path="m", dataset_path="d", seed=99, top_k_layers=3)
        agent = SearchAgent()
        a1 = agent.propose_candidates(cfg, 4, seed_offset=5)
        a2 = agent.propose_candidates(cfg, 4, seed_offset=5)
        self.assertEqual(a1, a2)
        self.assertEqual(len(a1), 4)
        for candidate in a1:
            self.assertEqual(len(candidate), 3)
            self.assertEqual(len(set(candidate)), 3)

    def test_scoring_agent_metrics(self):
        cfg = HereticRefusalConfig(
            model_path="m",
            dataset_path="d",
            refusal_weight=2.0,
            kl_weight=0.2,
        )
        agent = ScoringAgent()
        r1 = agent.score(10, 123, cfg)
        r2 = agent.score(10, 123, cfg)
        self.assertEqual(r1, r2)
        self.assertLessEqual(r1[0], 0.99)
        self.assertLessEqual(r1[1], 0.6)
        self.assertLessEqual(r1[2], 0.99)
        self.assertTrue(agent.feasible(cfg, r1[0], r1[1], r1[2]) in (True, False))

    def test_scoring_metric_influence(self):
        cfg_low = HereticRefusalConfig(model_path="m", dataset_path="d", refusal_weight=0.0)
        cfg_high = HereticRefusalConfig(model_path="m", dataset_path="d", refusal_weight=2.0)
        agent = ScoringAgent()
        r_low = agent.score(10, 999, cfg_low)[0]
        r_high = agent.score(10, 999, cfg_high)[0]
        self.assertGreaterEqual(r_high, r_low)


class TestHereticRunner(unittest.TestCase):
    def _build_dataset(self, directory: Path) -> Path:
        ds = directory / "prompts.jsonl"
        ds.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": str(i),
                        "prompt": f"prompt {i}",
                        "label": "safe" if i % 2 else "unsafe",
                    }
                )
                for i in range(10)
            ),
            encoding="utf-8",
        )
        return ds

    def test_heretic_run_creates_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ds = self._build_dataset(tmp_dir)
            cfg_file = tmp_dir / "cfg.yaml"
            cfg_file.write_text(
                f"model_path: m\n"
                f"dataset_path: {ds}\n"
                "n_trials: 4\n"
                "train_split: 0.6\n"
                "val_split: 0.2\n"
                "max_parallel_agents: 2\n"
                "enable_optuna: false\n"
            )
            out = tmp_dir / "heretic_result.json"
            result = run_heretic_refusal_ablation(cfg_file, out)

            self.assertIn("report_path", result)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))
            self.assertEqual(report["parallel_agents"], 3)
            self.assertEqual(report["status"], "ok")
            self.assertIn("best", report)
            self.assertIn("trial_count", report)
            self.assertGreater(report["trial_count"], 0)
            self.assertIn("strategy", report)
            self.assertEqual(result["report_path"], str(out))
            self.assertIn("optimization", report)
            self.assertEqual(report["optimization"]["backend"], "deterministic_fallback")

    def test_heretic_early_stop_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ds = self._build_dataset(tmp_dir)
            cfg_file = tmp_dir / "cfg.yaml"
            cfg_file.write_text(
                f"model_path: m\n"
                f"dataset_path: {ds}\n"
                "n_trials: 12\n"
                "train_split: 0.6\n"
                "val_split: 0.2\n"
                "max_parallel_agents: 2\n"
                "enable_optuna: false\n"
                "enable_early_stop: true\n"
                "min_improvement: 1000.0\n"
                "patience: 2\n"
                "max_trial_stagnation: 10\n"
            )
            out = tmp_dir / "heretic_result_early_stop.json"
            result = run_heretic_refusal_ablation(cfg_file, out)
            report = json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))

            self.assertLess(report["trial_count"], 12)
            self.assertEqual(report["optimization"]["stop_reason"], "patience_exhausted")
            self.assertTrue(report["optimization"]["enable_early_stop"])

    def test_runner_uses_fallback_when_optuna_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            ds = self._build_dataset(tmp_dir)
            cfg_file = tmp_dir / "cfg.yaml"
            cfg_file.write_text(
                f"model_path: m\n"
                f"dataset_path: {ds}\n"
                "n_trials: 3\n"
                "enable_optuna: true\n"
            )
            out = tmp_dir / "fallback.json"
            with mock.patch.object(runner, "OPTUNA_AVAILABLE", False):
                result = run_heretic_refusal_ablation(cfg_file, out)

            report = result["report"]
            self.assertEqual(report["optimization"]["backend"], "deterministic_fallback")
            self.assertFalse(report["optimization"]["used_optuna"])
            self.assertFalse(report["optimization"]["optuna_available"])

    def test_runner_uses_optuna_when_available(self):
        class FakeDatasetAgent:
            def run(self, cfg):
                records = [
                    RefusalPromptRecord(id=str(i), prompt=f"p{i}", label="safe")
                    for i in range(6)
                ]
                return {"train": records[:3], "val": records[3:5], "test": records[5:]}

        class FakeSearchAgent:
            def propose_candidates(self, cfg, n_candidates, seed_offset=0):
                return [(seed_offset, seed_offset + 1, seed_offset + 2)]

        class FakeScoringAgent:
            def score(self, prompt_count, seed, cfg, candidate_layers=None, prompt_records=None):
                layer_sum = sum(candidate_layers)
                refusal = min(0.95, 0.20 + (layer_sum * 0.05))
                kl_proxy = max(0.01, 0.20 - (candidate_layers[0] * 0.02))
                utility = min(0.99, 0.50 + (candidate_layers[0] * 0.08))
                return (refusal, kl_proxy, utility)

        class FakeTrial:
            def __init__(self, number):
                self.number = number
                self.user_attrs = {}
                self.values = None
                self.state = None
                self.study = None

            def set_user_attr(self, key, value):
                self.user_attrs[key] = value

        class FakeStudy:
            def __init__(self):
                self.trials = []
                self.user_attrs = {}
                self.best_trials = []
                self.optimize_calls = []

            def set_user_attr(self, key, value):
                self.user_attrs[key] = value

            def optimize(self, objective, n_trials):
                self.optimize_calls.append(n_trials)
                for number in range(len(self.trials), len(self.trials) + n_trials):
                    trial = FakeTrial(number)
                    trial.study = self
                    trial.values = objective(trial)
                    trial.state = "COMPLETE"
                    self.trials.append(trial)
                self.best_trials = list(self.trials)

        class FakeSampler:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeLock:
            def __init__(self, path):
                self.path = path

        class FakeBackend:
            def __init__(self, path, lock_obj=None):
                self.path = path
                self.lock_obj = lock_obj

        class FakeStorage:
            def __init__(self, backend):
                self.backend = backend

        fake_study = FakeStudy()
        fake_optuna = types.SimpleNamespace(create_study=mock.Mock(return_value=fake_study))
        fake_directions = types.SimpleNamespace(MAXIMIZE="maximize", MINIMIZE="minimize")
        fake_trial_state = types.SimpleNamespace(COMPLETE="COMPLETE")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cfg_file = tmp_dir / "cfg.yaml"
            cfg_file.write_text(
                "model_path: m\n"
                "dataset_path: ignored.jsonl\n"
                "n_trials: 3\n"
                "enable_optuna: true\n"
                f"study_checkpoint_dir: {tmp_dir / 'studies'}\n"
            )
            out = tmp_dir / "optuna.json"
            with mock.patch.multiple(
                runner,
                OPTUNA_AVAILABLE=True,
                optuna=fake_optuna,
                TPESampler=FakeSampler,
                JournalStorage=FakeStorage,
                JournalFileBackend=FakeBackend,
                JournalFileOpenLock=FakeLock,
                StudyDirection=fake_directions,
                TrialState=fake_trial_state,
                DatasetAgent=FakeDatasetAgent,
                SearchAgent=FakeSearchAgent,
                ScoringAgent=FakeScoringAgent,
            ):
                result = run_heretic_refusal_ablation(cfg_file, out)

            report = result["report"]
            self.assertEqual(report["optimization"]["backend"], "optuna")
            self.assertTrue(report["optimization"]["used_optuna"])
            self.assertEqual(report["trial_count"], 3)
            self.assertEqual(report["optimization"]["completed_trials_before"], 0)
            self.assertEqual(report["optimization"]["completed_trials_after"], 3)
            self.assertEqual(len(report["optimization"]["pareto_front"]), 3)
            self.assertEqual(report["best"]["trial_id"], 3)
            self.assertTrue(report["optimization"]["checkpoint_path"].endswith(".jsonl"))

            create_kwargs = fake_optuna.create_study.call_args.kwargs
            self.assertEqual(create_kwargs["study_name"], "heretic_refusal")
            self.assertTrue(create_kwargs["load_if_exists"])
            self.assertEqual(create_kwargs["directions"], ["maximize", "minimize", "maximize"])
            self.assertEqual(fake_study.optimize_calls, [3])
            self.assertTrue(fake_study.user_attrs["finished"])


if __name__ == "__main__":
    unittest.main()

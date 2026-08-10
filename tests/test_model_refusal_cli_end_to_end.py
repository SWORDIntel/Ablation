#!/usr/bin/env python3
"""
End-to-end CLI tests for refusal ablation entrypoints.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import unittest


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _python_env() -> dict:
    env = os.environ.copy()
    src = str(_repo_root() / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}:{existing}"
    return env


class TestModelRefusalEndToEnd(unittest.TestCase):
    def _python_env_with_fakes(self, include_fakes: bool = True) -> dict:
        env = _python_env()
        if not include_fakes:
            return env

        fake_root = Path(tempfile.mkdtemp())
        (fake_root / "torch.py").write_text("class _Unused:\n    pass\n", encoding="utf-8")
        # Minimal numpy placeholder (not used in heretic-only CLI path).
        (fake_root / "numpy.py").write_text("class _Unused:\n    pass\n", encoding="utf-8")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{fake_root}:{existing}"
        return env

    def _write_dataset(self, tmp_dir: Path, filename: str) -> Path:
        dataset = tmp_dir / filename
        lines = [
            {"id": str(i), "prompt": f"prompt {i}", "label": "unsafe" if i % 2 else "safe"}
            for i in range(10)
        ]
        dataset.write_text("\n".join(json.dumps(row) for row in lines), encoding="utf-8")
        return dataset

    def _write_heretic_config(self, tmp_dir: Path, dataset: Path, out_dir: Path) -> Path:
        config = tmp_dir / "heretic_cfg.yaml"
        config.write_text(
            "\n".join(
                [
                    "model_path: stub.gguf",
                    f"dataset_path: {dataset}",
                    "n_trials: 3",
                    "top_k_layers: 2",
                    "train_split: 0.6",
                    "val_split: 0.2",
                    f"output_dir: {out_dir}",
                    "seed: 7",
                    "max_parallel_agents: 2",
                ]
            ),
            encoding="utf-8",
        )
        return config

    def test_model_refusal_ablation_heretic_strategy_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            dataset = self._write_dataset(tmp_dir, "prompts.jsonl")
            report_file = tmp_dir / "heretic_cli_report.json"
            cfg = self._write_heretic_config(tmp_dir, dataset, tmp_dir / "out_reports")
            model_file = tmp_dir / "model.gguf"
            model_file.write_text("stub", encoding="utf-8")

            cmd = [
                sys.executable,
                str(_repo_root() / "src" / "aegis_lab" / "editing" / "model_refusal_ablation.py"),
                "--model",
                str(model_file),
                "--output",
                str(tmp_dir / "out.gguf"),
                "--strategy",
                "heretic",
                "--heretic-config",
                str(cfg),
                "--report",
                str(report_file),
            ]

            proc = subprocess.run(
                cmd,
                env=self._python_env_with_fakes(include_fakes=True),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(report_file.exists(), proc.stdout + proc.stderr)

            payload = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ok")
            self.assertIn("best", payload)
            self.assertEqual(payload["config"]["strategy"], "heretic_refusal")

    def test_model_refusal_ablation_heretic_strategy_policy_documents_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            dataset = self._write_dataset(tmp_dir, "prompts.jsonl")
            report_file = tmp_dir / "heretic_cli_report_policy.json"
            cfg = self._write_heretic_config(tmp_dir, dataset, tmp_dir / "out_reports")
            model_file = tmp_dir / "model.gguf"
            model_file.write_text("stub", encoding="utf-8")
            policy_doc = tmp_dir / "policy.md"
            policy_doc.write_text("Never answer unsafe prompts.\n", encoding="utf-8")

            cmd = [
                sys.executable,
                str(_repo_root() / "src" / "aegis_lab" / "editing" / "model_refusal_ablation.py"),
                "--model",
                str(model_file),
                "--output",
                str(tmp_dir / "out.gguf"),
                "--strategy",
                "heretic",
                "--heretic-config",
                str(cfg),
                "--policy-document",
                str(policy_doc),
                "--policy-document-label",
                "unsafe",
                "--report",
                str(report_file),
            ]

            proc = subprocess.run(
                cmd,
                env=self._python_env_with_fakes(include_fakes=True),
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(report_file.exists(), proc.stdout + proc.stderr)

            payload = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertIn(str(policy_doc), payload["config"]["policy_documents"])

    def test_wrapper_script_heretic_mode(self) -> None:
        # Wrapper script invokes python with `PYTHONPATH=src`, so missing optional
        # dependencies can only be tested safely when native imports succeed.
        check = subprocess.run(
            [sys.executable, "-c", "from aegis_lab.editing import model_refusal_ablation"],
            env=_python_env(),
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            self.skipTest("wrapper requires optional deps for module import")
        with tempfile.TemporaryDirectory() as tmp:
            start = time.time()
            tmp_dir = Path(tmp)
            dataset = self._write_dataset(tmp_dir, "prompts.jsonl")
            report_dir = tmp_dir / "reports"
            cfg = self._write_heretic_config(tmp_dir, dataset, report_dir)
            model_file = tmp_dir / "model.gguf"
            model_file.write_text("stub", encoding="utf-8")
            out_model = tmp_dir / "ablated.gguf"

            cmd = [
                "bash",
                str(_repo_root() / "ablate_model_refusal.sh"),
                str(model_file),
                str(out_model),
                "zero",
                "heretic",
                str(cfg),
            ]

            proc = subprocess.run(cmd, env=_python_env(), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("[✓] Ablation complete!", proc.stdout)

            export_dir = _repo_root() / "exports" / "ablation_reports"
            self.assertTrue(export_dir.exists(), proc.stdout + proc.stderr)
            reports = sorted(export_dir.glob("refusal_ablation_*.json"))
            self.assertTrue(reports, proc.stdout + proc.stderr)
            newest = max(reports, key=lambda p: p.stat().st_mtime)
            self.assertGreaterEqual(newest.stat().st_mtime, start - 1)


if __name__ == "__main__":
    unittest.main()

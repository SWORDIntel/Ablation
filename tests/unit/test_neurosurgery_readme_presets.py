"""Ensure the public README preset catalogue matches the executable registry."""
import re
from pathlib import Path
import unittest

from aegis_lab.editing.neurosurgery.neuron_steering import PRESETS


class TestDocumentedSteeringPresets(unittest.TestCase):
    def test_readme_catalogue_matches_registry(self):
        root = Path(__file__).resolve().parents[2]
        readme = (root / "README.md").read_text(encoding="utf-8")
        matcher = re.compile(
            r"^\| \x60(?P<id>qwen[a-z0-9-]+)\x60 \|[^|]*\|\s*"
            r"\x60(?P<doses>[^\x60]+)\x60\s*\|$"
        )
        doses = re.compile(r"L(\d+)N(\d+)([+-]\d+(?:\.\d+)?)")
        reported = {}
        for line in readme.splitlines():
            m = matcher.fullmatch(line)
            if m:
                key = m.group("id")
                self.assertNotIn(key, reported, f"README duplicates preset {key}")
                tokens = m.group("doses").split()
                matched = [doses.fullmatch(token) for token in tokens]
                self.assertTrue(all(matched), f"README malformed neuron dose {key}")
                reported[key] = [
                    {"layer": int(x.group(1)), "neuron": int(x.group(2)),
                     "dose": float(x.group(3))} for x in matched
                ]
        self.assertEqual(set(reported), set(PRESETS), "README and preset registry have drifted")
        self.assertEqual(len(reported), 19)
        for key, recipe in PRESETS.items():
            with self.subTest(preset=key):
                self.assertEqual(reported[key], recipe["neurons"])
                self.assertTrue(recipe["source"].startswith("https://"))
                self.assertIn("not locally reproduced", recipe["evidence"])
        self.assertIn("--max-pairs 3", readme)
        self.assertIn("docs/neurosurgery/NEURON_STEERING.md", readme)


if __name__ == "__main__":
    unittest.main()

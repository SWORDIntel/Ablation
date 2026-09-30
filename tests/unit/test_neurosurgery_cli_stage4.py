import unittest

from aegis_lab.editing.neurosurgery.cli import build_parser


class TestNeurosurgeryStage4CLI(unittest.TestCase):
    def test_optimize_command(self):
        args = build_parser().parse_args([
            "optimize",
            "--model", "m",
            "--keep", "k",
            "--out", "o",
            "--mlp-profile", "mlp.pt",
            "--mlp-ratios", "1,0.9,0.8",
            "--strategy", "frontier",
        ])
        self.assertEqual(args.cmd, "optimize")
        self.assertEqual(args.mlp_ratios, [1.0, 0.9, 0.8])
        self.assertEqual(args.strategy, "frontier")

    def test_apply_profile_is_optional_for_structural_plan(self):
        args = build_parser().parse_args([
            "apply",
            "--model", "m",
            "--plan", "p.yaml",
            "--out", "o",
        ])
        self.assertIsNone(args.profile)


if __name__ == "__main__":
    unittest.main()

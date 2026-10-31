import unittest

from aegis_lab.editing.neurosurgery.cli import build_parser


class TestNeurosurgeryCLI(unittest.TestCase):
    def test_parses_stage3_commands(self):
        args = build_parser().parse_args([
            "search-attention", "--model", "m", "--keep", "k", "--profile", "p", "--out", "o", "--ratios", "0.75,0.5"
        ])
        self.assertEqual(args.ratios, [0.75, 0.5])
        args = build_parser().parse_args([
            "search-moe", "--model", "m", "--keep", "k", "--profile", "p", "--out", "o"
        ])
        self.assertEqual(args.cmd, "search-moe")

    def test_parses_steering_a_b_arguments(self):
        args = build_parser().parse_args([
            "steer", "--model", "m", "--model-id", "Qwen/Qwen3.5-4B",
            "--preset", "qwen35-4b-evil-pair", "--workload", "questions.txt",
            "--scales", "-0.5,0.5,1", "--out", "report.json",
        ])
        self.assertEqual(args.cmd, "steer")
        self.assertEqual(args.scales, "-0.5,0.5,1")
        self.assertIsNone(args.config)
        self.assertEqual(args.max_new_tokens, 128)
        listing = build_parser().parse_args(["steer-presets", "--model-id", "Qwen/Qwen3.5-4B"])
        self.assertEqual(listing.cmd, "steer-presets")

    def test_parses_read_only_preview_command(self):
        args = build_parser().parse_args([
            "preview", "--model", "m", "--plan", "plan.yaml"
        ])
        self.assertEqual(args.cmd, "preview")
        self.assertIsNone(args.profile)
        self.assertEqual(args.device, "cpu")


if __name__ == "__main__":
    unittest.main()

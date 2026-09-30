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

    def test_parses_read_only_preview_command(self):
        args = build_parser().parse_args([
            "preview", "--model", "m", "--plan", "plan.yaml"
        ])
        self.assertEqual(args.cmd, "preview")
        self.assertIsNone(args.profile)
        self.assertEqual(args.device, "cpu")


if __name__ == "__main__":
    unittest.main()

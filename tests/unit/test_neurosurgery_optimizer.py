import unittest

from aegis_lab.editing.neurosurgery.optimizer import (
    CandidateState,
    choose_best_feasible,
    constrained_frontier_search,
    normalize_ratios,
    pareto_front,
)


class TestStage4Optimizer(unittest.TestCase):
    def test_normalize_ratios_includes_identity_and_orders_aggressive(self):
        self.assertEqual(normalize_ratios([0.8, 0.95, 0.8]), [1.0, 0.95, 0.8])

    def test_pareto_front_filters_dominated_and_infeasible(self):
        trials = [
            {"trial_id": 1, "feasible": True, "bytes_saved": 100, "estimated_macs_saved_per_token": 100, "mean_kl": 0.01},
            {"trial_id": 2, "feasible": True, "bytes_saved": 90, "estimated_macs_saved_per_token": 90, "mean_kl": 0.02},
            {"trial_id": 3, "feasible": True, "bytes_saved": 120, "estimated_macs_saved_per_token": 80, "mean_kl": 0.008},
            {"trial_id": 4, "feasible": False, "bytes_saved": 1000, "estimated_macs_saved_per_token": 1000, "mean_kl": 1.0},
        ]
        front = pareto_front(trials)
        ids = {row["trial_id"] for row in front}
        self.assertEqual(ids, {1, 3})

    def test_frontier_expands_only_passing_states(self):
        sizes = (3, 3, 1, 1)

        def estimate(state):
            aggressiveness = state.layer_level + state.mlp_level
            return {
                "bytes_saved": aggressiveness * 100,
                "estimated_macs_saved_per_token": aggressiveness * 10,
            }

        counter = {"value": 0}

        def evaluate(state):
            counter["value"] += 1
            score = state.layer_level + state.mlp_level
            return {
                "trial_id": counter["value"],
                "state": list(state.as_tuple()),
                "feasible": score <= 2,
                "bytes_saved": score * 100,
                "estimated_macs_saved_per_token": score * 10,
                "mean_kl": score * 0.01,
                "top1_agreement": 1.0,
            }

        trials = constrained_frontier_search(sizes, estimate, evaluate, max_trials=20)
        self.assertTrue(any(row["state"] == [2, 0, 0, 0] for row in trials))
        self.assertTrue(any(row["state"] == [1, 1, 0, 0] for row in trials))
        self.assertFalse(any(row["state"] == [2, 2, 0, 0] for row in trials))

        best = choose_best_feasible(trials)
        self.assertEqual(best["bytes_saved"], 200)


if __name__ == "__main__":
    unittest.main()

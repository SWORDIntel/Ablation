import copy
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn
import yaml

from aegis_lab.editing.neurosurgery.stage4d_hypertuning import (
    BaselineCache,
    CategoricalDim,
    FloatDim,
    HypertuningCandidate,
    HypertuningStudyResult,
    IntDim,
    MultiObjectiveCriteria,
    ObjectiveCriterion,
    ObjectiveDirection,
    SearchSpace,
    SearchStrategyKind,
    StudyBudget,
    StudyJournal,
    StudyStatus,
    TrialRecord,
    TrialStatus,
    apply_candidate_surgery,
    coarse_to_fine_search_candidates,
    compute_pareto_front,
    create_default_criteria,
    create_default_search_space,
    execute_candidate_neurosurgery,
    export_winning_plan,
    generate_operator_report,
    measure_model_latency,
    measure_peak_memory_mb,
    replay_winning_plan,
    run_hypertuning_study,
    seeded_random_search,
    select_winning_candidate,
)


class TinyMLP(nn.Module):
    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, out_features)
        self.up_proj = nn.Linear(in_features, out_features)
        self.down_proj = nn.Linear(out_features, in_features)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=32, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.mlp = TinyMLP(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        h = self.mlp(h)
        h = self.o_proj(h)
        return self.lm_head(h)


# ==============================================================================
# 1. Search Space & Candidate Configuration Tests
# ==============================================================================


class TestSearchSpaceAndCandidate(unittest.TestCase):
    def test_categorical_dim(self):
        dim = CategoricalDim("lora_r", [0, 4, 8, 16])
        self.assertEqual(dim.name, "lora_r")
        grid = dim.grid(count=3)
        self.assertEqual(len(grid), 3)

        with self.assertRaises(ValueError):
            CategoricalDim("empty", [])

    def test_int_dim(self):
        dim = IntDim("rank", low=0, high=16, step=4)
        grid = dim.grid(count=5)
        self.assertEqual(grid, [0, 4, 8, 12, 16])

        # Step validation
        with self.assertRaises(ValueError):
            IntDim("bad_step", low=0, high=10, step=0)
        with self.assertRaises(ValueError):
            IntDim("bad_bounds", low=10, high=2)

    def test_float_dim_linear_and_log(self):
        dim_lin = FloatDim("strength", low=0.0, high=1.0, step=0.25)
        grid_lin = dim_lin.grid(count=5)
        self.assertEqual(grid_lin, [0.0, 0.25, 0.5, 0.75, 1.0])

        dim_log = FloatDim("lr", low=1e-4, high=1e-2, log_scale=True)
        grid_log = dim_log.grid(count=3)
        self.assertEqual(len(grid_log), 3)
        self.assertAlmostEqual(grid_log[0], 1e-4, places=6)
        self.assertAlmostEqual(grid_log[-1], 1e-2, places=4)

        # Log scale with non-positive low raises
        with self.assertRaises(ValueError):
            FloatDim("bad_log", low=0.0, high=1.0, log_scale=True)

    def test_search_space_sample_and_grid(self):
        space = create_default_search_space()
        rng = torch.Generator().manual_seed(42)
        import random
        py_rng = random.Random(42)

        sample = space.sample(py_rng)
        self.assertIn("ablation_strength", sample)
        self.assertIn("lora_r", sample)
        self.assertIn("lr", sample)

        small_space = SearchSpace()
        small_space.add(CategoricalDim("a", [1, 2]))
        small_space.add(IntDim("b", 10, 20, step=10))
        grid = small_space.coarse_grid()
        self.assertEqual(len(grid), 4)

    def test_candidate_to_and_from_dict(self):
        cand = HypertuningCandidate(
            ablation_strength=0.75,
            preservation_rank=8,
            mlp_keep_ratio=0.85,
            lora_r=16,
            lora_alpha=32.0,
            lr=5e-5,
            training_steps=40,
            custom_params={"extra_param": 123},
        )
        d = cand.to_dict()
        self.assertEqual(d["ablation_strength"], 0.75)
        self.assertEqual(d["lora_r"], 16)
        self.assertEqual(d["custom_params"]["extra_param"], 123)

        cand2 = HypertuningCandidate.from_dict(d)
        self.assertEqual(cand2.ablation_strength, 0.75)
        self.assertEqual(cand2.lora_r, 16)
        self.assertEqual(cand2.custom_params["extra_param"], 123)

    def test_candidate_content_derived_id(self):
        cand1 = HypertuningCandidate(ablation_strength=0.5, lora_r=8)
        cand2 = HypertuningCandidate(ablation_strength=0.5, lora_r=8)
        cand3 = HypertuningCandidate(ablation_strength=1.0, lora_r=8)

        # Determinism and uniqueness
        self.assertEqual(cand1.compute_trial_id(), cand2.compute_trial_id())
        self.assertNotEqual(cand1.compute_trial_id(), cand3.compute_trial_id())

        # Salt modifies hash
        self.assertNotEqual(
            cand1.compute_trial_id(study_salt="study_A"),
            cand1.compute_trial_id(study_salt="study_B"),
        )


# ==============================================================================
# 2. Multi-Objective Criteria & Hard Thresholds Tests
# ==============================================================================


class TestMultiObjectiveCriteriaAndFeasibility(unittest.TestCase):
    def test_criterion_evaluations(self):
        crit_max = ObjectiveCriterion(
            name="keep_retention",
            direction=ObjectiveDirection.MAXIMIZE,
            hard_threshold=0.85,
        )
        self.assertTrue(crit_max.passes_threshold(0.90))
        self.assertTrue(crit_max.passes_threshold(0.85))
        self.assertFalse(crit_max.passes_threshold(0.80))
        self.assertTrue(crit_max.is_better(0.95, 0.90))

        crit_min = ObjectiveCriterion(
            name="mean_kl",
            direction=ObjectiveDirection.MINIMIZE,
            hard_threshold=0.05,
        )
        self.assertTrue(crit_min.passes_threshold(0.02))
        self.assertFalse(crit_min.passes_threshold(0.08))
        self.assertTrue(crit_min.is_better(0.01, 0.04))

    def test_feasibility_evaluation_passes_and_fails(self):
        criteria = create_default_criteria(
            min_keep_retention=0.85,
            min_drop_suppression=0.50,
            max_mean_kl=0.05,
        )

        passing_metrics = {
            "keep_retention": 0.90,
            "drop_suppression": 0.65,
            "mean_kl": 0.02,
            "bytes_saved": 1000.0,
            "peak_memory_mb": 120.0,
            "latency_ms": 15.0,
        }
        feasible, violations = criteria.evaluate_feasibility(passing_metrics)
        self.assertTrue(feasible)
        self.assertEqual(len(violations), 0)

        failing_metrics = {
            "keep_retention": 0.70,  # Fails threshold 0.85
            "drop_suppression": 0.40,  # Fails threshold 0.50
            "mean_kl": 0.08,  # Fails threshold 0.05
            "bytes_saved": 1000.0,
        }
        feasible, violations = criteria.evaluate_feasibility(failing_metrics)
        self.assertFalse(feasible)
        self.assertEqual(len(violations), 3)

    def test_dominance_logic(self):
        criteria = MultiObjectiveCriteria()
        criteria.add(ObjectiveCriterion("keep", ObjectiveDirection.MAXIMIZE))
        criteria.add(ObjectiveCriterion("kl", ObjectiveDirection.MINIMIZE))

        # A is strictly better than B in both
        metrics_a = {"keep": 0.90, "kl": 0.01}
        metrics_b = {"keep": 0.80, "kl": 0.05}
        self.assertTrue(criteria.dominates(metrics_a, metrics_b))
        self.assertFalse(criteria.dominates(metrics_b, metrics_a))

        # C is better in keep, D is better in kl -> neither dominates
        metrics_c = {"keep": 0.95, "kl": 0.08}
        metrics_d = {"keep": 0.85, "kl": 0.01}
        self.assertFalse(criteria.dominates(metrics_c, metrics_d))
        self.assertFalse(criteria.dominates(metrics_d, metrics_c))

        # Identical metrics do not dominate
        self.assertFalse(criteria.dominates(metrics_a, metrics_a))


# ==============================================================================
# 3. Pareto Frontier and Candidate Selection Tests
# ==============================================================================


class TestParetoFrontierAndSelection(unittest.TestCase):
    def setUp(self):
        self.criteria = MultiObjectiveCriteria()
        self.criteria.add(
            ObjectiveCriterion("keep_retention", ObjectiveDirection.MAXIMIZE, hard_threshold=0.80)
        )
        self.criteria.add(
            ObjectiveCriterion("bytes_saved", ObjectiveDirection.MAXIMIZE, hard_threshold=0.0)
        )
        self.criteria.add(
            ObjectiveCriterion("mean_kl", ObjectiveDirection.MINIMIZE, hard_threshold=0.05)
        )

    def test_pareto_front_calculation(self):
        trials = [
            TrialRecord(
                trial_id="t1",
                candidate=HypertuningCandidate(),
                metrics={"keep_retention": 0.95, "bytes_saved": 100.0, "mean_kl": 0.01},
                feasible=True,
            ),
            TrialRecord(
                trial_id="t2",
                candidate=HypertuningCandidate(),
                metrics={"keep_retention": 0.90, "bytes_saved": 80.0, "mean_kl": 0.02},  # Dominated by t1
                feasible=True,
            ),
            TrialRecord(
                trial_id="t3",
                candidate=HypertuningCandidate(),
                metrics={"keep_retention": 0.85, "bytes_saved": 500.0, "mean_kl": 0.03},  # More bytes saved
                feasible=True,
            ),
            TrialRecord(
                trial_id="t4",
                candidate=HypertuningCandidate(),
                metrics={"keep_retention": 0.50, "bytes_saved": 1000.0, "mean_kl": 0.10},  # Infeasible
                feasible=False,
            ),
        ]

        front = compute_pareto_front(trials, self.criteria, feasible_only=True)
        front_ids = {t.trial_id for t in front}
        self.assertEqual(front_ids, {"t1", "t3"})

    def test_select_winning_candidate(self):
        t1 = TrialRecord(
            trial_id="t1",
            candidate=HypertuningCandidate(lora_r=4),
            metrics={"keep_retention": 0.95, "bytes_saved": 100.0, "mean_kl": 0.01},
            feasible=True,
        )
        t2 = TrialRecord(
            trial_id="t2",
            candidate=HypertuningCandidate(lora_r=8),
            metrics={"keep_retention": 0.92, "bytes_saved": 300.0, "mean_kl": 0.02},
            feasible=True,
        )
        winner = select_winning_candidate([t1, t2], self.criteria)
        self.assertIsNotNone(winner)
        self.assertIn(winner.trial_id, ["t1", "t2"])

    def test_empty_pareto_front_returns_none(self):
        winner = select_winning_candidate([], self.criteria)
        self.assertIsNone(winner)


# ==============================================================================
# 4. Study Budget and Resource Limits Tests
# ==============================================================================


class TestStudyBudgetAndLimits(unittest.TestCase):
    def test_max_trials_limit(self):
        budget = StudyBudget(max_trials=5)
        limit_hit, msg = budget.check_limits(trials_completed=5, start_time=0.0)
        self.assertTrue(limit_hit)
        self.assertIn("Max trials budget reached", msg)

        limit_hit, _ = budget.check_limits(trials_completed=4, start_time=0.0)
        self.assertFalse(limit_hit)

    def test_max_time_seconds_limit(self):
        budget = StudyBudget(max_trials=100, max_time_seconds=1.0)
        # Simulate time 2 seconds past start
        limit_hit, msg = budget.check_limits(trials_completed=1, start_time=0.0)
        # Real time won't exceed unless time.time() - start_time >= 1.0
        import time
        t_start = time.time() - 2.0
        limit_hit, msg = budget.check_limits(trials_completed=1, start_time=t_start)
        self.assertTrue(limit_hit)
        self.assertIn("Max time budget exceeded", msg)

    def test_max_peak_memory_limit(self):
        budget = StudyBudget(max_trials=100, max_peak_memory_mb=500.0)
        limit_hit, msg = budget.check_limits(trials_completed=1, start_time=0.0, current_memory_mb=600.0)
        self.assertTrue(limit_hit)
        self.assertIn("Peak memory budget exceeded", msg)


# ==============================================================================
# 5. Search Strategies: Seeded Random, Coarse-to-Fine, and Successive Halving
# ==============================================================================


class TestSearchStrategies(unittest.TestCase):
    def test_seeded_random_search_reproducibility(self):
        space = SearchSpace()
        space.add(FloatDim("ablation_strength", 0.0, 1.0))
        space.add(CategoricalDim("lora_r", [4, 8, 16]))

        cands1 = seeded_random_search(space, count=5, seed=123)
        cands2 = seeded_random_search(space, count=5, seed=123)
        cands3 = seeded_random_search(space, count=5, seed=999)

        ids1 = [c.compute_trial_id() for c in cands1]
        ids2 = [c.compute_trial_id() for c in cands2]
        ids3 = [c.compute_trial_id() for c in cands3]

        self.assertEqual(ids1, ids2)
        self.assertNotEqual(ids1, ids3)

    def test_coarse_to_fine_generation(self):
        space = SearchSpace()
        space.add(FloatDim("ablation_strength", 0.0, 1.0, step=0.5))
        space.add(IntDim("preservation_rank", 0, 8, step=4))

        coarse_cands, gen_fine = coarse_to_fine_search_candidates(
            space=space,
            coarse_steps=2,
            top_k=1,
            fine_samples_per_center=2,
            seed=42,
        )
        self.assertGreater(len(coarse_cands), 0)

        mock_coarse_trials = [
            TrialRecord(
                trial_id="c1",
                candidate=coarse_cands[0],
                metrics={"keep_retention": 0.95, "mean_kl": 0.01},
                feasible=True,
                status=TrialStatus.COMPLETED.value,
            )
        ]
        fine_cands = gen_fine(mock_coarse_trials)
        self.assertEqual(len(fine_cands), 3)  # center + 2 perturbed

    def test_successive_halving_pruning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            call_counts = {"eval": 0}

            def mock_evaluator(cand: HypertuningCandidate, budget_steps: int, split: str):
                call_counts["eval"] += 1
                # Higher lora_r gives better retention
                retention = 0.80 + (cand.lora_r / 32.0) * 0.15
                return {
                    "keep_retention": retention,
                    "drop_suppression": 0.70,
                    "mean_kl": 0.02,
                    "bytes_saved": 500.0,
                }

            initial_cands = [
                HypertuningCandidate(lora_r=2),
                HypertuningCandidate(lora_r=4),
                HypertuningCandidate(lora_r=8),
                HypertuningCandidate(lora_r=16),
            ]

            budget = StudyBudget(
                max_trials=20,
                successive_halving_rungs=2,
                reduction_factor=2,
                min_budget=10,
                max_budget=20,
            )

            result = run_hypertuning_study(
                study_id="test_halving",
                evaluator_fn=mock_evaluator,
                initial_candidates=initial_cands,
                strategy=SearchStrategyKind.SUCCESSIVE_HALVING,
                budget=budget,
                study_dir=tmpdir,
            )

            self.assertEqual(result.status, StudyStatus.SUCCESS.value)
            self.assertGreater(result.pruned_trials, 0)
            pruned_records = [t for t in result.all_trials if t.pruned]
            self.assertTrue(all(pr.pruning_reason is not None for pr in pruned_records))


# ==============================================================================
# 6. Study Journal, Baseline Caching, and Resumption Tests
# ==============================================================================


class TestStudyJournalAndResumption(unittest.TestCase):
    def test_journal_record_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            journal_path = Path(tmpdir) / "journal.jsonl"
            journal = StudyJournal(journal_path)

            cand = HypertuningCandidate(ablation_strength=0.5)
            t_id = cand.compute_trial_id()
            trial = TrialRecord(
                trial_id=t_id,
                candidate=cand,
                rung=0,
                budget_allocated=10,
                metrics={"keep_retention": 0.92},
                feasible=True,
            )
            journal.record(trial)

            # Reload journal in new instance
            journal2 = StudyJournal(journal_path)
            loaded = journal2.get_completed(t_id, budget=10)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.trial_id, t_id)
            self.assertEqual(loaded.metrics["keep_retention"], 0.92)

    def test_study_resumption_skips_completed_trials(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_counter = {"calls": 0}

            def evaluator(cand: HypertuningCandidate, budget: int, split: str):
                eval_counter["calls"] += 1
                return {
                    "keep_retention": 0.90,
                    "drop_suppression": 0.60,
                    "bytes_saved": 100.0,
                }

            cands = [
                HypertuningCandidate(ablation_strength=0.1),
                HypertuningCandidate(ablation_strength=0.2),
            ]

            # Run 1: evaluate both candidates
            res1 = run_hypertuning_study(
                study_id="resumption_study",
                evaluator_fn=evaluator,
                initial_candidates=cands,
                study_dir=tmpdir,
            )
            self.assertEqual(eval_counter["calls"], 3)  # baseline + 2 trials
            self.assertEqual(res1.completed_trials, 2)

            # Run 2: resume in the exact same directory with identical candidates
            res2 = run_hypertuning_study(
                study_id="resumption_study",
                evaluator_fn=evaluator,
                initial_candidates=cands,
                study_dir=tmpdir,
            )
            # Evaluator calls should NOT increase because trials are loaded from journal & baseline from cache
            self.assertEqual(eval_counter["calls"], 3)
            self.assertEqual(res2.completed_trials, 2)

    def test_baseline_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "baseline.json"
            cache = BaselineCache(cache_path)

            calls = {"count": 0}

            def compute():
                calls["count"] += 1
                return {"keep_retention": 1.0, "mean_kl": 0.0}

            val1 = cache.get_or_compute("val", compute)
            val2 = cache.get_or_compute("val", compute)
            self.assertEqual(calls["count"], 1)
            self.assertEqual(val1, val2)


# ==============================================================================
# 7. Validation vs. Held-Out Test Split Isolation Tests
# ==============================================================================


class TestValidationVsTestSplitIsolation(unittest.TestCase):
    def test_test_split_evaluated_strictly_post_selection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            splits_seen_during_search = []
            test_eval_calls = []

            def search_evaluator(cand: HypertuningCandidate, budget: int, split: str):
                splits_seen_during_search.append(split)
                return {
                    "keep_retention": 0.90,
                    "drop_suppression": 0.60,
                    "bytes_saved": 200.0,
                    "mean_kl": 0.01,
                }

            def held_out_test_evaluator(cand: HypertuningCandidate):
                test_eval_calls.append(cand.compute_trial_id(study_salt="test_split_isolation"))
                return {"test_accuracy": 0.94}

            cands = [
                HypertuningCandidate(lora_r=4),
                HypertuningCandidate(lora_r=8),
            ]

            result = run_hypertuning_study(
                study_id="test_split_isolation",
                evaluator_fn=search_evaluator,
                test_evaluator_fn=held_out_test_evaluator,
                initial_candidates=cands,
                study_dir=tmpdir,
            )

            # Search MUST only see validation
            self.assertTrue(all(s == "validation" for s in splits_seen_during_search))
            # Test evaluator MUST be called exactly once post-selection for the winning trial
            self.assertEqual(len(test_eval_calls), 1)
            self.assertEqual(test_eval_calls[0], result.winning_trial.trial_id)
            self.assertIsNotNone(result.winning_trial.test_metrics)
            self.assertEqual(result.winning_trial.test_metrics["test_accuracy"], 0.94)


# ==============================================================================
# 8. Infeasible Study Handling & Operator Report Tests
# ==============================================================================


class TestInfeasibleStudyHandlingAndReports(unittest.TestCase):
    def test_infeasible_case_handled_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set impossible keep_retention threshold of 0.999
            criteria = create_default_criteria(min_keep_retention=0.999)

            def evaluator(cand: HypertuningCandidate, budget: int, split: str):
                return {
                    "keep_retention": 0.85,  # Violates 0.999
                    "drop_suppression": 0.70,
                    "bytes_saved": 100.0,
                }

            cands = [HypertuningCandidate(ablation_strength=0.5)]

            result = run_hypertuning_study(
                study_id="infeasible_study",
                evaluator_fn=evaluator,
                criteria=criteria,
                initial_candidates=cands,
                study_dir=tmpdir,
            )

            self.assertEqual(result.status, StudyStatus.INFEASIBLE_NO_CANDIDATE_PASSED.value)
            self.assertIsNone(result.winning_trial)
            self.assertIsNone(result.winning_plan_path)
            self.assertEqual(len(result.pareto_front), 0)

            # Verify report diagnosis
            report_text = generate_operator_report(result)
            self.assertIn("INFEASIBLE: NO CANDIDATE SATISFIED ALL HARD CONSTRAINTS", report_text)
            self.assertIn("Diagnostics for closest candidates", report_text)

    def test_operator_report_and_plan_export_on_feasible_study(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def evaluator(cand: HypertuningCandidate, budget: int, split: str):
                return {
                    "keep_retention": 0.92,
                    "drop_suppression": 0.65,
                    "bytes_saved": 500.0,
                    "mean_kl": 0.015,
                }

            cands = [HypertuningCandidate(ablation_strength=0.2, lora_r=8)]

            result = run_hypertuning_study(
                study_id="feasible_study",
                evaluator_fn=evaluator,
                initial_candidates=cands,
                study_dir=tmpdir,
            )

            self.assertEqual(result.status, StudyStatus.SUCCESS.value)
            self.assertIsNotNone(result.winning_trial)
            self.assertIsNotNone(result.winning_plan_path)

            report_file = Path(tmpdir) / "study_report.txt"
            self.assertTrue(report_file.exists())
            self.assertIn("STAGE 4D HYPERTUNING REPORT", report_file.read_text())

            plan_file = Path(result.winning_plan_path)
            self.assertTrue(plan_file.exists())
            loaded_plan = yaml.safe_load(plan_file.read_text())
            self.assertEqual(loaded_plan["version"], "4d.1")
            self.assertEqual(loaded_plan["surgery"]["ablation"]["strength"], 0.2)
            self.assertEqual(loaded_plan["recovery"]["lora_r"], 8)


# ==============================================================================
# 9. Winning Plan Replayability and Metric Reproduction Tests
# ==============================================================================


class TestWinningPlanReplay(unittest.TestCase):
    def test_replay_winning_plan_matches_within_tolerance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            def deterministic_eval(cand: HypertuningCandidate, budget: int, split: str):
                return {
                    "keep_retention": round(0.90 + cand.lora_r * 0.005, 6),
                    "drop_suppression": 0.75,
                    "mean_kl": 0.02,
                }

            winner = TrialRecord(
                trial_id="winning_t1",
                candidate=HypertuningCandidate(lora_r=8, ablation_strength=0.5),
                metrics={"keep_retention": 0.94, "drop_suppression": 0.75, "mean_kl": 0.02},
                feasible=True,
            )

            plan_path = export_winning_plan(
                study_id="replay_study",
                winning_trial=winner,
                baseline_val_metrics={"keep_retention": 1.0, "mean_kl": 0.0},
                baseline_test_metrics=None,
                out_path=Path(tmpdir) / "plan.yaml",
            )

            # Replay plan
            passed, details = replay_winning_plan(
                plan_path_or_dict=plan_path,
                evaluator_fn=deterministic_eval,
                tolerance=1e-3,
            )
            self.assertTrue(passed)
            self.assertEqual(len(details["discrepancies"]), 0)

    def test_replay_winning_plan_detects_intolerable_drift(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            winner = TrialRecord(
                trial_id="winning_t2",
                candidate=HypertuningCandidate(lora_r=8),
                metrics={"keep_retention": 0.95},
                feasible=True,
            )

            plan_path = export_winning_plan(
                study_id="drift_study",
                winning_trial=winner,
                baseline_val_metrics={},
                baseline_test_metrics=None,
                out_path=Path(tmpdir) / "plan.yaml",
            )

            # Drifting evaluator
            def drifting_eval(cand: HypertuningCandidate, budget: int, split: str):
                return {"keep_retention": 0.80}  # Deviates by 0.15

            passed, details = replay_winning_plan(
                plan_path_or_dict=plan_path,
                evaluator_fn=drifting_eval,
                tolerance=0.01,
            )
            self.assertFalse(passed)
            self.assertIn("keep_retention", details["discrepancies"])


# ==============================================================================
# 10. Hardware Latency and Memory Measurement Tests
# ==============================================================================


class TestHardwareLatencyAndMemory(unittest.TestCase):
    def test_measure_model_latency(self):
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
        batch = torch.randn(4, 16)
        lat_ms = measure_model_latency(model, batch, num_warmup=2, num_repeats=5)
        self.assertIsInstance(lat_ms, float)
        self.assertGreater(lat_ms, 0.0)

    def test_measure_peak_memory_mb(self):
        mem = measure_peak_memory_mb()
        self.assertIsInstance(mem, float)
        self.assertGreaterEqual(mem, 0.0)


# ==============================================================================
# 11. End-to-End Model Neurosurgery Execution Tests
# ==============================================================================


class TestEndToEndNeurosurgeryExecution(unittest.TestCase):
    def test_apply_candidate_surgery(self):
        model = TinyCausalLM(vocab_size=16, hidden_dim=8)
        old_weight = model.o_proj.weight.data.clone()

        direction = torch.randn(8)
        profile = {"direction": direction}

        cand = HypertuningCandidate(
            ablation_strength=0.5,
            target_modules=["o_proj"],
            norm_preserve=True,
            preserve_subspace=False,
        )

        apply_candidate_surgery(model, cand, directional_profiles=profile)
        new_weight = model.o_proj.weight.data

        # Weights must be modified by surgery
        self.assertFalse(torch.allclose(old_weight, new_weight))

    def test_execute_candidate_neurosurgery_with_recovery(self):
        model = TinyCausalLM(vocab_size=16, hidden_dim=8)
        keep_data = [{"input_ids": torch.randint(0, 16, (2, 4))}]
        sample_batch = {"input_ids": torch.randint(0, 16, (2, 4))}

        cand = HypertuningCandidate(
            ablation_strength=0.2,
            target_modules=["o_proj"],
            lora_r=4,
            lora_alpha=8.0,
            lora_target_modules=["mlp.down_proj"],
            lr=1e-3,
            training_steps=2,
        )

        def dummy_val(m):
            return {"keep_retention": 0.92, "drop_suppression": 0.65}

        metrics = execute_candidate_neurosurgery(
            model=model,
            candidate=cand,
            keep_data=keep_data,
            directional_profiles={"direction": torch.randn(8)},
            validation_evaluator=dummy_val,
            sample_batch_for_latency=sample_batch,
        )

        self.assertEqual(metrics["keep_retention"], 0.92)
        self.assertIn("latency_ms", metrics)
        self.assertIn("peak_memory_mb", metrics)


if __name__ == "__main__":
    unittest.main()

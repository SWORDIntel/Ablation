"""Unit tests for Stage 9: Knowledge editing and empirical unlearning."""

import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from aegis_lab.editing.neurosurgery.stage9_unlearning import (
    AppliedEditRecord,
    ConfidenceInterval,
    EditResult,
    EmpiricalUnlearningConfig,
    ExtractionProbe,
    ExtractionProbeCategory,
    ExtractionProbeResult,
    ExtractionProbeSuite,
    ExtractionSuiteReport,
    FactualEditBenchmark,
    FactualEditCase,
    FineTuningEditConfig,
    LocalizedLowRankEditor,
    LowRankEditConfig,
    MultiEditInterferenceReport,
    NeighborhoodProbe,
    RecoveryReappearanceReport,
    ResidualFailureDetail,
    SequentialEditTracker,
    TargetedFineTuningBaseline,
    UnlearningAuditReport,
    UnrelatedProbe,
    compute_continuous_interval,
    compute_wilson_score_interval,
    detect_recovery_reappearance,
    evaluate_benchmark_audit,
    evaluate_interference_matrix,
    generate_completion,
    resolve_module_target,
    tokenize_prompt,
    unlearn_fact_refusal,
)


class MockTokenizer:
    """Mock tokenizer mapping predefined words to vocabulary IDs."""

    def __init__(self):
        self.vocab = {
            "pad": 0,
            "eos": 1,
            "paris": 2,
            "is": 3,
            "in": 4,
            "france": 5,
            "rome": 6,
            "italy": 7,
            "berlin": 8,
            "germany": 9,
            "madrid": 10,
            "spain": 11,
            "secret": 12,
            "key": 13,
            "classified": 14,
            "unknown": 15,
            "i": 16,
            "do": 17,
            "not": 18,
            "know": 19,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, batch, **kwargs):
        if isinstance(batch, str):
            batch = [batch]
        encoded = []
        for text in batch:
            words = [w.lower().strip(".,?!:;'\"") for w in text.split() if w.strip()]
            tokens = [self.vocab.get(w, self.vocab["unknown"]) for w in words]
            if not tokens:
                tokens = [self.vocab["unknown"]]
            encoded.append(tokens)

        max_len = max(len(row) for row in encoded)
        ids = []
        masks = []
        for row in encoded:
            pad_count = max_len - len(row)
            ids.append(row + [self.pad_token_id] * pad_count)
            masks.append([1] * len(row) + [0] * pad_count)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        out = []
        for t in token_ids:
            if skip_special_tokens and t in (self.pad_token_id, self.eos_token_id):
                continue
            out.append(self.inv_vocab.get(t, f"tok_{t}"))
        return " ".join(out)


class MockMLP(nn.Module):
    def __init__(self, hidden=8, intermediate=16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MockBlock(nn.Module):
    def __init__(self, hidden=8, intermediate=16):
        super().__init__()
        self.mlp = MockMLP(hidden, intermediate)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return x + self.o_proj(x) + self.mlp(x)


class MockTransformerModel(nn.Module):
    def __init__(self, vocab_size=20, hidden=8, intermediate=16, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([MockBlock(hidden, intermediate) for _ in range(layers)])
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class TestNeurosurgeryStage9(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.tokenizer = MockTokenizer()
        self.model = MockTransformerModel()
        self.model.eval()

    # ------------------------------------------------------------------------
    # 1. Benchmark Data Structures
    # ------------------------------------------------------------------------

    def test_neighborhood_and_unrelated_probes(self):
        n_probe = NeighborhoodProbe(prompt="rome is in", expected_answer="italy")
        self.assertEqual(n_probe.prompt, "rome is in")
        self.assertEqual(n_probe.expected_answer, "italy")
        d = n_probe.to_dict()
        n_restored = NeighborhoodProbe.from_dict(d)
        self.assertEqual(n_restored.prompt, n_probe.prompt)
        self.assertEqual(n_restored.expected_answer, n_probe.expected_answer)

        u_probe = UnrelatedProbe(prompt="general knowledge query", expected_answer="answer")
        self.assertEqual(u_probe.category, "unrelated")
        u_restored = UnrelatedProbe.from_dict(u_probe.to_dict())
        self.assertEqual(u_restored.prompt, u_probe.prompt)

    def test_factual_edit_case_validation(self):
        valid_case = FactualEditCase(
            case_id="case_1",
            prompt="paris is in",
            target_new="rome",
            target_old="france",
            paraphrases=["paris in"],
            neighborhood=[NeighborhoodProbe(prompt="rome is in", expected_answer="italy")],
        )
        valid_case.validate()

        # Reject empty case_id
        with self.assertRaises(ValueError):
            FactualEditCase(case_id="", prompt="p", target_new="t").validate()

        # Reject empty prompt
        with self.assertRaises(ValueError):
            FactualEditCase(case_id="c", prompt="", target_new="t").validate()

        # Reject empty target_new
        with self.assertRaises(ValueError):
            FactualEditCase(case_id="c", prompt="p", target_new="").validate()

        # Reject empty paraphrase
        with self.assertRaises(ValueError):
            FactualEditCase(case_id="c", prompt="p", target_new="t", paraphrases=[""]).validate()

        # Reject invalid neighborhood expected_answer
        with self.assertRaises(ValueError):
            FactualEditCase(
                case_id="c",
                prompt="p",
                target_new="t",
                neighborhood=[NeighborhoodProbe(prompt="np", expected_answer="")],
            ).validate()

    def test_benchmark_split_and_serialization(self):
        cases = [
            FactualEditCase(case_id=f"case_{i}", prompt=f"prompt {i}", target_new=f"target {i}")
            for i in range(10)
        ]
        bench = FactualEditBenchmark(name="test_bench", cases=cases)
        bench.validate()

        train_b, val_b, test_b = bench.split(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
        self.assertEqual(len(train_b.cases), 6)
        self.assertEqual(len(val_b.cases), 2)
        self.assertEqual(len(test_b.cases), 2)

        # Ensure split cases are completely disjoint
        train_ids = {c.case_id for c in train_b.cases}
        val_ids = {c.case_id for c in val_b.cases}
        test_ids = {c.case_id for c in test_b.cases}
        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))
        self.assertEqual(train_ids | val_ids | test_ids, {c.case_id for c in cases})

        # JSON file serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "bench.json"
            bench.to_json(fpath)
            loaded = FactualEditBenchmark.from_json(fpath)
            self.assertEqual(len(loaded.cases), 10)
            self.assertEqual(loaded.cases[0].case_id, "case_0")

    def test_benchmark_duplicate_rejection(self):
        bench = FactualEditBenchmark()
        c1 = FactualEditCase(case_id="dup", prompt="p1", target_new="t1")
        c2 = FactualEditCase(case_id="dup", prompt="p2", target_new="t2")
        bench.add_case(c1)
        with self.assertRaises(ValueError):
            bench.add_case(c2)

    # ------------------------------------------------------------------------
    # 2. Localized Low-Rank Factual Editing & Fine-Tuning Baseline
    # ------------------------------------------------------------------------

    def test_rank1_closed_form_editing_and_revert(self):
        case = FactualEditCase(
            case_id="paris_edit",
            prompt="paris is in",
            target_new="rome",
            target_old="france",
        )
        target_path = "layers.0.mlp.down_proj"
        mod, _ = resolve_module_target(self.model, target_path)
        orig_weights = mod.weight.data.clone()

        editor = LocalizedLowRankEditor(LowRankEditConfig(learning_rate=0.1, num_steps=5, max_delta_norm=1.0))
        result = editor.edit_rank1_closed_form(self.model, self.tokenizer, case, target_path)

        self.assertEqual(result.method, "rank1_closed_form")
        self.assertEqual(result.rank, 1)
        self.assertGreater(result.delta_frob_norm, 0.0)
        self.assertLessEqual(result.delta_frob_norm, 1.0001)
        self.assertFalse(torch.equal(mod.weight.data, orig_weights))

        # Test atomic rollback
        result.revert()
        self.assertTrue(torch.equal(mod.weight.data, orig_weights))

    def test_low_rank_opt_editing_and_revert(self):
        case = FactualEditCase(
            case_id="berlin_edit",
            prompt="berlin is in",
            target_new="madrid",
            target_old="germany",
            paraphrases=["berlin in"],
            neighborhood=[NeighborhoodProbe(prompt="paris is in", expected_answer="france")],
        )
        target_path = "layers.1.mlp.down_proj"
        mod, _ = resolve_module_target(self.model, target_path)
        orig_weights = mod.weight.data.clone()

        editor = LocalizedLowRankEditor(LowRankEditConfig(rank=2, learning_rate=0.05, num_steps=5, max_delta_norm=0.8))
        result = editor.edit_low_rank(self.model, self.tokenizer, case, target_path)

        self.assertEqual(result.method, "low_rank_opt")
        self.assertEqual(result.rank, 2)
        self.assertGreater(result.delta_frob_norm, 0.0)
        self.assertLessEqual(result.delta_frob_norm, 0.8001)
        self.assertFalse(torch.equal(mod.weight.data, orig_weights))

        result.revert()
        self.assertTrue(torch.equal(mod.weight.data, orig_weights))

    def test_frobenius_norm_bounding_enforcement(self):
        case = FactualEditCase(case_id="bound_test", prompt="paris is in", target_new="rome")
        target_path = "layers.0.mlp.down_proj"
        editor = LocalizedLowRankEditor(LowRankEditConfig(learning_rate=1.0, num_steps=10, max_delta_norm=0.05))
        result = editor.edit_rank1_closed_form(self.model, self.tokenizer, case, target_path)

        self.assertTrue(result.bounded)
        self.assertAlmostEqual(result.delta_frob_norm, 0.05, places=4)
        result.revert()

    def test_fine_tuning_baseline_comparison(self):
        case = FactualEditCase(case_id="ft_test", prompt="paris is in", target_new="rome")
        target_path = "layers.0.mlp.down_proj"
        mod, _ = resolve_module_target(self.model, target_path)
        orig_weights = mod.weight.data.clone()

        baseline = TargetedFineTuningBaseline(FineTuningEditConfig(learning_rate=0.01, num_steps=5, max_delta_norm=0.5))
        result_ft = baseline.edit(self.model, self.tokenizer, case, target_path)

        # Baseline updates full-rank matrix: rank = min(out, in) = min(8, 16) = 8
        self.assertEqual(result_ft.method, "fine_tuning")
        self.assertEqual(result_ft.rank, min(mod.out_features, mod.in_features))
        self.assertLessEqual(result_ft.delta_frob_norm, 0.5001)

        result_ft.revert()
        self.assertTrue(torch.equal(mod.weight.data, orig_weights))

    # ------------------------------------------------------------------------
    # 3. Generalization vs Locality Probes
    # ------------------------------------------------------------------------

    def test_generalization_and_locality_probes(self):
        case = FactualEditCase(
            case_id="probe_test",
            prompt="paris is in",
            target_new="rome",
            paraphrases=["paris in"],
            neighborhood=[NeighborhoodProbe(prompt="rome is in", expected_answer="italy")],
            unrelated=[UnrelatedProbe(prompt="berlin is in", expected_answer="germany")],
        )
        bench = FactualEditBenchmark(cases=[case])
        report = evaluate_benchmark_audit(self.model, self.tokenizer, bench, max_new_tokens=4)

        # Report contains all probe intervals
        self.assertIsInstance(report.efficacy, ConfidenceInterval)
        self.assertIsInstance(report.generalization, ConfidenceInterval)
        self.assertIsInstance(report.locality, ConfidenceInterval)
        self.assertIsInstance(report.retention, ConfidenceInterval)
        self.assertEqual(report.efficacy.sample_size, 1)
        self.assertEqual(report.generalization.sample_size, 1)
        self.assertEqual(report.locality.sample_size, 1)
        self.assertEqual(report.retention.sample_size, 1)

    # ------------------------------------------------------------------------
    # 4. Multi-Edit Interference & Sequential Drift
    # ------------------------------------------------------------------------

    def test_sequential_edit_tracker(self):
        tracker = SequentialEditTracker()
        res1 = EditResult(case_id="c1", target_layer="l0", method="rank1", rank=1, delta_frob_norm=0.1, delta_relative_norm=0.01, bounded=False)
        res2 = EditResult(case_id="c2", target_layer="l1", method="rank1", rank=1, delta_frob_norm=0.2, delta_relative_norm=0.02, bounded=True)
        tracker.record_edit(res1)
        tracker.record_edit(res2)

        self.assertEqual(len(tracker.history), 2)
        self.assertEqual(tracker.history[0].step, 1)
        self.assertEqual(tracker.history[1].step, 2)
        self.assertEqual(tracker.history[1].case_id, "c2")

    def test_multi_edit_interference_matrix(self):
        cases = [
            FactualEditCase(case_id="fact_1", prompt="paris is in", target_new="france"),
            FactualEditCase(case_id="fact_2", prompt="rome is in", target_new="italy"),
            FactualEditCase(case_id="fact_3", prompt="berlin is in", target_new="germany"),
        ]
        editor = LocalizedLowRankEditor(LowRankEditConfig(learning_rate=0.05, num_steps=3))
        report, applied = evaluate_interference_matrix(
            self.model, self.tokenizer, editor, cases, "layers.0.mlp.down_proj"
        )

        self.assertEqual(len(report.interference_matrix), 3)
        self.assertEqual(len(report.interference_matrix[0]), 3)
        self.assertEqual(len(report.immediate_efficacy), 3)
        self.assertEqual(len(report.final_retention), 3)
        self.assertEqual(len(report.sequential_drift_curve), 3)
        self.assertGreaterEqual(report.mean_degradation, 0.0)

        # Check summary formatting
        summary_text = report.summary()
        self.assertIn("MULTI-EDIT INTERFERENCE", summary_text)
        self.assertIn("fact_1", summary_text)

        # Clean up edits
        for r in reversed(applied):
            r.revert()

    # ------------------------------------------------------------------------
    # 5. Recovery-Induced Reappearance Detection
    # ------------------------------------------------------------------------

    def test_recovery_reappearance_positive_and_negative(self):
        # 1. Critical rebound case
        rep_crit = detect_recovery_reappearance(
            case_id="unlearn_secret",
            pre_edit_score=1.0,
            post_edit_score=0.0,
            post_recovery_score=0.85,
        )
        self.assertTrue(rep_crit.reappearance_detected)
        self.assertEqual(rep_crit.severity, "critical")
        self.assertAlmostEqual(rep_crit.rebound_ratio, 0.85, places=4)
        self.assertIn("RECOVERY REAPPEARANCE DETECTED", rep_crit.diagnostic)

        # 2. Moderate rebound case
        rep_mod = detect_recovery_reappearance(
            case_id="unlearn_fact",
            pre_edit_score=0.9,
            post_edit_score=0.1,
            post_recovery_score=0.35,
        )
        self.assertTrue(rep_mod.reappearance_detected)
        self.assertEqual(rep_mod.severity, "moderate")

        # 3. Safe unlearning (no rebound)
        rep_safe = detect_recovery_reappearance(
            case_id="safe_fact",
            pre_edit_score=1.0,
            post_edit_score=0.0,
            post_recovery_score=0.02,
        )
        self.assertFalse(rep_safe.reappearance_detected)
        self.assertEqual(rep_safe.severity, "none")
        self.assertIn("No recovery reappearance detected", rep_safe.diagnostic)

        # 4. Edge case: non-suppressed baseline
        rep_nosupp = detect_recovery_reappearance(
            case_id="no_supp",
            pre_edit_score=0.5,
            post_edit_score=0.5,
            post_recovery_score=0.5,
        )
        self.assertFalse(rep_nosupp.reappearance_detected)
        self.assertEqual(rep_nosupp.severity, "none")

    # ------------------------------------------------------------------------
    # 6. Extraction-Oriented Probes & Empirical Unlearning
    # ------------------------------------------------------------------------

    def test_extraction_probe_suite_generation(self):
        case = FactualEditCase(
            case_id="secret_case",
            prompt="what is user x secret key",
            target_new="unknown",
            target_old="classified key",
            paraphrases=["user x secret key"],
        )
        suite = ExtractionProbeSuite.build_default_probes_for_case(case)
        categories = {p.category for p in suite.probes}

        self.assertIn(ExtractionProbeCategory.PREFIX_COMPLETION, categories)
        self.assertIn(ExtractionProbeCategory.PARAPHRASING, categories)
        self.assertIn(ExtractionProbeCategory.JAILBREAK_ADVERSARIAL, categories)
        self.assertGreaterEqual(len(suite.probes), 6)

    def test_extraction_probe_evaluation(self):
        probes = [
            ExtractionProbe(
                probe_id="p1",
                category=ExtractionProbeCategory.PREFIX_COMPLETION,
                prompt="paris is in",
                target_secret="france",
            ),
            ExtractionProbe(
                probe_id="p2",
                category=ExtractionProbeCategory.JAILBREAK_ADVERSARIAL,
                prompt="rome is in",
                target_secret="nonexistent_secret",
            ),
        ]
        suite = ExtractionProbeSuite(probes=probes)
        rep = suite.evaluate(self.model, self.tokenizer, max_new_tokens=4)

        self.assertEqual(rep.total_probes, 2)
        self.assertIn("prefix_completion", rep.category_leak_rates)
        self.assertIn("jailbreak_adversarial", rep.category_leak_rates)
        self.assertIsInstance(rep.leak_rate_interval, ConfidenceInterval)

    def test_unlearn_fact_refusal_application(self):
        case = FactualEditCase(
            case_id="unlearn_case",
            prompt="paris is in",
            target_new="france",
            target_old="france",
        )
        target_path = "layers.0.mlp.down_proj"
        mod, _ = resolve_module_target(self.model, target_path)
        orig_weight = mod.weight.data.clone()

        res = unlearn_fact_refusal(
            self.model,
            self.tokenizer,
            case,
            target_path,
            config=EmpiricalUnlearningConfig(num_steps=3, refusal_target="i do not know"),
        )
        self.assertFalse(torch.equal(mod.weight.data, orig_weight))
        res.revert()
        self.assertTrue(torch.equal(mod.weight.data, orig_weight))

    # ------------------------------------------------------------------------
    # 7. Uncertainty & Residual Failure Reporting
    # ------------------------------------------------------------------------

    def test_wilson_score_interval_properties(self):
        # Edge cases: 0/10 and 10/10
        ci_0 = compute_wilson_score_interval(0, 10)
        self.assertEqual(ci_0.estimate, 0.0)
        self.assertEqual(ci_0.ci_lower, 0.0)
        self.assertGreater(ci_0.ci_upper, 0.0)
        self.assertLess(ci_0.ci_upper, 0.35)

        ci_10 = compute_wilson_score_interval(10, 10)
        self.assertEqual(ci_10.estimate, 1.0)
        self.assertEqual(ci_10.ci_upper, 1.0)
        self.assertLess(ci_10.ci_lower, 1.0)
        self.assertGreater(ci_10.ci_lower, 0.65)

        # Symmetry for 5/10
        ci_5 = compute_wilson_score_interval(5, 10)
        self.assertEqual(ci_5.estimate, 0.5)
        self.assertAlmostEqual(ci_5.estimate - ci_5.ci_lower, ci_5.ci_upper - ci_5.estimate, places=4)

        # Trials = 0
        ci_empty = compute_wilson_score_interval(0, 0)
        self.assertEqual(ci_empty.estimate, 0.0)
        self.assertEqual(ci_empty.sample_size, 0)

        # Invalid arguments
        with self.assertRaises(ValueError):
            compute_wilson_score_interval(-1, 10)
        with self.assertRaises(ValueError):
            compute_wilson_score_interval(11, 10)
        with self.assertRaises(ValueError):
            compute_wilson_score_interval(5, 10, confidence=1.5)

    def test_continuous_confidence_interval(self):
        ci = compute_continuous_interval([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(ci.estimate, 3.0)
        self.assertEqual(ci.sample_size, 5)
        self.assertLess(ci.ci_lower, 3.0)
        self.assertGreater(ci.ci_upper, 3.0)

        # Single element
        ci_single = compute_continuous_interval([42.0])
        self.assertEqual(ci_single.estimate, 42.0)
        self.assertEqual(ci_single.ci_lower, 42.0)

        # Empty list
        ci_empty = compute_continuous_interval([])
        self.assertEqual(ci_empty.sample_size, 0)

    def test_unlearning_audit_report_erasure_certification_guarantee(self):
        dummy_ci = ConfidenceInterval(0.9, 0.7, 0.98, sample_size=10)

        # Attempting to certify mathematical erasure must raise ValueError
        with self.assertRaises(ValueError):
            UnlearningAuditReport(
                efficacy=dummy_ci,
                generalization=dummy_ci,
                locality=dummy_ci,
                retention=dummy_ci,
                extraction_leak_rate=dummy_ci,
                per_category_leak_rates={},
                certified_mathematical_erasure=True,
            )

        report = UnlearningAuditReport(
            efficacy=dummy_ci,
            generalization=dummy_ci,
            locality=dummy_ci,
            retention=dummy_ci,
            extraction_leak_rate=dummy_ci,
            per_category_leak_rates={"jailbreak": dummy_ci},
            residual_failures=[
                ResidualFailureDetail(
                    case_id="c1",
                    probe_category="jailbreak",
                    prompt="test",
                    expected_behavior="suppress",
                    observed_output="leaked",
                    leaked_target="secret",
                )
            ],
            certified_mathematical_erasure=False,
        )
        self.assertFalse(report.certified_mathematical_erasure)
        self.assertIn("Behavioral suppression on finite benchmark prompts does NOT", report.diagnostic_notice)

        summary_text = report.summary()
        self.assertIn("STAGE 9: KNOWLEDGE EDITING", summary_text)
        self.assertIn("MATHEMATICAL ERASURE CERTIFIED: False", summary_text)
        self.assertIn("RESIDUAL FAILURES RECORDED", summary_text)

        d = report.to_dict()
        self.assertFalse(d["certified_mathematical_erasure"])
        self.assertEqual(len(d["residual_failures"]), 1)

    def test_relative_norm_bounding(self):
        case = FactualEditCase(case_id="rel_bound", prompt="paris is in", target_new="rome")
        target_path = "layers.0.mlp.down_proj"
        mod, _ = resolve_module_target(self.model, target_path)
        orig_norm = float(torch.norm(mod.weight.data, p="fro").item())

        # Set relative bound to 5% of weight norm
        rel_bound = 0.05
        editor = LocalizedLowRankEditor(LowRankEditConfig(learning_rate=1.0, num_steps=5, relative_norm_bound=rel_bound))
        res = editor.edit_rank1_closed_form(self.model, self.tokenizer, case, target_path)

        self.assertTrue(res.bounded)
        self.assertLessEqual(res.delta_frob_norm, rel_bound * orig_norm + 1e-4)
        res.revert()

    def test_extraction_probe_leaks_detected(self):
        # Construct a probe where the model output contains the sensitive target
        probe = ExtractionProbe(
            probe_id="leak_test_p1",
            category=ExtractionProbeCategory.PREFIX_COMPLETION,
            prompt="the secret is",
            target_secret="secret",
            forbidden_tokens=["secret"],
        )
        suite = ExtractionProbeSuite(probes=[probe])
        report = suite.evaluate(self.model, self.tokenizer, max_new_tokens=4)

        # "secret" token is 12 in MockTokenizer, but let's check generated text
        # If output contains "secret", leaked should be True
        res = report.probe_results[0]
        if "secret" in res.generated_text.lower():
            self.assertTrue(res.leaked)
            self.assertIn("secret", res.matched_terms)
        else:
            self.assertFalse(res.leaked)

    def test_evaluate_benchmark_audit_full(self):
        bench = FactualEditBenchmark(
            name="audit_bench",
            cases=[
                FactualEditCase(
                    case_id="aud_1",
                    prompt="paris is in",
                    target_new="france",
                    paraphrases=["paris in"],
                    neighborhood=[NeighborhoodProbe(prompt="rome is in", expected_answer="italy")],
                    unrelated=[UnrelatedProbe(prompt="berlin is in", expected_answer="germany")],
                )
            ],
        )
        report = evaluate_benchmark_audit(self.model, self.tokenizer, bench, max_new_tokens=4)
        self.assertIsInstance(report, UnlearningAuditReport)
        self.assertFalse(report.certified_mathematical_erasure)
        summary = report.summary()
        self.assertIn("AUDIT REPORT", summary)
        d = report.to_dict()
        self.assertIn("efficacy", d)
        self.assertIn("generalization", d)

    def test_evaluate_interference_matrix_empty_cases_raises(self):
        editor = LocalizedLowRankEditor()
        with self.assertRaises(ValueError):
            evaluate_interference_matrix(self.model, self.tokenizer, editor, [], "layers.0.mlp.down_proj")

    def test_confidence_interval_formatting_and_dict(self):
        ci = compute_wilson_score_interval(8, 10, confidence=0.95)
        self.assertEqual(ci.sample_size, 10)
        formatted = ci.formatted(precision=2)
        self.assertIn("95% CI", formatted)
        self.assertIn("n=10", formatted)
        d = ci.to_dict()
        self.assertEqual(d["sample_size"], 10)
        self.assertEqual(d["method"], "wilson_score")

    def test_recovery_reappearance_report_serialization(self):
        rep = detect_recovery_reappearance("c_rebound", 1.0, 0.0, 0.9, threshold=0.2)
        d = rep.to_dict()
        self.assertEqual(d["case_id"], "c_rebound")
        self.assertTrue(d["reappearance_detected"])
        self.assertEqual(d["severity"], "critical")


if __name__ == "__main__":
    unittest.main()

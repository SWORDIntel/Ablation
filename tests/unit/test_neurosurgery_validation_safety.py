import os
import pickle
import tempfile
import unittest
from pathlib import Path

import torch

from aegis_lab.editing.neurosurgery.common import load_tensor_artifact, trust_remote_code_enabled
from aegis_lab.editing.neurosurgery.validate import (
    assert_compatible_tokenizers,
    compare_logprobs,
)


class UnsafePayload:
    pass


class FakeTokenizer:
    def __init__(self, vocab, special_tokens=None):
        self._vocab = vocab
        self.special_tokens_map = special_tokens or {}

    def get_vocab(self):
        return self._vocab


class TestNeurosurgeryValidationSafety(unittest.TestCase):
    def test_tensor_artifact_loads_with_restricted_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.pt"
            torch.save({"indices": torch.tensor([1, 3]), "version": 1}, path)
            value = load_tensor_artifact(path)
        self.assertEqual(value["version"], 1)
        self.assertEqual(value["indices"].tolist(), [1, 3])

    def test_restricted_loader_rejects_custom_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsafe.pt"
            torch.save({"value": UnsafePayload()}, path)
            with self.assertRaises((RuntimeError, pickle.UnpicklingError)):
                load_tensor_artifact(path)

    def test_remote_code_requires_explicit_environment_opt_in(self):
        prior = os.environ.pop("AEGIS_TRUST_REMOTE_CODE", None)
        try:
            self.assertFalse(trust_remote_code_enabled())
            os.environ["AEGIS_TRUST_REMOTE_CODE"] = "1"
            self.assertTrue(trust_remote_code_enabled())
        finally:
            if prior is None:
                os.environ.pop("AEGIS_TRUST_REMOTE_CODE", None)
            else:
                os.environ["AEGIS_TRUST_REMOTE_CODE"] = prior

    def test_compare_rejects_empty_or_malformed_scores(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            compare_logprobs([], [])
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            compare_logprobs([torch.tensor([0.0, -1.0])], [torch.tensor([0.0])])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_logprobs([torch.tensor([0.0, float("nan")])], [torch.tensor([0.0, -1.0])])

    def test_compare_reports_finite_metrics(self):
        scores = [torch.log_softmax(torch.tensor([2.0, 1.0, -1.0]), dim=0)]
        result = compare_logprobs(scores, scores)
        self.assertEqual(result["samples"], 1)
        self.assertAlmostEqual(result["mean_kl"], 0.0, places=6)
        self.assertEqual(result["top1_agreement"], 1.0)

    def test_tokenizer_vocab_and_special_tokens_must_match(self):
        base = FakeTokenizer({"a": 0, "b": 1}, {"eos_token": "b"})
        same = FakeTokenizer({"a": 0, "b": 1}, {"eos_token": "b"})
        assert_compatible_tokenizers(base, same)
        with self.assertRaisesRegex(ValueError, "vocabularies"):
            assert_compatible_tokenizers(base, FakeTokenizer({"a": 0, "c": 1}, {"eos_token": "c"}))
        with self.assertRaisesRegex(ValueError, "special tokens"):
            assert_compatible_tokenizers(base, FakeTokenizer({"a": 0, "b": 1}, {"eos_token": "a"}))


if __name__ == "__main__":
    unittest.main()

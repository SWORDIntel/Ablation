"""Exercise core search, optimizer preview and saved HF checkpoint reload offline."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from aegis_lab.editing.neurosurgery.apply import run_apply
from aegis_lab.editing.neurosurgery.mlp import run_mlp_profile, run_mlp_search
from aegis_lab.editing.neurosurgery.optimizer import run_optimizer
from aegis_lab.editing.neurosurgery.preview import build_preview
from aegis_lab.editing.neurosurgery.validate import run_validate


class TestCoreSurgeryWorkflow(unittest.TestCase):
    def test_mlp_search_optimizer_preview_apply_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp, torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            root = Path(tmp)
            base = root / "base"
            vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
                     "hello": 4, "world": 5, "one": 6, "two": 7, "three": 8, "four": 9}
            raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]",
                bos_token="[BOS]", eos_token="[EOS]",
            )
            tokenizer.save_pretrained(base)
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=len(vocab), hidden_size=32, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=8, pad_token_id=0, bos_token_id=2, eos_token_id=3,
            ))
            model.save_pretrained(base)
            source_weights = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            keep, drop, held_out = (root / name for name in ("keep.txt", "drop.txt", "test.txt"))
            keep.write_text("hello world\none two three\n")
            drop.write_text("four three\ntwo world four\n")
            held_out.write_text("hello two\nworld three four\n")
            run_mlp_profile(str(base), str(keep), str(drop), str(root / "profile"), device="cpu")
            profile = root / "profile" / "mlp_profile.pt"
            run_mlp_search(str(base), str(keep), str(profile), str(root / "search"),
                           [1.0, 0.5], device="cpu")
            search = json.loads((root / "search" / "mlp_search.json").read_text())
            self.assertTrue(any(row["channels_kept"] == 64 for row in search["results"]))
            # Deliberately loose fixture gates force a physical cut; this is not a quality test.
            run_optimizer(str(base), str(keep), str(root / "opt"), mlp_profile_path=str(profile),
                          mlp_ratios=[1.0, 0.5], max_mean_kl=10, min_top1_agreement=0,
                          max_trials=4, device="cpu")
            plan = root / "opt" / "optimized_plan.yaml"
            plan_bytes = plan.read_bytes()
            self.assertEqual(yaml.safe_load(plan_bytes)["version"], 4)
            preview = build_preview(model, str(plan))
            self.assertEqual(preview["summary"]["operation_count"], 6)
            self.assertEqual(plan.read_bytes(), plan_bytes)
            candidate = root / "candidate"
            run_apply(str(base), None, str(plan), str(candidate), device="cpu")
            reloaded = LlamaForCausalLM.from_pretrained(candidate)
            self.assertEqual(reloaded.config.intermediate_size, 64)
            self.assertEqual(reloaded.model.layers[0].mlp.down_proj.weight.shape, (32, 64))
            result = run_validate(str(base), str(candidate), str(held_out), 2, None, 10,
                                  str(root / "validation.json"), "cpu")
            self.assertTrue(result["passed"])
            reloaded.eval()
            with torch.no_grad():
                inputs = tokenizer("hello world", return_tensors="pt")
                inputs.pop("token_type_ids", None)
                generated = reloaded.generate(**inputs,
                                              max_new_tokens=2, do_sample=False, use_cache=True)
            self.assertGreater(generated.shape[-1], 2)
            restored_source = LlamaForCausalLM.from_pretrained(base)
            for name, tensor in restored_source.state_dict().items():
                self.assertTrue(torch.equal(tensor, source_weights[name]))


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import yaml

from aegis_lab.editing.neurosurgery.selectors import _write_selector_outputs


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.config = SimpleNamespace(model_type="llama", intermediate_size=6, num_hidden_layers=2)


class TestNeurosurgerySelectors(unittest.TestCase):
    def _compile(self, folder, spec):
        selector = folder / "selectors.yaml"
        selector.write_text(yaml.safe_dump(spec), encoding="utf-8")
        return _write_selector_outputs(TinyModel(), selector, str(folder / "compiled"), "tiny-model", None)

    def test_compiles_channel_ids_plan_and_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = self._compile(folder, {
                "version": 1,
                "mlp": {"keep_indices": {"0": [1, 2, 4], "1": [0, 3, 5]}},
            })
            plan = yaml.safe_load(Path(result["plan"]).read_text(encoding="utf-8"))
            self.assertEqual(plan["structured"]["mlp"]["enabled"], True)
            preview = yaml.safe_load(Path(result["preview"]).read_text(encoding="utf-8"))
            self.assertEqual(preview["summary"]["operation_count"], 6)
            payload = torch.load(Path(result["plan"]).parent / "mlp_selection.pt", weights_only=True)
            self.assertEqual([x.tolist() for x in payload["keep_indices"]], [[1, 2, 4], [0, 3, 5]])

    def test_rejects_incomplete_layer_map_and_keeps_output_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            selector = folder / "selectors.yaml"
            selector.write_text(yaml.safe_dump({
                "version": 1, "mlp": {"keep_indices": {"0": [1, 2, 4]}},
            }), encoding="utf-8")
            out = folder / "compiled"
            with self.assertRaisesRegex(ValueError, "every supported layer"):
                _write_selector_outputs(TinyModel(), selector, str(out), "tiny-model", None)
            self.assertFalse(out.exists())

    def test_rejects_unknown_selector_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "unsupported selector fields"):
                self._compile(Path(tmp), {"version": 1, "surprise": True})

    def test_refuses_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            out = folder / "compiled"
            out.mkdir()
            sentinel = out / "keep.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            selector = folder / "selectors.yaml"
            selector.write_text(yaml.safe_dump({"version": 1, "drop_layers": [0]}), encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _write_selector_outputs(TinyModel(), selector, str(out), "tiny-model", None)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()

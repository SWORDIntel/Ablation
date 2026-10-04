from __future__ import annotations

from typing import Optional, Union

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable

import torch

LOG = logging.getLogger("neurosurgery")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def load_prompts(path: Union[str, Path], max_prompts: Optional[int] = None) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    rows.append(obj)
                else:
                    rows.append(obj.get("text") or obj.get("prompt") or obj.get("content"))
        prompts = [x.strip() for x in rows if isinstance(x, str) and x.strip()]
    elif p.suffix == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError("JSON prompt file must be a list")
        prompts = []
        for row in obj:
            if isinstance(row, str):
                prompts.append(row.strip())
            elif isinstance(row, dict):
                value = row.get("text") or row.get("prompt") or row.get("content")
                if isinstance(value, str) and value.strip():
                    prompts.append(value.strip())
    else:
        prompts = [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError(f"No prompts loaded from {p}")
    return prompts


def batches(items: list[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("batch size must be > 0")
    for i in range(0, len(items), size):
        yield items[i : i + size]


def resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def nested_getattr(root, path: str):
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def nested_setattr(root, path: str, value) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def find_layer_path(model) -> str:
    candidates = (
        "model.language_model.layers",
        "model.model.layers",
        "model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "transformer.h",
        "layers",
    )
    for path in candidates:
        try:
            value = nested_getattr(model, path)
            if isinstance(value, torch.nn.ModuleList) or isinstance(value, (list, tuple)):
                return path
        except (AttributeError, TypeError):
            pass
    raise ValueError(f"Could not locate transformer blocks for {type(model).__name__}")


def get_layers(model):
    path = find_layer_path(model)
    return path, nested_getattr(model, path)


def set_layers(model, layer_path: str, layers: list[torch.nn.Module]) -> None:
    nested_setattr(model, layer_path, torch.nn.ModuleList(layers))
    n = len(layers)
    # HF cache slots follow current layer order, not source checkpoint indices.
    for index, layer in enumerate(layers):
        for module in layer.modules():
            if hasattr(module, "layer_idx"):
                module.layer_idx = index
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is not None and hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = n


def parameter_bytes(module: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def file_sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def load_tensor_artifact(path: Union[str, Path]):
    """Load tensor-and-primitive artifacts without enabling Python pickle globals."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def trust_remote_code_enabled() -> bool:
    """Remote model code is disabled unless the operator explicitly opts in."""
    enabled = os.environ.get("AEGIS_TRUST_REMOTE_CODE", "").strip() == "1"
    if enabled:
        LOG.warning(
            "AEGIS_TRUST_REMOTE_CODE=1 enables execution of code from the selected model repository"
        )
    return enabled

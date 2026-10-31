"""Byte-bound provenance for local steering checkpoints.

The result identifies *which bytes were tested*. An asserted Hugging Face name
does not establish that these bytes are an authentic upstream release.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional

from .measured import bind_files

_TOKEN_ASSETS = {
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
    "tokenizer.model", "spiece.model", "sentencepiece.bpe.model",
    "chat_template.jinja", "chat_template.json",
}
_WEIGHT_BIN = re.compile(r"^(?:pytorch_model|model|consolidated)(?:[.-].*)?\.bin$")
_WEIGHT_PTH = re.compile(r"^(?:pytorch_model|model|consolidated)(?:[.-].*)?\.pth$")
_SHA256 = re.compile(r"^(?:sha256:)?([a-fA-F0-9]{64})$")


def _canonical_digest(data: Mapping[str, str]) -> str:
    """Unambiguous, path-sorted digest of name -> SHA256; not a git tree hash."""
    if not data:
        raise ValueError("Cannot hash an empty checkpoint asset set")
    payload = json.dumps(dict(sorted(data.items())), sort_keys=True,
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _weight_file(name: str) -> bool:
    # Exclude index JSON and optimizer/adapter unrelated tensors.
    stem = Path(name).name
    return (stem.endswith(".safetensors") and
            (stem.startswith("model") or stem.startswith("pytorch_model"))) or bool(
        _WEIGHT_BIN.match(stem) or _WEIGHT_PTH.match(stem)
    )


def _tokenizer_file(name: str) -> bool:
    stem = Path(name).name
    return stem in _TOKEN_ASSETS or stem.startswith("vocab.") or stem.startswith("tokenizer.")


def checkpoint_binding(
    path: str, *,
    audited_inputs: Optional[Mapping[str, Any]] = None,
    expected_checkpoint_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Record exact local model assets, or reuse the operator's byte-level audit.

    Expected checksum is of the canonical JSON of relative-name -> file-SHA256
    for every file under the model folder. It is *not* an HF commit SHA or an
    individual weight-shard hash.
    """
    root = Path(path)
    if not root.is_dir():
        raise ValueError("Steering checkpoint must be a local model directory")
    if not (root / "config.json").is_file():
        raise ValueError("Missing config.json in local checkpoint root")
    if audited_inputs is not None:
        record = audited_inputs.get("model")
        if not isinstance(record, dict) or not isinstance(record.get("files"), dict):
            raise ValueError("Operator model binding is missing")
        if Path(record.get("path", "")).resolve() != root.resolve():
            raise ValueError("Operator model binding points to a different checkpoint")
        files = record["files"]
    else:
        files = bind_files({"model": root})["model"]["files"]

    if not isinstance(files, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{64}", v)
        for k, v in files.items()
    ):
        raise ValueError("Malformed model file hashes")

    config = {k: v for k, v in files.items() if Path(k).name == "config.json"}
    weights = {k: v for k, v in files.items() if _weight_file(k)}
    tokens = {k: v for k, v in files.items() if _tokenizer_file(k)}
    if not config:
        raise ValueError("Missing config.json in local checkpoint")
    if not weights:
        raise ValueError("Checkpoint contains no supported model weight shards")
    if not tokens:
        raise ValueError("Checkpoint contains no tokenizer assets")

    digest = _canonical_digest(files)
    supplied = None
    verified = False
    if expected_checkpoint_sha256 is not None:
        match = _SHA256.fullmatch(expected_checkpoint_sha256)
        if match is None:
            raise ValueError("Expected checkpoint digest must be 64 hexadecimal SHA256 characters")
        supplied = "sha256:" + match.group(1).lower()
        if supplied != digest:
            raise ValueError(f"Checkpoint digest mismatch: expected {supplied}, actual {digest}")
        verified = True

    return {
        "schema_version": 1,
        "checkpoint_sha256": digest,
        "weights_sha256": _canonical_digest(weights),
        "config_sha256": _canonical_digest(config),
        "tokenizer_assets_sha256": _canonical_digest(tokens),
        "file_hashes": dict(sorted(files.items())),
        "weight_files": sorted(weights),
        "config_files": sorted(config),
        "tokenizer_files": sorted(tokens),
        "expected_checkpoint_sha256": supplied,
        "matched_expected_digest": verified,
        "identity_attestation": (
            "matched operator-supplied digest; trust depends on independent provenance"
            if verified else "local bytes bound, but no independently trusted upstream identity provided"
        ),
    }


def runtime_semantic_binding(model, tokenizer) -> dict[str, str]:
    """Reuse Stage 4A's canonical config/tokenizer semantic hashing."""
    from .stage4a_provenance import compute_config_hash, compute_tokenizer_hash
    return {
        "runtime_config_sha256": compute_config_hash(model.config),
        "runtime_tokenizer_sha256": compute_tokenizer_hash(tokenizer),
    }

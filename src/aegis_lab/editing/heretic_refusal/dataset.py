"""
Prompt corpus loading/splitting utilities for Heretic-style refusal ablation.
"""

from __future__ import annotations

import re
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from framewerx.aegis_lab.editing.heretic_refusal.config import HereticRefusalConfig


@dataclass(frozen=True)
class RefusalPromptRecord:
    id: str
    prompt: str
    label: str
    metadata: Optional[Dict[str, Any]] = None


def _normalize_row(row: Dict[str, Any], cfg: HereticRefusalConfig, idx: int) -> RefusalPromptRecord:
    label_map = dict(cfg.prompt_label_map or {})
    label_raw = str(row.get(cfg.label_field, "")).strip().lower()
    label = label_map.get(label_raw, label_raw) if label_map else label_raw
    prompt = str(row.get(cfg.prompt_field, "")).strip()

    if not prompt:
        raise ValueError(f"Empty prompt found at row {idx}")
    if not label:
        raise ValueError(f"Empty label found at row {idx}")

    return RefusalPromptRecord(
        id=str(row.get("id", idx)),
        prompt=prompt,
        label=label,
        metadata={k: v for k, v in row.items() if k not in {cfg.prompt_field, cfg.label_field}},
    )


def _is_remote_path(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {"http", "https"}


def _normalize_document_url(path: str) -> str:
    parsed = urlparse(path)
    if "docs.google.com" in parsed.netloc.lower() and "/document/d/" in parsed.path:
        query = parse_qs(parsed.query)
        # Convert shared edit links to deterministic text-export URLs.
        export_query = "&".join(
            f"{key}={value[0]}"
            for key, value in query.items()
            if value
        )
        if "format" not in query:
            export_query = "format=txt"
        elif query:
            export_query = "&".join(
                f"{key}={value[0]}" for key, value in query.items() if key != "format"
            )
            if "format=txt" not in export_query:
                if export_query:
                    export_query = f"{export_query}&format=txt"
                else:
                    export_query = "format=txt"
        path_parts = parsed.path.split("/edit")
        normalized_path = path_parts[0] + "/export"
        return parsed._replace(path=normalized_path, query=export_query).geturl()
    return path


def _normalize_doc_prompt_line(line: str) -> str:
    line = re.sub(r"^\s*[\-\*\u2022]+\s*", "", line)
    return line.strip()


def _load_remote_text(path: str) -> str:
    url = _normalize_document_url(path)
    try:
        with urlopen(url, timeout=20) as response:  # noqa: S310 - repository policy allows stdlib URL usage
            encoding = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(encoding, errors="replace")
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise ValueError(
                "Failed to load policy document: "
                f"HTTP {exc.code}. Ensure this URL is publicly accessible "
                "or use an exported format endpoint before retrying."
            ) from exc
        raise ValueError(f"Failed to download remote source {path}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise ValueError(f"Failed to download remote source {path}: {exc}") from exc


def _build_unstructured_records(
    text: str,
    cfg: HereticRefusalConfig,
    source: str,
    start_index: int,
) -> List[RefusalPromptRecord]:
    candidates: List[str] = []
    raw_lines = text.splitlines()
    for line in raw_lines:
        normalized = _normalize_doc_prompt_line(line)
        if not normalized:
            continue
        if len(normalized) < 2:
            continue
        candidates.append(normalized)

    # Fallback for single-line docs without visible separators.
    if not candidates:
        for sentence in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
            normalized = _normalize_doc_prompt_line(sentence)
            if normalized:
                candidates.append(normalized)

    label = str(cfg.policy_document_label).strip().lower()
    return [
        RefusalPromptRecord(
            id=f"{source}#{start_index + offset}",
            prompt=prompt,
            label=label,
            metadata={"source": source},
        )
        for offset, prompt in enumerate(candidates)
    ]


def _load_unstructured_source(path: str, cfg: HereticRefusalConfig, start_index: int) -> List[RefusalPromptRecord]:
    if _is_remote_path(path):
        text = _load_remote_text(path)
    else:
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        text = source_path.read_text(encoding="utf-8")
    return _build_unstructured_records(text, cfg, str(path), start_index)


def load_prompt_records(cfg: HereticRefusalConfig) -> List[RefusalPromptRecord]:
    """
    Load prompt/label records from JSON, JSONL, or CSV.
    """
    records: List[RefusalPromptRecord] = []
    dataset_source = str(cfg.dataset_path)

    if _is_remote_path(dataset_source):
        records.extend(_load_unstructured_source(dataset_source, cfg, len(records) + 1))
    else:
        path = Path(cfg.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        suffix = path.suffix.lower()

        if suffix == ".jsonl":
            lines = path.read_text(encoding="utf-8").splitlines()
            for idx, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"JSONL row must be object at line {idx}")
                records.append(_normalize_row(raw, cfg, idx))
        elif suffix == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("JSON dataset must be a list of records")
            for idx, row in enumerate(raw, start=1):
                if not isinstance(row, dict):
                    raise ValueError(f"JSON row must be object at index {idx}")
                records.append(_normalize_row(row, cfg, idx))
        elif suffix in {".csv", ".tsv"}:
            rows = csv.DictReader(path.read_text(encoding="utf-8").splitlines())
            for idx, row in enumerate(rows, start=1):
                records.append(_normalize_row(row, cfg, idx))
        elif suffix in {".txt", ".md", ".markdown"}:
            text = path.read_text(encoding="utf-8")
            records.extend(_build_unstructured_records(text, cfg, str(path), len(records) + 1))
        else:
            raise ValueError("Unsupported dataset format. Use .jsonl, .json, .csv, .tsv, .txt, .md, .markdown, or a URL")

    if cfg.policy_documents:
        for policy_document in cfg.policy_documents:
            records.extend(
                _load_unstructured_source(
                    policy_document,
                    cfg,
                    len(records) + 1,
                )
            )

    if cfg.max_prompts is not None:
        records = records[: cfg.max_prompts]
    return records


def split_records(records: List[RefusalPromptRecord], cfg: HereticRefusalConfig) -> dict[str, List[RefusalPromptRecord]]:
    """
    Split into train/val/test with deterministic shuffling by config seed.
    """
    rng = random.Random(cfg.seed)
    ordered = list(records)
    rng.shuffle(ordered)

    if not ordered:
        return {"train": [], "val": [], "test": []}

    total = len(ordered)
    train_end = int(total * cfg.train_split)
    val_end = train_end + int(total * cfg.val_split)

    return {
        "train": ordered[:train_end],
        "val": ordered[train_end:val_end],
        "test": ordered[val_end:],
    }

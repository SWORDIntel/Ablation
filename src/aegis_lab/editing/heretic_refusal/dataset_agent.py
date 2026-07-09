from __future__ import annotations

"""
DatasetAgent
------------
Loads labeled prompt records and produces train/val/test splits.
"""

from typing import Dict, List

from framewerx.aegis_lab.editing.heretic_refusal.config import HereticRefusalConfig
from framewerx.aegis_lab.editing.heretic_refusal.dataset import RefusalPromptRecord, load_prompt_records, split_records


class DatasetAgent:
    """Agent 1: corpus ingestion and deterministic partitioning."""

    def run(self, cfg: HereticRefusalConfig) -> Dict[str, List[RefusalPromptRecord]]:
        records = load_prompt_records(cfg)
        if not records:
            raise ValueError("No records loaded from dataset.")
        return split_records(records, cfg)

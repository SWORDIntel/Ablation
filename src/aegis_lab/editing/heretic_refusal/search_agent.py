from __future__ import annotations

"""
SearchAgent
-----------
Generates candidate interventions for each trial.
"""

import random
from typing import List, Tuple

from framewerx.aegis_lab.editing.heretic_refusal.config import HereticRefusalConfig


class SearchAgent:
    """Agent 2: candidate generation (scaffold for future Optuna-backed search)."""

    def propose_candidates(
        self,
        cfg: HereticRefusalConfig,
        attempt_count: int,
        *,
        seed_offset: int = 0,
    ) -> List[Tuple[int, ...]]:
        rng = random.Random(cfg.seed + attempt_count + seed_offset)
        candidates: List[Tuple[int, ...]] = []
        for _ in range(attempt_count):
            count = max(1, cfg.top_k_layers)
            # Keep each candidate stable across retries by using fixed pseudo-random stream.
            layers = set()
            while len(layers) < count:
                # Keep layer ids broad and deterministic for the scaffolded search space.
                layers.add(rng.randrange(0, 32))
            candidate = tuple(sorted(layers))
            # Stable shape, sorted layers for comparison/logging.
            candidates.append(candidate)
        return candidates

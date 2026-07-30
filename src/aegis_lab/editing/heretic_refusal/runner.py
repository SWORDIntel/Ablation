"""
Orchestration for a Heretic-inspired refusal ablation study.

This implementation keeps the branch compatible while explicitly enforcing a
three-agent split with a concurrent search-to-scoring pipeline:

1. DatasetAgent: deterministic load/split stage.
2. SearchAgent: generates candidate layer tuples.
3. ScoringAgent: evaluates candidates in parallel worker threads.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Tuple

from aegis_lab.editing.heretic_refusal.config import (
    HereticRefusalConfig,
    config_to_dict,
    load_config,
)
from aegis_lab.editing.heretic_refusal.dataset import RefusalPromptRecord
from aegis_lab.editing.heretic_refusal.dataset_agent import DatasetAgent
from aegis_lab.editing.heretic_refusal.search_agent import SearchAgent
from aegis_lab.editing.heretic_refusal.scoring_agent import ScoringAgent

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    from optuna.study import StudyDirection
    from optuna.trial import TrialState

    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TPESampler = None
    JournalStorage = None
    JournalFileBackend = None
    JournalFileOpenLock = None
    StudyDirection = None
    TrialState = None
    OPTUNA_AVAILABLE = False


@dataclass(frozen=True)
class TrialOutcome:
    trial_id: int
    refusal_score: float
    kl_proxy: float
    utility: float
    feasible: bool
    layers: Tuple[int, ...]


_SENTINEL = object()


def _heretic_study_signature(
    cfg: HereticRefusalConfig,
    splits: Dict[str, List[RefusalPromptRecord]],
) -> str:
    return (
        f"{cfg.model_path}:{cfg.dataset_path}:"
        f"{len(splits['train'])}:{len(splits['val'])}:{len(splits['test'])}:"
        f"{cfg.seed}:{cfg.n_trials}"
    )


def _sanitize_checkpoint_stem(value: str) -> str:
    sanitized = [c if (c.isalnum() or c in {"_", "-"}) else "-" for c in value]
    result = "".join(sanitized).strip("-")
    return result or "heretic-refusal"


def _study_checkpoint_path(cfg: HereticRefusalConfig, signature: str) -> Path:
    if cfg.study_checkpoint_file:
        return Path(cfg.study_checkpoint_file)
    checkpoint_dir = Path(cfg.study_checkpoint_dir)
    stem = _sanitize_checkpoint_stem(f"{cfg.study_name}-{signature}")
    return checkpoint_dir / f"{stem}.jsonl"


def _is_better_trial(current: Optional[TrialOutcome], challenger: TrialOutcome) -> bool:
    if current is None:
        return True
    if challenger.feasible != current.feasible:
        return challenger.feasible
    if challenger.refusal_score != current.refusal_score:
        return challenger.refusal_score > current.refusal_score
    if challenger.utility != current.utility:
        return challenger.utility > current.utility
    return challenger.kl_proxy < current.kl_proxy


def _trial_objective_value(trial: TrialOutcome) -> float:
    feasible_rank = 1.0 if trial.feasible else 0.0
    return (
        (feasible_rank * 1000.0)
        + (trial.refusal_score * 100.0)
        + (trial.utility * 10.0)
        - trial.kl_proxy
    )


def _trial_outcome_to_dict(trial: TrialOutcome) -> Dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "refusal_score": trial.refusal_score,
        "kl_proxy": trial.kl_proxy,
        "utility": trial.utility,
        "feasible": trial.feasible,
        "layers": list(trial.layers),
    }


def _build_trial_outcome(
    cfg: HereticRefusalConfig,
    trial_id: int,
    layers: Tuple[int, ...],
    refusal_score: float,
    kl_proxy: float,
    utility: float,
) -> TrialOutcome:
    feasible = (
        refusal_score >= cfg.feasible_refusal_min
        and kl_proxy <= cfg.feasible_kl_max
        and utility >= cfg.feasible_utility_min
    )
    return TrialOutcome(
        trial_id=trial_id,
        refusal_score=refusal_score,
        kl_proxy=kl_proxy,
        utility=utility,
        feasible=feasible,
        layers=layers,
    )


def _search_worker(
    cfg: HereticRefusalConfig,
    search_agent: SearchAgent,
    candidate_queue: Queue,
    stop_event: threading.Event,
    scoring_workers: int,
    timeout: Optional[float],
    start: float,
) -> None:
    """Agent-2 worker: generate candidates into the shared queue."""
    for trial_id in range(1, cfg.n_trials + 1):
        if stop_event.is_set():
            break
        if timeout is not None and (time.time() - start) > timeout:
            break

        candidate_lists = search_agent.propose_candidates(cfg, 1, seed_offset=trial_id)
        if not candidate_lists:
            continue

        candidate_queue.put((trial_id, candidate_lists[0]))

    for _ in range(scoring_workers):
        candidate_queue.put(_SENTINEL)


def _scoring_worker(
    cfg: HereticRefusalConfig,
    prompt_count: int,
    prompt_records: Tuple[RefusalPromptRecord, ...],
    score_agent: ScoringAgent,
    candidate_queue: Queue,
    result_queue: Queue,
    stop_event: threading.Event,
) -> None:
    """Agent-3 worker: consume candidates and emit outcomes."""
    while True:
        try:
            payload = candidate_queue.get(timeout=0.1)
        except Empty:
            if stop_event.is_set():
                break
            continue

        if payload is _SENTINEL:
            break

        if stop_event.is_set():
            continue

        trial_id, layers = payload
        refusal_score, kl_proxy, utility = score_agent.score(
            prompt_count,
            cfg.seed + trial_id,
            cfg,
            candidate_layers=layers,
            prompt_records=prompt_records,
        )
        result_queue.put(
            _build_trial_outcome(
                cfg,
                trial_id=trial_id,
                layers=layers,
                refusal_score=refusal_score,
                kl_proxy=kl_proxy,
                utility=utility,
            )
        )


def _count_completed_trials(study: Any) -> int:
    if TrialState is None:
        return len(getattr(study, "trials", []))
    return sum(1 for trial in study.trials if trial.state == TrialState.COMPLETE)


def _extract_pareto_trials(study: Any) -> List[Any]:
    candidate_trials = list(getattr(study, "best_trials", []) or [])
    if not candidate_trials:
        all_trials = getattr(study, "trials", [])
        if TrialState is None:
            candidate_trials = list(all_trials)
        else:
            candidate_trials = [
                trial for trial in all_trials if trial.state == TrialState.COMPLETE
            ]

    def trial_key(trial: Any) -> Tuple[Any, ...]:
        attrs = getattr(trial, "user_attrs", {})
        return (
            not attrs.get("feasible", False),
            -attrs.get("refusal_score", 0.0),
            attrs.get("kl_proxy", float("inf")),
            -attrs.get("utility", 0.0),
            getattr(trial, "number", attrs.get("trial_id", 0)),
        )

    return sorted(candidate_trials, key=trial_key)


def _optuna_trial_to_outcome(trial: Any) -> TrialOutcome:
    attrs = getattr(trial, "user_attrs", {})
    return TrialOutcome(
        trial_id=attrs.get("trial_id", getattr(trial, "number", 0) + 1),
        refusal_score=attrs["refusal_score"],
        kl_proxy=attrs["kl_proxy"],
        utility=attrs["utility"],
        feasible=attrs["feasible"],
        layers=tuple(attrs["layers"]),
    )


def _build_report(
    cfg: HereticRefusalConfig,
    splits: Dict[str, List[RefusalPromptRecord]],
    signature: str,
    best: TrialOutcome,
    trial_results: List[TrialOutcome],
    runtime_seconds: float,
    optimization: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "config": config_to_dict(cfg),
        "dataset_split": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "signature": signature,
        "status": "ok",
        "best": _trial_outcome_to_dict(best),
        "runtime_seconds": round(runtime_seconds, 6),
        "strategy": cfg.strategy,
        "parallel_agents": optimization.get("parallel_agents", cfg.max_parallel_agents),
        "trial_count": len(trial_results),
        "optimization": optimization,
    }


def _run_deterministic_study(
    cfg: HereticRefusalConfig,
    splits: Dict[str, List[RefusalPromptRecord]],
    signature: str,
    search_agent: SearchAgent,
    score_agent: ScoringAgent,
) -> Dict[str, Any]:
    prompt_count = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    prompt_records: Tuple[RefusalPromptRecord, ...] = tuple(
        list(splits["train"]) + list(splits["val"]) + list(splits["test"])
    )
    best: Optional[TrialOutcome] = None
    trial_results: List[TrialOutcome] = []
    stop_reason = "completed"
    processed_trials = 0
    trials_since_improvement = 0
    trials_since_best_update = 0
    pending_by_id: Dict[int, TrialOutcome] = {}
    expected_trial_id = 1

    queue_maxsize = max(4, cfg.max_parallel_agents)
    candidate_queue: Queue[Any] = Queue(maxsize=queue_maxsize)
    result_queue: Queue[Any] = Queue(maxsize=queue_maxsize)
    stop_event = threading.Event()
    start = time.time()
    timeout = cfg.max_runtime_seconds
    scoring_workers = min(max(1, cfg.max_parallel_agents), cfg.n_trials)

    with ThreadPoolExecutor(max_workers=1 + scoring_workers) as executor:
        search_future = executor.submit(
            _search_worker,
            cfg,
            search_agent,
            candidate_queue,
            stop_event,
            scoring_workers,
            timeout,
            start,
        )

        scoring_futures = [
            executor.submit(
                _scoring_worker,
                cfg,
                prompt_count,
                prompt_records,
                score_agent,
                candidate_queue,
                result_queue,
                stop_event,
            )
            for _ in range(scoring_workers)
        ]

        while True:
            if timeout is not None and (time.time() - start) > timeout:
                stop_event.set()

            try:
                trial = result_queue.get(timeout=0.1)
            except Empty:
                all_done = search_future.done() and all(f.done() for f in scoring_futures)
                queue_empty = result_queue.empty()
                if all_done and queue_empty:
                    break
                if timeout is not None and stop_event.is_set():
                    all_done = search_future.done() and all(f.done() for f in scoring_futures)
                    if all_done:
                        break
                continue

            trial_results.append(trial)
            pending_by_id[trial.trial_id] = trial

            while expected_trial_id in pending_by_id:
                ordered_trial = pending_by_id.pop(expected_trial_id)
                processed_trials += 1
                trials_since_improvement += 1
                trials_since_best_update += 1

                if _is_better_trial(best, ordered_trial):
                    previous_best = best
                    best = ordered_trial
                    trials_since_best_update = 0
                    if previous_best is None:
                        trials_since_improvement = 0
                    else:
                        delta = _trial_objective_value(best) - _trial_objective_value(previous_best)
                        if delta >= cfg.min_improvement:
                            trials_since_improvement = 0

                expected_trial_id += 1

                if cfg.enable_early_stop:
                    if trials_since_improvement >= cfg.patience:
                        stop_reason = "patience_exhausted"
                        stop_event.set()
                        break
                    if trials_since_best_update >= cfg.max_trial_stagnation:
                        stop_reason = "trial_stagnation"
                        stop_event.set()
                        break
            if stop_event.is_set():
                continue

    search_future.result()
    for future in scoring_futures:
        future.result()

    if best is None:
        raise RuntimeError("Heretic pipeline did not generate any trials.")

    optimization = {
        "backend": "deterministic_fallback",
        "optuna_available": OPTUNA_AVAILABLE,
        "used_optuna": False,
        "min_improvement": cfg.min_improvement,
        "patience": cfg.patience,
        "feasible_refusal_min": cfg.feasible_refusal_min,
        "feasible_kl_max": cfg.feasible_kl_max,
        "feasible_utility_min": cfg.feasible_utility_min,
        "max_trial_stagnation": cfg.max_trial_stagnation,
        "enable_early_stop": cfg.enable_early_stop,
        "stop_reason": stop_reason,
        "processed_trials": processed_trials,
        "trials_since_improvement": trials_since_improvement,
        "trials_since_best_update": trials_since_best_update,
        "parallel_agents": 1 + scoring_workers,
        "study_name": cfg.study_name,
        "study_resume": cfg.study_resume,
        "checkpoint_path": None,
        "pareto_front": [_trial_outcome_to_dict(best)],
    }
    report = _build_report(
        cfg,
        splits,
        signature,
        best,
        trial_results,
        time.time() - start,
        optimization,
    )
    return {"report": report, "trial_results": trial_results}


def _run_optuna_study(
    cfg: HereticRefusalConfig,
    splits: Dict[str, List[RefusalPromptRecord]],
    signature: str,
    search_agent: SearchAgent,
    score_agent: ScoringAgent,
) -> Dict[str, Any]:
    prompt_count = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    prompt_records: Tuple[RefusalPromptRecord, ...] = tuple(
        list(splits["train"]) + list(splits["val"]) + list(splits["test"])
    )
    checkpoint_path = _study_checkpoint_path(cfg, signature)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    lock_obj = JournalFileOpenLock(str(checkpoint_path))
    backend = JournalFileBackend(str(checkpoint_path), lock_obj=lock_obj)
    storage = JournalStorage(backend)

    directions = [
        StudyDirection.MAXIMIZE,
        StudyDirection.MINIMIZE,
        StudyDirection.MAXIMIZE,
    ]
    sampler = TPESampler(
        n_startup_trials=cfg.optuna_startup_trials,
        multivariate=True,
        seed=cfg.seed,
    )

    study = optuna.create_study(
        sampler=sampler,
        directions=directions,
        storage=storage,
        study_name=cfg.study_name,
        load_if_exists=cfg.study_resume,
    )
    completed_before = _count_completed_trials(study)
    start = time.time()

    def objective(trial: Any) -> Tuple[float, float, float]:
        trial_id = getattr(trial, "number", 0) + 1
        candidate_lists = search_agent.propose_candidates(cfg, 1, seed_offset=trial_id)
        if not candidate_lists:
            raise RuntimeError("SearchAgent did not return any candidate layers.")

        layers = tuple(candidate_lists[0])
        refusal_score, kl_proxy, utility = score_agent.score(
            prompt_count,
            cfg.seed + trial_id,
            cfg,
            candidate_layers=layers,
            prompt_records=prompt_records,
        )
        outcome = _build_trial_outcome(
            cfg,
            trial_id=trial_id,
            layers=layers,
            refusal_score=refusal_score,
            kl_proxy=kl_proxy,
            utility=utility,
        )
        trial.set_user_attr("trial_id", outcome.trial_id)
        trial.set_user_attr("layers", list(outcome.layers))
        trial.set_user_attr("refusal_score", outcome.refusal_score)
        trial.set_user_attr("kl_proxy", outcome.kl_proxy)
        trial.set_user_attr("utility", outcome.utility)
        trial.set_user_attr("feasible", outcome.feasible)
        trial.set_user_attr("objective_value", _trial_objective_value(outcome))
        return (outcome.refusal_score, outcome.kl_proxy, outcome.utility)

    study.set_user_attr("config", config_to_dict(cfg))
    study.set_user_attr("signature", signature)
    study.set_user_attr("finished", False)

    remaining_trials = max(0, cfg.n_trials - completed_before)
    if remaining_trials:
        study.optimize(objective, n_trials=remaining_trials)

    completed_after = _count_completed_trials(study)
    if completed_after >= cfg.n_trials:
        study.set_user_attr("finished", True)

    completed_trials = [
        trial
        for trial in getattr(study, "trials", [])
        if TrialState is None or trial.state == TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("Optuna study did not complete any trials.")

    trial_results = [_optuna_trial_to_outcome(trial) for trial in completed_trials]
    best: Optional[TrialOutcome] = None
    for outcome in trial_results:
        if _is_better_trial(best, outcome):
            best = outcome

    pareto_trials = _extract_pareto_trials(study)
    pareto_front = [_trial_outcome_to_dict(_optuna_trial_to_outcome(trial)) for trial in pareto_trials]

    optimization = {
        "backend": "optuna",
        "optuna_available": True,
        "used_optuna": True,
        "study_name": cfg.study_name,
        "study_resume": cfg.study_resume,
        "checkpoint_path": str(checkpoint_path),
        "completed_trials_before": completed_before,
        "completed_trials_after": completed_after,
        "requested_trials": cfg.n_trials,
        "startup_trials": cfg.optuna_startup_trials,
        "stop_reason": "completed" if completed_after >= cfg.n_trials else "incomplete",
        "parallel_agents": 3,
        "pareto_front": pareto_front,
        "directions": ["maximize", "minimize", "maximize"],
    }
    report = _build_report(
        cfg,
        splits,
        signature,
        best if best is not None else trial_results[0],
        trial_results,
        time.time() - start,
        optimization,
    )
    return {"report": report, "trial_results": trial_results}


def run_heretic_refusal_ablation(
    config_path: str,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the Heretic-like optimization workflow and return a structured report.
    """
    cfg = load_config(config_path)
    return run_heretic_refusal_ablation_from_config(cfg, report_path)


def run_heretic_refusal_ablation_from_config(
    cfg: HereticRefusalConfig,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute workflow from a config object (test-friendly/CLI override entrypoint).
    """
    data_agent = DatasetAgent()
    search_agent = SearchAgent()
    score_agent = ScoringAgent()

    splits = data_agent.run(cfg)
    signature = _heretic_study_signature(cfg, splits)

    if cfg.enable_optuna and OPTUNA_AVAILABLE:
        result = _run_optuna_study(cfg, splits, signature, search_agent, score_agent)
    else:
        result = _run_deterministic_study(cfg, splits, signature, search_agent, score_agent)

    report = result["report"]
    if report_path is None:
        report_path = Path(cfg.output_dir) / "heretic_refusal_result.json"
    else:
        report_path = Path(report_path)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    return {"report_path": str(report_path), "report": report}


__all__ = [
    "DatasetAgent",
    "SearchAgent",
    "ScoringAgent",
    "run_heretic_refusal_ablation",
    "run_heretic_refusal_ablation_from_config",
]

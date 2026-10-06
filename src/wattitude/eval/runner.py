"""Benchmark runner: evaluate estimators over the BROAD trials and cache results.

One row of the result table is one (algorithm, parameter set, trial) evaluation.
Everything downstream -- the trial-agnostic and per-trial tuning, the per-group
tables, the plots -- is a reduction over that table, so the expensive part runs
once and is cached to disk.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines import ESTIMATORS
from ..data import broad
from ..eskf import FilterDivergenceError
from ..init import find_static_window
from .metrics import orientation_errors

__all__ = [
    "param_combinations",
    "init_window_for",
    "evaluate_one",
    "run_grid",
    "default_cache",
]

METRICS = ("total_rmse_deg", "heading_rmse_deg", "inclination_rmse_deg")


def default_cache() -> Path:
    path = Path(__file__).resolve().parents[3] / "reports" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def param_combinations(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a parameter grid, as a list of dicts."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def init_window_for(trial: broad.BroadTrial, max_seconds: float = 2.0) -> slice:
    """Static window used to initialise every estimator on a trial.

    Detected from the IMU signals alone, so no ground truth leaks into
    initialisation, and shared by all algorithms so none is advantaged.
    """
    window = find_static_window(
        trial.gyr, trial.acc, trial.rate, min_duration=0.5, max_search=10.0
    )
    limit = int(round(max_seconds * trial.rate))
    stop = min(window.stop, window.start + limit)
    return slice(window.start, max(stop, window.start + 1))


def _key(algorithm: str, params: dict) -> str:
    return json.dumps({"algorithm": algorithm, **params}, sort_keys=True, default=float)


def evaluate_one(task: tuple[str, str, dict, bool]) -> dict:
    """Evaluate one (trial, algorithm, params) combination.

    Returns a row with the three BROAD metrics plus the heading diagnostics.  A
    diverged filter is recorded as infinite error rather than aborting the sweep.
    """
    trial_name, algorithm, params, use_mag = task
    trial = broad.load_trial(trial_name)
    window = init_window_for(trial)
    fn = ESTIMATORS[algorithm]

    row = {
        "algorithm": algorithm,
        "trial": trial_name,
        "params": _key(algorithm, params),
        "n_total": len(trial),
        "rate": trial.rate,
    }
    row.update({k: float(v) for k, v in params.items()})

    start = time.perf_counter()
    try:
        quats = fn(
            trial.gyr,
            trial.acc,
            trial.rate,
            mag=trial.mag if use_mag else None,
            init_slice=window,
            **params,
        )
        errors = orientation_errors(
            quats, trial.opt_quat, trial.movement, trial.rate,
            align_heading=not use_mag,
        )
        row.update(errors.as_dict())
        row["diverged"] = False
    except (FilterDivergenceError, FloatingPointError) as exc:
        row.update({m: float("inf") for m in METRICS})
        row.update(
            heading_offset_deg=float("nan"),
            heading_drift_deg_per_min=float("nan"),
            n_samples=0,
            diverged=True,
            error=str(exc),
        )
    row["runtime_s"] = time.perf_counter() - start
    return row


def run_grid(
    algorithm: str,
    grid: dict[str, list],
    trials: list[str] | None = None,
    use_mag: bool = False,
    workers: int | None = None,
    cache: Path | str | None = None,
    tag: str | None = None,
    force: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate a parameter grid for one algorithm over a set of trials.

    Results are cached as a CSV keyed by ``tag`` (default: the algorithm name),
    so re-running is free unless ``force`` is set.
    """
    trials = broad.trial_names() if trials is None else list(trials)
    cache_dir = default_cache() if cache is None else Path(cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{tag or algorithm}.csv"
    if path.is_file() and not force:
        return pd.read_csv(path)

    combos = param_combinations(grid)
    tasks = [(t, algorithm, p, use_mag) for p in combos for t in trials]
    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    t0 = time.perf_counter()
    if workers == 1:
        rows = [evaluate_one(task) for task in tasks]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = []
            for i, row in enumerate(pool.imap_unordered(evaluate_one, tasks, chunksize=1), 1):
                rows.append(row)
                if progress and (i % max(1, len(tasks) // 20) == 0 or i == len(tasks)):
                    done = time.perf_counter() - t0
                    print(
                        f"  {algorithm}: {i}/{len(tasks)} "
                        f"({100 * i / len(tasks):.0f}%) in {done:.0f}s, "
                        f"eta {done * (len(tasks) - i) / i:.0f}s",
                        flush=True,
                    )

    df = pd.DataFrame(rows).sort_values(["params", "trial"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    elapsed = time.perf_counter() - t0
    if progress:
        print(
            f"  {algorithm}: {len(combos)} parameter sets x {len(trials)} trials "
            f"in {elapsed:.0f}s -> {path.name}",
            flush=True,
        )
    return df

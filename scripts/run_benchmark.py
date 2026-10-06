"""Full BROAD benchmark: grid-search every estimator over all 39 trials.

Each variant is evaluated over its own parameter grid and cached to
``reports/cache/<tag>.csv``.  Re-running is free; pass ``--force`` to recompute.

Every estimator gets exactly one tuned scalar, matching BROAD's protocol, so
no variant wins by being allowed to search a larger space.  The two ESKF
variants differ only in whether the measurement covariance adapts, and they are
run with the *same* nominal sigma_acc grid: this is what makes the
adaptive-versus-fixed comparison meaningful, since the adaptive estimate uses
the nominal as its floor and a nominal already inflated to the disturbance scale
would mask the adaptation entirely.
"""

from __future__ import annotations

import argparse

import pandas as pd

from wattitude.baselines import PARAM_GRIDS
from wattitude.data import broad
from wattitude.eval.runner import run_grid
from wattitude.eval.tuning import METRICS, SELECTION_METRIC, itop, tagp

# (tag, algorithm, extra fixed parameters, uses magnetometer)
SPECS: list[tuple[str, str, dict, bool]] = [
    ("gyro_only", "gyro_only", {}, False),
    ("madgwick_6d", "madgwick", {}, False),
    ("madgwick_adaptive_6d", "madgwick_adaptive", {}, False),
    ("mahony_6d", "mahony", {}, False),
    ("vqf_6d", "vqf", {}, False),
    ("eskf_adaptive", "eskf", {"adaptive": True}, False),
    ("eskf_fixed", "eskf", {"adaptive": False}, False),
    # 9-axis Madgwick reproduces BROAD's published reference numbers and so
    # validates the harness against an external result.
    ("madgwick_9d", "madgwick", {}, True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", nargs="*", help="restrict to these tags")
    ap.add_argument("--trials", type=int, default=0, help="use only the first N trials")
    args = ap.parse_args()

    trials = broad.trial_names()
    if args.trials:
        trials = trials[: args.trials]
    print(f"{len(trials)} trials, {args.workers} workers")

    summary = []
    for tag, algorithm, fixed, use_mag in SPECS:
        if args.only and tag not in args.only:
            continue
        grid = {k: list(v) for k, v in PARAM_GRIDS[algorithm].items()}
        grid.update({k: [v] for k, v in fixed.items()})
        print(f"\n{tag}: {algorithm}, grid {({k: len(v) for k, v in grid.items()})}")
        df = run_grid(
            algorithm, grid, trials=trials, use_mag=use_mag, tag=tag,
            workers=args.workers, force=args.force,
        )
        row = {"variant": tag, "n_param_sets": df["params"].nunique()}
        try:
            _, params = tagp(df, SELECTION_METRIC)
            best = df[df["params"] == tagp(df, SELECTION_METRIC)[0]]
            row.update({f"tagp_{m}": float(best[m].mean()) for m in METRICS})
            row["tagp_params"] = {k: v for k, v in params.items() if k not in fixed}
            # ITOP picks each trial's parameters on the selection metric and
            # then reports that row's three errors, rather than minimising each
            # metric independently, which would mix incompatible parameters.
            tuned = itop(df, SELECTION_METRIC)
            row.update({f"itop_{m}": float(tuned[m].mean()) for m in METRICS})
        except ValueError as exc:
            row["error"] = str(exc)
        summary.append(row)

    out = pd.DataFrame(summary)
    pd.set_option("display.width", 250)
    print(f"\n=== TAGP / ITOP summary, selected on {SELECTION_METRIC} ===")
    print(out.to_string(index=False))
    path = run_grid.__globals__["default_cache"]() / "summary.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

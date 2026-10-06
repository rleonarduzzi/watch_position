"""Trial-agnostic and per-trial tuning, following BROAD's benchmark protocol.

BROAD defines two benchmark metrics:

* **TAGP** (trial-agnostic parameters): the smallest average error over all 39
  trials achievable with one single parameter set.  This is the number that
  matters in practice, where per-recording tuning is impossible.
* **ITOP** (individually tuned optimal parameters): the average of the per-trial
  minima.  The gap between the two measures how well an algorithm generalises
  across motions.

One deviation, made explicit because it changes which parameters win.  BROAD
selects the TAGP by minimising mean *total* RMSE, which is appropriate for its
9-axis reference algorithms: they get absolute heading from the magnetometer, so
total error measures something the algorithm controls.  A 6-axis filter cannot
observe heading at all, so its total error is dominated by an unobservable
random walk and selecting on it would tune the wrong thing.  The default
selection metric here is therefore ``inclination_rmse_deg``; pass
``metric="total_rmse_deg"`` to reproduce BROAD's rule.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..data import broad

__all__ = ["SELECTION_METRIC", "METRICS", "tagp", "itop", "per_group", "summarise"]

SELECTION_METRIC = "inclination_rmse_deg"
METRICS = ("total_rmse_deg", "heading_rmse_deg", "inclination_rmse_deg")


def _mean_by_params(df: pd.DataFrame, metric: str) -> pd.Series:
    """Mean of ``metric`` over trials, for each parameter set.

    A parameter set that did not complete every trial is dropped rather than
    averaged over a subset, which would otherwise reward divergence.
    """
    n_trials = df["trial"].nunique()
    grouped = df.groupby("params")[metric]
    means = grouped.mean()
    counts = grouped.size()
    return means[counts == n_trials]


def tagp(df: pd.DataFrame, metric: str = SELECTION_METRIC) -> tuple[str, dict]:
    """Parameter set minimising the mean ``metric`` over all trials."""
    means = _mean_by_params(df, metric)
    if means.empty or not np.isfinite(means.min()):
        raise ValueError("no parameter set completed all trials with a finite error")
    best = str(means.idxmin())
    params = {k: v for k, v in json.loads(best).items() if k != "algorithm"}
    return best, params


def itop(df: pd.DataFrame, metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Per-trial best row under ``metric`` (individually tuned optimum)."""
    idx = df.groupby("trial")[metric].idxmin()
    return df.loc[idx].set_index("trial")


def per_group(rows: pd.DataFrame, root=None) -> pd.DataFrame:
    """Average the metrics over each BROAD trial group.

    ``rows`` must hold exactly one row per trial.
    """
    rows = rows.set_index("trial") if "trial" in rows.columns else rows
    out = {}
    for group in broad.group_names(root):
        members = [t for t in broad.trials_in_group(group, root) if t in rows.index]
        if not members:
            continue
        subset = rows.loc[members]
        entry = {m: float(subset[m].mean()) for m in METRICS if m in subset}
        entry["n_trials"] = len(members)
        if "heading_drift_deg_per_min" in subset:
            entry["abs_heading_drift_deg_per_min"] = float(
                subset["heading_drift_deg_per_min"].abs().mean()
            )
        out[group] = entry
    return pd.DataFrame(out).T


def summarise(df: pd.DataFrame, metric: str = SELECTION_METRIC, root=None) -> dict:
    """Full summary for one algorithm: TAGP rows, ITOP rows and group tables."""
    best_key, best_params = tagp(df, metric)
    tagp_rows = df[df["params"] == best_key].copy()
    itop_rows = itop(df, metric).reset_index()

    return {
        "algorithm": str(df["algorithm"].iloc[0]),
        "tagp_params": best_params,
        "tagp_rows": tagp_rows,
        "tagp_groups": per_group(tagp_rows, root),
        "itop_rows": itop_rows,
        "itop_groups": per_group(itop_rows, root),
        "n_param_sets": int(df["params"].nunique()),
        "selection_metric": metric,
    }

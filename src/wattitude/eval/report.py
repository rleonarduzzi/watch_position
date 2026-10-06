"""Tables, figures and acceptance checks built from the cached benchmark results.

Reductions over the result table live here; the expensive evaluation lives in
:mod:`wattitude.eval.runner`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data import broad
from ..eskf import run_batch
from ..eval.metrics import error_series
from ..eval.runner import default_cache, init_window_for
from ..eval.tuning import METRICS, SELECTION_METRIC, per_group, tagp

__all__ = [
    "load_variant",
    "tagp_table",
    "group_table",
    "plot_adaptation",
    "plot_group_errors",
    "figures_dir",
]

# Groups worth reporting separately: the motion taxonomy plus the mechanical
# disturbances.  The magnetic groups are omitted from the 6-axis tables because
# a 6-axis filter cannot be affected by a magnet.
REPORT_GROUPS = [
    "all_trials", "undisturbed", "disturbed",
    "rotation", "translation", "combined",
    "slow", "fast", "no_breaks", "with_breaks",
    "tapping", "vibration", "mixed",
]

PRETTY = {
    "gyro_only": "Gyro only (open loop)",
    "madgwick_6d": "Madgwick 6D",
    "madgwick_adaptive_6d": "Madgwick 6D, adaptive beta",
    "mahony_6d": "Mahony 6D",
    "vqf_6d": "VQF 6D",
    "eskf_fixed": "ESKF, fixed R",
    "eskf_adaptive": "ESKF, adaptive R",
    "madgwick_9d": "Madgwick 9D (harness check)",
}


def figures_dir() -> Path:
    path = Path(__file__).resolve().parents[3] / "reports" / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_variant(tag: str, cache: Path | None = None) -> pd.DataFrame:
    path = (cache or default_cache()) / f"{tag}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run scripts/run_benchmark.py")
    return pd.read_csv(path)


def tagp_table(tags: list[str], metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """One row per variant: its trial-agnostic parameters and resulting errors."""
    rows = []
    for tag in tags:
        df = load_variant(tag)
        key, params = tagp(df, metric)
        best = df[df["params"] == key]
        row = {"variant": PRETTY.get(tag, tag), "tag": tag}
        row.update({m: float(best[m].mean()) for m in METRICS})
        row["drift_deg_per_min"] = float(best["heading_drift_deg_per_min"].abs().mean())
        row["params"] = ", ".join(f"{k}={v:g}" for k, v in sorted(params.items()))
        rows.append(row)
    return pd.DataFrame(rows)


def group_table(tags: list[str], metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Per-group mean of ``metric`` at each variant's trial-agnostic parameters."""
    out = {}
    for tag in tags:
        df = load_variant(tag)
        key, _ = tagp(df, metric)
        groups = per_group(df[df["params"] == key])
        out[PRETTY.get(tag, tag)] = groups[metric]
    table = pd.DataFrame(out)
    present = [g for g in REPORT_GROUPS if g in table.index]
    return table.loc[present]


def nominal_sweep(metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Error against nominal ``sigma_acc`` for both ESKF variants.

    This is the comparison that actually isolates the adaptive mechanism.  At
    the trial-agnostic optimum the two variants are the same filter, because the
    innovation-based estimate uses the nominal as a floor and the optimum
    nominal sits far above anything the innovations ever produce.  Sweeping the
    nominal instead shows what the adaptation is worth as a function of how well
    the nominal was chosen.
    """
    out = {}
    for tag in ("eskf_adaptive", "eskf_fixed"):
        df = load_variant(tag)
        out[tag.split("_")[1]] = df.groupby("sigma_acc")[metric].mean()
    table = pd.DataFrame(out)
    table["ratio"] = table["fixed"] / table["adaptive"]
    return table


def matched_nominal_groups(sigma_acc: float, metric: str = SELECTION_METRIC) -> pd.DataFrame:
    """Per-group error for both ESKF variants at one shared nominal ``sigma_acc``."""
    out = {}
    for tag in ("eskf_fixed", "eskf_adaptive"):
        df = load_variant(tag)
        rows = df[np.isclose(df["sigma_acc"], sigma_acc)]
        if rows.empty:
            raise ValueError(f"sigma_acc={sigma_acc} not in the {tag} grid")
        out[PRETTY[tag]] = per_group(rows)[metric]
    table = pd.DataFrame(out)
    table["gap"] = table[PRETTY["eskf_fixed"]] - table[PRETTY["eskf_adaptive"]]
    table["ratio"] = table[PRETTY["eskf_fixed"]] / table[PRETTY["eskf_adaptive"]]
    present = [g for g in REPORT_GROUPS if g in table.index]
    return table.loc[present]


def plot_nominal_sweep(out: Path | None = None, metric: str = SELECTION_METRIC):
    """Plot the nominal sweep, which is the adaptive mechanism's actual effect."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = nominal_sweep(metric)
    gyro = load_variant("gyro_only")[metric].mean()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.semilogx(table.index, table["fixed"], "o-", color="tab:red", label="fixed $R$")
    ax.semilogx(table.index, table["adaptive"], "s-", color="tab:green",
                label="adaptive $R$")
    ax.axhline(gyro, ls=":", color="k", lw=1, label="open loop (gyro only)")
    ax.set_xlabel(r"nominal $\sigma_{\mathrm{acc}}$  (m/s$^2$)")
    ax.set_ylabel(f"mean {metric.replace('_', ' ')} (deg)")
    ax.set_title("Error vs how well the nominal is chosen")
    ax.legend(fontsize=8)

    ax2.semilogx(table.index, table["ratio"], "d-", color="tab:blue")
    ax2.axhline(1.0, ls="--", color="k", lw=0.8)
    ax2.set_xlabel(r"nominal $\sigma_{\mathrm{acc}}$  (m/s$^2$)")
    ax2.set_ylabel("fixed / adaptive")
    ax2.set_title("What the adaptation is worth")
    for a in (ax, ax2):
        a.grid(alpha=0.25, which="both")
    fig.tight_layout()
    out = out or figures_dir() / "nominal_sweep.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out, table


def plot_adaptation(trial_name: str, out: Path | None = None, **eskf_kwargs):
    """Show the adaptive covariance reacting to an accelerometer disturbance.

    Three stacked panels over a window containing a disturbance: the excess
    accelerometer magnitude that the filter is reacting to, the trace of the
    adapted measurement covariance, and the resulting inclination error of both
    the adaptive and the fixed-covariance filter.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trial = broad.load_trial(trial_name)
    window = init_window_for(trial)
    kwargs = {"init_samples": window.stop, **eskf_kwargs}

    q_ad, _, diag = run_batch(trial.gyr, trial.acc, trial.rate, diagnostics=True,
                              adaptive=True, **kwargs)
    q_fx, _, _ = run_batch(trial.gyr, trial.acc, trial.rate, adaptive=False, **kwargs)

    def incl_series(quats):
        series = error_series(quats, trial.opt_quat, trial.movement, align_heading=True)
        return np.degrees(series["inclination"])

    incl_ad, incl_fx = incl_series(q_ad), incl_series(q_fx)
    t = np.arange(len(trial)) / trial.rate

    # Centre the view on the most violent stretch of the recording, since the
    # whole trial is mostly undisturbed and the reaction would be invisible.
    span = int(20.0 * trial.rate)
    kernel = np.ones(min(span, len(trial))) / min(span, len(trial))
    centre = int(np.argmax(np.convolve(diag.acc_excess, kernel, mode="same")))
    lo = max(0, centre - span // 2)
    hi = min(len(trial), centre + span // 2)
    sl = slice(lo, hi)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(t[sl], diag.acc_excess[sl], lw=0.7, color="tab:gray")
    axes[0].set_ylabel(r"$|\,\|a_k\| - g\,|$  (m/s$^2$)")
    axes[0].set_title(f"Adaptive measurement covariance on {trial_name}")

    axes[1].semilogy(t[sl], diag.R_trace[sl], lw=0.8, color="tab:blue")
    nominal = 3.0 * eskf_kwargs.get("sigma_acc", 0.15) ** 2
    axes[1].axhline(nominal, ls="--", lw=0.8, color="k",
                    label=f"nominal $\\mathrm{{tr}}\\,R = {nominal:.3g}$")
    axes[1].set_ylabel(r"$\mathrm{tr}\,R_k$  (m$^2$/s$^4$)")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t[sl], incl_fx[sl], lw=0.8, color="tab:red", label="fixed $R$")
    axes[2].plot(t[sl], incl_ad[sl], lw=0.8, color="tab:green", label="adaptive $R$")
    axes[2].set_ylabel("inclination error (deg)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()

    out = out or figures_dir() / f"adaptation_{trial_name.split('_')[0]}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out, {
        "R_trace_median": float(np.median(diag.R_trace)),
        "R_trace_p99": float(np.percentile(diag.R_trace, 99)),
        "nominal": nominal,
        "incl_adaptive": float(np.sqrt(np.nanmean(incl_ad[trial.movement] ** 2))),
        "incl_fixed": float(np.sqrt(np.nanmean(incl_fx[trial.movement] ** 2))),
    }


def plot_group_errors(tags: list[str], out: Path | None = None,
                      metric: str = SELECTION_METRIC):
    """Grouped bar chart of per-group error at each variant's TAGP parameters."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = group_table(tags, metric)
    fig, ax = plt.subplots(figsize=(12, 5))
    n = len(table.columns)
    x = np.arange(len(table.index))
    width = 0.8 / n
    for i, col in enumerate(table.columns):
        ax.bar(x + i * width - 0.4 + width / 2, table[col], width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=30, ha="right")
    ax.set_ylabel(f"{metric.replace('_', ' ')} (deg)")
    ax.set_yscale("log")
    ax.set_title("Per-group error at trial-agnostic parameters (lower is better)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = out or figures_dir() / "group_errors.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out, table

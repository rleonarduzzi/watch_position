"""Tables, figures and acceptance checks built from the cached benchmark results.

Reductions over the result table live here; the expensive evaluation lives in
:mod:`wattitude.eval.runner`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines import ESTIMATORS
from ..data import broad
from ..eskf import run_batch
from ..eval.metrics import (
    apply_heading_offset,
    error_series,
    optimal_heading_offset,
    orientation_errors,
)
from ..eval.runner import default_cache, init_window_for
from ..eval.tuning import METRICS, SELECTION_METRIC, per_group, tagp
from ..quaternion import euler_gimbal_risk, quat_to_euler

__all__ = [
    "load_variant",
    "tagp_table",
    "group_table",
    "nominal_sweep",
    "matched_nominal_groups",
    "euler_tracking",
    "plot_euler_tracking",
    "plot_nominal_sweep",
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
    "eskf_fixed": "ESKF, fixed R, eq. (8)",
    "eskf_adaptive": "ESKF, adaptive R, eq. (8)",
    "eskf_exactphi": "ESKF, adaptive R, exact Phi",
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


def _break_wraps(angle_deg: np.ndarray, jump: float = 180.0) -> np.ndarray:
    """Insert NaN at +-180 degree wraps so no vertical line is drawn across them.

    Keeps the plotted values inside the natural range instead of unwrapping them
    into the hundreds of degrees, at the cost of a one-sample gap at each wrap.
    """
    out = np.asarray(angle_deg, dtype=float).copy()
    d = np.abs(np.diff(out))
    out[1:][d > jump] = np.nan
    return out


def _decimate(n: int, limit: int) -> slice:
    return slice(None, None, max(1, int(np.ceil(n / limit))))


def euler_tracking(trial_name: str, eskf_params: dict | None = None,
                   vqf_params: dict | None = None) -> dict:
    """Ground-truth and estimated attitude for one trial, as Euler angles.

    Both estimators are magnetometer-free, so their heading is expressed in
    their own initial frame rather than the optical system's.  Each estimate
    therefore has its optimal constant heading offset removed first -- the same
    alignment the error metric applies -- otherwise the yaw panel would show a
    meaningless constant difference and nothing else.
    """
    trial = broad.load_trial(trial_name)
    window = init_window_for(trial)
    movement = trial.movement

    series = {}
    for name, params in (
        ("eskf", eskf_params or {"sigma_acc": 200.0}),
        ("vqf", vqf_params or {"tauAcc": 3.0}),
    ):
        quats = ESTIMATORS[name](
            trial.gyr, trial.acc, trial.rate, init_slice=window, **params
        )
        offset = optimal_heading_offset(quats, trial.opt_quat, movement)
        aligned = apply_heading_offset(quats, offset)
        series[name] = {
            "euler": np.rad2deg(quat_to_euler(aligned)),
            "errors": orientation_errors(
                quats, trial.opt_quat, movement, trial.rate, align_heading=True
            ),
            "total_series": np.rad2deg(
                error_series(quats, trial.opt_quat, movement)["total"]
            ),
            "heading_offset_deg": float(np.rad2deg(offset)),
            "params": params,
        }

    return {
        "trial": trial,
        "truth": np.rad2deg(quat_to_euler(trial.opt_quat)),
        "gimbal_risk": euler_gimbal_risk(trial.opt_quat, margin_deg=10.0),
        "series": series,
    }


def plot_euler_tracking(trial_name: str, out: Path | None = None,
                        max_points: int = 6000, zoom: tuple[float, float] | None = None,
                        **kwargs):
    """Plot roll, pitch and yaw of both estimates against the optical reference.

    ``zoom`` restricts the view to a ``(start, stop)`` time range in seconds,
    which is the only way to see the sample-level behaviour on a recording of a
    few hundred thousand samples.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = euler_tracking(trial_name, **kwargs)
    trial = data["trial"]
    t = np.arange(len(trial)) / trial.rate

    # Restrict to the annotated movement phase: the long rest periods either
    # side carry no information and would compress everything interesting.
    movement = trial.movement
    lo, hi = int(np.argmax(movement)), int(len(movement) - np.argmax(movement[::-1]))
    if zoom is not None:
        lo = max(lo, int(zoom[0] * trial.rate))
        hi = min(hi, int(zoom[1] * trial.rate))
    view = slice(lo, hi)
    step = _decimate(hi - lo, max_points)

    tv = t[view][step]
    truth = data["truth"][view][step]
    risk = data["gimbal_risk"][view][step]

    styles = {
        "eskf": dict(color="tab:green", lw=0.9, label="ESKF (this filter)"),
        "vqf": dict(color="tab:purple", lw=0.9, label="VQF"),
    }

    fig, axes = plt.subplots(4, 1, figsize=(12, 9.5), sharex=True)
    for i, name in enumerate(("roll", "pitch", "yaw")):
        ax = axes[i]
        ax.plot(tv, _break_wraps(truth[:, i]), color="k", lw=1.4, alpha=0.75,
                label="optical ground truth", zorder=3)
        for key, style in styles.items():
            euler = data["series"][key]["euler"][view][step]
            ax.plot(tv, _break_wraps(euler[:, i]), zorder=4, **style)
        ax.set_ylabel(f"{name} (deg)")
        ax.grid(alpha=0.25)
        # Roll and yaw are ill-conditioned near vertical pitch; shade it so the
        # resulting swings are not mistaken for estimation error.
        if name in ("roll", "yaw") and np.any(risk):
            ax.fill_between(tv, *ax.get_ylim(), where=risk, color="tab:orange",
                            alpha=0.15, step="mid", zorder=0,
                            label="pitch within 10 deg of vertical")
        ax.legend(loc="upper right", fontsize=7, ncol=2)

    ax = axes[3]
    for key, style in styles.items():
        err = data["series"][key]["total_series"][view][step]
        ax.plot(tv, err, **style)
    ax.set_ylabel("total error (deg)")
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)

    incl = {k: v["errors"].inclination_rmse_deg for k, v in data["series"].items()}
    axes[0].set_title(
        f"{trial_name} -- inclination RMSE: "
        f"ESKF {incl['eskf']:.2f} deg, VQF {incl['vqf']:.2f} deg"
        + ("" if zoom is None else f"  (zoom {zoom[0]:.0f}-{zoom[1]:.0f} s)")
    )
    fig.tight_layout()

    suffix = "" if zoom is None else "_zoom"
    out = out or figures_dir() / f"euler_{trial_name.split('_')[0]}{suffix}.png"
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out, data


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

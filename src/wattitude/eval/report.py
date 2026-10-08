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
    "plot_euler_interactive",
    "plot_euler_dropdown",
    "quat_tracking",
    "plot_quat_dropdown",
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
                   vqf_params: dict | None = None, trial=None) -> dict:
    """Ground-truth and estimated attitude for one trial, as Euler angles.

    Both estimators are magnetometer-free, so their heading is expressed in
    their own initial frame rather than the optical system's.  Each estimate
    therefore has its optimal constant heading offset removed first -- the same
    alignment the error metric applies -- otherwise the yaw panel would show a
    meaningless constant difference and nothing else.

    ``trial`` overrides the BROAD lookup.  Any object with ``gyr``, ``acc``,
    ``opt_quat``, ``movement``, ``rate`` and ``name`` is accepted, which is how
    the Vicon sequences are plotted with the same figure.
    """
    if trial is None:
        trial = broad.load_trial(trial_name)
    else:
        trial_name = getattr(trial, "name", trial_name)
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
                        trial=None, **kwargs):
    """Plot roll, pitch and yaw of both estimates against the optical reference.

    ``zoom`` restricts the view to a ``(start, stop)`` time range in seconds,
    which is the only way to see the sample-level behaviour on a recording of a
    few hundred thousand samples.  Pass ``trial`` to plot a recording that is
    not part of BROAD; see :func:`euler_tracking`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = euler_tracking(trial_name, trial=trial, **kwargs)
    trial = data["trial"]
    trial_name = getattr(trial, "name", trial_name)
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
    token = trial_name.split("_")[0]
    # BROAD trials are numbered (``01_...``).  Other names must keep the full
    # stem or every sequence in a family overwrites one file.
    stem = token if token.isdigit() else trial_name
    out = out or figures_dir() / f"euler_{stem}{suffix}.png"
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out, data


def _true_spans(mask: np.ndarray, t: np.ndarray) -> list[tuple[float, float]]:
    """Time intervals where ``mask`` is true, widened by half a sample."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    half = 0.5 * dt
    return [
        (float(t[s] - half), float(t[e] + half))
        for s, e in zip(starts, ends)
    ]


def _euler_figure(data: dict, trial_name: str | None = None,
                  max_points: int | None = None):
    """Plotly figure of one Euler-tracking result. See :func:`plot_euler_interactive`."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    trial = data["trial"]
    trial_name = trial_name or getattr(trial, "name", "trial")
    t = np.arange(len(trial)) / trial.rate

    movement = np.asarray(trial.movement, dtype=bool)
    lo = int(np.argmax(movement))
    hi = int(len(movement) - np.argmax(movement[::-1]))
    view = slice(lo, hi)
    step = _decimate(hi - lo, max_points) if max_points else slice(None)

    tv = t[view][step]
    truth = data["truth"][view][step]
    risk = np.asarray(data["gimbal_risk"][view][step], dtype=bool)

    colors = {"eskf": "#2ca02c", "vqf": "#9467bd"}
    labels = {"eskf": "ESKF (this filter)", "vqf": "VQF"}

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
    )
    hover = "%{y:.1f}°<extra>%{fullData.name}</extra>"
    for i, angle in enumerate(("roll", "pitch", "yaw")):
        row = i + 1
        fig.add_trace(go.Scatter(
            x=tv, y=_break_wraps(truth[:, i]), name="optical ground truth",
            legendgroup="truth", showlegend=(i == 0),
            line=dict(color="black", width=1.6), hovertemplate=hover,
        ), row=row, col=1)
        for key in ("eskf", "vqf"):
            euler = data["series"][key]["euler"][view][step]
            fig.add_trace(go.Scatter(
                x=tv, y=_break_wraps(euler[:, i]), name=labels[key],
                legendgroup=key, showlegend=(i == 0),
                line=dict(color=colors[key], width=1.2), hovertemplate=hover,
            ), row=row, col=1)
        fig.update_yaxes(title_text=f"{angle} (deg)", row=row, col=1)

    for key in ("eskf", "vqf"):
        err = data["series"][key]["total_series"][view][step]
        fig.add_trace(go.Scatter(
            x=tv, y=err, name=labels[key], legendgroup=key, showlegend=False,
            line=dict(color=colors[key], width=1.2), hovertemplate=hover,
        ), row=4, col=1)
    fig.update_yaxes(title_text="total error (deg)", row=4, col=1)
    fig.update_xaxes(title_text="time (s)", row=4, col=1)

    # One legend entry for the bands. The rectangles themselves are shapes.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=12, color="rgba(255, 127, 14, 0.55)", symbol="square"),
        name="pitch within 10° of vertical",
        legendgroup="risk", showlegend=True, hoverinfo="skip",
    ), row=1, col=1)
    for x0, x1 in _true_spans(risk, tv):
        for row in (1, 3):
            fig.add_vrect(
                x0=x0, x1=x1, fillcolor="rgba(255, 127, 14, 0.18)",
                line_width=0, layer="below", row=row, col=1,
            )

    incl = {k: v["errors"].inclination_rmse_deg for k, v in data["series"].items()}
    fig.update_layout(
        height=960,
        title=dict(
            text=(
                f"{trial_name} -- inclination RMSE: "
                f"ESKF {incl['eskf']:.2f}°, VQF {incl['vqf']:.2f}°"
            ),
            x=0.5,
        ),
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        margin=dict(l=70, r=24, t=100, b=48),
        template="plotly_white",
    )
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikethickness=1,
        spikedash="dot", spikecolor="gray",
    )
    # The slider is the overview; box-zoom and scroll-zoom inspect a window.
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=4, col=1)

    return fig


_PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False, "doubleClick": "reset"}


def _ensure_plotlyjs(directory: Path) -> None:
    """Copy ``plotly.min.js`` next to an HTML figure so the page works offline."""
    dest = directory / "plotly.min.js"
    if dest.is_file():
        return
    import plotly.graph_objects as go

    scratch = directory / "_plotly_js.html"
    go.Figure().write_html(scratch, include_plotlyjs="directory", full_html=True)
    scratch.unlink(missing_ok=True)


def plot_euler_interactive(trial_name: str, out: Path | None = None,
                           trial=None, data: dict | None = None,
                           max_points: int | None = None, **kwargs) -> Path:
    """Write a zoomable HTML figure of the same Euler traces as the PNG.

    The four panels share a time axis. In a browser, drag a rectangle to zoom,
    scroll to zoom further, and double-click to reset. ``data`` may be the dict
    already returned by :func:`plot_euler_tracking`, so the filters are not run
    a second time. ``max_points`` decimates for display; the default keeps
    every sample so a zoom shows the recording rate.
    """
    if data is None:
        data = euler_tracking(trial_name, trial=trial, **kwargs)
    name = getattr(data["trial"], "name", trial_name)
    fig = _euler_figure(data, name, max_points=max_points)
    stem = name.split("_")[0]
    stem = stem if stem.isdigit() else name
    out = out or figures_dir() / f"euler_{stem}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    _ensure_plotlyjs(out.parent)
    fig.write_html(
        out,
        include_plotlyjs=False,
        full_html=True,
        config=_PLOTLY_CONFIG,
    )
    # write_html(include_plotlyjs=False) omits the script tag entirely.
    text = out.read_text()
    if "plotly.min.js" not in text:
        text = text.replace(
            "</head>",
            '    <script src="plotly.min.js"></script>\n</head>',
            1,
        )
        out.write_text(text)
    return out


def _write_dropdown_page(payload: dict[str, dict], out: Path, title: str,
                         note: str = "") -> Path:
    """One HTML page; the menu replaces the Plotly figure."""
    import json

    if not payload:
        raise ValueError("dropdown page needs at least one figure")
    blob = json.dumps(payload).replace("<", "\\u003c")
    options = "\n".join(
        f'      <option value="{label}">{label}</option>' for label in payload
    )
    note_html = f'\n    <span class="note">{note}</span>' if note else ""
    page = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="plotly.min.js"></script>
  <style>
    body {{ margin: 0; font-family: sans-serif; }}
    .bar {{ padding: 10px 16px 0; font-size: 15px; }}
    select {{ font-size: 15px; margin-left: 8px; }}
    .note {{ margin-left: 16px; color: #444; }}
  </style>
</head>
<body>
  <div class="bar">
    <label for="seq">Sequence</label>
    <select id="seq">
{options}
    </select>{note_html}
  </div>
  <div id="plot"></div>
  <script id="euler-figures" type="application/json">{blob}</script>
  <script>
    const figures = JSON.parse(document.getElementById("euler-figures").textContent);
    const select = document.getElementById("seq");
    const config = {{scrollZoom: true, displaylogo: false, doubleClick: "reset"}};
    function show(name) {{
      const fig = figures[name];
      Plotly.react("plot", fig.data, fig.layout, config);
    }}
    select.addEventListener("change", () => show(select.value));
    show(select.value);
  </script>
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    _ensure_plotlyjs(out.parent)
    out.write_text(page)
    return out


def _figures_payload(figures: list[tuple[str, object]]) -> dict[str, dict]:
    import json

    import plotly.io as pio

    return {
        label: json.loads(pio.to_json(fig, validate=False))
        for label, fig in figures
    }


def plot_euler_dropdown(entries: list[tuple[str, dict]], out: Path) -> Path:
    """Write one HTML page whose dropdown switches the Euler figure.

    ``entries`` is ``(label, data)`` in menu order. ``data`` is an
    :func:`euler_tracking` result. Choosing a label replaces the figure; zoom
    works the same way as in :func:`plot_euler_interactive`.
    """
    if not entries:
        raise ValueError("plot_euler_dropdown needs at least one sequence")
    payload = _figures_payload(
        [(label, _euler_figure(data, label)) for label, data in entries]
    )
    return _write_dropdown_page(payload, out, "Euler tracking")


def _continuous_quaternions(q: np.ndarray) -> np.ndarray:
    """Flip signs so successive samples stay in one hemisphere.

    ``q`` and ``-q`` are the same rotation. The first sample is taken with a
    non-negative scalar part, and each later sample is flipped when its dot
    product with the previous sample is negative.
    """
    q = np.array(q, dtype=float, copy=True)
    if len(q) == 0:
        return q
    if np.isfinite(q[0, 0]) and q[0, 0] < 0.0:
        q[0] *= -1.0
    dots = np.sum(q[1:] * q[:-1], axis=1)
    steps = np.ones(len(q))
    steps[1:] = np.where(dots < 0.0, -1.0, 1.0)
    return q * np.cumprod(steps)[:, None]


def _match_quaternion_sign(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Flip ``other`` samplewise onto the hemisphere of ``reference``."""
    dots = np.sum(other * reference, axis=1)
    sign = np.where(dots < 0.0, -1.0, 1.0)
    return other * sign[:, None]


def quat_tracking(trial_name: str, eskf_params: dict | None = None,
                  vqf_params: dict | None = None, trial=None) -> dict:
    """Ground-truth and estimated attitude for one trial, as quaternion components.

    The heading alignment is the same one :func:`euler_tracking` applies. Signs
    are then chosen so the optical quaternion is continuous in time and each
    estimate lies in that hemisphere: ``q`` and ``-q`` would otherwise draw a
    false jump in every component.
    """
    if trial is None:
        trial = broad.load_trial(trial_name)
    else:
        trial_name = getattr(trial, "name", trial_name)
    window = init_window_for(trial)
    movement = trial.movement
    truth = _continuous_quaternions(trial.opt_quat)

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
            "quat": _match_quaternion_sign(truth, aligned),
            "errors": orientation_errors(
                quats, trial.opt_quat, movement, trial.rate, align_heading=True
            ),
            "total_series": np.rad2deg(
                error_series(quats, trial.opt_quat, movement)["total"]
            ),
            "heading_offset_deg": float(np.rad2deg(offset)),
            "params": params,
        }

    return {"trial": trial, "truth": truth, "series": series}


def _quat_figure(data: dict, trial_name: str | None = None,
                 max_points: int | None = None):
    """Plotly figure of scalar-first quaternion components."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    trial = data["trial"]
    trial_name = trial_name or getattr(trial, "name", "trial")
    t = np.arange(len(trial)) / trial.rate
    movement = np.asarray(trial.movement, dtype=bool)
    lo = int(np.argmax(movement))
    hi = int(len(movement) - np.argmax(movement[::-1]))
    view = slice(lo, hi)
    step = _decimate(hi - lo, max_points) if max_points else slice(None)

    tv = t[view][step]
    truth = data["truth"][view][step]
    colors = {"eskf": "#2ca02c", "vqf": "#9467bd"}
    labels = {"eskf": "ESKF (this filter)", "vqf": "VQF"}

    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
    )
    hover = "%{y:.3f}<extra>%{fullData.name}</extra>"
    for i, component in enumerate(("w", "x", "y", "z")):
        row = i + 1
        fig.add_trace(go.Scatter(
            x=tv, y=truth[:, i], name="optical ground truth",
            legendgroup="truth", showlegend=(i == 0),
            line=dict(color="black", width=1.6), hovertemplate=hover,
        ), row=row, col=1)
        for key in ("eskf", "vqf"):
            quat = data["series"][key]["quat"][view][step]
            fig.add_trace(go.Scatter(
                x=tv, y=quat[:, i], name=labels[key],
                legendgroup=key, showlegend=(i == 0),
                line=dict(color=colors[key], width=1.2), hovertemplate=hover,
            ), row=row, col=1)
        fig.update_yaxes(title_text=component, row=row, col=1)

    hover_err = "%{y:.1f}°<extra>%{fullData.name}</extra>"
    for key in ("eskf", "vqf"):
        err = data["series"][key]["total_series"][view][step]
        fig.add_trace(go.Scatter(
            x=tv, y=err, name=labels[key], legendgroup=key, showlegend=False,
            line=dict(color=colors[key], width=1.2), hovertemplate=hover_err,
        ), row=5, col=1)
    fig.update_yaxes(title_text="total error (deg)", row=5, col=1)
    fig.update_xaxes(title_text="time (s)", row=5, col=1)

    incl = {k: v["errors"].inclination_rmse_deg for k, v in data["series"].items()}
    fig.update_layout(
        height=1180,
        title=dict(
            text=(
                f"{trial_name} -- w, x, y, z -- inclination RMSE: "
                f"ESKF {incl['eskf']:.2f}°, VQF {incl['vqf']:.2f}°"
            ),
            x=0.5,
        ),
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        margin=dict(l=70, r=24, t=100, b=48),
        template="plotly_white",
    )
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikethickness=1,
        spikedash="dot", spikecolor="gray",
    )
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.04), row=5, col=1)
    return fig


def plot_quat_dropdown(entries: list[tuple[str, dict]], out: Path) -> Path:
    """Write one HTML page whose dropdown switches the quaternion figure.

    ``entries`` is ``(label, data)`` in menu order. ``data`` is a
    :func:`quat_tracking` result.
    """
    if not entries:
        raise ValueError("plot_quat_dropdown needs at least one sequence")
    payload = _figures_payload(
        [(label, _quat_figure(data, label)) for label, data in entries]
    )
    return _write_dropdown_page(
        payload,
        out,
        "Quaternion tracking",
        note=(
            "Scalar-first w, x, y, z. The constant heading offset is removed, "
            "and each sample uses the sign closest to the optical quaternion. "
            "The bottom panel is the total orientation error after that alignment."
        ),
    )


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

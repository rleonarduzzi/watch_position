"""Interactive check that the gyroscope correlation keeps its sign.

Section 3.3 of the axis note reports one correlation per axis, averaged over
the samples with a large gyroscope magnitude.  This page draws those same
samples after the centering and scaling inside that correlation, so a stretch
where the two rates oppose each other is visible instead of being folded into
the average.

Two HTML pages, each with the sequence chosen from a menu, as for the Euler
and quaternion pages.  The second page draws the product of the two scaled
rates, which is the term averaged by the correlation, and opens on the bulk
of that product rather than on its spikes.
"""

from __future__ import annotations

import numpy as np

from wattitude.data.vicon import (
    _body_rate,
    load_trial,
    sequence_names,
    short_root,
    write_prefix_dataset,
)
from wattitude.eval.report import _figures_payload, _write_dropdown_page, figures_dir

DT = 0.01
# Same gate as needs_rz180 and the correlation in the note.
RATE_GATE = 0.5


def _rates(trial):
    """Optical body rate and gyroscope midpoint, both length n-1."""
    omega = _body_rate(trial.opt_quat, DT)
    gyro = 0.5 * (trial.gyr[:-1] + trial.gyr[1:])
    return omega, gyro


def _zscore_masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Center and scale on ``mask``, with the sample standard deviation.

    ``np.corrcoef`` divides by that deviation, so the correlation of the two
    masked series equals the mean of their product times ``n / (n - 1)``.
    Samples outside the mask are left blank.
    """
    out = np.full(len(values), np.nan)
    kept = values[mask]
    scale = float(kept.std(ddof=1))
    if scale < 1e-8:
        return out
    out[mask] = (kept - kept.mean()) / scale
    return out


def _correlation_series(omega: np.ndarray, gyro: np.ndarray):
    """Per-axis z-scored rates and the correlations those samples produce."""
    moving = np.linalg.norm(gyro, axis=1) > RATE_GATE
    optical = np.column_stack([_zscore_masked(omega[:, i], moving) for i in range(3)])
    measured = np.column_stack([_zscore_masked(gyro[:, i], moving) for i in range(3)])
    corr = np.array([
        float(np.corrcoef(omega[moving, i], gyro[moving, i])[0, 1])
        for i in range(3)
    ])
    return optical, measured, corr


def _figure(name: str, optical: np.ndarray, measured: np.ndarray, corr: np.ndarray):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    k = np.arange(len(optical))
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    hover = "%{y:.2f}<extra>%{fullData.name}</extra>"
    for i, axis in enumerate("xyz"):
        row = i + 1
        fig.add_trace(go.Scatter(
            x=k, y=optical[:, i], name="optical ω",
            legendgroup="optical", showlegend=(i == 0),
            line=dict(color="black", width=1.3), hovertemplate=hover,
            connectgaps=False,
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=k, y=measured[:, i], name="gyroscope ω̃",
            legendgroup="gyro", showlegend=(i == 0),
            line=dict(color="#1f77b4", width=1.1), hovertemplate=hover,
            connectgaps=False,
        ), row=row, col=1)
        fig.update_yaxes(title_text=f"{axis}", row=row, col=1)
    fig.update_xaxes(title_text="sample k", row=3, col=1)
    signed = ", ".join(f"{c:+.2f}" for c in corr)
    fig.update_layout(
        height=820,
        title=dict(text=f"{name} — correlation ({signed})", x=0.5),
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        margin=dict(l=60, r=24, t=90, b=40),
        template="plotly_white",
    )
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikethickness=1,
        spikedash="dot", spikecolor="gray",
    )
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=3, col=1)
    return fig


def _bulk_range(values: np.ndarray) -> tuple[float, float]:
    """Y limits covering the central 98% of finite samples, and zero."""
    finite = values[np.isfinite(values)]
    lo, hi = (float(v) for v in np.percentile(finite, [1.0, 99.0]))
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = 0.05 * (hi - lo) if hi > lo else 0.5
    return lo - pad, hi + pad


def _product_figure(name: str, optical: np.ndarray, measured: np.ndarray, corr: np.ndarray):
    """Product of the two scaled rates.  The y range drops the outer 1% each side."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    product = optical * measured
    k = np.arange(len(product))
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04)
    hover = "%{y:.2f}<extra>%{fullData.name}</extra>"
    for i, axis in enumerate("xyz"):
        row = i + 1
        fig.add_trace(go.Scatter(
            x=k, y=product[:, i], name="ω̃ ω",
            legendgroup="product", showlegend=(i == 0),
            line=dict(color="#1f77b4", width=1.1), hovertemplate=hover,
            connectgaps=False,
        ), row=row, col=1)
        fig.add_hline(y=0.0, row=row, col=1, line_width=1, line_color="gray", opacity=0.7)
        lo, hi = _bulk_range(product[:, i])
        fig.update_yaxes(title_text=f"{axis}", range=[lo, hi], autorange=False, row=row, col=1)
    fig.update_xaxes(title_text="sample k", row=3, col=1)
    signed = ", ".join(f"{c:+.2f}" for c in corr)
    fig.update_layout(
        height=820,
        title=dict(text=f"{name} — product, correlation ({signed})", x=0.5),
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        margin=dict(l=60, r=24, t=90, b=40),
        template="plotly_white",
    )
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikethickness=1,
        spikedash="dot", spikecolor="gray",
    )
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=3, col=1)
    return fig


def main() -> None:
    root = short_root()
    if not any(root.glob("v3_*.csv")):
        written = write_prefix_dataset()
        print(f"wrote {len(written)} prefixes to {root}")

    separate = []
    products = []
    for name in sequence_names(root):
        trial = load_trial(name, root=root, align_axes=False)
        optical, measured, corr = _correlation_series(*_rates(trial))
        separate.append((name, _figure(name, optical, measured, corr)))
        products.append((name, _product_figure(name, optical, measured, corr)))
        print(f"{name}  c = ({corr[0]:+.2f}, {corr[1]:+.2f}, {corr[2]:+.2f})")

    out = figures_dir() / "vicon_gyro_corr.html"
    _write_dropdown_page(
        _figures_payload(separate),
        out,
        "Gyroscope correlation",
        note=(
            "Each axis is centered and scaled on the samples with "
            "a gyroscope magnitude above 0.5 rad/s. Gaps are the samples left out of the correlation."
        ),
    )
    print(f"wrote {out}")

    product_out = figures_dir() / "vicon_gyro_corr_product.html"
    _write_dropdown_page(
        _figures_payload(products),
        product_out,
        "Gyroscope correlation product",
        note=(
            "Product of the centered and scaled rates. "
            "The opening view omits the outer 1% on each side; autoscale shows the spikes."
        ),
    )
    print(f"wrote {product_out}")


if __name__ == "__main__":
    main()

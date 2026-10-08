"""The interactive Euler figure is a self-contained HTML file."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("plotly")

from wattitude.eval.report import plot_euler_dropdown, plot_euler_interactive


def _series(n: int, scale: float) -> dict:
    euler = np.column_stack([
        scale * np.sin(np.linspace(0, 8, n)),
        -40 + 5 * np.cos(np.linspace(0, 6, n)),
        np.linspace(-170, 190, n),
    ])
    return {
        "euler": euler,
        "total_series": np.abs(scale) * np.ones(n),
        "errors": SimpleNamespace(inclination_rmse_deg=abs(scale)),
    }


class _Trial:
    def __init__(self, n: int):
        self.name = "toy"
        self.rate = 100.0
        self.movement = np.ones(n, dtype=bool)

    def __len__(self) -> int:
        return len(self.movement)


def test_interactive_euler_writes_html(tmp_path):
    n = 40
    data = {
        "trial": _Trial(n),
        "truth": np.zeros((n, 3)),
        "gimbal_risk": np.arange(n) % 10 < 2,
        "series": {"eskf": _series(n, 1.5), "vqf": _series(n, 2.5)},
    }
    out = tmp_path / "toy.html"
    written = plot_euler_interactive("toy", data=data, out=out)
    text = written.read_text()
    assert written == out
    assert "toy -- inclination RMSE" in text
    assert "optical ground truth" in text
    assert "ESKF (this filter)" in text
    assert "plotly" in text.lower()
    assert (tmp_path / "plotly.min.js").is_file()


def test_dropdown_switches_sequences(tmp_path):
    n = 40
    entries = []
    for name, scale in (("v3_01", 1.5), ("v3_02", 2.5)):
        trial = _Trial(n)
        trial.name = name
        entries.append((name, {
            "trial": trial,
            "truth": np.zeros((n, 3)),
            "gimbal_risk": np.arange(n) % 10 < 2,
            "series": {"eskf": _series(n, scale), "vqf": _series(n, scale + 1)},
        }))
    out = tmp_path / "vicon_euler.html"
    written = plot_euler_dropdown(entries, out)
    text = written.read_text()
    assert written == out
    assert text.count("<select") == 1
    assert '<option value="v3_01">v3_01</option>' in text
    assert '<option value="v3_02">v3_02</option>' in text
    assert "v3_01 -- inclination RMSE" in text
    assert "v3_02 -- inclination RMSE" in text
    assert 'id="plot"' in text

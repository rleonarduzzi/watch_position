"""Quaternion-component figure: signs stay continuous, and the page has a menu."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from wattitude.eval.report import (
    _continuous_quaternions,
    _match_quaternion_sign,
    plot_quat_dropdown,
)


def test_continuous_quaternions_remove_double_cover_flips():
    q = np.tile(np.array([0.6, 0.8, 0.0, 0.0]), (5, 1))
    q[2:] *= -1.0
    out = _continuous_quaternions(q)
    dots = np.sum(out[1:] * out[:-1], axis=1)
    assert np.all(dots > 0.0)
    assert out[0, 0] > 0.0


def test_estimate_is_flipped_onto_the_reference_hemisphere():
    reference = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    other = -reference
    matched = _match_quaternion_sign(reference, other)
    assert np.allclose(matched, reference)


class _Trial:
    def __init__(self, name: str, n: int):
        self.name = name
        self.rate = 100.0
        self.movement = np.ones(n, dtype=bool)

    def __len__(self) -> int:
        return len(self.movement)


def test_quat_dropdown_lists_components(tmp_path):
    pytest.importorskip("plotly")
    n = 30
    truth = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    truth[10:15, 0] = -1.0

    def series(scale: float) -> dict:
        quat = truth.copy()
        quat[:, 1] = scale * 0.01
        return {
            "quat": quat,
            "total_series": np.full(n, abs(scale)),
            "errors": SimpleNamespace(inclination_rmse_deg=abs(scale)),
        }

    entries = []
    for name, scale in (("v3_01", 1.5), ("v3_02", 2.5)):
        entries.append((name, {
            "trial": _Trial(name, n),
            "truth": truth,
            "series": {"eskf": series(scale), "vqf": series(scale + 1)},
        }))

    out = plot_quat_dropdown(entries, tmp_path / "vicon_quat.html")
    text = out.read_text()
    assert '<option value="v3_01">v3_01</option>' in text
    assert '<option value="v3_02">v3_02</option>' in text
    assert "w, x, y, z" in text
    assert "Scalar-first" in text
    assert "v3_01 -- w, x, y, z" in text
    assert "total error (deg)" in text

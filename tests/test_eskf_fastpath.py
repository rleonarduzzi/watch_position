"""Pin the filter's inlined quaternion algebra to the reference implementation.

:meth:`AdaptiveESKF.update` inlines the propagation and error-injection
quaternion maths for speed, duplicating logic that already exists in
:mod:`wattitude.quaternion`.  These tests exist so the duplication cannot drift:
every inlined expression is checked against the reference version it replaced,
including the regimes the fast path special-cases.

The motivating failure is worth stating, since it is invisible on gentle data: a
propagation error scales with angular rate, so a broken fast path still looks
correct on a slow-rotation recording and only shows up at several hundred
degrees per second.
"""

from __future__ import annotations

import numpy as np
import pytest

from wattitude.eskf import AdaptiveESKF, GRAVITY
from wattitude.quaternion import (
    exp_q,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
)

RATE = 285.714


def _filter(**kwargs) -> AdaptiveESKF:
    est = AdaptiveESKF(rate=RATE, **kwargs)
    est.quat = quat_normalize(np.array([0.3, -0.5, 0.7, 0.41]))
    return est


# Spans the small-angle branch (below the 1e-16 threshold on |omega*dt|^2), the
# ordinary regime, and rates beyond the full-scale range of a consumer gyroscope.
RATES = [
    np.zeros(3),
    np.array([1e-9, -2e-9, 5e-10]),
    np.array([0.01, -0.02, 0.005]),
    np.array([1.0, -2.0, 0.5]),
    np.array([12.0, -9.0, 15.0]),
    np.array([40.0, 40.0, -40.0]),
]


@pytest.mark.parametrize("omega", RATES, ids=lambda w: f"|w|={np.linalg.norm(w):.3g}")
def test_inlined_propagation_matches_reference_exp_map(omega):
    """One propagation step must equal q (x) exp_q(omega dt) from the reference."""
    dt = 1.0 / RATE
    est = _filter(adaptive=False)
    q0 = est.quat.copy()

    # Isolate propagation: a measurement equal to the predicted gravity leaves a
    # zero innovation, so the update contributes nothing to the quaternion.
    est.bias_gyr[:] = 0.0
    q_pred = quat_normalize(quat_multiply(q0, exp_q(omega * dt)))
    acc = quat_to_matrix(q_pred).T @ np.array([0.0, 0.0, est.gravity])

    est.update(omega, acc, dt)

    assert np.allclose(est.quat, q_pred, atol=1e-13)


def test_small_angle_branch_agrees_with_the_general_branch_at_the_threshold():
    """The two branches must not disagree where the fast path switches between them."""
    dt = 1.0 / RATE
    # |omega*dt|^2 straddles the 1e-16 cutoff.
    below = np.array([1e-9, 0.0, 0.0]) / dt
    above = np.array([1e-7, 0.0, 0.0]) / dt
    results = []
    for omega in (below * 1e-2, below, above):
        est = _filter(adaptive=False)
        est.bias_gyr[:] = 0.0
        q_pred = quat_normalize(quat_multiply(est.quat, exp_q(omega * dt)))
        acc = quat_to_matrix(q_pred).T @ np.array([0.0, 0.0, est.gravity])
        est.update(omega, acc, dt)
        results.append(est.quat.copy())
        assert np.allclose(est.quat, q_pred, atol=1e-13)
    # Continuity: a 100x smaller rotation must give a 100x smaller displacement,
    # not a discontinuous jump caused by the branch.
    assert np.linalg.norm(results[0] - results[1]) < np.linalg.norm(results[1] - results[2])


def test_bias_is_subtracted_before_propagation():
    """The inlined omega must be gyr - bias_gyr, not gyr."""
    dt = 1.0 / RATE
    gyr = np.array([0.7, -0.3, 1.1])
    bias = np.array([0.02, -0.01, 0.03])

    est = _filter(adaptive=False)
    est.bias_gyr[:] = bias
    q_pred = quat_normalize(quat_multiply(est.quat, exp_q((gyr - bias) * dt)))
    acc = quat_to_matrix(q_pred).T @ np.array([0.0, 0.0, est.gravity])
    est.update(gyr, acc, dt)

    assert np.allclose(est.quat, q_pred, atol=1e-13)


@pytest.mark.parametrize("left", [False, True])
def test_error_injection_matches_reference_quaternion_product(left):
    """_apply_increment must equal the reference product on the chosen side."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        est = _filter(adaptive=False, inject_left=left)
        q0 = est.quat.copy()
        dtheta = rng.normal(scale=0.05, size=3)

        # The increment the filter composes is the small-angle form 1 + dtheta/2,
        # unnormalised, exactly as the inlined code builds it.
        inc = np.array([1.0, *(0.5 * dtheta)])
        expected = quat_normalize(
            quat_multiply(inc, q0) if left else quat_multiply(q0, inc)
        )
        if expected[0] < 0.0:
            expected = -expected

        est._apply_increment(1.0, *(0.5 * dtheta), left=left)
        assert np.allclose(est.quat, expected, atol=1e-14)


def test_increment_keeps_the_quaternion_normalised_and_in_the_positive_half():
    """Repeated in-place composition must not drift off the unit sphere."""
    rng = np.random.default_rng(1)
    est = _filter(adaptive=False)
    for _ in range(10_000):
        est._apply_increment(1.0, *(0.5 * rng.normal(scale=0.1, size=3)), left=False)
        assert est.quat[0] >= 0.0
    assert abs(np.linalg.norm(est.quat) - 1.0) < 1e-12


def test_full_update_tracks_a_fast_rotation_that_would_expose_a_propagation_error():
    """End-to-end guard at a rate where a wrong exp map is unmistakable.

    A first-order (dropped sin/cos) propagation accumulates roughly
    |omega dt|^3/24 per step, which at 10 rad/s over two seconds is far larger
    than the tolerance below.
    """
    dt = 1.0 / RATE
    n = int(2.0 * RATE)
    omega = np.array([6.0, -8.0, 10.0])

    est = _filter(adaptive=False)
    est.bias_gyr[:] = 0.0
    truth = est.quat.copy()
    for _ in range(n):
        truth = quat_normalize(quat_multiply(truth, exp_q(omega * dt)))
        acc = quat_to_matrix(truth).T @ np.array([0.0, 0.0, est.gravity])
        est.update(omega, acc, dt)

    # Zero innovation throughout, so the filter must reproduce the reference
    # integration to numerical precision.
    err = quat_multiply(np.array([truth[0], *(-truth[1:])]), est.quat)
    assert 2.0 * np.arcsin(min(1.0, np.linalg.norm(err[1:]))) < 1e-9

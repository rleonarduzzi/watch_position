"""Regression tests for the linearisation of Section IV.A.

These lock in the findings of ``reports/derivation_review.md``: the analytic
Jacobians the filter uses must match finite differences of the exact nonlinear
models, and the printed equations (9), (13) and (20) must be shown to deviate.
"""

from __future__ import annotations

import numpy as np
import pytest

from wattitude.eskf import GRAVITY, AdaptiveESKF
from wattitude.quaternion import (
    exp_q,
    log_q,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
    skew,
)

G_W = np.array([0.0, 0.0, GRAVITY])
STEP = 1e-7


@pytest.fixture
def rng():
    return np.random.default_rng(2024)


def measurement(q, bias_acc):
    """Exact nonlinear accelerometer model, equation (11) at rest."""
    return quat_to_matrix(q).T @ G_W + bias_acc


def test_measurement_jacobian_matches_finite_differences(rng):
    est = AdaptiveESKF(rate=285.7)
    for _ in range(100):
        q = quat_normalize(rng.normal(size=4))
        ba = rng.normal(scale=0.1, size=3)
        est.quat, est.bias_acc = q, ba

        H = est._measurement_jacobian(quat_to_matrix(q))
        H_fd = np.zeros((3, 9))
        for j in range(3):
            e = np.zeros(3)
            e[j] = STEP
            H_fd[:, j] = (
                measurement(quat_multiply(q, exp_q(e)), ba)
                - measurement(quat_multiply(q, exp_q(-e)), ba)
            ) / (2 * STEP)
            H_fd[:, 6 + j] = (measurement(q, ba + e) - measurement(q, ba - e)) / (2 * STEP)
        np.testing.assert_allclose(H, H_fd, atol=1e-5)


def test_measurement_jacobian_bias_block_is_plus_identity():
    """Equation (13) prints -I; the error convention of equation (5) needs +I."""
    corrected = AdaptiveESKF(rate=100.0, paper_faithful=False)
    faithful = AdaptiveESKF(rate=100.0, paper_faithful=True)
    R = np.eye(3)
    np.testing.assert_allclose(corrected._measurement_jacobian(R)[:, 6:9], np.eye(3))
    np.testing.assert_allclose(faithful._measurement_jacobian(R)[:, 6:9], -np.eye(3))


def test_six_state_jacobian_drops_the_bias_block():
    est = AdaptiveESKF(rate=100.0, estimate_acc_bias=False)
    H = est._measurement_jacobian(np.eye(3))
    assert H.shape == (3, 6)
    assert est.dim == 6
    assert est.P.shape == (6, 6)


def test_transition_matrix_matches_finite_differences_of_error_dynamics(rng):
    """Phi must linearise dtheta_{k+1} = f(dtheta_k, db_g) correctly."""
    dt = 1.0 / 285.7
    est = AdaptiveESKF(rate=285.7, exact_phi=True)
    for _ in range(50):
        omega = rng.normal(scale=2.0, size=3)
        Phi = est._transition(omega, dt)

        # Propagate a perturbed truth and the nominal estimate, then read off the
        # resulting error.  q_true = q_est * exp_q(dtheta) by convention.
        q_est = quat_normalize(rng.normal(size=4))

        def error_after(dtheta, dbg):
            q_true = quat_multiply(q_est, exp_q(dtheta))
            omega_true = omega - dbg
            q_true_next = quat_multiply(q_true, exp_q(omega_true * dt))
            q_est_next = quat_multiply(q_est, exp_q(omega * dt))
            return log_q(quat_multiply(np.array([q_est_next[0], *(-q_est_next[1:])]), q_true_next))

        J = np.zeros((3, 6))
        for j in range(3):
            e = np.zeros(3)
            e[j] = 1e-6
            J[:, j] = (error_after(e, np.zeros(3)) - error_after(-e, np.zeros(3))) / 2e-6
            J[:, 3 + j] = (error_after(np.zeros(3), e) - error_after(np.zeros(3), -e)) / 2e-6
        np.testing.assert_allclose(Phi[0:3, 0:6], J, atol=1e-6)


def test_first_order_phi_approximates_the_exact_one():
    dt = 1.0 / 285.7
    omega = np.array([1.0, -2.0, 0.5])
    approx = AdaptiveESKF(rate=285.7, exact_phi=False)._transition(omega, dt)
    exact = AdaptiveESKF(rate=285.7, exact_phi=True)._transition(omega, dt)
    # Equation (8) is first order in dt, so the gap must scale like (|w| dt)^2.
    assert np.max(np.abs(approx - exact)) < (np.linalg.norm(omega) * dt) ** 2


def test_injection_must_be_right_multiplication(rng):
    """Equation (20) prints dq (x) q; the convention requires q (x) dq."""
    for _ in range(100):
        q_est = quat_normalize(rng.normal(size=4))
        dtheta = rng.normal(scale=1e-3, size=3)
        q_true = quat_multiply(q_est, exp_q(dtheta))
        dq = quat_normalize(np.concatenate(([1.0], 0.5 * dtheta)))

        def residual(q):
            inv = np.array([q_true[0], *(-q_true[1:])])
            return float(np.linalg.norm(log_q(quat_multiply(inv, quat_normalize(q)))))

        right = residual(quat_multiply(q_est, dq))
        left = residual(quat_multiply(dq, q_est))
        assert right < 1e-8
        assert left > 100 * right


def test_paper_process_noise_is_singular():
    """Equation (9) gives rank 6 of 9 with perfectly correlated dtheta and db_g."""
    est = AdaptiveESKF(rate=285.7, paper_faithful=True)
    Q = est._process_noise(1.0 / 285.7)
    assert np.linalg.matrix_rank(Q, tol=1e-18) == 6
    corr = Q[0, 3] / np.sqrt(Q[0, 0] * Q[3, 3])
    assert corr == pytest.approx(-1.0, abs=1e-12)


def test_standard_process_noise_is_positive_definite():
    est = AdaptiveESKF(rate=285.7, paper_faithful=False)
    Q = est._process_noise(1.0 / 285.7)
    assert np.all(np.linalg.eigvalsh(Q) > 0.0)


def test_gain_uses_the_inverse_not_the_transpose():
    """||K|| must shrink as R grows; equation (14) as printed does the opposite."""
    P = np.diag([1e-4, 1e-4, 1e-2, 1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3])
    H = np.zeros((3, 9))
    H[:, 0:3] = skew(G_W)
    H[:, 6:9] = np.eye(3)

    norms_inv, norms_printed = [], []
    for sigma in (0.05, 0.15, 0.5, 1.5, 5.0):
        S = H @ P @ H.T + sigma**2 * np.eye(3)
        np.testing.assert_allclose(S, S.T, atol=1e-15)
        norms_inv.append(np.linalg.norm(np.linalg.solve(S, H @ P).T))
        norms_printed.append(np.linalg.norm(P @ H.T @ S.T))
    assert all(b < a for a, b in zip(norms_inv, norms_inv[1:]))
    assert all(b > a for a, b in zip(norms_printed, norms_printed[1:]))


def test_gravity_update_is_rank_deficient_in_tilt():
    """Yaw is unobservable, and tilt/bias are confounded in a single update."""
    H_tilt = skew(G_W)
    assert np.linalg.matrix_rank(H_tilt, tol=1e-9) == 2
    # The null space is the gravity direction in the body frame.
    _, _, vt = np.linalg.svd(H_tilt)
    null = vt[-1]
    assert abs(abs(np.dot(null, G_W / GRAVITY)) - 1.0) < 1e-9

    H_full = np.hstack([H_tilt, np.zeros((3, 3)), np.eye(3)])
    assert np.linalg.matrix_rank(H_full, tol=1e-9) == 3


def test_covariance_stays_symmetric_positive_definite(rng):
    """Joseph-form updates must survive unbounded growth of the yaw variance."""
    est = AdaptiveESKF(rate=200.0).initialize(acc_window=np.array([[0.0, 0.0, GRAVITY]]))
    for _ in range(4000):
        gyr = rng.normal(scale=0.5, size=3)
        acc = np.array([0.0, 0.0, GRAVITY]) + rng.normal(scale=0.05, size=3)
        est.update(gyr, acc)
    np.testing.assert_allclose(est.P, est.P.T, atol=1e-18)
    assert np.all(np.linalg.eigvalsh(est.P) > 0.0)

"""Verify the linearisation of Section IV.A symbolically and numerically.

Checks, in order:

1. Equation (5): the error-state system matrix ``F``, against a symbolic
   derivation of the attitude-error dynamics.
2. Equation (13): the measurement Jacobian ``H``, against central finite
   differences of the exact nonlinear measurement function.
3. Equation (20): which side the small-angle correction must be applied on.
4. Equation (9): the rank of the resulting process-noise covariance.
5. Equation (14): the numerical consequence of the printed transpose.

Run with ``python scripts/verify_derivation.py``.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from wattitude.quaternion import (
    exp_q,
    log_q,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
    skew,
)

np.set_printoptions(precision=10, suppress=True)
G_W = np.array([0.0, 0.0, 9.81])
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def sym_skew(v):
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ---------------------------------------------------------------------------
# 1. Equation (5): attitude error dynamics
# ---------------------------------------------------------------------------
def check_F_symbolic() -> None:
    """Derive d/dt dtheta from R_true = R_est expm(skew(dtheta)) symbolically.

    Both sides of ``Rdot = R skew(omega_true)`` are expanded to first order in
    the error, with ``omega_true = omega_hat - db_g - n_g`` from equation (3).
    After cancelling the common factor ``R_est``, the O(eps) terms give

        skew(omega_hat) skew(th) + skew(thdot)
            = skew(th) skew(omega_hat) - skew(db_g + n_g)

    which must be satisfied by ``thdot = -skew(omega_hat) th - db_g - n_g``.
    """
    w = sp.Matrix(sp.symbols("w1 w2 w3", real=True))  # omega_hat = gyr - b_g
    dbg = sp.Matrix(sp.symbols("dbg1 dbg2 dbg3", real=True))
    ng = sp.Matrix(sp.symbols("ng1 ng2 ng3", real=True))
    th = sp.Matrix(sp.symbols("th1 th2 th3", real=True))
    thdot = sp.Matrix(sp.symbols("thd1 thd2 thd3", real=True))

    residual = sp.expand(
        sym_skew(w) * sym_skew(th)
        + sym_skew(thdot)
        - sym_skew(th) * sym_skew(w)
        + sym_skew(dbg + ng)
    )
    claimed = -sym_skew(w) * th - dbg - ng
    substituted = sp.simplify(residual.subs(dict(zip(thdot, claimed))))
    ok_dynamics = substituted == sp.zeros(3, 3)

    # The system matrix is the Jacobian of the dynamics w.r.t. the error state.
    dba = sp.Matrix(sp.symbols("dba1 dba2 dba3", real=True))
    state = sp.Matrix([*th, *dbg, *dba])
    F_sym = claimed.subs({s: 0 for s in ng}).jacobian(state)
    F_paper = sp.zeros(3, 9)
    F_paper[:, 0:3] = -sym_skew(w)
    F_paper[:, 3:6] = -sp.eye(3)
    ok_F = sp.simplify(F_sym - F_paper) == sp.zeros(3, 9)

    record(
        "eq(5) F matrix",
        bool(ok_dynamics and ok_F),
        "d/dt dtheta = -skew(gyr - b_g) dtheta - db_g - n_g holds, and its "
        "Jacobian reproduces F = [[-skew(gyr - b_g), -I, 0], [0, 0, 0], [0, 0, 0]] "
        "as printed; this is what fixes the error convention to the body-frame side",
    )


# ---------------------------------------------------------------------------
# 2. Equation (13): measurement Jacobian
# ---------------------------------------------------------------------------
def measurement(q: np.ndarray, bias_acc: np.ndarray) -> np.ndarray:
    """Exact nonlinear accelerometer model of equation (11), at rest."""
    return quat_to_matrix(q).T @ G_W + bias_acc


def check_H_numeric() -> None:
    rng = np.random.default_rng(0)
    worst_tilt, worst_bias = 0.0, 0.0
    for _ in range(200):
        q = quat_normalize(rng.normal(size=4))
        ba = rng.normal(scale=0.1, size=3)
        R = quat_to_matrix(q)

        H_tilt_analytic = skew(R.T @ G_W)
        H_bias_analytic = np.eye(3)

        h = 1e-7
        H_tilt_fd = np.empty((3, 3))
        H_bias_fd = np.empty((3, 3))
        for j in range(3):
            e = np.zeros(3)
            e[j] = h
            # true minus estimate, body-frame error: q_true = q_est * exp_q(dtheta)
            zp = measurement(quat_multiply(q, exp_q(+e)), ba)
            zm = measurement(quat_multiply(q, exp_q(-e)), ba)
            H_tilt_fd[:, j] = (zp - zm) / (2 * h)
            zp = measurement(q, ba + e)
            zm = measurement(q, ba - e)
            H_bias_fd[:, j] = (zp - zm) / (2 * h)

        worst_tilt = max(worst_tilt, np.max(np.abs(H_tilt_fd - H_tilt_analytic)))
        worst_bias = max(worst_bias, np.max(np.abs(H_bias_fd - H_bias_analytic)))

    record(
        "eq(13) tilt block",
        worst_tilt < 1e-5,
        f"skew(R^T g_w) matches finite differences, max abs error {worst_tilt:.2e}",
    )
    record(
        "eq(13) bias block sign",
        worst_bias < 1e-5,
        f"d(acc)/d(db_a) = +I, max abs error {worst_bias:.2e} "
        "-> the printed -I has the wrong sign for the error convention of eq(5)",
    )


# ---------------------------------------------------------------------------
# 3. Equation (20): injection side
# ---------------------------------------------------------------------------
def check_injection_side() -> None:
    rng = np.random.default_rng(1)
    err_right, err_left = [], []
    for _ in range(200):
        q_est = quat_normalize(rng.normal(size=4))
        dtheta = rng.normal(scale=1e-3, size=3)
        # Ground truth by definition of the error convention of eq (5).
        q_true = quat_multiply(q_est, exp_q(dtheta))

        dq = quat_normalize(np.concatenate(([1.0], 0.5 * dtheta)))
        q_right = quat_normalize(quat_multiply(q_est, dq))
        q_left = quat_normalize(quat_multiply(dq, q_est))

        err_right.append(
            np.linalg.norm(log_q(quat_multiply(_conj(q_true), q_right)))
        )
        err_left.append(np.linalg.norm(log_q(quat_multiply(_conj(q_true), q_left))))

    mr, ml = float(np.max(err_right)), float(np.max(err_left))
    record(
        "eq(20) injection side",
        mr < 1e-8 and ml > 1e-6,
        f"right multiplication residual {mr:.2e} rad vs left multiplication "
        f"{ml:.2e} rad -> q (x) dq is required, the printed dq (x) q is wrong",
    )


def _conj(q):
    out = np.asarray(q, dtype=float).copy()
    out[1:] *= -1
    return out


# ---------------------------------------------------------------------------
# 4. Equation (9): rank of the process noise
# ---------------------------------------------------------------------------
def check_Q_rank() -> None:
    sg, sa, dt = 5e-3, 0.15, 1.0 / 285.7
    G = np.zeros((9, 6))
    G[0:3, 0:3] = -np.eye(3)
    G[3:6, 0:3] = np.eye(3)
    G[6:9, 3:6] = np.eye(3)
    noise = np.diag(np.concatenate([np.full(3, sg**2), np.full(3, sa**2)]))
    Q_paper = G @ noise @ G.T * dt

    rank = np.linalg.matrix_rank(Q_paper, tol=1e-18)
    corr = Q_paper[0, 3] / np.sqrt(Q_paper[0, 0] * Q_paper[3, 3])
    record(
        "eq(9) Q rank",
        rank == 6,
        f"rank {rank} of 9; dtheta/db_g correlation {corr:+.1f} "
        "-> gyro bias has no independent random walk and the block is singular",
    )

    Q_std = np.diag(np.concatenate([np.full(3, sg**2), np.full(3, 1e-5**2), np.full(3, 1e-4**2)])) * dt
    record(
        "standard Q is full rank",
        np.linalg.matrix_rank(Q_std, tol=1e-30) == 9,
        "diag(sigma_g^2, sigma_bg^2, sigma_ba^2) * dt is positive definite",
    )


# ---------------------------------------------------------------------------
# 5. Equation (14): the printed transpose
# ---------------------------------------------------------------------------
def check_gain_transpose() -> None:
    rng = np.random.default_rng(2)
    P = np.diag([1e-4, 1e-4, 1e-2, 1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3])
    q = quat_normalize(rng.normal(size=4))
    H = np.zeros((3, 9))
    H[:, 0:3] = skew(quat_to_matrix(q).T @ G_W)
    H[:, 6:9] = np.eye(3)
    # Sweep the measurement noise: a correct gain must shrink as R grows.
    correct, printed, asym = [], [], 0.0
    for sigma in (0.05, 0.15, 0.5, 1.5, 5.0):
        S = H @ P @ H.T + sigma**2 * np.eye(3)
        asym = max(asym, float(np.max(np.abs(S - S.T))))
        correct.append(float(np.linalg.norm(P @ H.T @ np.linalg.inv(S))))
        printed.append(float(np.linalg.norm(P @ H.T @ S.T)))  # equation (14) literally

    correct_decreasing = all(b < a for a, b in zip(correct, correct[1:]))
    printed_increasing = all(b > a for a, b in zip(printed, printed[1:]))
    record(
        "eq(14) transpose",
        asym < 1e-15 and correct_decreasing and printed_increasing,
        f"S is symmetric (asymmetry {asym:.1e}) so S^T = S and the transpose is a "
        f"no-op; ||K|| over sigma_a = 0.05..5 m/s^2 goes "
        f"{correct[0]:.3f} -> {correct[-1]:.4f} with the inverse but "
        f"{printed[0]:.3f} -> {printed[-1]:.1f} as printed, i.e. the printed form "
        "trusts the accelerometer more the noisier it gets -> a typo for the inverse",
    )


# ---------------------------------------------------------------------------
# 6. Observability of the gravity measurement
# ---------------------------------------------------------------------------
def check_observability() -> None:
    q = quat_normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    R = quat_to_matrix(q)
    H_tilt = skew(R.T @ G_W)
    rank_tilt = np.linalg.matrix_rank(H_tilt, tol=1e-9)
    null_dir = R.T @ G_W / np.linalg.norm(G_W)

    H_full = np.hstack([H_tilt, np.zeros((3, 3)), np.eye(3)])
    rank_full = np.linalg.matrix_rank(H_full, tol=1e-9)
    record(
        "observability of a single gravity update",
        rank_tilt == 2 and rank_full == 3,
        f"skew(R^T g) has rank {rank_tilt} with null space along R^T g "
        f"{np.round(null_dir, 3)} (yaw unobservable); the full 3x9 H has rank "
        f"{rank_full} against 5 unknowns, so tilt and accel bias are only "
        "separable through motion",
    )


def check_transition_orthogonality() -> None:
    """Equation (8) is not orthogonal, so it inflates the covariance.

    The exact attitude block is a rotation and preserves the covariance norm;
    the printed first-order form stretches it by ``1 + theta^2/2`` per step,
    which is spurious growth unrelated to any noise source.
    """
    from wattitude.eskf import AdaptiveESKF

    dt = 7.0 / 2000.0
    rate = 12.7  # peak angular rate on BROAD's fastest trial, 728 deg/s
    omega = np.array([rate, 0.0, 0.0])
    theta = rate * dt

    first_order = np.eye(3) - skew(omega) * dt
    exact = AdaptiveESKF(rate=1.0 / dt, exact_phi=True)._transition(omega, dt)[0:3, 0:3]

    s_first = np.linalg.svd(first_order, compute_uv=False)
    s_exact = np.linalg.svd(exact, compute_uv=False)
    predicted = np.sqrt(1.0 + theta * theta)

    record(
        "equation (8) is not orthogonal, the exact form is",
        np.allclose(s_exact, 1.0, atol=1e-12)
        and abs(s_first.max() - predicted) < 1e-12
        and s_first.max() > 1.0,
        f"at {np.degrees(rate):.0f} deg/s the printed Phi has singular values "
        f"{np.round(s_first, 9)} (max {s_first.max():.9f} = sqrt(1+theta^2)), "
        f"against {np.round(s_exact, 12)} for the closed form; the covariance "
        f"therefore grows {s_first.max() ** 2:.7f} per step, "
        f"{(s_first.max() ** 2) ** int(60 / dt):.2e} over a minute",
    )

    # The two must still agree to first order, or the fast path would be wrong
    # rather than merely non-orthogonal.
    slow = np.array([0.01, 0.0, 0.0])
    a = np.eye(3) - skew(slow) * dt
    b = AdaptiveESKF(rate=1.0 / dt, exact_phi=True)._transition(slow, dt)[0:3, 0:3]
    err = np.max(np.abs(a - b))
    bound = (np.linalg.norm(slow) * dt) ** 2
    record(
        "the two transition forms agree to O(theta^2)",
        err < bound,
        f"at {np.degrees(np.linalg.norm(slow)):.2f} deg/s they differ by "
        f"{err:.3e}, below theta^2 = {bound:.3e}",
    )


def main() -> int:
    print("=" * 78)
    print("Verification of Section IV.A (Bai et al., adaptive ESKF attitude)")
    print("=" * 78)
    check_F_symbolic()
    check_H_numeric()
    check_injection_side()
    check_Q_rank()
    check_gain_transpose()
    check_observability()
    check_transition_orthogonality()
    print("-" * 78)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

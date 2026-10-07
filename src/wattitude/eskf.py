"""Adaptive error-state Kalman filter for 6-axis IMU attitude estimation.

Implements Section IV.A of Bai et al., "Watch Your Position: Neural Inertial
Localization with a Single Wrist-worn Device".

State and conventions
---------------------
Nominal state: quaternion ``q`` (body to world), gyroscope bias ``b_g``,
accelerometer bias ``b_a``.  Error state ordering is
``[dtheta (0:3), db_g (3:6), db_a (6:9)]`` with the *true minus estimate*
definition and the body-frame (right) attitude error
``R_true = R_est @ expm(skew(dtheta))``.

Sensor models (paper equations 3 and 11)::

    gyr_meas = gyr_true + b_g + n_g
    acc_meas = R^T @ g_w + b_a + n_a        with g_w = [0, 0, +g]

so ``acc_meas`` reads ``+g`` on the vertical axis at rest, matching an ENU
world frame with the z axis pointing up.

Linearisation
-------------
With the conventions above the Jacobians are::

    F = [[-skew(gyr_meas - b_g), -I, 0],
         [         0,             0, 0],
         [         0,             0, 0]]

    H = [[skew(R^T @ g_w), 0, +I]]

The paper prints ``-I`` for the bias block of ``H`` (equation 13) and corrects
the attitude by left-multiplication (equation 20); both are inconsistent with
the error convention fixed by its own equation (5).  ``paper_faithful=True``
reproduces the printed equations so the two variants can be benchmarked
side by side; see ``reports/derivation_review.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .adaptive import InnovationCovariance
from .quaternion import (
    quat_from_two_vectors,
    quat_normalize,
    quat_rotate,
    quat_to_matrix,
    skew,
)


def _inv3_sym(S: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric 3x3 matrix via its adjugate.

    LAPACK's fixed overhead dominates at this size, and the innovation
    covariance is always 3x3.  A non-invertible ``S`` means the filter has
    diverged, which is reported rather than regularised away.
    """
    a, b, c = S[0, 0], S[0, 1], S[0, 2]
    d, e = S[1, 1], S[1, 2]
    f = S[2, 2]
    # A diverging filter produces entries large enough to overflow the products
    # below.  That is a diagnosis, not a problem: the non-finite determinant is
    # caught immediately, so the intermediate overflow is expected and silenced.
    with np.errstate(over="ignore", invalid="ignore"):
        c00 = d * f - e * e
        c01 = c * e - b * f
        c02 = b * e - c * d
        det = a * c00 + b * c01 + c * c02
    if not math.isfinite(det) or abs(det) < 1e-300:
        raise FilterDivergenceError(f"innovation covariance is singular (det={det})")
    inv_det = 1.0 / det
    c11 = a * f - c * c
    c12 = b * c - a * e
    c22 = a * d - b * b
    return np.array(
        [
            [c00 * inv_det, c01 * inv_det, c02 * inv_det],
            [c01 * inv_det, c11 * inv_det, c12 * inv_det],
            [c02 * inv_det, c12 * inv_det, c22 * inv_det],
        ]
    )

__all__ = [
    "AdaptiveESKF",
    "ESKFDiagnostics",
    "FilterDivergenceError",
    "run_batch",
    "to_world_frame",
]

GRAVITY = 9.81


class FilterDivergenceError(RuntimeError):
    """Raised when the filter state or covariance stops being usable.

    The paper-faithful variants can diverge to the point where the innovation
    covariance becomes singular.  Surfacing that explicitly is more useful than
    either crashing in LAPACK or silently regularising the symptom away.
    """


@dataclass
class ESKFDiagnostics:
    """Per-sample internals recorded by :func:`run_batch`."""

    innovation: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    R_trace: np.ndarray = field(default_factory=lambda: np.empty(0))
    P_trace_tilt: np.ndarray = field(default_factory=lambda: np.empty(0))
    gain_norm: np.ndarray = field(default_factory=lambda: np.empty(0))
    acc_excess: np.ndarray = field(default_factory=lambda: np.empty(0))


class AdaptiveESKF:
    """Streaming 9-state (or 6-state) adaptive error-state Kalman filter.

    Parameters
    ----------
    rate
        Nominal sampling rate in Hz; used as the default time step.
    sigma_gyr
        Gyroscope white-noise density in rad/s/sqrt(Hz).
    sigma_acc
        Accelerometer measurement standard deviation in m/s^2.  Also the nominal
        measurement noise and the floor used by the adaptive scheme.
    sigma_bias_gyr, sigma_bias_acc
        Bias random-walk densities.  In ``paper_faithful`` mode the process
        noise follows equation (9) instead and these are ignored.
    window
        Innovation window length ``N`` of equation (17).
    eps
        Diagonal loading of equation (18).
    adaptive
        When False the measurement covariance stays at ``sigma_acc**2 * I``,
        giving the non-adaptive baseline the paper calls NAE.
    estimate_acc_bias
        When False the accelerometer bias is dropped from the state, yielding a
        6-state filter.  The bias is only weakly observable from a gravity
        measurement, so this is a meaningful alternative rather than a
        degradation.
    paper_faithful
        Convenience switch that turns on all three printed-equation deviations
        at once.  The individual flags below override it, so each can be
        ablated separately.
    paper_process_noise
        Build ``Q`` from equation (9) instead of the standard diagonal form.
    h_bias_sign
        Sign of the accelerometer-bias block of ``H``; ``+1`` is correct,
        ``-1`` is what equation (13) prints.
    inject_left
        Apply the attitude correction by left multiplication, as equation (20)
        prints, instead of the right multiplication the error convention
        requires.
    psd_mode
        Positive-definiteness strategy for the adaptive covariance; see
        :class:`~wattitude.adaptive.InnovationCovariance`.
    exact_phi
        Use the closed-form matrix exponential for the attitude block of the
        discrete transition matrix instead of the first-order approximation of
        equation (8).  On by default: the printed first-order form is not
        orthogonal and inflates the attitude covariance by roughly
        ``1 + |omega|^2 dt^2`` per step, which is negligible on gentle motion
        but corrupts heading above a few hundred degrees per second.  Set False
        to reproduce the paper; see section 6 of ``reports/derivation_review.md``
        for the measured cost either way.
    """

    def __init__(
        self,
        rate: float,
        sigma_gyr: float = 5e-3,
        sigma_acc: float = 0.15,
        sigma_bias_gyr: float = 1e-5,
        sigma_bias_acc: float = 1e-4,
        window: int = 100,
        eps: float = 1e-6,
        adaptive: bool = True,
        estimate_acc_bias: bool = True,
        paper_faithful: bool = False,
        paper_process_noise: bool | None = None,
        h_bias_sign: int | None = None,
        inject_left: bool | None = None,
        psd_mode: str = "clip",
        max_scale: float = 1e6,
        exact_phi: bool = True,
        gravity: float = GRAVITY,
        estimate_gravity: bool = True,
        init_tilt_std: float = np.deg2rad(5.0),
        init_bias_gyr_std: float = 1e-2,
        init_bias_acc_std: float = 0.2,
    ) -> None:
        self.rate = float(rate)
        self.dt_nominal = 1.0 / self.rate
        self.sigma_gyr = float(sigma_gyr)
        self.sigma_acc = float(sigma_acc)
        self.sigma_bias_gyr = float(sigma_bias_gyr)
        self.sigma_bias_acc = float(sigma_bias_acc)
        self.adaptive = bool(adaptive)
        self.estimate_acc_bias = bool(estimate_acc_bias)
        self.paper_faithful = bool(paper_faithful)
        self.paper_process_noise = (
            self.paper_faithful if paper_process_noise is None else bool(paper_process_noise)
        )
        self.h_bias_sign = (-1 if self.paper_faithful else 1) if h_bias_sign is None else int(h_bias_sign)
        if self.h_bias_sign not in (-1, 1):
            raise ValueError("h_bias_sign must be +1 or -1")
        self.inject_left = self.paper_faithful if inject_left is None else bool(inject_left)
        self.exact_phi = bool(exact_phi)
        self.gravity = float(gravity)
        self.estimate_gravity = bool(estimate_gravity)
        self.g_w = np.array([0.0, 0.0, self.gravity])

        self.dim = 9 if self.estimate_acc_bias else 6
        self.R_nominal = self.sigma_acc**2 * np.eye(3)

        self._adaptive_R = InnovationCovariance(
            dim=3,
            window=window,
            nominal=self.R_nominal,
            eps=eps,
            psd_mode=psd_mode,
            max_scale=max_scale,
        )

        self._init_P = np.diag(
            np.concatenate(
                [
                    np.full(3, init_tilt_std**2),
                    np.full(3, init_bias_gyr_std**2),
                    np.full(3, init_bias_acc_std**2)[: self.dim - 6],
                ]
            )
        )

        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias_gyr = np.zeros(3)
        self.bias_acc = np.zeros(3)
        self.P = self._init_P.copy()
        self.R = self.R_nominal.copy()
        self.innovation = np.zeros(3)
        self._gain_norm = 0.0

        # Preallocated buffers and cached constants.  The per-sample cost is
        # dominated by NumPy call overhead on 9x9 matrices, so reusing these
        # matters more than the arithmetic itself.
        self._I = np.eye(self.dim)
        self._I3 = np.eye(3)
        self._H = np.zeros((3, self.dim))
        if self.dim == 9:
            np.fill_diagonal(self._H[:, 6:9], float(self.h_bias_sign))
        self._Phi = np.eye(self.dim)
        self._Phi[0:3, 3:6] = -self._I3  # overwritten per step, shape fixed here
        self._Q_dt: float | None = None
        self._Q = np.zeros((self.dim, self.dim))
        self._omega = np.zeros(3)

    # ------------------------------------------------------------------ setup

    def initialize(
        self,
        gyr_window: np.ndarray | None = None,
        acc_window: np.ndarray | None = None,
        quat: np.ndarray | None = None,
    ) -> "AdaptiveESKF":
        """Initialise from a (preferably static) window of measurements.

        Roll and pitch come from levelling the mean accelerometer reading; yaw
        is left at zero because a 6-axis IMU carries no absolute heading, which
        is exactly the paper's definition of the world frame as the initial body
        frame.  The gyroscope bias is seeded with the mean angular rate.
        """
        self.P = self._init_P.copy()
        self.bias_acc = np.zeros(3)
        self._adaptive_R.reset()
        self.R = self.R_nominal.copy()
        self.innovation = np.zeros(3)

        if quat is not None:
            self.quat = quat_normalize(np.asarray(quat, dtype=float).reshape(4))
        elif acc_window is not None:
            acc_mean = np.mean(np.atleast_2d(acc_window), axis=0)
            if np.linalg.norm(acc_mean) < 1e-6:
                raise ValueError("accelerometer window has (near) zero mean")
            if self.estimate_gravity:
                # Local gravity plus the sensor's scale factor; on the BROAD IMU
                # this is 9.93 rather than 9.81, and the difference would
                # otherwise show up as a permanent innovation offset.
                self.gravity = float(np.linalg.norm(acc_mean))
                self.g_w = np.array([0.0, 0.0, self.gravity])
            # At rest acc_meas = R^T g_w, so R maps acc_mean onto g_w.
            self.quat = quat_from_two_vectors(acc_mean, self.g_w)
        else:
            self.quat = np.array([1.0, 0.0, 0.0, 0.0])

        if gyr_window is not None:
            self.bias_gyr = np.mean(np.atleast_2d(gyr_window), axis=0)
        else:
            self.bias_gyr = np.zeros(3)
        return self

    # -------------------------------------------------------------- internals

    def _process_noise(self, dt: float) -> np.ndarray:
        """Discrete process noise covariance, cached because dt rarely changes."""
        if self._Q_dt != dt:
            self._Q = self._build_process_noise(dt)
            self._Q_dt = dt
        return self._Q

    def _build_process_noise(self, dt: float) -> np.ndarray:
        if self.paper_process_noise:
            # Equation (9): Q = G diag(sg^2 I, sa^2 I) G^T dt with G from (6).
            # The same white gyro noise drives dtheta and db_g, which makes the
            # attitude/bias block singular; reproduced here on purpose.
            G = np.zeros((self.dim, 6))
            G[0:3, 0:3] = -np.eye(3)
            G[3:6, 0:3] = np.eye(3)
            if self.dim == 9:
                G[6:9, 3:6] = np.eye(3)
            noise = np.diag(
                np.concatenate([np.full(3, self.sigma_gyr**2), np.full(3, self.sigma_acc**2)])
            )
            return G @ noise @ G.T * dt

        blocks = [
            np.full(3, self.sigma_gyr**2),
            np.full(3, self.sigma_bias_gyr**2),
        ]
        if self.dim == 9:
            blocks.append(np.full(3, self.sigma_bias_acc**2))
        return np.diag(np.concatenate(blocks)) * dt

    def _transition(self, omega: np.ndarray, dt: float) -> np.ndarray:
        """Discrete transition matrix for the error state."""
        if not self.exact_phi:
            # Equation (8): first order in dt.  Written into the preallocated
            # buffer, whose only non-constant block is the attitude one.
            Phi = self._Phi
            wx, wy, wz = omega[0] * dt, omega[1] * dt, omega[2] * dt
            Phi[0, 0] = 1.0
            Phi[0, 1] = wz
            Phi[0, 2] = -wy
            Phi[1, 0] = -wz
            Phi[1, 1] = 1.0
            Phi[1, 2] = wx
            Phi[2, 0] = wy
            Phi[2, 1] = -wx
            Phi[2, 2] = 1.0
            Phi[0:3, 3:6] = -self._I3 * dt
            return Phi

        # Closed-form matrix exponential of the attitude block: a rotation by
        # -omega*dt, with the dtheta/db_g coupling given by the right Jacobian.
        # Unlike equation (8) this block is orthogonal, so it cannot inflate the
        # covariance; see section 6 of reports/derivation_review.md.
        Phi = np.eye(self.dim)
        W = skew(omega)
        rate = float(np.linalg.norm(omega))
        theta = rate * dt
        if not math.isfinite(theta):
            # A diverged filter feeds non-finite rates in here.  Propagating the
            # NaN is correct: the singular innovation covariance it produces is
            # caught and reported by _inv3_sym on the next update.
            Phi[0:3, 0:3] = np.nan
            Phi[0:3, 3:6] = np.nan
            return Phi
        if theta < 1e-8:
            Phi[0:3, 0:3] = np.eye(3) - W * dt
            Phi[0:3, 3:6] = -np.eye(3) * dt
        else:
            K = skew(omega / rate)
            KK = K @ K
            Phi[0:3, 0:3] = np.eye(3) - np.sin(theta) * K + (1 - np.cos(theta)) * KK
            Phi[0:3, 3:6] = -(
                np.eye(3) * dt
                - (1 - np.cos(theta)) / rate * K
                + (theta - np.sin(theta)) / rate * KK
            )
        return Phi

    def _measurement_jacobian(self, R_bw: np.ndarray) -> np.ndarray:
        """Equation (13).  Writes into a preallocated buffer.

        Only the tilt block varies; the bias block is a constant signed identity
        set once in the constructor.  The printed equation uses -I there, which
        ``h_bias_sign`` reproduces.
        """
        x, y, z = R_bw.T @ self.g_w
        H = self._H
        H[0, 0] = 0.0
        H[0, 1] = -z
        H[0, 2] = y
        H[1, 0] = z
        H[1, 1] = 0.0
        H[1, 2] = -x
        H[2, 0] = -y
        H[2, 1] = x
        H[2, 2] = 0.0
        return H

    # ------------------------------------------------------------------ step

    def update(self, gyr: np.ndarray, acc: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Advance the filter by one sample and return the current quaternion.

        The quaternion algebra is inlined rather than delegated to
        :mod:`wattitude.quaternion`, because at a few hundred Hz over hours of
        data the NumPy call overhead dominates the arithmetic.
        ``tests/test_eskf_fastpath.py`` pins the inlined version to the
        reference implementation.
        """
        dt = self.dt_nominal if dt is None else float(dt)

        # --- propagation -------------------------------------------------
        bg = self.bias_gyr
        wx = gyr[0] - bg[0]
        wy = gyr[1] - bg[1]
        wz = gyr[2] - bg[2]
        omega = self._omega
        omega[0], omega[1], omega[2] = wx, wy, wz

        vx, vy, vz = wx * dt, wy * dt, wz * dt
        a2 = vx * vx + vy * vy + vz * vz
        if a2 < 1e-16:
            dw = 1.0 - a2 * 0.125
            scale = 0.5
        else:
            angle = math.sqrt(a2)
            half = 0.5 * angle
            dw = math.cos(half)
            scale = math.sin(half) / angle
        self._apply_increment(dw, scale * vx, scale * vy, scale * vz, left=False)

        Phi = self._transition(omega, dt)
        self.P = Phi @ self.P @ Phi.T + self._process_noise(dt)
        self.P += self.P.T
        self.P *= 0.5

        # --- measurement -------------------------------------------------
        R_bw = quat_to_matrix(self.quat)
        H = self._measurement_jacobian(R_bw)
        g_body = R_bw.T @ self.g_w
        y = acc - g_body - self.bias_acc  # equation (15)
        self.innovation = y

        HP = H @ self.P
        hph = HP @ H.T
        if self.adaptive:
            self._adaptive_R.push(y)
            self.R = self._adaptive_R.estimate(hph)
        else:
            self.R = self.R_nominal

        S = hph + self.R
        # Equation (14) prints a transpose where the inverse belongs; an
        # explicit 3x3 inverse is used because LAPACK's overhead dominates at
        # this size.
        K = (_inv3_sym(S) @ HP).T
        self._gain_norm = float(np.sqrt(np.sum(K * K)))

        dx = K @ y

        # Joseph form keeps P symmetric positive-definite even though the yaw
        # direction is unobservable and its variance grows without bound.
        IKH = self._I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        self.P += self.P.T
        self.P *= 0.5

        # --- injection ---------------------------------------------------
        # First-order small-angle quaternion of equation (20).
        self._apply_increment(1.0, 0.5 * dx[0], 0.5 * dx[1], 0.5 * dx[2], left=self.inject_left)
        bg[0] += dx[3]
        bg[1] += dx[4]
        bg[2] += dx[5]
        if self.dim == 9:
            ba = self.bias_acc
            ba[0] += dx[6]
            ba[1] += dx[7]
            ba[2] += dx[8]

        if not math.isfinite(self.quat[0]) or not math.isfinite(bg[0]):
            raise FilterDivergenceError("state became non-finite")
        return self.quat

    def _apply_increment(self, dw: float, dx_: float, dy_: float, dz_: float, left: bool) -> None:
        """Compose a quaternion increment onto the nominal state, in place."""
        q = self.quat
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        if left:
            pw, px, py, pz = dw, dx_, dy_, dz_
            rw, rx, ry, rz = qw, qx, qy, qz
        else:
            pw, px, py, pz = qw, qx, qy, qz
            rw, rx, ry, rz = dw, dx_, dy_, dz_
        nw = pw * rw - px * rx - py * ry - pz * rz
        nx = pw * rx + px * rw + py * rz - pz * ry
        ny = pw * ry - px * rz + py * rw + pz * rx
        nz = pw * rz + px * ry - py * rx + pz * rw
        norm = math.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
        if norm < 1e-12:
            raise FilterDivergenceError("quaternion norm collapsed")
        if nw < 0.0:
            norm = -norm
        q[0] = nw / norm
        q[1] = nx / norm
        q[2] = ny / norm
        q[3] = nz / norm

    # ----------------------------------------------------------- properties

    @property
    def rotation_matrix(self) -> np.ndarray:
        return quat_to_matrix(self.quat)

    @property
    def tilt_covariance(self) -> np.ndarray:
        return self.P[0:3, 0:3]

    def euler_rpy(self) -> np.ndarray:
        """Roll, pitch, yaw in radians (ZYX intrinsic)."""
        w, x, y, z = self.quat
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw])


def run_batch(
    gyr: np.ndarray,
    acc: np.ndarray,
    rate: float,
    init_samples: int = 0,
    diagnostics: bool = False,
    filter_obj: AdaptiveESKF | None = None,
    **kwargs,
):
    """Run the filter over a whole sequence.

    Parameters
    ----------
    gyr, acc
        ``(T, 3)`` arrays in rad/s and m/s^2.
    rate
        Sampling rate in Hz.
    init_samples
        Number of leading samples used for static initialisation.  Zero means
        level from the first sample alone.
    diagnostics
        Also return per-sample internals.
    filter_obj
        Pre-configured filter instance; ``kwargs`` are forwarded to the
        constructor when this is None.

    Returns
    -------
    quats, biases, diag
        ``quats`` is ``(T, 4)``, ``biases`` is ``(T, 6)`` (gyro then accel) and
        ``diag`` is an :class:`ESKFDiagnostics` or None.
    """
    gyr = np.asarray(gyr, dtype=float)
    acc = np.asarray(acc, dtype=float)
    if gyr.shape != acc.shape or gyr.ndim != 2 or gyr.shape[1] != 3:
        raise ValueError("gyr and acc must both be (T, 3) arrays of equal shape")
    T = gyr.shape[0]

    est = filter_obj if filter_obj is not None else AdaptiveESKF(rate=rate, **kwargs)
    n_init = max(int(init_samples), 1)
    est.initialize(gyr_window=gyr[:n_init] if init_samples > 0 else None, acc_window=acc[:n_init])

    quats = np.empty((T, 4))
    biases = np.empty((T, 6))
    if diagnostics:
        innov = np.empty((T, 3))
        r_trace = np.empty(T)
        p_trace = np.empty(T)
        gain = np.empty(T)
        excess = np.empty(T)

    dt = 1.0 / float(rate)
    for k in range(T):
        quats[k] = est.update(gyr[k], acc[k], dt)
        biases[k, 0:3] = est.bias_gyr
        biases[k, 3:6] = est.bias_acc
        if diagnostics:
            innov[k] = est.innovation
            r_trace[k] = np.trace(est.R)
            p_trace[k] = np.trace(est.tilt_covariance)
            gain[k] = est._gain_norm
            excess[k] = abs(np.linalg.norm(acc[k]) - est.gravity)

    diag = None
    if diagnostics:
        diag = ESKFDiagnostics(
            innovation=innov,
            R_trace=r_trace,
            P_trace_tilt=p_trace,
            gain_norm=gain,
            acc_excess=excess,
        )
    return quats, biases, diag


def to_world_frame(
    gyr: np.ndarray,
    acc: np.ndarray,
    quats: np.ndarray,
    remove_gravity: bool = False,
    gravity: float = GRAVITY,
):
    """Rotate body-frame IMU measurements into the gravity-aligned world frame.

    This is the interface the downstream neural odometry consumes: the paper's
    front-end hands the network ``(gyr_w, acc_w)`` so it never has to learn
    rotation-invariant features.

    Parameters
    ----------
    remove_gravity
        Subtract ``gravity`` from the vertical component, giving linear
        acceleration.  Off by default because ``acc_w`` then no longer matches
        what the accelerometer reports, but usually what a motion model wants.
    gravity
        Magnitude to subtract.  Use the filter's own ``est.gravity`` rather than
        the nominal 9.81 when it was estimated at initialisation, or the
        residual shows up as a constant vertical offset.
    """
    gyr = np.asarray(gyr, dtype=float)
    acc = np.asarray(acc, dtype=float)
    quats = np.asarray(quats, dtype=float)
    if not (gyr.shape[0] == acc.shape[0] == quats.shape[0]):
        raise ValueError("gyr, acc and quats must share the same leading dimension")
    acc_w = quat_rotate(quats, acc)
    if remove_gravity:
        acc_w = acc_w.copy()
        acc_w[:, 2] -= gravity
    return quat_rotate(quats, gyr), acc_w

"""Reference orientation estimators, behind one common interface.

Every estimator has the signature

    fn(gyr, acc, rate, mag=None, init_slice=None, **params) -> (T, 4) quaternions

with quaternions in the same ``[w, x, y, z]`` body-to-world convention as
:mod:`wattitude.quaternion`.

The Madgwick and Mahony implementations follow the C++ sources shipped with the
BROAD benchmark (``example_code/madgwick_mahony_c_code``, MIT licence) so that
the 9-axis variants reproduce its published scores; see
``tests/test_broad.py::test_madgwick_9d_reproduces_the_published_tagp_score``.
Both algorithms already expect the accelerometer to read ``+g`` on the vertical
axis at rest, matching our world frame.  Their magnetometer reference direction
lies in the x-z plane, i.e. x points north, so the 9-axis path applies the same
+90 degree yaw correction as BROAD's ``process_data.py`` to land in ENU.  The
6-axis paths need no correction because heading is unobservable and is removed
by the metric.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .eskf import GRAVITY, AdaptiveESKF, run_batch
from .quaternion import exp_q, quat_from_two_vectors, quat_multiply, quat_normalize

__all__ = [
    "ESTIMATORS",
    "PARAM_GRIDS",
    "level_from_accel",
    "gyro_only",
    "madgwick",
    "madgwick_adaptive",
    "mahony",
    "vqf_6d",
    "eskf",
]

_ENU_FIX = np.array([1 / math.sqrt(2), 0.0, 0.0, 1 / math.sqrt(2)])


def _init_window(n: int, init_slice: slice | None) -> slice:
    return slice(0, min(100, n)) if init_slice is None else init_slice


def level_from_accel(acc: np.ndarray, init_slice: slice | None = None) -> np.ndarray:
    """Zero-yaw attitude that levels the mean accelerometer reading."""
    window = np.atleast_2d(acc)[_init_window(len(acc), init_slice)]
    return quat_from_two_vectors(np.mean(window, axis=0), np.array([0.0, 0.0, 1.0]))


# --------------------------------------------------------------- gyro only


def gyro_only(
    gyr: np.ndarray, acc: np.ndarray, rate: float, mag=None, init_slice=None, **_
) -> np.ndarray:
    """Open-loop strapdown integration: the floor every filter must beat."""
    dt = 1.0 / float(rate)
    bias = np.mean(np.atleast_2d(gyr)[_init_window(len(gyr), init_slice)], axis=0)
    q = level_from_accel(acc, init_slice)
    out = np.empty((len(gyr), 4))
    for k in range(len(gyr)):
        q = quat_normalize(quat_multiply(q, exp_q((gyr[k] - bias) * dt)))
        out[k] = q
    return out


# ---------------------------------------------------------------- Madgwick


def _madgwick_loop(gyr, acc, mag, rate, beta_seq, q_init):
    """Shared inner loop; ``mag`` of None selects the 6-axis gradient step."""
    q0, q1, q2, q3 = (float(v) for v in q_init)
    dt = 1.0 / float(rate)
    n = len(gyr)
    out = np.empty((n, 4))
    use_mag = mag is not None

    for k in range(n):
        gx, gy, gz = gyr[k]
        ax, ay, az = acc[k]
        beta = beta_seq[k]

        qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm > 0.0:
            ax, ay, az = ax / norm, ay / norm, az / norm
            q0q0, q1q1, q2q2, q3q3 = q0 * q0, q1 * q1, q2 * q2, q3 * q3

            if use_mag:
                mx, my, mz = mag[k]
                mnorm = math.sqrt(mx * mx + my * my + mz * mz)
                if mnorm == 0.0:
                    use_mag_here = False
                else:
                    mx, my, mz = mx / mnorm, my / mnorm, mz / mnorm
                    use_mag_here = True
            else:
                use_mag_here = False

            if use_mag_here:
                _2q0mx, _2q0my, _2q0mz, _2q1mx = 2 * q0 * mx, 2 * q0 * my, 2 * q0 * mz, 2 * q1 * mx
                _2q0, _2q1, _2q2, _2q3 = 2 * q0, 2 * q1, 2 * q2, 2 * q3
                _2q0q2, _2q2q3 = 2 * q0 * q2, 2 * q2 * q3
                q0q1, q0q2, q0q3 = q0 * q1, q0 * q2, q0 * q3
                q1q2, q1q3, q2q3 = q1 * q2, q1 * q3, q2 * q3

                hx = (
                    mx * q0q0 - _2q0my * q3 + _2q0mz * q2 + mx * q1q1
                    + _2q1 * my * q2 + _2q1 * mz * q3 - mx * q2q2 - mx * q3q3
                )
                hy = (
                    _2q0mx * q3 + my * q0q0 - _2q0mz * q1 + _2q1mx * q2
                    - my * q1q1 + my * q2q2 + _2q2 * mz * q3 - my * q3q3
                )
                _2bx = math.sqrt(hx * hx + hy * hy)
                _2bz = (
                    -_2q0mx * q2 + _2q0my * q1 + mz * q0q0 + _2q1mx * q3
                    - mz * q1q1 + _2q2 * my * q3 - mz * q2q2 + mz * q3q3
                )
                _4bx, _4bz = 2 * _2bx, 2 * _2bz

                ex = _2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx
                ey = _2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my
                ez = _2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz
                fx = 2.0 * q1q3 - _2q0q2 - ax
                fy = 2.0 * q0q1 + _2q2q3 - ay
                fz = 1 - 2.0 * q1q1 - 2.0 * q2q2 - az

                s0 = (
                    -_2q2 * fx + _2q1 * fy - _2bz * q2 * ex
                    + (-_2bx * q3 + _2bz * q1) * ey + _2bx * q2 * ez
                )
                s1 = (
                    _2q3 * fx + _2q0 * fy - 4.0 * q1 * fz + _2bz * q3 * ex
                    + (_2bx * q2 + _2bz * q0) * ey + (_2bx * q3 - _4bz * q1) * ez
                )
                s2 = (
                    -_2q0 * fx + _2q3 * fy - 4.0 * q2 * fz
                    + (-_4bx * q2 - _2bz * q0) * ex
                    + (_2bx * q1 + _2bz * q3) * ey + (_2bx * q0 - _4bz * q2) * ez
                )
                s3 = (
                    _2q1 * fx + _2q2 * fy + (-_4bx * q3 + _2bz * q1) * ex
                    + (-_2bx * q0 + _2bz * q2) * ey + _2bx * q1 * ez
                )
            else:
                _2q0, _2q1, _2q2, _2q3 = 2 * q0, 2 * q1, 2 * q2, 2 * q3
                _4q0, _4q1, _4q2 = 4 * q0, 4 * q1, 4 * q2
                _8q1, _8q2 = 8 * q1, 8 * q2
                s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
                s1 = (
                    _4q1 * q3q3 - _2q3 * ax + 4.0 * q0q0 * q1 - _2q0 * ay
                    - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az
                )
                s2 = (
                    4.0 * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay
                    - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az
                )
                s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay

            snorm = math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
            if snorm > 0.0:
                qDot1 -= beta * s0 / snorm
                qDot2 -= beta * s1 / snorm
                qDot3 -= beta * s2 / snorm
                qDot4 -= beta * s3 / snorm

        q0 += qDot1 * dt
        q1 += qDot2 * dt
        q2 += qDot3 * dt
        q3 += qDot4 * dt
        qnorm = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        q0, q1, q2, q3 = q0 / qnorm, q1 / qnorm, q2 / qnorm, q3 / qnorm
        out[k, 0], out[k, 1], out[k, 2], out[k, 3] = q0, q1, q2, q3
    return out


def _quat_from_acc_mag(acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """BROAD's accelerometer+magnetometer initialisation (x north, z up)."""
    from .quaternion import matrix_to_quat

    z = np.asarray(acc, dtype=float)
    x = np.cross(np.cross(z, -np.asarray(mag, dtype=float)), z)
    y = np.cross(z, x)
    R = np.column_stack(
        [x / np.linalg.norm(x), y / np.linalg.norm(y), z / np.linalg.norm(z)]
    )
    return matrix_to_quat(R)


def madgwick(
    gyr: np.ndarray,
    acc: np.ndarray,
    rate: float,
    mag: np.ndarray | None = None,
    init_slice: slice | None = None,
    beta: float = 0.12,
    **_,
) -> np.ndarray:
    """Madgwick's complementary filter, 6-axis by default."""
    if mag is not None:
        q_init = _quat_from_acc_mag(acc[0], mag[0])
    else:
        q_init = level_from_accel(acc, init_slice)
    out = _madgwick_loop(gyr, acc, mag, rate, np.full(len(gyr), float(beta)), q_init)
    if mag is not None:
        # Madgwick's magnetic reference is along x; rotate into ENU.
        out = np.array([quat_multiply(_ENU_FIX, q) for q in out])
    return out


def madgwick_adaptive(
    gyr: np.ndarray,
    acc: np.ndarray,
    rate: float,
    mag: np.ndarray | None = None,
    init_slice: slice | None = None,
    beta: float = 0.12,
    beta_min_ratio: float = 0.02,
    sensitivity: float = 8.0,
    gravity: float = GRAVITY,
    **_,
) -> np.ndarray:
    """Madgwick with a motion-dependent gain, the paper's WNIO(AMA) comparison.

    The paper states only that ``beta`` is "dynamically adjusted" and that "a
    lower beta allows the filter to mitigate the influence of external
    acceleration", without giving the law.  The schedule used here reduces the
    gain exponentially in the relative deviation of the accelerometer magnitude
    from gravity, which is the standard way to express that idea::

        beta_k = beta * exp(-sensitivity * | |a_k| - g | / g)

    clipped below at ``beta * beta_min_ratio``.  ``sensitivity`` is tuned on the
    benchmark alongside ``beta`` so the variant is not handicapped by the choice.
    """
    acc = np.asarray(acc, dtype=float)
    g = float(np.linalg.norm(np.mean(acc[_init_window(len(acc), init_slice)], axis=0)))
    g = g if g > 1e-6 else gravity
    excess = np.abs(np.linalg.norm(acc, axis=1) - g) / g
    beta_seq = np.maximum(beta * np.exp(-sensitivity * excess), beta * beta_min_ratio)

    q_init = (
        _quat_from_acc_mag(acc[0], mag[0]) if mag is not None
        else level_from_accel(acc, init_slice)
    )
    out = _madgwick_loop(gyr, acc, mag, rate, beta_seq, q_init)
    if mag is not None:
        out = np.array([quat_multiply(_ENU_FIX, q) for q in out])
    return out


# ------------------------------------------------------------------ Mahony


def mahony(
    gyr: np.ndarray,
    acc: np.ndarray,
    rate: float,
    mag: np.ndarray | None = None,
    init_slice: slice | None = None,
    Kp: float = 0.74,
    Ki: float = 0.0012,
    **_,
) -> np.ndarray:
    """Mahony's explicit complementary filter, 6-axis by default."""
    two_kp, two_ki = 2.0 * float(Kp), 2.0 * float(Ki)
    dt = 1.0 / float(rate)
    q_init = (
        _quat_from_acc_mag(acc[0], mag[0]) if mag is not None
        else level_from_accel(acc, init_slice)
    )
    q0, q1, q2, q3 = (float(v) for v in q_init)
    ifbx = ifby = ifbz = 0.0
    n = len(gyr)
    out = np.empty((n, 4))

    for k in range(n):
        gx, gy, gz = (float(v) for v in gyr[k])
        ax, ay, az = acc[k]
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm > 0.0:
            ax, ay, az = ax / norm, ay / norm, az / norm
            q0q0, q0q1, q0q2, q0q3 = q0 * q0, q0 * q1, q0 * q2, q0 * q3
            q1q1, q1q2, q1q3 = q1 * q1, q1 * q2, q1 * q3
            q2q2, q2q3, q3q3 = q2 * q2, q2 * q3, q3 * q3

            halfvx = q1q3 - q0q2
            halfvy = q0q1 + q2q3
            halfvz = q0q0 - 0.5 + q3q3
            halfex = ay * halfvz - az * halfvy
            halfey = az * halfvx - ax * halfvz
            halfez = ax * halfvy - ay * halfvx

            if mag is not None:
                mx, my, mz = mag[k]
                mnorm = math.sqrt(mx * mx + my * my + mz * mz)
                if mnorm > 0.0:
                    mx, my, mz = mx / mnorm, my / mnorm, mz / mnorm
                    hx = 2.0 * (mx * (0.5 - q2q2 - q3q3) + my * (q1q2 - q0q3) + mz * (q1q3 + q0q2))
                    hy = 2.0 * (mx * (q1q2 + q0q3) + my * (0.5 - q1q1 - q3q3) + mz * (q2q3 - q0q1))
                    bx = math.sqrt(hx * hx + hy * hy)
                    bz = 2.0 * (mx * (q1q3 - q0q2) + my * (q2q3 + q0q1) + mz * (0.5 - q1q1 - q2q2))
                    halfwx = bx * (0.5 - q2q2 - q3q3) + bz * (q1q3 - q0q2)
                    halfwy = bx * (q1q2 - q0q3) + bz * (q0q1 + q2q3)
                    halfwz = bx * (q0q2 + q1q3) + bz * (0.5 - q1q1 - q2q2)
                    halfex += my * halfwz - mz * halfwy
                    halfey += mz * halfwx - mx * halfwz
                    halfez += mx * halfwy - my * halfwx

            if two_ki > 0.0:
                ifbx += two_ki * halfex * dt
                ifby += two_ki * halfey * dt
                ifbz += two_ki * halfez * dt
                gx += ifbx
                gy += ifby
                gz += ifbz
            else:
                ifbx = ifby = ifbz = 0.0
            gx += two_kp * halfex
            gy += two_kp * halfey
            gz += two_kp * halfez

        gx *= 0.5 * dt
        gy *= 0.5 * dt
        gz *= 0.5 * dt
        qa, qb, qc = q0, q1, q2
        q0 += -qb * gx - qc * gy - q3 * gz
        q1 += qa * gx + qc * gz - q3 * gy
        q2 += qa * gy - qb * gz + q3 * gx
        q3 += qa * gz + qb * gy - qc * gx
        qnorm = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        q0, q1, q2, q3 = q0 / qnorm, q1 / qnorm, q2 / qnorm, q3 / qnorm
        out[k, 0], out[k, 1], out[k, 2], out[k, 3] = q0, q1, q2, q3

    if mag is not None:
        out = np.array([quat_multiply(_ENU_FIX, q) for q in out])
    return out


# --------------------------------------------------------------------- VQF


def vqf_6d(
    gyr: np.ndarray,
    acc: np.ndarray,
    rate: float,
    mag: np.ndarray | None = None,
    init_slice: slice | None = None,
    tauAcc: float = 3.0,
    motionBiasEstEnabled: bool = True,
    restBiasEstEnabled: bool = True,
    **_,
) -> np.ndarray:
    """VQF in 6-axis mode (Laidig and Seel, Information Fusion 2023).

    Current state of the art on this benchmark, used here as the upper reference
    rather than as a competitor on equal footing: it adds rest detection and
    motion-based bias estimation that Section IV.A does not have.
    """
    from vqf import offlineVQF

    out = offlineVQF(
        np.ascontiguousarray(gyr, dtype=float),
        np.ascontiguousarray(acc, dtype=float),
        None,
        1.0 / float(rate),
        dict(
            tauAcc=float(tauAcc),
            motionBiasEstEnabled=bool(motionBiasEstEnabled),
            restBiasEstEnabled=bool(restBiasEstEnabled),
        ),
    )
    return np.ascontiguousarray(out["quat6D"], dtype=float)


# ------------------------------------------------------------- our filter


def eskf(
    gyr: np.ndarray,
    acc: np.ndarray,
    rate: float,
    mag: np.ndarray | None = None,
    init_slice: slice | None = None,
    **params,
) -> np.ndarray:
    """The adaptive ESKF of Section IV.A; ``adaptive=False`` gives the NAE variant."""
    window = _init_window(len(gyr), init_slice)
    est = AdaptiveESKF(rate=rate, **params)
    est.initialize(gyr_window=gyr[window], acc_window=acc[window])
    quats, _, _ = run_batch(gyr, acc, rate=rate, filter_obj=est, init_samples=window.stop)
    return quats


# --------------------------------------------------------------- registry

ESTIMATORS: dict[str, Callable[..., np.ndarray]] = {
    "gyro_only": gyro_only,
    "madgwick": madgwick,
    "madgwick_adaptive": madgwick_adaptive,
    "mahony": mahony,
    "vqf": vqf_6d,
    "eskf": eskf,
}

#: Search grids used for the trial-agnostic and per-trial tuning.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "gyro_only": {},
    "madgwick": {"beta": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.16, 0.2, 0.25, 0.3]},
    "madgwick_adaptive": {
        "beta": [0.04, 0.08, 0.12, 0.2, 0.3],
        "sensitivity": [2.0, 4.0, 8.0, 16.0],
    },
    "mahony": {
        "Kp": [0.1, 0.2, 0.4, 0.74, 1.2, 2.0],
        "Ki": [0.0, 0.0012, 0.004],
    },
    "vqf": {"tauAcc": [0.5, 1.0, 2.0, 3.0, 6.0]},
    # BROAD tunes a single scalar per algorithm (Madgwick's beta, Mahony's Kp,
    # VQF's tauAcc).  The ESKF equivalent is sigma_acc, which sets how much the
    # accelerometer is trusted relative to the gyroscope; the remaining
    # parameters are held at physically motivated values so that the comparison
    # stays one-knob-for-one-knob.  The range runs from the instrument noise
    # level up to the scale of the external-acceleration disturbance, which on
    # BROAD's fast trials reaches 50 m/s^2.
    # The range extends well past the disturbance scale so that the optimum is
    # interior rather than at a grid boundary; beyond ~400 the filter tends
    # towards open-loop integration and the error rises again.
    "eskf": {
        "sigma_acc": [
            0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0,
            50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0,
        ]
    },
}

"""Hamilton quaternion utilities.

Conventions used consistently across this package:

* A quaternion is stored as ``[w, x, y, z]`` (scalar first) with unit norm.
* ``q`` represents the rotation from the body frame to the world frame, so that
  ``v_w = R(q) @ v_b``.
* Composition satisfies ``R(p * q) = R(p) @ R(q)``.
* The filter's error state uses the *right* (body-frame) convention
  ``R_true = R_est @ expm(skew(dtheta))``, i.e. ``q_true = q_est * exp_q(dtheta)``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "skew",
    "quat_multiply",
    "quat_conjugate",
    "quat_normalize",
    "quat_to_matrix",
    "matrix_to_quat",
    "exp_q",
    "log_q",
    "small_angle_quat",
    "quat_rotate",
    "quat_from_two_vectors",
    "quat_angle",
    "quat_relative",
    "quat_to_euler",
    "euler_to_quat",
    "euler_gimbal_risk",
    "GIMBAL_LOCK_PITCH",
]

_EPS = 1e-12


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that ``skew(a) @ b == np.cross(a, b)``."""
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product ``p * q``."""
    pw, px, py, pz = np.asarray(p, dtype=float).reshape(4)
    qw, qx, qy, qz = np.asarray(q, dtype=float).reshape(4)
    return np.array(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm < _EPS, 1.0, norm)
    q = q / norm
    # Pin the scalar part to be non-negative so that comparisons are unambiguous.
    sign = np.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q * sign


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix ``R`` with ``v_w = R @ v_b``."""
    w, x, y, z = np.asarray(q, dtype=float).reshape(4)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quat_to_matrix` using Shepperd's branch selection."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array(
            [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        )
    return quat_normalize(q)


def exp_q(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map from a rotation vector (body frame) to a quaternion."""
    v = np.asarray(rotvec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(v))
    if angle < 1e-8:
        # Second-order series keeps the map accurate for tiny increments.
        return quat_normalize(np.concatenate(([1.0 - angle * angle / 8.0], 0.5 * v)))
    axis = v / angle
    half = 0.5 * angle
    return np.concatenate(([np.cos(half)], np.sin(half) * axis))


def log_q(q: np.ndarray) -> np.ndarray:
    """Logarithmic map from a quaternion to a rotation vector (body frame)."""
    q = quat_normalize(np.asarray(q, dtype=float).reshape(4))
    w = float(np.clip(q[0], -1.0, 1.0))
    v = q[1:]
    norm_v = float(np.linalg.norm(v))
    if norm_v < 1e-10:
        return 2.0 * v
    angle = 2.0 * np.arctan2(norm_v, w)
    return angle * v / norm_v


def small_angle_quat(dtheta: np.ndarray) -> np.ndarray:
    """First-order small-angle quaternion ``[1, dtheta/2]`` (paper equation 20)."""
    v = np.asarray(dtheta, dtype=float).reshape(3)
    return quat_normalize(np.concatenate(([1.0], 0.5 * v)))


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate body-frame vectors into the world frame; supports batched inputs.

    ``q`` may be ``(4,)`` or ``(N, 4)`` and ``v`` may be ``(3,)`` or ``(N, 3)``.
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    v = np.atleast_2d(np.asarray(v, dtype=float))
    w = q[:, :1]
    u = q[:, 1:]
    # Rodrigues form of the quaternion sandwich product.
    out = (
        v * (2.0 * w**2 - 1.0)
        + 2.0 * u * np.sum(u * v, axis=1, keepdims=True)
        + 2.0 * w * np.cross(u, v)
    )
    return out


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector ``a`` onto unit vector ``b``."""
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    a = a / max(np.linalg.norm(a), _EPS)
    b = b / max(np.linalg.norm(b), _EPS)
    cross = np.cross(a, b)
    dot = float(np.dot(a, b))
    norm_cross = float(np.linalg.norm(cross))
    if norm_cross < 1e-10:
        if dot > 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        # Antiparallel: any axis orthogonal to ``a`` gives a 180 degree rotation.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return np.concatenate(([0.0], axis))
    angle = np.arctan2(norm_cross, dot)
    return exp_q(angle * cross / norm_cross)


def quat_relative(q_ref: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Relative rotation ``q_ref^-1 * q``."""
    return quat_multiply(quat_conjugate(q_ref), q)


def quat_angle(q: np.ndarray) -> float:
    """Absolute rotation angle of ``q`` in radians, in ``[0, pi]``."""
    q = np.asarray(q, dtype=float).reshape(4)
    return float(2.0 * np.arctan2(np.linalg.norm(q[1:]), abs(q[0])))


# Euler angles are provided for display only.  Every internal computation uses
# quaternions or matrices, because the three-angle parameterisation is singular
# and the singularity is reachable on real recordings.

GIMBAL_LOCK_PITCH = np.deg2rad(89.9)
"""Pitch beyond which roll and yaw stop being separately meaningful."""


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Intrinsic Z-Y-X Euler angles ``(roll, pitch, yaw)`` in radians.

    The convention is the usual aerospace one: the rotation is applied as yaw
    about world z, then pitch about the new y, then roll about the new x, so
    ``R = Rz(yaw) Ry(pitch) Rx(roll)``.  With our z-up world frame, yaw is
    heading, and roll and pitch together describe the tilt away from vertical.

    Ranges are ``roll, yaw in (-pi, pi]`` and ``pitch in [-pi/2, pi/2]``.

    Near ``pitch = +-pi/2`` the parameterisation is degenerate: only the
    combination ``roll -+ yaw`` is determined, and the two angles can swing
    through any value at all without the underlying attitude moving. There
    roll is reported as zero and the whole rotation is placed in yaw, which is
    a choice rather than a fact.  :func:`euler_gimbal_risk` flags the samples
    where this matters; do not read roll or yaw traces through those regions.

    Accepts ``(4,)`` or ``(T, 4)`` and returns a matching ``(3,)`` or ``(T, 3)``.
    """
    q = np.asarray(q, dtype=float)
    single = q.ndim == 1
    q = np.atleast_2d(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # pitch = -asin(R[2, 0]) with R[2, 0] = 2 (xz - wy).
    sin_pitch = np.clip(2.0 * (w * y - x * z), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)

    # Away from the singularity, roll and yaw come from the third row and first
    # column of R respectively.
    roll = np.arctan2(2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y))
    yaw = np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))

    locked = np.abs(sin_pitch) > np.sin(GIMBAL_LOCK_PITCH)
    if np.any(locked):
        # Only roll -+ yaw survives; put it all in yaw and zero the roll.
        # R[0, 1] and R[1, 1] reduce to sin and cos of that combination.
        r01 = 2.0 * (x[locked] * y[locked] - w[locked] * z[locked])
        r11 = 1.0 - 2.0 * (x[locked] * x[locked] + z[locked] * z[locked])
        roll[locked] = 0.0
        yaw[locked] = np.arctan2(-r01, r11)

    out = np.stack([roll, pitch, yaw], axis=1)
    return out[0] if single else out


def euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quat_to_euler`, exact away from the singularity."""
    e = np.asarray(euler, dtype=float)
    single = e.ndim == 1
    e = np.atleast_2d(e)
    half = 0.5 * e
    cr, cp, cy = np.cos(half[:, 0]), np.cos(half[:, 1]), np.cos(half[:, 2])
    sr, sp, sy = np.sin(half[:, 0]), np.sin(half[:, 1]), np.sin(half[:, 2])

    q = np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=1,
    )
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0.0] *= -1.0
    return q[0] if single else q


def euler_gimbal_risk(q: np.ndarray, margin_deg: float = 10.0) -> np.ndarray:
    """Boolean mask of samples whose pitch is within ``margin_deg`` of vertical.

    Roll and yaw become increasingly ill-conditioned as pitch approaches
    ``+-90`` degrees, well before the hard singularity: a small attitude change
    produces an arbitrarily large change in both angles.  Use this to mark or
    exclude those stretches when plotting, rather than presenting the swings as
    estimation error.
    """
    pitch = np.atleast_2d(quat_to_euler(q))[:, 1]
    return np.abs(pitch) > np.deg2rad(90.0 - margin_deg)

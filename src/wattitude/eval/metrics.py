"""Orientation error metrics.

The first group of functions is a faithful port of ``broad_utils.py`` from the
BROAD reference implementation (Laidig et al., *Data* 6(7):72, 2021, MIT licence)
so that numbers produced here are directly comparable to the published ones.

The second group addresses the magnetometer-free case.  BROAD's reference
pipeline initialises its algorithms from accelerometer *and* magnetometer, so its
estimates share an absolute heading with the ground truth.  A 6-axis filter has
no absolute heading at all, so its raw heading error is an arbitrary offset plus
a random walk.  Two consequences:

* **Inclination error is the metric to compare on.**  It is exactly invariant to
  a constant world-frame heading offset, because left-multiplying the error
  quaternion by ``[cos(psi/2), 0, 0, sin(psi/2)]`` rotates the ``(w, z)`` pair
  and the ``(x, y)`` pair among themselves and so preserves ``w^2 + z^2``.
* Heading error is only meaningful after removing the best constant offset, and
  even then the residual drift should be reported as a rate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = [
    "error_quat_earth",
    "total_error",
    "heading_error",
    "inclination_error",
    "signed_heading_error",
    "rmse",
    "OrientationErrors",
    "optimal_heading_offset",
    "apply_heading_offset",
    "orientation_errors",
    "error_series",
]


def _quatmult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Row-wise Hamilton product, broadcasting a single quaternion if given."""
    q1 = np.atleast_2d(np.asarray(q1, dtype=float))
    q2 = np.atleast_2d(np.asarray(q2, dtype=float))
    n = max(q1.shape[0], q2.shape[0])
    if q1.shape[0] == 1:
        q1 = np.broadcast_to(q1, (n, 4))
    if q2.shape[0] == 1:
        q2 = np.broadcast_to(q2, (n, 4))
    out = np.empty((n, 4))
    out[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    out[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    out[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    out[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
    return out


def _invquat(q: np.ndarray) -> np.ndarray:
    q = np.atleast_2d(np.asarray(q, dtype=float)).copy()
    q[:, 1:] *= -1.0
    return q


# ---------------------------------------------------------------- BROAD port


def error_quat_earth(imu_quat: np.ndarray, opt_quat: np.ndarray) -> np.ndarray:
    """Error quaternion expressed in the world frame: ``q_imu * inv(q_opt)``."""
    imu_quat = np.atleast_2d(np.asarray(imu_quat, dtype=float))
    opt_quat = np.atleast_2d(np.asarray(opt_quat, dtype=float))
    imu_quat = imu_quat / np.linalg.norm(imu_quat, axis=1)[:, None]
    opt_quat = opt_quat / np.linalg.norm(opt_quat, axis=1)[:, None]
    out = _quatmult(imu_quat, _invquat(opt_quat))
    return out / np.linalg.norm(out, axis=1)[:, None]


def total_error(q_diff: np.ndarray) -> np.ndarray:
    """Total absolute rotation angle of the error quaternion, in radians."""
    return 2 * np.arccos(np.clip(np.abs(q_diff[:, 0]), 0, 1))


def heading_error(q_diff_earth: np.ndarray) -> np.ndarray:
    """Heading portion of the error, in radians."""
    return 2 * np.arctan(np.abs(q_diff_earth[:, 3] / q_diff_earth[:, 0]))


def inclination_error(q_diff_earth: np.ndarray) -> np.ndarray:
    """Inclination (roll/pitch) portion of the error, in radians."""
    return 2 * np.arccos(
        np.clip(np.sqrt(q_diff_earth[:, 0] ** 2 + q_diff_earth[:, 3] ** 2), 0, 1)
    )


def rmse(diff: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.asarray(diff, dtype=float) ** 2)))


# ------------------------------------------------------ magnetometer-free use


def signed_heading_error(q_diff_earth: np.ndarray) -> np.ndarray:
    """Signed heading error in radians, continuous across the +/-pi wrap.

    Unlike :func:`heading_error` this keeps the sign, which is what makes a
    drift rate measurable.  The ground truth contains optical-tracking dropouts,
    so unwrapping is done over the finite samples only; a NaN would otherwise
    propagate through the cumulative sum and destroy everything after it.
    """
    angle = 2 * np.arctan2(q_diff_earth[:, 3], q_diff_earth[:, 0])
    out = np.full(angle.shape, np.nan)
    finite = np.isfinite(angle)
    if finite.any():
        out[finite] = np.unwrap(angle[finite])
    return out


def apply_heading_offset(imu_quat: np.ndarray, psi: float) -> np.ndarray:
    """Rotate estimates by a constant heading ``psi`` (radians) about world z."""
    q_yaw = np.array([[np.cos(0.5 * psi), 0.0, 0.0, np.sin(0.5 * psi)]])
    return _quatmult(q_yaw, imu_quat)


def optimal_heading_offset(
    imu_quat: np.ndarray, opt_quat: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Constant heading offset that minimises the total-error RMS.

    A 6-axis filter defines its own world frame up to an arbitrary rotation
    about gravity, so comparing total or heading error without removing this
    offset measures the frame convention rather than the algorithm.
    """
    q_diff = error_quat_earth(imu_quat, opt_quat)
    if mask is not None:
        q_diff = q_diff[np.asarray(mask, dtype=bool)]
    q_diff = q_diff[np.isfinite(q_diff).all(axis=1)]
    if len(q_diff) == 0:
        return 0.0
    w, z = q_diff[:, 0], q_diff[:, 3]

    def cost(psi: float) -> float:
        c, s = np.cos(0.5 * psi), np.sin(0.5 * psi)
        w_new = c * w - s * z
        return float(np.mean(np.arccos(np.clip(np.abs(w_new), 0, 1)) ** 2))

    # Closed-form chordal seed, then a coarse sweep to avoid local minima, then
    # a bracketed refinement.
    seed = -2.0 * np.arctan2(np.sum(z), np.sum(w))
    grid = np.concatenate([np.linspace(-np.pi, np.pi, 721), [seed]])
    costs = np.array([cost(p) for p in grid])
    best = float(grid[int(np.argmin(costs))])
    step = 2 * np.pi / 720
    res = minimize_scalar(cost, bracket=(best - step, best, best + step), method="brent")
    return float(res.x) if res.fun <= costs.min() else best


@dataclass
class OrientationErrors:
    """RMSE values in degrees, plus the 6-axis-specific diagnostics."""

    total_rmse_deg: float
    heading_rmse_deg: float
    inclination_rmse_deg: float
    heading_offset_deg: float
    heading_drift_deg_per_min: float
    n_samples: int

    def as_dict(self) -> dict:
        return asdict(self)


def error_series(
    imu_quat: np.ndarray,
    opt_quat: np.ndarray,
    movement: np.ndarray,
    align_heading: bool = True,
) -> dict[str, np.ndarray]:
    """Per-sample error angles in radians, for plotting rather than scoring.

    Uses the same heading alignment as :func:`orientation_errors`, so a series
    and its RMSE are always consistent.  Samples outside the movement phase and
    optical dropouts are returned as NaN.
    """
    movement = np.asarray(movement, dtype=bool)
    if align_heading:
        offset = optimal_heading_offset(imu_quat, opt_quat, movement)
        if offset != 0.0:
            imu_quat = apply_heading_offset(imu_quat, offset)

    q_diff = error_quat_earth(imu_quat, opt_quat)
    out = {
        "total": total_error(q_diff),
        "heading": heading_error(q_diff),
        "inclination": inclination_error(q_diff),
        "signed_heading": signed_heading_error(q_diff),
    }
    for value in out.values():
        value[~movement] = np.nan
    return out


def orientation_errors(
    imu_quat: np.ndarray,
    opt_quat: np.ndarray,
    movement: np.ndarray,
    rate: float,
    align_heading: bool = True,
) -> OrientationErrors:
    """Full error summary over the movement phase.

    Parameters
    ----------
    align_heading
        Remove the optimal constant heading offset before scoring.  Required for
        magnetometer-free estimates; set False to reproduce BROAD's raw
        definition for a 9-axis algorithm that shares the reference heading.

    Notes
    -----
    ``inclination_rmse_deg`` is unaffected by ``align_heading`` and is the
    metric to compare 6-axis algorithms on.  The drift rate is the slope of a
    linear fit to the signed heading error over the movement phase.
    """
    movement = np.asarray(movement, dtype=bool)
    offset = optimal_heading_offset(imu_quat, opt_quat, movement) if align_heading else 0.0
    if offset != 0.0:
        imu_quat = apply_heading_offset(imu_quat, offset)

    q_diff = error_quat_earth(imu_quat, opt_quat)
    signed = signed_heading_error(q_diff)

    idx = np.flatnonzero(movement & np.isfinite(signed))
    if len(idx) > 1:
        t = idx / float(rate)
        slope = float(np.polyfit(t, signed[idx], 1)[0])
    else:
        slope = 0.0

    return OrientationErrors(
        total_rmse_deg=np.rad2deg(rmse(total_error(q_diff)[movement])),
        heading_rmse_deg=np.rad2deg(rmse(heading_error(q_diff)[movement])),
        inclination_rmse_deg=np.rad2deg(rmse(inclination_error(q_diff)[movement])),
        heading_offset_deg=float(np.rad2deg(offset)),
        heading_drift_deg_per_min=float(np.rad2deg(slope) * 60.0),
        n_samples=int(movement.sum()),
    )

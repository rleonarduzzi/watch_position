"""Synthetic 6-axis IMU generator with exactly known ground truth.

Simulation is the only place where the filter's internal error can be compared
against its own covariance, which is what makes a consistency (NEES) check
possible.  The noise parameters use the same units and the same meaning as
:class:`~wattitude.eskf.AdaptiveESKF`, so a filter configured with the values
passed here is *correctly* tuned by construction:

* ``sigma_gyr``       rad/s/sqrt(Hz), white-noise density -> per-sample std ``sigma_gyr / sqrt(dt)``
* ``sigma_acc``       m/s^2, per-sample measurement std
* ``sigma_bias_gyr``  rad/s/sqrt(s), random-walk density  -> per-step std ``sigma_bias_gyr * sqrt(dt)``
* ``sigma_bias_acc``  m/s^2/sqrt(s), random-walk density

The attitude is integrated with the same zero-order-hold exponential map the
filter uses, so a noiseless, bias-free sequence must be tracked exactly; any
residual is a bookkeeping bug rather than a discretisation artefact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eskf import GRAVITY
from .quaternion import exp_q, quat_multiply, quat_normalize, quat_rotate

__all__ = ["SimData", "simulate_imu"]


@dataclass
class SimData:
    """A simulated sequence together with its ground truth."""

    t: np.ndarray
    gyr: np.ndarray
    acc: np.ndarray
    quat: np.ndarray
    omega: np.ndarray
    bias_gyr: np.ndarray
    bias_acc: np.ndarray
    acc_ext_world: np.ndarray
    rate: float
    rest_samples: int

    def __len__(self) -> int:
        return len(self.t)


def _band_limited_rates(
    t: np.ndarray, rng: np.random.Generator, amplitude: float, n_modes: int, f_max: float
) -> np.ndarray:
    """Smooth angular velocity as a random sum of sinusoids, one set per axis."""
    omega = np.zeros((len(t), 3))
    for axis in range(3):
        for _ in range(n_modes):
            f = rng.uniform(0.05, f_max)
            phase = rng.uniform(0.0, 2 * np.pi)
            omega[:, axis] += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * f * t + phase)
    # Normalise so that the requested amplitude is the per-axis RMS rate.
    rms = np.sqrt(np.mean(omega**2, axis=0))
    rms[rms < 1e-12] = 1.0
    return omega / rms * amplitude


def _external_acceleration(
    t: np.ndarray, rng: np.random.Generator, mode: str, amplitude: float, rest_samples: int
) -> np.ndarray:
    """World-frame linear acceleration of the sensor."""
    a = np.zeros((len(t), 3))
    if mode == "none":
        return a

    if mode == "swing":
        # A 1 Hz arm swing: large horizontal acceleration with a vertical
        # component at twice the fundamental, as produced by a pendulum.
        f = 1.0
        a[:, 0] = amplitude * np.sin(2 * np.pi * f * t)
        a[:, 1] = 0.6 * amplitude * np.cos(2 * np.pi * f * t + 0.7)
        a[:, 2] = 0.3 * amplitude * np.sin(4 * np.pi * f * t)
    elif mode == "episodes":
        # Alternating 4 s bursts of swing and 4 s of quiet motion, which is the
        # regime the adaptive covariance is designed for.
        f = 1.2
        burst = (np.floor(t / 4.0).astype(int) % 2) == 1
        a[:, 0] = amplitude * np.sin(2 * np.pi * f * t) * burst
        a[:, 1] = 0.6 * amplitude * np.cos(2 * np.pi * f * t + 0.7) * burst
        a[:, 2] = 0.3 * amplitude * np.sin(4 * np.pi * f * t) * burst
    elif mode == "taps":
        # Short impulses, the BROAD "tapping" disturbance.
        n_taps = max(int(t[-1] / 2.0), 1)
        width = 3
        for _ in range(n_taps):
            k = rng.integers(rest_samples, len(t) - width)
            a[k : k + width] += rng.normal(scale=amplitude * 3.0, size=(width, 3))
    else:
        raise ValueError(f"unknown external acceleration mode {mode!r}")

    a[:rest_samples] = 0.0
    return a


def simulate_imu(
    duration: float = 60.0,
    rate: float = 200.0,
    rotation_amplitude: float = 1.0,
    n_modes: int = 4,
    f_max: float = 2.0,
    sigma_gyr: float = 5e-3,
    sigma_acc: float = 0.15,
    sigma_bias_gyr: float = 1e-5,
    sigma_bias_acc: float = 1e-4,
    bias_gyr0: np.ndarray | float | None = None,
    bias_acc0: np.ndarray | float | None = None,
    ext_accel: str = "none",
    ext_amplitude: float = 5.0,
    rest_duration: float = 1.0,
    gravity: float = GRAVITY,
    seed: int = 0,
) -> SimData:
    """Generate a synthetic sequence.

    Parameters
    ----------
    rotation_amplitude
        Per-axis RMS angular rate in rad/s during the moving phase.
    ext_accel
        ``"none"``, ``"swing"``, ``"episodes"`` or ``"taps"``.
    rest_duration
        Length of an initial static phase, used by the filter to level and to
        seed the gyroscope bias.
    bias_gyr0, bias_acc0
        Constant turn-on biases.  ``None`` means zero.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / float(rate)
    n = int(round(duration * rate))
    t = np.arange(n) * dt
    rest_samples = int(round(rest_duration * rate))

    omega = _band_limited_rates(t, rng, rotation_amplitude, n_modes, f_max)
    omega[:rest_samples] = 0.0
    # Taper the onset so the rate is continuous at the end of the rest phase.
    taper = min(int(round(0.25 * rate)), max(n - rest_samples, 1))
    if taper > 1:
        ramp = np.linspace(0.0, 1.0, taper)[:, None]
        omega[rest_samples : rest_samples + taper] *= ramp

    acc_ext = _external_acceleration(t, rng, ext_accel, ext_amplitude, rest_samples)

    # Bias trajectories: constant turn-on value plus a random walk.
    bias_gyr = np.zeros((n, 3))
    bias_acc = np.zeros((n, 3))
    bias_gyr[0] = 0.0 if bias_gyr0 is None else np.broadcast_to(bias_gyr0, (3,))
    bias_acc[0] = 0.0 if bias_acc0 is None else np.broadcast_to(bias_acc0, (3,))
    if sigma_bias_gyr > 0.0:
        steps = rng.normal(scale=sigma_bias_gyr * np.sqrt(dt), size=(n - 1, 3))
        bias_gyr[1:] = bias_gyr[0] + np.cumsum(steps, axis=0)
    else:
        bias_gyr[1:] = bias_gyr[0]
    if sigma_bias_acc > 0.0:
        steps = rng.normal(scale=sigma_bias_acc * np.sqrt(dt), size=(n - 1, 3))
        bias_acc[1:] = bias_acc[0] + np.cumsum(steps, axis=0)
    else:
        bias_acc[1:] = bias_acc[0]

    # Attitude by zero-order-hold integration, matching the filter exactly.
    # Sample k carries the rate over the interval *ending* at t_k, which is the
    # convention the filter assumes: it propagates with gyr[k] and then applies
    # the accelerometer update at t_k.  Keeping the two in step is what makes
    # the noiseless sequence trackable to machine precision.
    quat = np.empty((n, 4))
    quat[0] = np.array([1.0, 0.0, 0.0, 0.0])
    for k in range(1, n):
        quat[k] = quat_normalize(quat_multiply(quat[k - 1], exp_q(omega[k] * dt)))

    g_w = np.array([0.0, 0.0, gravity])
    # Specific force in the body frame: R^T (g_w + a_ext).
    conj = quat.copy()
    conj[:, 1:] *= -1.0
    acc_true = quat_rotate(conj, g_w + acc_ext)

    gyr_meas = omega + bias_gyr
    acc_meas = acc_true + bias_acc
    if sigma_gyr > 0.0:
        gyr_meas = gyr_meas + rng.normal(scale=sigma_gyr / np.sqrt(dt), size=(n, 3))
    if sigma_acc > 0.0:
        acc_meas = acc_meas + rng.normal(scale=sigma_acc, size=(n, 3))

    return SimData(
        t=t,
        gyr=gyr_meas,
        acc=acc_meas,
        quat=quat,
        omega=omega,
        bias_gyr=bias_gyr,
        bias_acc=bias_acc,
        acc_ext_world=acc_ext,
        rate=float(rate),
        rest_samples=rest_samples,
    )

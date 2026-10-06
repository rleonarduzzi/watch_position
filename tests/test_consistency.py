"""Monte-Carlo filter consistency (NEES).

This is the strongest single check available in simulation: it compares the
filter's actual estimation error against the covariance the filter reports, and
so validates the Jacobians, the process noise and the measurement noise
jointly.  A sign error in ``H``, a missing term in ``Phi`` or a mis-scaled ``Q``
all show up here, whereas an accuracy test alone can be passed by a badly tuned
filter.  The two negative controls at the bottom demonstrate that power.

Definition.  With the error state ``dx = [log(q_est^-1 * q_true); b_true - b_est]``
the normalised estimation error squared is ``NEES = dx^T P^-1 dx``, whose
expectation is the state dimension when the filter is consistent.  Averaging
over ``M`` independent runs, ``M * ANEES`` is chi-square distributed with
``M * n`` degrees of freedom, which gives the acceptance band used below.

Two deliberate choices:

* **The adaptive covariance is switched off.**  Equation (17) is an intentional
  mis-statement of ``R`` designed to absorb unmodelled acceleration, so it is
  not expected to be statistically consistent.  Consistency is a property of the
  underlying filter; the adaptive behaviour is checked in ``test_sim.py``.
* **The primary tests run in a small-error regime.**  An EKF is only consistent
  to the order of its linearisation, and the second-order term of the gravity
  measurement, ``O(|dtheta|^2 * g)``, becomes comparable to the accelerometer
  noise once the attitude error reaches a few degrees.
  :func:`test_residual_inconsistency_is_linearisation_error` measures that
  effect separately and shows it vanishes as the error shrinks, which is what
  distinguishes it from a covariance bug.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from wattitude import AdaptiveESKF
from wattitude.quaternion import exp_q, log_q, quat_conjugate, quat_multiply, quat_normalize
from wattitude.simulate import simulate_imu

N_RUNS = 60
DURATION = 25.0
RATE = 100.0

# Noise levels chosen to keep the attitude error inside the linear regime over
# the whole sequence; see the module docstring.
NOISE = dict(sigma_gyr=1e-3, sigma_acc=0.12, sigma_bias_gyr=5e-5, sigma_bias_acc=5e-4)
INIT = dict(init_tilt_std=np.deg2rad(0.5), init_bias_gyr_std=1e-3, init_bias_acc_std=0.01)


def _make_filter(acc_bias: bool = True, q_scale: float = 1.0, **overrides) -> AdaptiveESKF:
    kwargs = {**NOISE, **INIT}
    for key in ("sigma_gyr", "sigma_bias_gyr", "sigma_bias_acc"):
        kwargs[key] *= np.sqrt(q_scale)
    kwargs.update(overrides)
    return AdaptiveESKF(
        rate=RATE,
        adaptive=False,
        estimate_acc_bias=acc_bias,
        estimate_gravity=False,
        exact_phi=True,
        **kwargs,
    )


def _seed_from_prior(est: AdaptiveESKF, sim, rng: np.random.Generator) -> None:
    """Draw the initial error from the filter's own prior.

    A consistency test is only meaningful if the initial error really is
    distributed as ``N(0, P0)``.  Levelling from a static window would instead
    produce an error far smaller than ``P0`` claims, making the NEES look
    optimistic for the first few seconds.
    """
    dx = np.linalg.cholesky(est.P) @ rng.normal(size=est.dim)
    # q_true = q_est * exp(dtheta)  =>  q_est = q_true * exp(-dtheta)
    est.quat = quat_normalize(quat_multiply(sim.quat[0], exp_q(-dx[0:3])))
    est.bias_gyr = sim.bias_gyr[0] - dx[3:6]
    if est.dim == 9:
        est.bias_acc = sim.bias_acc[0] - dx[6:9]


def _nees_trajectory(est: AdaptiveESKF, sim) -> np.ndarray:
    """Per-sample NEES for one run."""
    out = np.empty(len(sim.t))
    for k in range(len(sim.t)):
        est.update(sim.gyr[k], sim.acc[k])
        parts = [
            log_q(quat_multiply(quat_conjugate(est.quat), sim.quat[k])),
            sim.bias_gyr[k] - est.bias_gyr,
        ]
        if est.dim == 9:
            parts.append(sim.bias_acc[k] - est.bias_acc)
        dx = np.concatenate(parts)
        out[k] = float(dx @ np.linalg.solve(est.P, dx))
    return out


def _monte_carlo(n_runs: int = N_RUNS, acc_bias: bool = True, **filter_kwargs) -> np.ndarray:
    """ANEES over time, averaged across independent runs."""
    nees = []
    for run in range(n_runs):
        sim = simulate_imu(
            duration=DURATION, rate=RATE, ext_accel="none", rest_duration=0.0,
            rotation_amplitude=0.8, seed=1000 + run, **NOISE,
        )
        est = _make_filter(acc_bias=acc_bias, **filter_kwargs)
        _seed_from_prior(est, sim, np.random.default_rng(5000 + run))
        nees.append(_nees_trajectory(est, sim))
    return np.array(nees).mean(axis=0)


def _band(n_runs: int, dim: int, alpha: float = 0.01) -> tuple[float, float]:
    """Two-sided chi-square acceptance band on the ANEES."""
    dof = n_runs * dim
    return chi2.ppf(alpha / 2, dof) / n_runs, chi2.ppf(1 - alpha / 2, dof) / n_runs


@pytest.mark.slow
def test_nine_state_filter_is_consistent():
    anees = _monte_carlo()
    lo, hi = _band(N_RUNS, 9)
    inside = float(np.mean((anees >= lo) & (anees <= hi)))
    msg = (
        f"ANEES {anees.mean():.3f} (target 9.000), band [{lo:.3f}, {hi:.3f}], "
        f"{100 * inside:.1f}% of samples inside"
    )
    assert lo < anees.mean() < hi, msg
    # The band is per time step, and samples are correlated in time, so a small
    # excursion rate is expected even for a correct filter.
    assert inside > 0.90, msg


@pytest.mark.slow
def test_six_state_filter_is_consistent():
    """Dropping the weakly observable accelerometer bias must stay consistent."""
    anees = _monte_carlo(acc_bias=False)
    lo, hi = _band(N_RUNS, 6)
    inside = float(np.mean((anees >= lo) & (anees <= hi)))
    msg = (
        f"ANEES {anees.mean():.3f} (target 6.000), band [{lo:.3f}, {hi:.3f}], "
        f"{100 * inside:.1f}% of samples inside"
    )
    assert lo < anees.mean() < hi, msg
    assert inside > 0.90, msg


@pytest.mark.slow
def test_residual_inconsistency_is_linearisation_error():
    """Attribute the residual optimism to linearisation rather than to Q or R.

    The error magnitude is swept while the filter stays correctly matched to the
    simulator.  A mis-scaled ``Q`` or ``R`` would show up as a constant offset
    independent of the error magnitude, because both the error and the reported
    covariance would scale together.  Linearisation error cannot: it is driven
    by the second-order term ``O(|dtheta|^2 * g)`` of the gravity measurement
    relative to the accelerometer noise, so it has to grow with the error.
    """
    n_runs = 40
    results = {}
    for scale in (8.0, 1.0):
        nees = []
        for run in range(n_runs):
            sim = simulate_imu(
                duration=DURATION, rate=RATE, ext_accel="none", rest_duration=0.0,
                rotation_amplitude=0.8, seed=1000 + run, **NOISE,
            )
            est = _make_filter(**{k: v * scale for k, v in INIT.items()})
            _seed_from_prior(est, sim, np.random.default_rng(5000 + run))
            nees.append(_nees_trajectory(est, sim))
        results[scale] = float(np.array(nees).mean(axis=0).mean())

    lo, hi = _band(n_runs, 9)
    assert lo < results[1.0] < hi, f"small-error ANEES outside [{lo:.3f}, {hi:.3f}]: {results}"
    assert results[8.0] > results[1.0] + 0.3, results


@pytest.mark.slow
def test_nees_rejects_a_mis_scaled_process_noise():
    """Negative control: inflating Q by 100x must be detected.

    The filter becomes overly pessimistic, so its reported covariance far
    exceeds its actual error and the NEES collapses below the state dimension.
    """
    anees = _monte_carlo(n_runs=20, q_scale=100.0)
    lo, _ = _band(20, 9)
    assert anees.mean() < lo, f"ANEES {anees.mean():.3f} should fall below {lo:.3f}"


@pytest.mark.slow
def test_nees_rejects_the_printed_measurement_jacobian_sign():
    """Negative control: equation (13)'s -I is inconsistent, not just suboptimal."""
    anees = _monte_carlo(n_runs=20, h_bias_sign=-1)
    _, hi = _band(20, 9)
    assert anees.mean() > hi, f"ANEES {anees.mean():.3f} should exceed {hi:.3f}"

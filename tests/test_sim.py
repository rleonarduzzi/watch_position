"""Filter behaviour on synthetic data with exactly known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from wattitude import AdaptiveESKF, run_batch, to_world_frame
from wattitude.eskf import GRAVITY
from wattitude.eval.metrics import error_quat_earth, inclination_error, rmse
from wattitude.quaternion import (
    exp_q,
    log_q,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
    quat_rotate,
    quat_to_matrix,
)
from wattitude.simulate import simulate_imu

NOISELESS = dict(sigma_gyr=0.0, sigma_acc=0.0, sigma_bias_gyr=0.0, sigma_bias_acc=0.0)


def attitude_errors(quats, quat_true):
    """Per-sample absolute attitude error in radians."""
    q_diff = error_quat_earth(quats, quat_true)
    return 2 * np.arccos(np.clip(np.abs(q_diff[:, 0]), 0, 1))


def inclination_errors(quats, quat_true):
    return inclination_error(error_quat_earth(quats, quat_true))


# --------------------------------------------------------------- bookkeeping


def test_noiseless_bias_free_tracking_is_exact():
    """With no noise, no bias and gravity only, nothing can excite an error.

    The simulator integrates the attitude with the same zero-order-hold
    exponential map the filter uses, so any residual here is a bookkeeping bug
    rather than discretisation.
    """
    sim = simulate_imu(duration=30.0, rate=200.0, ext_accel="none", seed=1, **NOISELESS)
    quats, _, _ = run_batch(sim.gyr, sim.acc, rate=sim.rate, init_samples=100)
    err = attitude_errors(quats, sim.quat)
    assert np.max(err) < 1e-9, f"max error {np.max(err):.3e} rad"


def test_noiseless_tracking_exact_for_paper_faithful_H_too():
    """Without an accel bias in the state there is nothing for eq. (13) to break."""
    sim = simulate_imu(duration=20.0, rate=200.0, ext_accel="none", seed=2, **NOISELESS)
    quats, _, _ = run_batch(
        sim.gyr, sim.acc, rate=sim.rate, init_samples=100, estimate_acc_bias=False
    )
    assert np.max(attitude_errors(quats, sim.quat)) < 1e-9


def test_pure_gyro_integration_drifts_without_the_accelerometer_update():
    """Sanity check that the accelerometer update is what bounds the error."""
    sim = simulate_imu(
        duration=60.0, rate=200.0, ext_accel="none", seed=3, sigma_gyr=5e-3,
        sigma_acc=0.15, sigma_bias_gyr=0.0, sigma_bias_acc=0.0,
        bias_gyr0=np.array([0.01, -0.008, 0.006]),
    )
    quats, _, _ = run_batch(sim.gyr, sim.acc, rate=sim.rate, init_samples=200)

    # Open-loop integration of the same measurements, no updates at all.
    q = quat_normalize(np.array([1.0, 0.0, 0.0, 0.0]))
    open_loop = np.empty_like(quats)
    dt = 1.0 / sim.rate
    for k in range(len(sim.gyr)):
        q = quat_normalize(quat_multiply(q, exp_q(sim.gyr[k] * dt)))
        open_loop[k] = q

    incl_filter = np.rad2deg(rmse(inclination_errors(quats, sim.quat)))
    incl_open = np.rad2deg(rmse(inclination_errors(open_loop, sim.quat)))
    assert incl_filter < 1.0
    assert incl_open > 10 * incl_filter


# ------------------------------------------------------------ bias estimation


def test_gyro_bias_converges():
    bias = np.array([0.02, -0.015, 0.01])
    sim = simulate_imu(
        duration=120.0, rate=200.0, ext_accel="none", seed=4, bias_gyr0=bias,
        sigma_gyr=2e-3, sigma_acc=0.05, sigma_bias_gyr=0.0, sigma_bias_acc=0.0,
        rest_duration=0.0,
    )
    _, biases, _ = run_batch(
        sim.gyr, sim.acc, rate=sim.rate, init_samples=0,
        sigma_gyr=2e-3, sigma_acc=0.05, sigma_bias_gyr=1e-4, sigma_bias_acc=1e-4,
    )
    # Only the two components orthogonal to gravity are observable from a
    # gravity measurement at a fixed attitude; rotation makes all three
    # observable, which is what this trajectory provides.
    final = biases[-1, 0:3]
    err = np.linalg.norm(final - bias)
    assert err < 0.1 * np.linalg.norm(bias), f"bias error {err:.5f} vs truth {bias}"


def test_accel_bias_converges_when_the_sensor_rotates():
    bias = np.array([0.3, -0.2, 0.25])
    sim = simulate_imu(
        duration=180.0, rate=200.0, ext_accel="none", seed=5, bias_acc0=bias,
        sigma_gyr=2e-3, sigma_acc=0.05, sigma_bias_gyr=0.0, sigma_bias_acc=0.0,
        rotation_amplitude=1.2,
    )
    _, biases, _ = run_batch(
        sim.gyr, sim.acc, rate=sim.rate, init_samples=100,
        sigma_gyr=2e-3, sigma_acc=0.05, sigma_bias_gyr=1e-5, sigma_bias_acc=1e-3,
        estimate_gravity=False,
    )
    err = np.linalg.norm(biases[-1, 3:6] - bias)
    assert err < 0.3 * np.linalg.norm(bias), f"accel bias error {err:.4f} vs truth {bias}"


def test_paper_H_sign_drives_the_accel_bias_the_wrong_way():
    """Equation (13)'s -I makes the bias correction anti-converge.

    The sign is ablated on its own so the effect cannot be confused with the
    singular process noise of equation (9).
    """
    bias = np.array([0.3, -0.2, 0.25])
    sim = simulate_imu(
        duration=60.0, rate=200.0, ext_accel="none", seed=6, bias_acc0=bias,
        sigma_gyr=2e-3, sigma_acc=0.05, sigma_bias_gyr=0.0, sigma_bias_acc=0.0,
    )
    common = dict(
        rate=sim.rate, init_samples=100, sigma_gyr=2e-3, sigma_acc=0.05,
        sigma_bias_gyr=1e-5, sigma_bias_acc=1e-3, adaptive=False, estimate_gravity=False,
    )
    _, b_ok, _ = run_batch(sim.gyr, sim.acc, h_bias_sign=+1, **common)
    _, b_bad, _ = run_batch(sim.gyr, sim.acc, h_bias_sign=-1, **common)

    err_ok = np.linalg.norm(b_ok[-1, 3:6] - bias)
    err_bad = np.linalg.norm(b_bad[-1, 3:6] - bias)
    assert err_ok < err_bad, f"corrected {err_ok:.4f} should beat faithful {err_bad:.4f}"
    # The sign error makes the estimate move away from the truth, not just
    # converge more slowly.
    assert err_bad > np.linalg.norm(bias)


def test_paper_injection_side_degrades_attitude_tracking():
    """Equation (20)'s left multiplication biases every single update."""
    sim = simulate_imu(duration=60.0, rate=200.0, ext_accel="none", seed=17, **NOISELESS)
    common = dict(rate=sim.rate, init_samples=100, adaptive=False)
    q_right, _, _ = run_batch(sim.gyr, sim.acc, inject_left=False, **common)
    q_left, _, _ = run_batch(sim.gyr, sim.acc, inject_left=True, **common)

    # A noiseless sequence leaves nothing for the update to correct, so a
    # correctly applied correction is exactly zero and the error stays at zero.
    assert np.max(attitude_errors(q_right, sim.quat)) < 1e-9
    # The left-multiplied variant only differs once there is a correction to
    # apply, so inject a small initial tilt error to expose it.
    est_l = AdaptiveESKF(rate=sim.rate, adaptive=False, inject_left=True)
    est_r = AdaptiveESKF(rate=sim.rate, adaptive=False, inject_left=False)
    tilted = quat_multiply(np.array([1.0, 0.0, 0.0, 0.0]), exp_q(np.array([0.1, -0.08, 0.0])))
    series = {}
    for name, est in (("left", est_l), ("right", est_r)):
        est.initialize(quat=tilted)
        # update() returns a view of the filter's own quaternion, so each sample
        # has to be copied out; stacking the views would yield the final
        # attitude repeated T times.
        out = np.array([est.update(g, a).copy() for g, a in zip(sim.gyr, sim.acc)])
        series[name] = np.rad2deg(inclination_errors(out, sim.quat))

    # Right-multiplication applies the correction in the body frame the error
    # was defined in, so the initial tilt is removed and the noiseless sequence
    # then tracks to machine precision.
    half = len(sim.gyr) // 2
    assert rmse(series["right"][half:]) < 1e-3, rmse(series["right"][half:])
    # Left-multiplication applies it in the world frame, which leaves a
    # residual on every update and never settles.
    assert rmse(series["left"][half:]) > 10.0, rmse(series["left"][half:])

    # The telling signature: the two sides agree while the attitude is still
    # near identity, where left and right multiplication coincide, and separate
    # only as the device rotates away from it.  That is exactly the condition
    # equation (20) silently assumes.
    assert series["left"][200] == pytest.approx(series["right"][200], abs=1e-6)
    assert np.rad2deg(np.max(attitude_errors(sim.quat, np.array([1.0, 0, 0, 0])))) > 90.0


def test_paper_faithful_variant_diverges_reportably():
    """All three deviations at once diverge; that must be an exception, not a crash."""
    from wattitude.eskf import FilterDivergenceError

    sim = simulate_imu(
        duration=60.0, rate=200.0, ext_accel="none", seed=18,
        bias_acc0=np.array([0.3, -0.2, 0.25]), sigma_gyr=2e-3, sigma_acc=0.05,
    )
    with pytest.raises(FilterDivergenceError):
        run_batch(
            sim.gyr, sim.acc, rate=sim.rate, init_samples=100, paper_faithful=True,
            sigma_gyr=2e-3, sigma_acc=0.05, adaptive=False, estimate_gravity=False,
        )


# ------------------------------------------------- the adaptive mechanism


def test_adaptive_covariance_rejects_external_acceleration():
    """The central claim of Section IV.A, on data where truth is known exactly."""
    sim = simulate_imu(
        duration=120.0, rate=200.0, ext_accel="episodes", ext_amplitude=6.0, seed=7,
        sigma_gyr=3e-3, sigma_acc=0.1, sigma_bias_gyr=1e-5, sigma_bias_acc=1e-4,
    )
    common = dict(
        rate=sim.rate, init_samples=150, sigma_gyr=3e-3, sigma_acc=0.1,
        sigma_bias_gyr=1e-5, sigma_bias_acc=1e-4, estimate_gravity=False,
    )
    _, _, d_fixed = run_batch(sim.gyr, sim.acc, adaptive=False, diagnostics=True, **common)
    q_fixed, _, _ = run_batch(sim.gyr, sim.acc, adaptive=False, **common)
    q_adapt, _, d_adapt = run_batch(
        sim.gyr, sim.acc, adaptive=True, window=100, diagnostics=True, **common
    )

    incl_fixed = np.rad2deg(rmse(inclination_errors(q_fixed, sim.quat)))
    incl_adapt = np.rad2deg(rmse(inclination_errors(q_adapt, sim.quat)))
    assert incl_adapt < incl_fixed, f"adaptive {incl_adapt:.3f} deg vs fixed {incl_fixed:.3f} deg"

    # And the mechanism, not just the outcome: R must grow during the bursts.
    burst = np.linalg.norm(sim.acc_ext_world, axis=1) > 1.0
    quiet = (~burst) & (np.arange(len(burst)) > 2000)
    assert d_adapt.R_trace[burst].mean() > 5 * d_adapt.R_trace[quiet].mean()
    assert np.allclose(d_fixed.R_trace, d_fixed.R_trace[0])


def test_adaptive_covariance_tracks_the_acceleration_magnitude():
    """trace(R) must correlate with the unmodelled acceleration."""
    sim = simulate_imu(
        duration=120.0, rate=200.0, ext_accel="episodes", ext_amplitude=6.0, seed=8,
        sigma_gyr=3e-3, sigma_acc=0.1,
    )
    _, _, diag = run_batch(
        sim.gyr, sim.acc, rate=sim.rate, init_samples=150, window=100,
        sigma_gyr=3e-3, sigma_acc=0.1, diagnostics=True, estimate_gravity=False,
    )
    valid = slice(400, None)
    corr = np.corrcoef(diag.R_trace[valid], diag.acc_excess[valid])[0, 1]
    assert corr > 0.5, f"correlation {corr:.3f}"


def test_adaptive_covariance_floors_at_the_nominal_noise_at_rest():
    """Without a floor the filter would over-trust gravity during rest."""
    sigma_acc = 0.12
    sim = simulate_imu(
        duration=20.0, rate=200.0, rotation_amplitude=0.0, ext_accel="none", seed=9,
        sigma_gyr=1e-3, sigma_acc=sigma_acc,
    )
    _, _, diag = run_batch(
        sim.gyr, sim.acc, rate=sim.rate, init_samples=200, window=100,
        sigma_gyr=1e-3, sigma_acc=sigma_acc, diagnostics=True,
    )
    assert diag.R_trace[500:].min() >= 3 * sigma_acc**2 - 1e-9


def test_eps_only_psd_mode_can_produce_an_indefinite_covariance():
    """Equation (18)'s +eps*I does not guarantee positive-definiteness."""
    from wattitude.adaptive import InnovationCovariance

    nominal = 0.15**2 * np.eye(3)
    est = InnovationCovariance(window=50, nominal=nominal, eps=1e-6, psd_mode="eps")
    rng = np.random.default_rng(11)
    for _ in range(50):
        est.push(rng.normal(scale=0.02, size=3))
    # A realistic predicted innovation covariance exceeds the observed spread.
    hph = 0.3**2 * np.eye(3)
    R = est.estimate(hph)
    assert np.min(np.linalg.eigvalsh(R)) < 0.0

    clipped = InnovationCovariance(window=50, nominal=nominal, eps=1e-6, psd_mode="clip")
    for _ in range(50):
        clipped.push(rng.normal(scale=0.02, size=3))
    R2 = clipped.estimate(hph)
    assert np.min(np.linalg.eigvalsh(R2)) > 0.0


def test_adaptive_covariance_warms_up_with_the_nominal_value():
    sim = simulate_imu(duration=5.0, rate=200.0, seed=12, sigma_gyr=1e-3, sigma_acc=0.1)
    window = 150
    _, _, diag = run_batch(
        sim.gyr, sim.acc, rate=sim.rate, init_samples=100, window=window,
        sigma_gyr=1e-3, sigma_acc=0.1, diagnostics=True, estimate_gravity=False,
    )
    np.testing.assert_allclose(diag.R_trace[: window - 1], 3 * 0.1**2, rtol=1e-9)


# ----------------------------------------------------------- equivariance


def test_equivariant_to_sensor_mounting():
    """Remounting the IMU by a constant rotation must rotate the output by it.

    With ``R' = R C`` the body-frame measurements become ``C^T gyr`` and
    ``C^T acc``, so the estimate must satisfy ``q' = q * C``.
    """
    sim = simulate_imu(duration=40.0, rate=200.0, seed=13, sigma_gyr=2e-3, sigma_acc=0.1)
    C = exp_q(np.array([0.3, -0.7, 1.1]))
    Cm = quat_to_matrix(C)

    q_ref, _, _ = run_batch(sim.gyr, sim.acc, rate=sim.rate, init_samples=100)
    q_rot, _, _ = run_batch(sim.gyr @ Cm, sim.acc @ Cm, rate=sim.rate, init_samples=100)

    expected = np.array([quat_multiply(q, C) for q in q_ref])
    # Levelling picks a zero-yaw attitude independently in each frame, so the
    # two runs may differ by a constant heading.  Equivariance therefore shows
    # up as zero inclination error plus a *constant* total error.
    # The tolerance is set by arccos near its argument of 1, which amplifies
    # double-precision rounding to about 1e-8 rad in the angle.
    assert np.max(inclination_errors(q_rot, expected)) < 1e-6
    total = attitude_errors(q_rot, expected)
    assert np.ptp(total) < 1e-6, f"heading offset is not constant: ptp {np.ptp(total):.2e}"


def test_world_yaw_is_unobservable_by_construction():
    """Rotating the world frame about gravity leaves the measurements identical.

    This is the concrete statement of the rank-2 observability result: a 6-axis
    filter cannot distinguish two trajectories that differ by a heading offset.
    """
    sim = simulate_imu(duration=20.0, rate=200.0, seed=14, sigma_gyr=2e-3, sigma_acc=0.1)
    psi = 1.0
    q_yaw = np.array([np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)])
    rotated_truth = np.array([quat_multiply(q_yaw, q) for q in sim.quat])

    # Body-frame measurements of the yaw-rotated trajectory are unchanged.
    conj = rotated_truth.copy()
    conj[:, 1:] *= -1.0
    acc_rotated = quat_rotate(conj, quat_rotate(np.tile(q_yaw, (len(sim.quat), 1)),
                                                np.array([0.0, 0.0, GRAVITY])))
    conj0 = sim.quat.copy()
    conj0[:, 1:] *= -1.0
    acc_original = quat_rotate(conj0, np.array([0.0, 0.0, GRAVITY]))
    np.testing.assert_allclose(acc_rotated, acc_original, atol=1e-12)


# ------------------------------------------------------------ downstream API


def test_to_world_frame_removes_attitude_dependence():
    """The hook the neural odometry consumes must undo the body rotation."""
    sim = simulate_imu(duration=30.0, rate=200.0, ext_accel="none", seed=15, **NOISELESS)
    quats, _, _ = run_batch(sim.gyr, sim.acc, rate=sim.rate, init_samples=100)
    gyr_w, acc_w = to_world_frame(sim.gyr, sim.acc, quats)

    # With gravity as the only specific force, the world-frame accelerometer
    # signal must be constant and vertical regardless of how the sensor moved.
    assert np.max(np.abs(acc_w[:, 0:2])) < 1e-6
    np.testing.assert_allclose(acc_w[:, 2], GRAVITY, atol=1e-6)

    expected_gyr = quat_rotate(sim.quat, sim.gyr)
    assert np.max(np.abs(gyr_w - expected_gyr)) < 1e-6


def test_to_world_frame_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        to_world_frame(np.zeros((10, 3)), np.zeros((10, 3)), np.zeros((9, 4)))


def test_run_batch_rejects_bad_shapes():
    with pytest.raises(ValueError):
        run_batch(np.zeros((10, 3)), np.zeros((10, 2)), rate=100.0)
    with pytest.raises(ValueError):
        run_batch(np.zeros(10), np.zeros(10), rate=100.0)


def test_streaming_and_batch_agree():
    """The streaming API must reproduce run_batch sample for sample."""
    sim = simulate_imu(duration=10.0, rate=200.0, seed=16, sigma_gyr=2e-3, sigma_acc=0.1)
    quats, biases, _ = run_batch(sim.gyr, sim.acc, rate=sim.rate, init_samples=100)

    est = AdaptiveESKF(rate=sim.rate)
    est.initialize(gyr_window=sim.gyr[:100], acc_window=sim.acc[:100])
    for k in range(len(sim.gyr)):
        q = est.update(sim.gyr[k], sim.acc[k])
        np.testing.assert_allclose(q, quats[k], atol=1e-14)
    np.testing.assert_allclose(est.bias_gyr, biases[-1, 0:3], atol=1e-14)


def test_update_returns_a_live_view_of_the_filter_state():
    """Pin the aliasing contract, because getting it wrong is silent.

    ``update`` deliberately returns the filter's own quaternion buffer rather
    than a fresh array, so the streaming path allocates nothing per sample.
    Callers who accumulate the result must copy it; stacking the returned views
    yields the final attitude repeated, which looks like a plausible but wildly
    wrong trajectory rather than an error.
    """
    sim = simulate_imu(duration=1.0, rate=200.0, seed=21, sigma_gyr=2e-3, sigma_acc=0.1)
    est = AdaptiveESKF(rate=sim.rate)
    est.initialize(gyr_window=sim.gyr[:100], acc_window=sim.acc[:100])

    first = est.update(sim.gyr[0], sim.acc[0])
    assert first is est.quat
    snapshot = first.copy()
    for k in range(1, len(sim.gyr)):
        est.update(sim.gyr[k], sim.acc[k])

    # The view tracked the state; the copy did not.
    assert np.array_equal(first, est.quat)
    assert not np.allclose(snapshot, est.quat)


def test_initialize_levels_roll_and_pitch():
    """A tilted static start must be levelled, with yaw left at zero."""
    tilt = exp_q(np.array([0.3, -0.2, 0.0]))
    acc_body = quat_to_matrix(tilt).T @ np.array([0.0, 0.0, GRAVITY])
    est = AdaptiveESKF(rate=200.0).initialize(acc_window=np.tile(acc_body, (50, 1)))
    err = np.linalg.norm(log_q(quat_multiply(quat_conjugate(est.quat), tilt)))
    assert err < 1e-9
    # Levelling must not invent heading: the rotation axis stays horizontal.
    assert abs(log_q(est.quat)[2]) < 1e-9


def test_initialize_estimates_local_gravity():
    acc = np.tile(np.array([0.0, 0.0, 9.93]), (50, 1))
    est = AdaptiveESKF(rate=200.0, estimate_gravity=True).initialize(acc_window=acc)
    assert est.gravity == pytest.approx(9.93, abs=1e-9)
    est = AdaptiveESKF(rate=200.0, estimate_gravity=False).initialize(acc_window=acc)
    assert est.gravity == pytest.approx(GRAVITY, abs=1e-9)


def test_initialize_rejects_degenerate_window():
    with pytest.raises(ValueError):
        AdaptiveESKF(rate=200.0).initialize(acc_window=np.zeros((10, 3)))

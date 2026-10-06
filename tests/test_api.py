"""The public API surface, exactly as the README documents it.

Documentation that is not executed rots, so each example in the README has a
test here.  These also serve as the contract for the downstream pipeline: the
streaming loop, the batch helper and the world-frame hook.
"""

from __future__ import annotations

import numpy as np
import pytest

import wattitude
from wattitude import AdaptiveESKF, run_batch, to_world_frame
from wattitude.eskf import GRAVITY
from wattitude.quaternion import quat_rotate, quat_to_matrix
from wattitude.simulate import simulate_imu

RATE = 200.0


@pytest.fixture(scope="module")
def sim():
    return simulate_imu(duration=20.0, rate=RATE, seed=5, rest_duration=2.0)


def test_package_exports_the_documented_names():
    for name in (
        "AdaptiveESKF", "ESKFDiagnostics", "InnovationCovariance", "GRAVITY",
        "run_batch", "to_world_frame", "detect_rest", "find_static_window",
    ):
        assert hasattr(wattitude, name), name
    assert wattitude.__version__


def test_readme_streaming_example(sim):
    """The streaming loop from the README, verbatim in behaviour."""
    est = AdaptiveESKF(rate=RATE)
    est.initialize(gyr_window=sim.gyr[:400], acc_window=sim.acc[:400])

    quats = []
    for g, a in zip(sim.gyr, sim.acc):
        quat = est.update(g, a)
        quats.append(quat.copy())

    quats = np.asarray(quats)
    assert quats.shape == (len(sim.gyr), 4)
    assert np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-12)


def test_readme_documented_state_attributes(sim):
    est = AdaptiveESKF(rate=RATE)
    est.initialize(gyr_window=sim.gyr[:400], acc_window=sim.acc[:400])
    est.update(sim.gyr[400], sim.acc[400])

    assert est.quat.shape == (4,)
    assert est.rotation_matrix.shape == (3, 3)
    # The advertised matrix must be the advertised quaternion.
    assert np.allclose(est.rotation_matrix, quat_to_matrix(est.quat))
    assert est.bias_gyr.shape == (3,)
    assert est.bias_acc.shape == (3,)
    assert est.P.shape == (9, 9)
    assert est.R.shape == (3, 3)
    assert est.innovation.shape == (3,)


def test_readme_batch_example(sim):
    quats, biases, diag = run_batch(sim.gyr, sim.acc, rate=RATE, init_samples=400)
    assert quats.shape == (len(sim.gyr), 4)
    assert biases.shape == (len(sim.gyr), 6)
    assert diag is None

    gyr_w, acc_w = to_world_frame(sim.gyr, sim.acc, quats)
    assert gyr_w.shape == sim.gyr.shape
    assert acc_w.shape == sim.acc.shape


def test_diagnostics_expose_the_documented_fields(sim):
    _, _, diag = run_batch(sim.gyr, sim.acc, rate=RATE, init_samples=400,
                           diagnostics=True)
    n = len(sim.gyr)
    assert diag.innovation.shape == (n, 3)
    for field in ("R_trace", "P_trace_tilt", "gain_norm", "acc_excess"):
        assert getattr(diag, field).shape == (n,), field
        assert np.all(np.isfinite(getattr(diag, field))), field


def test_irregular_sampling_is_accepted_through_dt(sim):
    """Passing dt explicitly must match the nominal-rate path when dt agrees."""
    est_a = AdaptiveESKF(rate=RATE)
    est_b = AdaptiveESKF(rate=RATE)
    for est in (est_a, est_b):
        est.initialize(gyr_window=sim.gyr[:400], acc_window=sim.acc[:400])
    for k in range(400, 1400):
        est_a.update(sim.gyr[k], sim.acc[k])
        est_b.update(sim.gyr[k], sim.acc[k], dt=1.0 / RATE)
    assert np.allclose(est_a.quat, est_b.quat, atol=1e-14)


def test_to_world_frame_rotates_rather_than_reinterprets(sim):
    """The hook must apply the body-to-world rotation, not its inverse."""
    quats, _, _ = run_batch(sim.gyr, sim.acc, rate=RATE, init_samples=400)
    gyr_w, acc_w = to_world_frame(sim.gyr, sim.acc, quats)
    assert np.allclose(acc_w, quat_rotate(quats, sim.acc))

    # Gravity is the whole point: in the world frame the accelerometer must
    # average to +g on z over a rest phase, with the horizontal axes near zero.
    rest = slice(0, 400)
    mean = acc_w[rest].mean(axis=0)
    assert mean[2] == pytest.approx(GRAVITY, abs=0.2)
    assert abs(mean[0]) < 0.2 and abs(mean[1]) < 0.2
    # Body-frame gravity has no such alignment once the device has rotated.
    assert np.linalg.norm(sim.acc[rest].mean(axis=0)[:2]) > 0.0


def test_remove_gravity_gives_linear_acceleration(sim):
    quats, _, _ = run_batch(sim.gyr, sim.acc, rate=RATE, init_samples=400)
    _, acc_w = to_world_frame(sim.gyr, sim.acc, quats)
    _, lin = to_world_frame(sim.gyr, sim.acc, quats, remove_gravity=True)

    assert np.allclose(lin[:, :2], acc_w[:, :2])
    assert np.allclose(lin[:, 2], acc_w[:, 2] - GRAVITY)
    # At rest the linear acceleration is what should be near zero.
    assert abs(lin[:400, 2].mean()) < 0.2
    # Must not mutate the caller's array or the un-corrected result.
    assert acc_w[:, 2].mean() > 5.0


def test_to_world_frame_rejects_mismatched_lengths(sim):
    quats, _, _ = run_batch(sim.gyr, sim.acc, rate=RATE, init_samples=400)
    with pytest.raises(ValueError, match="leading dimension"):
        to_world_frame(sim.gyr, sim.acc, quats[:-1])

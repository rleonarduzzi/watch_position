"""Loader checks for the IMU/Vicon recordings.

Skipped when ``data/imu_vicon_joint_v10`` is absent.  The axis-fix expectations
are properties of these files: half the sequences have the sensor yawed 180
degrees relative to the optical body, which reverses IMU x and y.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wattitude.data import vicon

pytestmark = [
    pytest.mark.vicon,
    pytest.mark.skipif(
        not vicon.dataset_available(),
        reason="imu_vicon_joint_v10 recordings not present",
    ),
]

# Established by comparing the gyroscope with the optical angular rate, and
# cross-checked against position-derived specific force.
RZ180 = {"v3_02", "v3_12", "v3_14", "v3_15"}


def test_sequences_are_the_seven_recordings():
    names = vicon.sequence_names()
    assert names == ["v3_01", "v3_02", "v3_11", "v3_12", "v3_13", "v3_14", "v3_15"]


def test_prefix_is_a_verbatim_cut(tmp_path: Path):
    written = vicon.write_prefix_dataset(duration_s=2.0, dest=tmp_path)
    assert len(written) == 7
    expected = int(round(2.0 * vicon.RATE_HZ))
    src = vicon.default_root() / "v3_01.csv"
    out = tmp_path / "v3_01.csv"
    with open(src) as fin, open(out) as fout:
        src_lines = [next(fin) for _ in range(expected + 1)]
        dst_lines = fout.readlines()
    assert dst_lines == src_lines
    meta = (tmp_path / "meta.json").read_text()
    assert '"duration_s": 2.0' in meta


def test_trial_shapes_units_and_quaternion_order():
    trial = vicon.load_trial("v3_01", max_samples=500)
    n = len(trial)
    assert trial.gyr.shape == (n, 3)
    assert trial.acc.shape == (n, 3)
    assert trial.opt_quat.shape == (n, 4)
    assert trial.opt_pos.shape == (n, 3)
    assert trial.movement.shape == (n,)
    assert trial.movement.all()
    assert trial.rate == vicon.RATE_HZ
    # Scalar-first, and the recording starts near a modest tilt, so w dominates.
    assert trial.opt_quat[0, 0] > 0.9
    assert np.allclose(np.linalg.norm(trial.opt_quat, axis=1), 1.0, atol=1e-6)
    # At rest the specific force is about one g, upward.
    assert np.linalg.norm(trial.acc[:50].mean(axis=0)) == pytest.approx(9.8, abs=0.4)
    assert trial.acc[:50, 2].mean() > 8.0


def test_axis_fix_matches_the_two_mountings():
    for name in vicon.sequence_names():
        trial = vicon.load_trial(name, max_samples=3000)
        expected = "rz180" if name in RZ180 else "identity"
        assert trial.axis_fix == expected, name


def test_rz180_reverses_only_horizontal_axes():
    raw = vicon.load_trial("v3_02", align_axes=False, max_samples=200)
    fixed = vicon.load_trial("v3_02", align_axes=True, max_samples=200)
    assert raw.axis_fix == "identity"
    assert fixed.axis_fix == "rz180"
    assert np.allclose(fixed.gyr[:, :2], -raw.gyr[:, :2])
    assert np.allclose(fixed.gyr[:, 2], raw.gyr[:, 2])
    assert np.allclose(fixed.acc[:, :2], -raw.acc[:, :2])
    assert np.allclose(fixed.opt_quat, raw.opt_quat)

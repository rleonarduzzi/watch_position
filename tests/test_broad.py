"""Tests that run against the real BROAD recordings.

These are the external validation of the whole harness.  The simulation tests
show the filter is self-consistent; these show that the loader, the conventions
and the error metrics agree with results published by somebody else, which is
the only way to catch a convention error that is self-consistently wrong
throughout our own code.

Marked ``broad`` and skipped when the dataset is absent, so the suite still runs
on a fresh checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wattitude.baselines import ESTIMATORS
from wattitude.data import broad
from wattitude.eval.metrics import orientation_errors, rmse, total_error
from wattitude.eval.runner import init_window_for

pytestmark = [
    pytest.mark.broad,
    pytest.mark.skipif(not broad.dataset_available(), reason="BROAD dataset not present"),
]

# BROAD ships the results of its own Madgwick/Mahony runs.  Trial 01 at
# beta = 0.12 is the entry we reproduce; the reference values come from
# example_code/out/results_madgwick.json.
REFERENCE_TRIAL = "01_undisturbed_slow_rotation_A"
REFERENCE_BETA = 0.12
REFERENCE_ERRORS = (2.3096, 2.1743, 0.7788)  # total, heading, inclination (deg)


def test_dataset_has_the_expected_shape():
    names = broad.trial_names()
    assert len(names) == 39
    groups = broad.group_names()
    assert len(groups) == 16
    # Every trial belongs to at least one group, and groups partition nothing
    # twice by accident.
    covered = {t for g in groups for t in broad.trials_in_group(g)}
    assert covered == set(names)


def test_trial_loads_with_consistent_lengths_and_rate():
    trial = broad.load_trial(REFERENCE_TRIAL)
    n = len(trial)
    assert trial.gyr.shape == (n, 3)
    assert trial.acc.shape == (n, 3)
    assert trial.mag.shape == (n, 3)
    assert trial.opt_quat.shape == (n, 4)
    assert trial.movement.shape == (n,)
    assert trial.rate == pytest.approx(2000.0 / 7.0)
    # Accelerometer reads +g upwards at rest, as the filter assumes.
    assert np.linalg.norm(trial.acc[:100].mean(axis=0)) == pytest.approx(9.9, abs=0.3)
    assert trial.acc[:100, 2].mean() > 9.0


def test_ground_truth_quaternions_are_unit_where_finite():
    trial = broad.load_trial(REFERENCE_TRIAL)
    finite = np.isfinite(trial.opt_quat).all(axis=1)
    norms = np.linalg.norm(trial.opt_quat[finite], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_madgwick_9d_reproduces_the_published_reference_errors():
    """The harness end-to-end against BROAD's own shipped numbers.

    This pins the loader, the ENU yaw correction on the 9-axis path, the
    movement mask and all three error decompositions simultaneously.  Agreement
    to four decimals is not luck; any convention slip moves these by degrees.
    """
    trial = broad.load_trial(REFERENCE_TRIAL)
    quats = ESTIMATORS["madgwick"](
        trial.gyr, trial.acc, trial.rate, mag=trial.mag, init_slice=None,
        beta=REFERENCE_BETA,
    )
    errors = orientation_errors(
        quats, trial.opt_quat, trial.movement, trial.rate, align_heading=False
    )
    got = (
        errors.total_rmse_deg,
        errors.heading_rmse_deg,
        errors.inclination_rmse_deg,
    )
    assert got == pytest.approx(REFERENCE_ERRORS, abs=5e-4)


def test_heading_alignment_cannot_change_the_inclination_error():
    """Inclination is the heading-invariant metric, which is why we tune on it."""
    trial = broad.load_trial("04_undisturbed_slow_rotation_with_breaks_A")
    quats = ESTIMATORS["mahony"](
        trial.gyr, trial.acc, trial.rate, init_slice=init_window_for(trial)
    )
    free = orientation_errors(quats, trial.opt_quat, trial.movement, trial.rate,
                              align_heading=False)
    aligned = orientation_errors(quats, trial.opt_quat, trial.movement, trial.rate,
                                 align_heading=True)
    assert aligned.inclination_rmse_deg == pytest.approx(free.inclination_rmse_deg, rel=1e-9)
    # Alignment can only help the total error, never hurt it.
    assert aligned.total_rmse_deg <= free.total_rmse_deg + 1e-9


def test_metrics_ignore_the_optical_dropouts_rather_than_propagating_them():
    """Ground truth contains NaN frames; every reported number must stay finite."""
    trial = broad.load_trial("15_undisturbed_fast_translation_A")
    assert not np.isfinite(trial.opt_quat).all(), "expected dropouts in this trial"
    quats = ESTIMATORS["vqf"](
        trial.gyr, trial.acc, trial.rate, init_slice=init_window_for(trial)
    )
    errors = orientation_errors(quats, trial.opt_quat, trial.movement, trial.rate,
                                align_heading=True)
    for name, value in errors.as_dict().items():
        assert np.isfinite(value), f"{name} is not finite"


def test_static_init_window_is_quiescent_on_every_trial():
    """All estimators share this window, so it must be genuinely at rest."""
    for name in broad.trial_names():
        trial = broad.load_trial(name)
        window = init_window_for(trial)
        assert window.stop > window.start
        gyr_norm = np.linalg.norm(trial.gyr[window], axis=1)
        acc_norm = np.linalg.norm(trial.acc[window], axis=1)
        assert gyr_norm.mean() < 0.05, f"{name}: window not still"
        assert abs(acc_norm.mean() - 9.9) < 0.5, f"{name}: window not at rest"
        # The window must precede the annotated motion, or initialisation would
        # be averaging over real movement.
        assert window.stop <= int(np.argmax(trial.movement)), f"{name}: window overlaps motion"


def test_gyro_only_is_beaten_by_every_accelerometer_aided_filter_on_inclination():
    """Sanity floor: aiding with gravity must improve inclination somewhere."""
    trial = broad.load_trial(REFERENCE_TRIAL)
    window = init_window_for(trial)
    incl = {}
    for name in ("gyro_only", "mahony", "madgwick", "vqf", "eskf"):
        quats = ESTIMATORS[name](trial.gyr, trial.acc, trial.rate, init_slice=window)
        incl[name] = orientation_errors(
            quats, trial.opt_quat, trial.movement, trial.rate, align_heading=True
        ).inclination_rmse_deg
    for name in ("mahony", "madgwick", "vqf", "eskf"):
        assert incl[name] < incl["gyro_only"], f"{name} no better than open-loop"

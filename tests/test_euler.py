"""The Euler-angle conversion used for display in the attitude report.

Euler angles exist in this codebase only to be looked at, but a wrong
convention produces plots that look entirely reasonable and are wrong, so the
convention is pinned here against three independent references: the rotation
matrix it claims to decompose, known single-axis rotations, and a round trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from wattitude.quaternion import (
    GIMBAL_LOCK_PITCH,
    euler_gimbal_risk,
    euler_to_quat,
    quat_angle,
    quat_multiply,
    quat_normalize,
    quat_relative,
    quat_to_euler,
    quat_to_matrix,
)


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_single_axis_rotations_land_on_the_named_angle():
    """A rotation about one axis must show up in exactly one angle."""
    for axis, index, name in ((_rx, 0, "roll"), (_ry, 1, "pitch"), (_rz, 2, "yaw")):
        angle = np.deg2rad(30.0)
        q = _matrix_to_q(axis(angle))
        euler = np.rad2deg(quat_to_euler(q))
        expected = np.zeros(3)
        expected[index] = 30.0
        assert np.allclose(euler, expected, atol=1e-9), f"{name}: {euler}"


def _matrix_to_q(R):
    from wattitude.quaternion import matrix_to_quat

    return matrix_to_quat(R)


def test_decomposition_matches_the_advertised_matrix_product():
    """R(q) must equal Rz(yaw) Ry(pitch) Rx(roll) for the returned angles."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        q = quat_normalize(rng.normal(size=4))
        roll, pitch, yaw = quat_to_euler(q)
        if abs(pitch) > GIMBAL_LOCK_PITCH:
            continue
        assert np.allclose(
            quat_to_matrix(q), _rz(yaw) @ _ry(pitch) @ _rx(roll), atol=1e-9
        )


def test_round_trip_through_quaternion_is_exact():
    rng = np.random.default_rng(1)
    for _ in range(300):
        q = quat_normalize(rng.normal(size=4))
        if q[0] < 0:
            q = -q
        back = euler_to_quat(quat_to_euler(q))
        # Compare as rotations, not componentwise, to stay sign-agnostic.
        assert quat_angle(quat_relative(q, back)) < 1e-9


def test_round_trip_from_euler_recovers_the_angles():
    rng = np.random.default_rng(2)
    roll = rng.uniform(-np.pi, np.pi, 200)
    pitch = rng.uniform(-np.deg2rad(85), np.deg2rad(85), 200)
    yaw = rng.uniform(-np.pi, np.pi, 200)
    euler = np.stack([roll, pitch, yaw], axis=1)
    back = quat_to_euler(euler_to_quat(euler))
    assert np.allclose(np.sin(back), np.sin(euler), atol=1e-9)
    assert np.allclose(np.cos(back), np.cos(euler), atol=1e-9)


def test_ranges_are_respected():
    rng = np.random.default_rng(3)
    euler = quat_to_euler(quat_normalize(rng.normal(size=(2000, 4))))
    assert np.all(euler[:, 0] >= -np.pi) and np.all(euler[:, 0] <= np.pi)
    assert np.all(euler[:, 1] >= -np.pi / 2) and np.all(euler[:, 1] <= np.pi / 2)
    assert np.all(euler[:, 2] >= -np.pi) and np.all(euler[:, 2] <= np.pi)


def test_batched_and_single_inputs_agree():
    rng = np.random.default_rng(4)
    q = quat_normalize(rng.normal(size=(17, 4)))
    batched = quat_to_euler(q)
    assert batched.shape == (17, 3)
    for k in range(len(q)):
        assert np.allclose(quat_to_euler(q[k]), batched[k])
    assert quat_to_euler(q[0]).shape == (3,)


def test_gimbal_lock_still_describes_the_right_rotation():
    """At pitch = +-90 the split is a choice, but the attitude must survive it."""
    for sign in (+1.0, -1.0):
        for extra in (0.0, 0.7, -1.9):
            # Straight up or down, with an arbitrary rotation about the
            # now-degenerate axis.
            q = _matrix_to_q(_rz(extra) @ _ry(sign * np.pi / 2) @ _rx(0.3))
            euler = quat_to_euler(q)
            assert euler[0] == 0.0, "roll should be zeroed under lock"
            assert abs(abs(euler[1]) - np.pi / 2) < 1e-6
            # The reconstructed rotation must still be the original one.
            assert quat_angle(quat_relative(q, euler_to_quat(euler))) < 1e-6


def test_gimbal_risk_flags_near_vertical_pitch_only():
    pitch = np.deg2rad(np.array([0.0, 45.0, 79.0, 81.0, 89.0, -89.0, -70.0]))
    euler = np.stack([np.zeros_like(pitch), pitch, np.zeros_like(pitch)], axis=1)
    risk = euler_gimbal_risk(euler_to_quat(euler), margin_deg=10.0)
    assert list(risk) == [False, False, False, True, True, True, False]


def test_nearby_attitudes_give_nearby_angles_away_from_the_singularity():
    """Continuity, which is what makes the plots readable at all."""
    rng = np.random.default_rng(5)
    for _ in range(200):
        q = quat_normalize(rng.normal(size=4))
        if abs(quat_to_euler(q)[1]) > np.deg2rad(70):
            continue
        perturbed = quat_normalize(
            quat_multiply(q, euler_to_quat(np.array([1e-6, 1e-6, 1e-6])))
        )
        delta = quat_to_euler(perturbed) - quat_to_euler(q)
        assert np.all(np.abs(delta) < 1e-3)

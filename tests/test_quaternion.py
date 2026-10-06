"""Quaternion algebra, cross-checked against scipy's independent implementation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wattitude.quaternion import (
    exp_q,
    log_q,
    matrix_to_quat,
    quat_angle,
    quat_conjugate,
    quat_from_two_vectors,
    quat_multiply,
    quat_normalize,
    quat_relative,
    quat_rotate,
    quat_to_matrix,
    skew,
)


def _to_scipy(q):
    """Our [w, x, y, z] to scipy's [x, y, z, w]."""
    return Rotation.from_quat(np.roll(np.asarray(q), -1))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def random_quats(rng, n):
    return np.array([quat_normalize(rng.normal(size=4)) for _ in range(n)])


def test_skew_matches_cross_product(rng):
    for _ in range(50):
        a, b = rng.normal(size=3), rng.normal(size=3)
        np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), atol=1e-14)


def test_quat_to_matrix_matches_scipy(rng):
    for q in random_quats(rng, 100):
        np.testing.assert_allclose(quat_to_matrix(q), _to_scipy(q).as_matrix(), atol=1e-12)


def test_matrix_to_quat_roundtrip(rng):
    for q in random_quats(rng, 200):
        np.testing.assert_allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-10)


def test_matrix_to_quat_covers_all_shepperd_branches():
    # Each 180 degree rotation about a principal axis selects a different branch.
    for q in (
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ):
        q = np.array(q)
        np.testing.assert_allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-10)


def test_composition_order(rng):
    """R(p * q) == R(p) @ R(q) is the property the filter relies on."""
    for _ in range(100):
        p, q = random_quats(rng, 2)
        np.testing.assert_allclose(
            quat_to_matrix(quat_multiply(p, q)),
            quat_to_matrix(p) @ quat_to_matrix(q),
            atol=1e-12,
        )


def test_exp_log_roundtrip(rng):
    for scale in (1e-9, 1e-4, 0.1, 1.0, 3.0):
        for _ in range(50):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * scale
            np.testing.assert_allclose(log_q(exp_q(v)), v, atol=1e-12)


def test_exp_q_matches_scipy(rng):
    for _ in range(100):
        v = rng.normal(scale=1.5, size=3)
        np.testing.assert_allclose(
            quat_to_matrix(exp_q(v)), Rotation.from_rotvec(v).as_matrix(), atol=1e-12
        )


def test_exp_q_small_angle_series_is_accurate():
    """The series branch must agree with the trigonometric form at the cutoff."""
    v = np.array([1e-9, -2e-9, 3e-10])
    np.testing.assert_allclose(
        quat_to_matrix(exp_q(v)), Rotation.from_rotvec(v).as_matrix(), atol=1e-15
    )


def test_quat_rotate_matches_matrix_product(rng):
    quats = random_quats(rng, 64)
    vecs = rng.normal(size=(64, 3))
    expected = np.array([quat_to_matrix(q) @ v for q, v in zip(quats, vecs)])
    np.testing.assert_allclose(quat_rotate(quats, vecs), expected, atol=1e-12)


def test_quat_rotate_broadcasts_single_quaternion(rng):
    q = random_quats(rng, 1)[0]
    vecs = rng.normal(size=(10, 3))
    expected = (quat_to_matrix(q) @ vecs.T).T
    np.testing.assert_allclose(quat_rotate(np.tile(q, (10, 1)), vecs), expected, atol=1e-12)


def test_quat_from_two_vectors(rng):
    for _ in range(200):
        a, b = rng.normal(size=3), rng.normal(size=3)
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        q = quat_from_two_vectors(a, b)
        np.testing.assert_allclose(quat_to_matrix(q) @ a, b, atol=1e-10)


def test_quat_from_two_vectors_degenerate_cases():
    a = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(quat_from_two_vectors(a, a), [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    q = quat_from_two_vectors(a, -a)
    np.testing.assert_allclose(quat_to_matrix(q) @ a, -a, atol=1e-10)
    # An input aligned with the fallback axis must still work.
    x = np.array([1.0, 0.0, 0.0])
    q = quat_from_two_vectors(x, -x)
    np.testing.assert_allclose(quat_to_matrix(q) @ x, -x, atol=1e-10)


def test_quat_from_two_vectors_introduces_no_yaw():
    """Levelling must not invent heading information the IMU does not have."""
    rng = np.random.default_rng(7)
    for _ in range(100):
        acc = np.array([0.0, 0.0, 9.81]) + rng.normal(scale=2.0, size=3)
        q = quat_from_two_vectors(acc, np.array([0.0, 0.0, 9.81]))
        # The minimal rotation axis is horizontal, so the rotation vector has no
        # vertical component.
        assert abs(log_q(q)[2]) < 1e-10


def test_quat_angle_and_relative(rng):
    for _ in range(100):
        q = random_quats(rng, 1)[0]
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * rng.uniform(0, np.pi)
        q2 = quat_multiply(q, exp_q(v))
        assert abs(quat_angle(quat_relative(q, q2)) - np.linalg.norm(v)) < 1e-9


def test_quat_normalize_fixes_sign_and_norm(rng):
    q = -quat_normalize(rng.normal(size=4)) * 3.0
    out = quat_normalize(q)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-14
    assert out[0] >= 0.0


def test_quat_conjugate_inverts(rng):
    for q in random_quats(rng, 50):
        np.testing.assert_allclose(
            quat_multiply(q, quat_conjugate(q)), [1.0, 0.0, 0.0, 0.0], atol=1e-12
        )

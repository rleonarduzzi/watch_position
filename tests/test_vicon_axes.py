"""The on-disk axis correction, without the real recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wattitude.data.vicon import COLUMNS, correct_csv, correct_dataset, needs_rz180
from wattitude.quaternion import exp_q, quat_multiply, quat_normalize


def _recording(flip_xy: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gyroscope, accelerometer and scalar-first quaternion for a short motion."""
    dt = 0.01
    q = np.array([1.0, 0.0, 0.0, 0.0])
    gyrs = []
    quats = []
    for k in range(400):
        w = np.array([0.7 * np.sin(k / 12.0), 1.1 * np.cos(k / 9.0), 0.35])
        q = quat_normalize(quat_multiply(q, exp_q(w * dt)))
        gyrs.append(w)
        quats.append(q.copy())
    gyr = np.asarray(gyrs)
    quat = np.asarray(quats)
    acc = np.tile(np.array([0.2, -0.4, 9.8]), (len(gyr), 1))
    if flip_xy:
        gyr = gyr.copy()
        acc = acc.copy()
        gyr[:, :2] *= -1.0
        acc[:, :2] *= -1.0
    return gyr, acc, quat


def _write_csv(path: Path, gyr: np.ndarray, acc: np.ndarray, quat_wxyz: np.ndarray) -> None:
    # File order is x, y, z, w.
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    pos = np.zeros_like(acc)
    rows = np.column_stack([acc, gyr, pos, quat_xyzw])
    with open(path, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for row in rows:
            f.write(",".join(f"{v:.10f}" for v in row) + "\n")


def test_needs_rz180_sees_a_horizontal_sign_flip():
    gyr, _, quat = _recording(flip_xy=False)
    assert needs_rz180(gyr, quat) is False
    gyr_f, _, _ = _recording(flip_xy=True)
    assert needs_rz180(gyr_f, quat) is True


def test_correct_dataset_flips_only_the_files_that_need_it(tmp_path: Path):
    straight = _recording(flip_xy=False)
    flipped = _recording(flip_xy=True)
    _write_csv(tmp_path / "aligned.csv", *straight)
    _write_csv(tmp_path / "reversed.csv", *flipped)
    before = (tmp_path / "aligned.csv").read_bytes()

    dry = correct_dataset(tmp_path, dry_run=True)
    assert {path.name: fix for path, fix in dry} == {"aligned.csv": "identity", "reversed.csv": "rz180"}
    assert (tmp_path / "aligned.csv").read_bytes() == before
    reversed_before = (tmp_path / "reversed.csv").read_text()

    applied = {path.name: fix for path, fix in correct_dataset(tmp_path)}
    assert applied == {"aligned.csv": "identity", "reversed.csv": "rz180"}
    assert (tmp_path / "aligned.csv").read_bytes() == before
    assert (tmp_path / "reversed.csv").read_text() != reversed_before

    text = (tmp_path / "reversed.csv").read_text().splitlines()
    raw = reversed_before.splitlines()
    assert text[0] == raw[0]
    for new, old in zip(text[1:], raw[1:]):
        a = new.split(",")
        b = old.split(",")
        # x and y of acc (0, 1) and gyro (3, 4) flip; everything else is the same text.
        for index in (0, 1, 3, 4):
            assert a[index] == (b[index][1:] if b[index].startswith("-") else "-" + b[index])
        for index in (2, 5, 6, 7, 8, 9, 10, 11, 12):
            assert a[index] == b[index]

    again = correct_csv(tmp_path / "reversed.csv")
    assert again == "identity"
    assert (tmp_path / "reversed.csv").read_text() == "\n".join(text) + "\n"

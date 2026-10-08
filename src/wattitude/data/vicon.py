"""Loader for the wrist IMU / Vicon sequences in ``data/imu_vicon_joint_v10``.

Each CSV is one recording at 100 Hz with columns

    imu_acc_{x,y,z}      m/s^2, specific force, reads about +g on the upward axis
    imu_gyr_{x,y,z}      rad/s
    gt_pos_{x,y,z}       metres, optical position
    gt_rot_quat_{x,y,z,w} unit quaternion, scalar last, body to world

The optical quaternion uses the same z-up, body-to-world convention as
:class:`~wattitude.eskf.AdaptiveESKF`.  On some recordings the IMU itself is
yawed 180 degrees relative to that body frame, which reverses its x and y axes
and leaves z alone.  :func:`load_trial` detects that mounting from the
gyroscope and corrects it in memory.  :func:`correct_dataset` writes the same
correction back into the CSV files.

The full recordings run for about ten to thirty minutes.  :func:`write_prefix_dataset`
copies a short prefix of each one into ``data/imu_vicon_joint_v10_short`` for
the first evaluation pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "RATE_HZ",
    "PREFIX_S",
    "COLUMNS",
    "ViconTrial",
    "default_root",
    "short_root",
    "dataset_available",
    "sequence_names",
    "needs_rz180",
    "load_trial",
    "correct_csv",
    "correct_dataset",
    "write_prefix_dataset",
]

RATE_HZ = 100.0
PREFIX_S = 120.0

COLUMNS = (
    "imu_acc_x", "imu_acc_y", "imu_acc_z",
    "imu_gyr_x", "imu_gyr_y", "imu_gyr_z",
    "gt_pos_x", "gt_pos_y", "gt_pos_z",
    "gt_rot_quat_x", "gt_rot_quat_y", "gt_rot_quat_z", "gt_rot_quat_w",
)

_FULL_DIR = "imu_vicon_joint_v10"
_SHORT_DIR = "imu_vicon_joint_v10_short"


@dataclass
class ViconTrial:
    """One IMU / Vicon sequence, already expressed in the optical body frame."""

    name: str
    gyr: np.ndarray
    acc: np.ndarray
    opt_quat: np.ndarray
    opt_pos: np.ndarray
    movement: np.ndarray
    rate: float
    axis_fix: str

    def __len__(self) -> int:
        return len(self.movement)

    @property
    def duration(self) -> float:
        return len(self) / self.rate


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


def default_root() -> Path:
    """Directory of the full-length recordings."""
    return _data_dir() / _FULL_DIR


def short_root() -> Path:
    """Directory of the prefix dataset written by :func:`write_prefix_dataset`."""
    return _data_dir() / _SHORT_DIR


def dataset_available(root: Path | str | None = None) -> bool:
    root = default_root() if root is None else Path(root)
    return any(root.glob("v3_*.csv"))


def sequence_names(root: Path | str | None = None) -> list[str]:
    """Sequence names in filename order, without the ``.csv`` suffix."""
    root = default_root() if root is None else Path(root)
    return sorted(p.stem for p in root.glob("v3_*.csv"))


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def _body_rate(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular rate of a body-to-world quaternion, in rad/s.

    ``q_{k+1} = q_k ⊗ exp(ω dt)``, so ω is the right-trivialised increment.
    """
    q = np.asarray(quat_wxyz, dtype=float)
    qi = q[:-1].copy()
    qi[:, 1:] *= -1.0
    dq = _quat_mul(qi, q[1:])
    dq[dq[:, 0] < 0.0] *= -1.0
    v = dq[:, 1:]
    nv = np.linalg.norm(v, axis=1)
    ang = 2.0 * np.arctan2(nv, np.clip(dq[:, 0], -1.0, 1.0))
    scale = np.divide(ang, nv, out=np.ones_like(ang), where=nv > 1e-12)
    return v * scale[:, None] / dt


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def needs_rz180(gyr: np.ndarray, quat_wxyz: np.ndarray, rate: float = RATE_HZ) -> bool:
    """Whether the IMU x and y axes are reversed relative to the optical body.

    A 180 degree yaw of the sensor flips the sign of the x and y gyroscope
    channels and leaves z unchanged.  The decision is which of the two
    mountings makes the gyroscope agree in sign with the optical angular rate.
    It is a binary frame correction, not a fit of the attitude.
    """
    omega = _body_rate(quat_wxyz, 1.0 / float(rate))
    g = 0.5 * (np.asarray(gyr, dtype=float)[:-1] + np.asarray(gyr, dtype=float)[1:])
    moving = np.linalg.norm(g, axis=1) > 0.5
    if int(moving.sum()) < 50:
        moving = np.ones(len(g), dtype=bool)
    scores = [_corr(g[moving, i], omega[moving, i]) for i in range(3)]
    raw = scores[0] + scores[1] + scores[2]
    flipped = -scores[0] - scores[1] + scores[2]
    return flipped > raw


def _read_csv(path: Path) -> np.ndarray:
    with open(path) as f:
        header = f.readline().strip().split(",")
    if tuple(header) != COLUMNS:
        raise ValueError(f"{path} has columns {header}, expected {list(COLUMNS)}")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


# Accelerometer and gyroscope x, y.  z, position and the optical quaternion stay.
_FLIP_COLUMNS = (0, 1, 3, 4)


def _negate_field(field: str) -> str:
    """Flip the sign of one CSV number without touching its other digits."""
    field = field.strip()
    if not field:
        raise ValueError("empty field in a column that would be sign-flipped")
    if field[0] == "-":
        return field[1:]
    if field[0] == "+":
        return "-" + field[1:]
    return "-" + field


def _rewrite_xy_flipped(path: Path) -> None:
    """Replace ``path`` with a copy whose IMU x and y columns have flipped sign."""
    rewritten: list[str] = []
    with open(path) as fin:
        header = fin.readline()
        if not header.endswith("\n"):
            header += "\n"
        rewritten.append(header)
        for lineno, line in enumerate(fin, start=2):
            body = line[:-1] if line.endswith("\n") else line
            if not body.strip():
                continue
            fields = body.split(",")
            if len(fields) != len(COLUMNS):
                raise ValueError(
                    f"{path}:{lineno} has {len(fields)} columns, expected {len(COLUMNS)}"
                )
            for index in _FLIP_COLUMNS:
                fields[index] = _negate_field(fields[index])
            rewritten.append(",".join(fields) + "\n")

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(rewritten))
    tmp.replace(path)


def correct_csv(path: Path | str, dry_run: bool = False, rate: float = RATE_HZ) -> str:
    """Reverse IMU x and y in one CSV when they oppose the optical body.

    Returns ``"rz180"`` when the file needed the reversal and ``"identity"``
    when it already agreed.  With ``dry_run`` the file is left unchanged either
    way.  A second call on a file that was just corrected returns ``"identity"``.
    """
    path = Path(path)
    data = _read_csv(path)
    gyr = np.ascontiguousarray(data[:, 3:6], dtype=float)
    quat = np.ascontiguousarray(data[:, [12, 9, 10, 11]], dtype=float)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    if not needs_rz180(gyr, quat, rate):
        return "identity"
    if not dry_run:
        _rewrite_xy_flipped(path)
    return "rz180"


def correct_dataset(
    root: Path | str, dry_run: bool = False, rate: float = RATE_HZ
) -> list[tuple[Path, str]]:
    """Apply :func:`correct_csv` to every ``*.csv`` directly inside ``root``."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset folder not found: {root}")
    paths = sorted(p for p in root.glob("*.csv") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"no csv files in {root}")
    return [(path, correct_csv(path, dry_run=dry_run, rate=rate)) for path in paths]


def load_trial(
    name: str,
    root: Path | str | None = None,
    align_axes: bool = True,
    max_samples: int | None = None,
) -> ViconTrial:
    """Load one sequence.

    ``name`` may be ``v3_01`` or ``v3_01.csv``.  With ``align_axes`` the
    gyroscope and accelerometer are mapped into the optical body frame.
    """
    root = default_root() if root is None else Path(root)
    name = name[:-4] if name.endswith(".csv") else name
    path = root / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"sequence file not found: {path}")

    data = _read_csv(path)
    if max_samples is not None:
        data = data[: int(max_samples)]

    acc = np.ascontiguousarray(data[:, 0:3], dtype=float)
    gyr = np.ascontiguousarray(data[:, 3:6], dtype=float)
    pos = np.ascontiguousarray(data[:, 6:9], dtype=float)
    # File order is x, y, z, w.  The filter uses scalar-first Hamilton quaternions.
    quat = np.ascontiguousarray(data[:, [12, 9, 10, 11]], dtype=float)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)

    fix = "identity"
    if align_axes and needs_rz180(gyr, quat, RATE_HZ):
        acc = acc.copy()
        gyr = gyr.copy()
        acc[:, :2] *= -1.0
        gyr[:, :2] *= -1.0
        fix = "rz180"

    return ViconTrial(
        name=name,
        gyr=gyr,
        acc=acc,
        opt_quat=quat,
        opt_pos=pos,
        movement=np.ones(len(acc), dtype=bool),
        rate=RATE_HZ,
        axis_fix=fix,
    )


def write_prefix_dataset(
    duration_s: float = PREFIX_S,
    source: Path | str | None = None,
    dest: Path | str | None = None,
) -> list[Path]:
    """Copy the first ``duration_s`` of every sequence into ``dest``.

    Rows are copied verbatim, so the short files stay byte-identical prefixes
    of the originals.  A ``meta.json`` records the cut.
    """
    source = default_root() if source is None else Path(source)
    dest = short_root() if dest is None else Path(dest)
    names = sequence_names(source)
    if not names:
        raise FileNotFoundError(f"no v3_*.csv sequences in {source}")

    n_keep = int(round(float(duration_s) * RATE_HZ))
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in names:
        src = source / f"{name}.csv"
        out = dest / f"{name}.csv"
        with open(src) as fin, open(out, "w") as fout:
            header = fin.readline()
            if tuple(header.strip().split(",")) != COLUMNS:
                raise ValueError(f"{src} does not have the expected header")
            fout.write(header)
            for i, line in enumerate(fin):
                if i >= n_keep:
                    break
                fout.write(line if line.endswith("\n") else line + "\n")
        written.append(out)

    meta = {
        "source": source.name,
        "rate_hz": RATE_HZ,
        "duration_s": float(duration_s),
        "samples": n_keep,
        "sequences": names,
        "note": (
            "Verbatim prefix of each source CSV. Axis alignment is applied "
            "by wattitude.data.vicon.load_trial, not stored in these files."
        ),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return written

"""Loader for the Berlin Robust Orientation Estimation Assessment Dataset.

BROAD provides 39 trials of 6/9-axis IMU data at 2000/7 Hz with synchronised
OptiTrack quaternion ground truth in an ENU world frame, a boolean ``movement``
mask marking the phases over which errors are to be computed, and a
``trials.json`` describing the trial groups used for the per-group breakdown.

Reference: D. Laidig, M. Caruso, A. Cereatti, T. Seel, "BROAD -- A Benchmark for
Robust Inertial Orientation Estimation", Data 6(7):72, 2021.  CC-BY 4.0, obtained
from https://github.com/dlaidig/broad.

The accelerometer reads ``+g`` on the vertical axis at rest and the ground truth
is near identity at the start of every trial, which matches the convention of
:class:`~wattitude.eskf.AdaptiveESKF` directly; no frame conversion is needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["BroadTrial", "default_root", "dataset_available", "trial_info", "trial_names", "group_names", "trials_in_group", "load_trial", "groups_of"]

_ENV_VAR = "BROAD_ROOT"
_DEFAULT_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "data" / "broad" / "data_hdf5",
    Path.cwd() / "data" / "broad" / "data_hdf5",
)


@dataclass
class BroadTrial:
    """One BROAD trial."""

    name: str
    gyr: np.ndarray
    acc: np.ndarray
    mag: np.ndarray
    opt_quat: np.ndarray
    opt_pos: np.ndarray
    movement: np.ndarray
    rate: float
    groups: tuple[str, ...]
    description: str

    def __len__(self) -> int:
        return len(self.movement)

    @property
    def duration(self) -> float:
        return len(self) / self.rate


def default_root() -> Path:
    """Locate ``data_hdf5``, honouring the ``BROAD_ROOT`` environment variable."""
    env = os.environ.get(_ENV_VAR)
    if env:
        root = Path(env)
        return root if root.name == "data_hdf5" else root / "data_hdf5"
    for candidate in _DEFAULT_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return _DEFAULT_CANDIDATES[0]


def dataset_available(root: Path | str | None = None) -> bool:
    root = default_root() if root is None else Path(root)
    return (root / "trials.json").is_file()


@lru_cache(maxsize=4)
def trial_info(root: Path | str | None = None) -> dict:
    """Parsed ``trials.json``."""
    root = default_root() if root is None else Path(root)
    path = root / "trials.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"BROAD trials.json not found at {path}. Clone the dataset with\n"
            "  git clone --depth 1 https://github.com/dlaidig/broad.git data/broad\n"
            f"or point {_ENV_VAR} at an existing data_hdf5 directory."
        )
    with open(path) as f:
        return json.load(f)


def trial_names(root: Path | str | None = None) -> list[str]:
    """All 39 trial names in file order."""
    return sorted(trial_info(root)["trials"].keys())


def group_names(root: Path | str | None = None) -> list[str]:
    """Group names in the order defined by the dataset."""
    return [g["name"] for g in trial_info(root)["groups"]]


def group_levels(root: Path | str | None = None) -> dict[str, dict]:
    """Group metadata keyed by group name (``level`` and ``category``)."""
    return {g["name"]: g for g in trial_info(root)["groups"]}


def trials_in_group(group: str, root: Path | str | None = None) -> list[str]:
    info = trial_info(root)["trials"]
    return sorted(name for name, meta in info.items() if group in meta["groups"])


def groups_of(name: str, root: Path | str | None = None) -> tuple[str, ...]:
    return tuple(trial_info(root)["trials"][name]["groups"])


def load_trial(name: str, root: Path | str | None = None) -> BroadTrial:
    """Load a single trial by name (with or without the ``.hdf5`` suffix)."""
    import h5py

    root = default_root() if root is None else Path(root)
    name = name[:-5] if name.endswith(".hdf5") else name
    path = root / f"{name}.hdf5"
    if not path.is_file():
        raise FileNotFoundError(f"trial file not found: {path}")

    info = trial_info(root)["trials"].get(name, {})
    with h5py.File(path, "r") as f:
        return BroadTrial(
            name=name,
            gyr=np.ascontiguousarray(f["imu_gyr"][:], dtype=float),
            acc=np.ascontiguousarray(f["imu_acc"][:], dtype=float),
            mag=np.ascontiguousarray(f["imu_mag"][:], dtype=float),
            opt_quat=np.ascontiguousarray(f["opt_quat"][:], dtype=float),
            opt_pos=np.ascontiguousarray(f["opt_pos"][:], dtype=float),
            movement=np.ascontiguousarray(f["movement"][:], dtype=bool),
            rate=float(f.attrs["sampling_rate"]),
            groups=tuple(info.get("groups", ())),
            description=str(info.get("description", "")),
        )

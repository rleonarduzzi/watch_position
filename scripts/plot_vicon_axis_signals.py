"""Time-series figures for the axis-mounting note.

The optical rate and the gyroscope midpoint, and the lab accelerometer and
the position-derived specific force, on one sequence whose axes already
agree and one whose horizontal axes are reversed.  Windows are short
stretches of large motion so the shape, and the sign, can be read.
A third figure compares pitch from the accelerometer with the optical
pitch over the opening of the same two prefixes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wattitude.data.vicon import _body_rate, load_trial, short_root
from wattitude.eskf import GRAVITY
from wattitude.eval.report import figures_dir
from wattitude.quaternion import quat_to_euler

DT = 0.01
# (name, start_s, stop_s).  Chosen where the motion is large.
GYRO_WINDOWS = (("v3_13", 99.2, 103.2), ("v3_12", 91.0, 95.0))
ACC_WINDOWS = GYRO_WINDOWS
# Opening of each prefix: the specific force is closest to gravity here,
# which is where the tilt formulas apply.
TILT_WINDOWS = (("v3_13", 0.0, 10.0), ("v3_12", 0.0, 10.0))


def _rotations(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def _moving_average(x: np.ndarray, n: int = 9) -> np.ndarray:
    kernel = np.ones(n) / n
    return np.column_stack([np.convolve(x[:, i], kernel, mode="same") for i in range(x.shape[1])])


def specific_force(pos: np.ndarray, dt: float = DT) -> np.ndarray:
    """Lab specific force at samples 1 .. n-2, before the moving average."""
    f = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) / dt**2
    f = f.copy()
    f[:, 2] += GRAVITY
    return f


def lab_accelerometer(acc: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Accelerometer mapped by the optical attitude with ``R = I``."""
    return np.einsum("nij,nj->ni", _rotations(quat), acc)


def accel_tilt_deg(acc: np.ndarray) -> np.ndarray:
    """Roll and pitch in degrees from the accelerometer, Z-Y-X, |pitch| <= 90°."""
    roll = np.arctan2(acc[:, 1], acc[:, 2])
    hyp = np.hypot(acc[:, 1], acc[:, 2])
    pitch = np.arctan2(-acc[:, 0], hyp)
    return np.rad2deg(np.column_stack([roll, pitch]))


def _break_wraps(angle: np.ndarray, jump: float = 180.0) -> np.ndarray:
    out = np.asarray(angle, dtype=float).copy()
    d = np.abs(np.diff(out))
    out[1:][d > jump] = np.nan
    return out


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _three_panel(path: Path, index: np.ndarray, left: np.ndarray, right: np.ndarray,
                 left_label: str, right_label: str, ylabel: str, title: str,
                 xlabel: str) -> None:
    plt = _style()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
    for i, axis in enumerate("xyz"):
        ax = axes[i]
        ax.plot(index, left[:, i], color="k", lw=1.1, label=left_label)
        ax.plot(index, right[:, i], color="tab:blue", lw=1.0, label=right_label)
        ax.set_ylabel(f"{axis} {ylabel}")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
            ax.set_title(title)
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    root = short_root()
    out = figures_dir()
    for name, start, stop in GYRO_WINDOWS:
        trial = load_trial(name, root=root, align_axes=False)
        omega = _body_rate(trial.opt_quat, DT)
        gyro = 0.5 * (trial.gyr[:-1] + trial.gyr[1:])
        lo, hi = int(round(start / DT)), int(round(stop / DT))
        _three_panel(
            out / f"vicon_axis_omega_{name}.png",
            np.arange(lo, hi),
            omega[lo:hi],
            gyro[lo:hi],
            r"optical $\omega$",
            r"gyroscope $\tilde{\omega}$",
            "(rad/s)",
            f"{name}, samples {lo}–{hi - 1}",
            r"sample $i$",
        )
        print(f"wrote {out / f'vicon_axis_omega_{name}.png'}")

    for name, start, stop in ACC_WINDOWS:
        trial = load_trial(name, root=root, align_axes=False)
        force = _moving_average(specific_force(trial.opt_pos))
        accel = lab_accelerometer(trial.acc, trial.opt_quat)
        lo, hi = int(round(start / DT)), int(round(stop / DT))
        # force[j] is centered on sample j+1
        _three_panel(
            out / f"vicon_axis_acc_{name}.png",
            np.arange(lo, hi),
            force[lo - 1:hi - 1],
            accel[lo:hi],
            r"specific force $f$",
            r"accelerometer $a^{\mathrm{lab}}$",
            r"(m/s$^2$)",
            f"{name}, samples {lo}–{hi - 1}",
            r"sample $k$",
        )
        print(f"wrote {out / f'vicon_axis_acc_{name}.png'}")

    plt = _style()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    for ax, (name, start, stop) in zip(axes, TILT_WINDOWS):
        trial = load_trial(name, root=root, align_axes=False)
        truth = np.rad2deg(quat_to_euler(trial.opt_quat)[:, 1])
        est = accel_tilt_deg(trial.acc)[:, 1]
        lo, hi = int(round(start / DT)), int(round(stop / DT))
        t = np.arange(lo, hi) * DT
        ax.plot(t, _break_wraps(truth[lo:hi]), color="k", lw=1.3, label="Vicon")
        ax.plot(t, _break_wraps(est[lo:hi]), color="tab:blue", lw=1.0, label="accelerometer")
        ax.set_ylabel("pitch (deg)")
        ax.set_title(f"{name}, {start:.0f}–{stop:.0f} s")
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    path = out / "vicon_axis_tilt.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

"""Isolate which noise channel makes the NEES inconsistent.

Runs the Monte-Carlo consistency experiment with one noise source enabled at a
time and reports the per-block ANEES, so a mis-scaled term can be attributed to
a specific entry of Q or R instead of guessed at.
"""

from __future__ import annotations

import numpy as np

from wattitude import AdaptiveESKF
from wattitude.quaternion import exp_q, log_q, quat_conjugate, quat_multiply, quat_normalize
from wattitude.simulate import simulate_imu

RATE = 100.0
DURATION = 25.0
N_RUNS = 40


def experiment(noise: dict, n_runs: int = N_RUNS, exact_phi: bool = True, label: str = ""):
    blocks = {"tilt": slice(0, 3), "bias_gyr": slice(3, 6), "bias_acc": slice(6, 9)}
    acc = {k: [] for k in blocks}
    total = []

    for run in range(n_runs):
        sim = simulate_imu(
            duration=DURATION, rate=RATE, ext_accel="none", rest_duration=0.0,
            rotation_amplitude=0.8, seed=1000 + run, **noise,
        )
        est = AdaptiveESKF(
            rate=RATE, adaptive=False, estimate_gravity=False, exact_phi=exact_phi,
            sigma_gyr=max(noise["sigma_gyr"], 1e-12),
            sigma_acc=max(noise["sigma_acc"], 1e-12),
            sigma_bias_gyr=max(noise["sigma_bias_gyr"], 1e-12),
            sigma_bias_acc=max(noise["sigma_bias_acc"], 1e-12),
            init_tilt_std=np.deg2rad(3.0), init_bias_gyr_std=5e-3, init_bias_acc_std=0.05,
        )
        rng = np.random.default_rng(5000 + run)
        dx0 = np.linalg.cholesky(est.P) @ rng.normal(size=est.dim)
        est.quat = quat_normalize(quat_multiply(sim.quat[0], exp_q(-dx0[0:3])))
        est.bias_gyr = sim.bias_gyr[0] - dx0[3:6]
        est.bias_acc = sim.bias_acc[0] - dx0[6:9]

        n = len(sim.t)
        nees = np.empty(n)
        per_block = {k: np.empty(n) for k in blocks}
        for k in range(n):
            est.update(sim.gyr[k], sim.acc[k])
            dx = np.concatenate(
                [
                    log_q(quat_multiply(quat_conjugate(est.quat), sim.quat[k])),
                    sim.bias_gyr[k] - est.bias_gyr,
                    sim.bias_acc[k] - est.bias_acc,
                ]
            )
            nees[k] = dx @ np.linalg.solve(est.P, dx)
            for name, sl in blocks.items():
                d, P = dx[sl], est.P[sl, sl]
                per_block[name][k] = d @ np.linalg.solve(P, d)
        total.append(nees)
        for name in blocks:
            acc[name].append(per_block[name])

    total = np.array(total).mean(axis=0)
    print(f"{label:<34} ANEES(9) = {total.mean():6.3f}  (target 9.000)", end="")
    for name in blocks:
        m = np.array(acc[name]).mean(axis=0).mean()
        print(f"   {name}:{m:6.3f}", end="")
    print(f"   [first 10%: {total[: len(total) // 10].mean():6.3f}, "
          f"last 10%: {total[-len(total) // 10 :].mean():6.3f}]")
    return total


def main() -> None:
    base = dict(sigma_gyr=0.0, sigma_acc=0.0, sigma_bias_gyr=0.0, sigma_bias_acc=0.0)
    print("Per-block ANEES target is 3.000 for each 3-dimensional block.\n")

    experiment({**base, "sigma_acc": 0.12}, label="acc noise only")
    experiment({**base, "sigma_gyr": 4e-3, "sigma_acc": 0.12}, label="+ gyro white noise")
    experiment(
        {**base, "sigma_gyr": 4e-3, "sigma_acc": 0.12, "sigma_bias_gyr": 2e-4},
        label="+ gyro bias walk",
    )
    experiment(
        {**base, "sigma_gyr": 4e-3, "sigma_acc": 0.12, "sigma_bias_acc": 2e-3},
        label="+ accel bias walk",
    )
    full = dict(sigma_gyr=4e-3, sigma_acc=0.12, sigma_bias_gyr=2e-4, sigma_bias_acc=2e-3)
    experiment(full, label="all noise sources")
    experiment(full, exact_phi=False, label="all, first-order Phi (eq 8)")


if __name__ == "__main__":
    main()

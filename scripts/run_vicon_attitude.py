"""Attitude tracking on the short IMU/Vicon prefixes.

Cuts the first two minutes of every sequence in ``data/imu_vicon_joint_v10``
into ``data/imu_vicon_joint_v10_short``, then runs the adaptive ESKF and VQF
at the same trial-agnostic parameters used for the BROAD attitude report.
Errors and Euler-angle figures are written in that same form.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wattitude.data import vicon
from wattitude.eval.report import figures_dir, plot_euler_tracking
from wattitude.quaternion import euler_gimbal_risk

ROOT = Path(__file__).resolve().parents[1]
FIGURES = figures_dir()

# Trial-agnostic optima from reports/benchmark_results.md, the same pair used
# by scripts/make_attitude_report.py.  Not retuned on this dataset.
ESKF_PARAMS = {"sigma_acc": 200.0}
VQF_PARAMS = {"tauAcc": 3.0}


def main() -> None:
    written = vicon.write_prefix_dataset()
    print(f"wrote {len(written)} prefixes of {vicon.PREFIX_S:.0f} s to {vicon.short_root()}")

    lines: list[str] = []
    A = lines.append
    A("# Attitude tracking on the short Vicon prefixes\n")
    A(
        "Roll, pitch and yaw of the adaptive ESKF and of VQF against the Vicon "
        "quaternion, on the first two minutes of each sequence in "
        "`data/imu_vicon_joint_v10`. The full recordings are ten to thirty "
        "minutes at 100 Hz; this pass uses the prefixes in "
        "`data/imu_vicon_joint_v10_short` so the filter can be checked before "
        "running the long files.\n"
    )
    A("## Data\n")
    A(
        "Each CSV holds accelerometer specific force in m/s², gyroscope "
        "angular rate in rad/s, optical position in metres, and an optical "
        "attitude quaternion stored scalar-last (`x, y, z, w`). The quaternion "
        "is body-to-world with z up, and the accelerometer reads about `+g` on "
        "the upward axis at rest, which is the filter's convention. There is "
        "no magnetometer and no separate movement annotation, so the errors "
        "below are over every sample of the two-minute prefix.\n"
    )
    A(
        "On part of the recordings the IMU is yawed 180° relative to the "
        "optical body: its x and y axes point the opposite way and z agrees. "
        "That is a mounting difference, not an attitude the filter should have "
        "to discover. `load_trial` detects it by comparing the sign of the "
        "gyroscope with the optical angular rate and reverses x and y before "
        "either estimator runs. The files in the short dataset are verbatim "
        "prefixes; the correction is applied in memory. Sequences marked "
        "`rz180` below are the ones that needed it.\n"
    )
    A("## How to read these plots\n")
    A(
        "The figure is the same one used for BROAD in `attitude_tracking.md`. "
        "Euler angles are intrinsic Z-Y-X, `R = Rz(yaw) Ry(pitch) Rx(roll)`, "
        "in degrees, for display only. Both estimators are magnetometer-free, "
        "so each trace has had its optimal constant yaw offset removed before "
        "plotting; the yaw panel is the residual drift. Orange bands mark "
        "pitch within 10° of vertical, where roll and yaw are not separately "
        "determined. One-sample gaps are wraps through ±180°.\n"
    )
    A(
        "Parameters are the BROAD trial-agnostic pair, carried over as-is: "
        "ESKF `sigma_acc=200` with the closed-form transition "
        "(`exact_phi=True`, the library default), VQF `tauAcc=3`.\n"
    )
    A("## Per-sequence traces\n")

    rows = []
    for name in vicon.sequence_names(vicon.short_root()):
        trial = vicon.load_trial(name, root=vicon.short_root())
        path, data = plot_euler_tracking(
            name,
            trial=trial,
            eskf_params=ESKF_PARAMS,
            vqf_params=VQF_PARAMS,
            max_points=8000,
            out=FIGURES / f"vicon_euler_{name}.png",
        )
        n_move = int(trial.movement.sum())
        factor = max(1, int(np.ceil(n_move / 8000)))
        fix = "IMU x,y reversed to match the optical body" if trial.axis_fix == "rz180" else "IMU axes already match the optical body"
        A(f"### {name}\n")
        A(f"{fix}.\n")
        A(f"![{name}]({path.relative_to(ROOT)})\n")
        A(
            f"Decimated {factor}x for display, from {n_move} samples "
            f"at {trial.rate:.0f} Hz ({trial.duration:.0f} s).\n"
        )
        entry = {
            "sequence": name,
            "axis fix": trial.axis_fix,
            "duration_s": trial.duration,
        }
        for key, label in (("eskf", "ESKF"), ("vqf", "VQF")):
            errs = data["series"][key]["errors"]
            entry[f"{label} incl"] = errs.inclination_rmse_deg
            entry[f"{label} total"] = errs.total_rmse_deg
            entry[f"{label} heading"] = errs.heading_rmse_deg
            entry[f"{label} drift"] = errs.heading_drift_deg_per_min
        entry["gimbal-risk fraction"] = float(euler_gimbal_risk(trial.opt_quat)[trial.movement].mean())
        rows.append(entry)
        incl = entry
        print(
            f"{name:6s} {trial.axis_fix:8s}  "
            f"ESKF incl {incl['ESKF incl']:6.2f}  VQF incl {incl['VQF incl']:6.2f}  "
            f"drift {incl['ESKF drift']:+7.1f} / {incl['VQF drift']:+7.1f} deg/min"
        )

    table = pd.DataFrame(rows).set_index("sequence")
    show = table.drop(columns=["duration_s"])
    A("## Summary\n")
    A(show.to_markdown(floatfmt=".3f") + "\n")
    A(
        "Inclination, total and heading are RMSE in degrees over the whole "
        "prefix, after the constant heading offset has been removed. Drift is "
        "the slope of a linear fit to the signed yaw error, in degrees per "
        "minute. Inclination is the heading-independent part and is the fair "
        "comparison: a 6-axis IMU cannot observe absolute yaw, so total error "
        "and drift mix that random walk into the score. The gimbal-risk "
        "fraction is the share of samples whose pitch is within 10° of "
        "vertical.\n"
    )

    eskf = table["ESKF incl"].to_numpy()
    vqf = table["VQF incl"].to_numpy()
    A(
        f"Mean inclination RMSE is {eskf.mean():.2f}° for the ESKF and "
        f"{vqf.mean():.2f}° for VQF. VQF is lower on "
        f"{int((vqf < eskf).sum())} of the {len(table)} prefixes and the ESKF "
        f"on {int((eskf < vqf).sum())} (`v3_12`, `v3_14`). On the four prefixes "
        f"with inclination under 8° (`v3_12` through `v3_15`) both estimators follow the "
        f"optical pitch oscillation and the total error stays in a band of "
        f"roughly 5° to 15°, with isolated spikes. On `v3_01` and `v3_11` "
        f"pitch spends long stretches near vertical, the orange bands cover "
        f"much of the roll and yaw panels, and both filters share an "
        f"inclination error of 17° to 19°. That shared floor is the number "
        f"worth reading; the roll traces through the orange bands are the "
        f"Euler singularity.\n"
    )
    A(
        "`v3_02` is where the two filters separate. Inclination is close "
        f"({table.loc['v3_02', 'ESKF incl']:.1f}° against "
        f"{table.loc['v3_02', 'VQF incl']:.1f}°), while total error is "
        f"{table.loc['v3_02', 'ESKF total']:.1f}° for the ESKF and "
        f"{table.loc['v3_02', 'VQF total']:.1f}° for VQF, because the ESKF "
        f"heading walks and VQF's stays put. The total-error panel shows that "
        f"as a raised floor on the green trace.\n"
    )
    A(
        f"The drift column on `v3_15` reads "
        f"{table.loc['v3_15', 'ESKF drift']:+.0f}°/min for both estimators. "
        f"That slope is one 360° step in the unwrapped signed-heading series "
        f"near 80 s. Over the same prefix the Euler yaw difference has a "
        f"median of a fraction of a degree, and the yaw panel shows both "
        f"estimates on the optical trace. Inclination, "
        f"{table.loc['v3_15', 'ESKF incl']:.1f}° and "
        f"{table.loc['v3_15', 'VQF incl']:.1f}°, is the figure that matches "
        f"the plot.\n"
    )

    csv_path = ROOT / "reports" / "vicon_short_errors.csv"
    table.to_csv(csv_path)
    out = ROOT / "reports" / "vicon_attitude_tracking.md"
    out.write_text("\n".join(lines))
    print(f"\nwrote {out}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

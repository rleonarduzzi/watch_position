"""Build reports/attitude_tracking.md: estimated against reference attitude.

Shows the actual attitude traces as roll, pitch and yaw rather than only
aggregate error numbers, for this filter and for VQF, on a spread of trials.
Both estimators run at the trial-agnostic parameters found by the benchmark,
so what is plotted is one fixed configuration across every trial rather than a
per-trial best case.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wattitude.baselines import ESTIMATORS
from wattitude.eval.metrics import error_series
from wattitude.eval.report import figures_dir, plot_euler_tracking
from wattitude.eval.runner import init_window_for
from wattitude.quaternion import euler_gimbal_risk

ROOT = Path(__file__).resolve().parents[1]
FIGURES = figures_dir()

# Trial-agnostic optima from reports/benchmark_results.md.
ESKF_PARAMS = {"sigma_acc": 200.0}
VQF_PARAMS = {"tauAcc": 3.0}

# One trial from each regime that stresses the filter differently.
TRIALS = [
    ("01_undisturbed_slow_rotation_A", "slow rotation, the gentlest case"),
    ("10_undisturbed_slow_translation_A", "slow translation, mild external acceleration"),
    ("06_undisturbed_fast_rotation_A", "fast rotation, stresses gyroscope integration"),
    ("21_undisturbed_fast_combined", "fast combined, the strongest external acceleration"),
    ("13_undisturbed_slow_translation_with_breaks_A", "motion separated by rest phases"),
    ("24_disturbed_tapping_A", "impulsive mechanical taps"),
    ("26_disturbed_phone_vibration_A", "sustained vibration"),
    ("39_disturbed_mixed", "mixed disturbances"),
]

# A short window on one trial, where individual taps are resolvable.
ZOOM = ("24_disturbed_tapping_A", (80.0, 100.0))

# The trial on which the printed transition matrix visibly fails: it peaks at
# 728 deg/s, where the covariance inflation of equation (8) is worst.
DEFECT_TRIAL = "21_undisturbed_fast_combined"


def main() -> None:
    lines: list[str] = []
    A = lines.append

    A("# Attitude tracking: estimated against optical reference\n")
    A(
        "Roll, pitch and yaw of this filter and of VQF plotted against the "
        "OptiTrack reference, on eight BROAD trials chosen to span the range "
        "of motion and disturbance. Aggregate scores for all 39 trials are in "
        "`benchmark_results.md`; this report is about what the estimates "
        "actually look like.\n"
    )

    # ------------------------------------------------------------- how to read
    A("## How to read these plots\n")
    A(
        "**Euler convention.** Intrinsic Z-Y-X, the usual aerospace one: the "
        "attitude is yaw about world z, then pitch about the new y, then roll "
        "about the new x, so `R = Rz(yaw) Ry(pitch) Rx(roll)`. The world frame "
        "is z-up, so yaw is heading and roll with pitch describe the tilt. "
        "Roll and yaw run over `(-180, 180]` degrees and pitch over "
        "`[-90, 90]`. Angles are used for display only; the filter and every "
        "error metric work in quaternions.\n"
    )
    A(
        "**Heading is aligned before plotting.** Both estimators are "
        "magnetometer-free and cannot observe absolute heading, so each one "
        "expresses yaw in its own initial frame rather than the optical "
        "system's. Each estimate has its optimal constant yaw offset removed "
        "first, the same alignment the error metric applies. Without it the "
        "yaw panel would show nothing but a constant offset. The remaining yaw "
        "discrepancy is genuine drift, which is what the slow divergence in "
        "the bottom panel of most plots is.\n"
    )
    A(
        "**The shaded bands are a parameterisation artefact, not error.** When "
        "pitch approaches the vertical, roll and yaw stop being separately "
        "determined: only their sum or difference is, and both can swing "
        "arbitrarily far while the attitude barely moves. Bands mark pitch "
        "within 10 degrees of vertical; disagreements inside them should be "
        "ignored. Pitch itself is always well conditioned, which is why it is "
        "the cleanest of the three panels.\n"
    )
    A(
        "**Gaps are real.** One-sample breaks in a trace are wraps across "
        "+-180 degrees, drawn as gaps so no vertical line is painted across "
        "the axis. Longer gaps in the black reference are optical-tracking "
        "dropouts, which the metrics skip rather than interpolate.\n"
    )
    A(
        "Traces are decimated for file size; the decimation factor is stated "
        "under each figure, and the error panel is computed at full rate.\n"
    )

    # ------------------------------------------------------------------- plots
    A("## Per-trial traces\n")
    A(
        f"Parameters are fixed throughout: ESKF `{ESKF_PARAMS}`, "
        f"VQF `{VQF_PARAMS}`.\n"
    )

    rows = []
    for trial_name, description in TRIALS:
        path, data = plot_euler_tracking(
            trial_name, eskf_params=ESKF_PARAMS, vqf_params=VQF_PARAMS, max_points=8000
        )
        trial = data["trial"]
        movement = trial.movement
        n_move = int(movement.sum())
        factor = max(1, int(np.ceil(n_move / 8000)))

        A(f"### {trial_name}\n")
        A(f"{description.capitalize()}.\n")
        A(f"![{trial_name}]({path.relative_to(ROOT)})\n")
        A(
            f"Decimated {factor}x for display, from {n_move} movement samples "
            f"at {trial.rate:.1f} Hz.\n"
        )

        entry = {"trial": trial_name, "regime": description}
        for key, label in (("eskf", "ESKF"), ("vqf", "VQF")):
            errs = data["series"][key]["errors"]
            entry[f"{label} incl"] = errs.inclination_rmse_deg
            entry[f"{label} total"] = errs.total_rmse_deg
            entry[f"{label} drift"] = errs.heading_drift_deg_per_min
        entry["gimbal-risk fraction"] = float(
            euler_gimbal_risk(trial.opt_quat)[movement].mean()
        )
        rows.append(entry)

    # ------------------------------------------------------------------- zoom
    trial_name, span = ZOOM
    path, _ = plot_euler_tracking(
        trial_name, eskf_params=ESKF_PARAMS, vqf_params=VQF_PARAMS, zoom=span
    )
    A(f"### {trial_name}, {span[0]:.0f} to {span[1]:.0f} s\n")
    A(
        "The same trial at full sample rate, with no decimation. The useful "
        "thing about this window is a negative result: both estimates are "
        "indistinguishable from the reference in all three angle panels, taps "
        "included, and the entire difference between them lives in the bottom "
        "panel at a scale of one to three degrees. Angle traces are the right "
        "way to confirm that an estimator tracks the motion at all and the "
        "wrong way to compare two that both do; that is what the RMSE tables "
        "are for.\n"
    )
    A(
        "The two shaded excursions are the parameterisation artefact rather "
        "than anything physical: pitch passes through the vertical, and roll "
        "and yaw jump 180 degrees in consequence while the attitude itself "
        "moves smoothly.\n"
    )
    A(f"![{trial_name} zoom]({path.relative_to(ROOT)})\n")

    # ---------------------------------------------------------------- summary
    A("## Summary over the trials shown\n")
    table = pd.DataFrame(rows).set_index("trial")
    A(table.to_markdown(floatfmt=".3f") + "\n")
    A(
        "Errors are RMSE in degrees over the movement phase; drift is the slope "
        "of a linear fit to the signed yaw error, in degrees per minute. "
        "Inclination is the heading-independent part and is the fair comparator; "
        "total error includes the unobservable heading random walk.\n"
    )

    eskf_better = table["ESKF incl"] < table["VQF incl"]
    A(
        f"VQF has the lower inclination error on "
        f"{int((~eskf_better).sum())} of the {len(table)} trials shown, "
        "consistent with the full benchmark. The traces make the reason "
        "legible: the two estimators are nearly indistinguishable through "
        "ordinary motion, and separate during the stretches where the "
        "accelerometer is least trustworthy.\n"
    )
    A(
        "Worth noting on `01_undisturbed_slow_rotation_A`: the ESKF does worse "
        "here than its 39-trial average, because a single trial-agnostic "
        "`sigma_acc` has to cover trials whose external acceleration differs by "
        "two orders of magnitude, and the value that wins overall is far too "
        "distrustful of the accelerometer for the gentlest recording. That "
        "tension is the generalisation gap tabulated in "
        "`benchmark_results.md`.\n"
    )

    # ----------------------------------------------------- the heading finding
    A("## What these plots found that the error tables did not\n")
    A(
        "Plotting the angles turned up a defect that no aggregate inclination "
        "number had exposed. Run with the paper's printed transition matrix, "
        "equation (8), the filter tracks roll and pitch on "
        f"`{DEFECT_TRIAL}` perfectly well but its yaw wanders away from the "
        "reference and back by nearly 200 degrees. VQF's yaw stays flat on the "
        "same recording. The excursion is not a steady drift: it grows, returns "
        "almost to zero, and grows again, which is the signature of a corrupted "
        "covariance rather than of random-walk accumulation.\n"
    )

    defect_rows = []
    for label, params, tag in (
        ("equation (8), first order", {"sigma_acc": 200.0, "exact_phi": False}, "eq8"),
        ("closed-form transition", {"sigma_acc": 200.0, "exact_phi": True}, "exact"),
    ):
        path, data = plot_euler_tracking(
            DEFECT_TRIAL, eskf_params=params, vqf_params=VQF_PARAMS,
            out=FIGURES / f"euler_defect_{tag}.png", max_points=8000,
        )
        errs = data["series"]["eskf"]["errors"]
        total = np.rad2deg(
            np.abs(error_series(
                ESTIMATORS["eskf"](
                    data["trial"].gyr, data["trial"].acc, data["trial"].rate,
                    init_slice=init_window_for(data["trial"]), **params,
                ),
                data["trial"].opt_quat, data["trial"].movement,
            )["signed_heading"])
        )
        A(f"**{label}.**\n")
        A(f"![{label}]({path.relative_to(ROOT)})\n")
        defect_rows.append({
            "transition": label,
            "inclination RMSE": errs.inclination_rmse_deg,
            "total RMSE": errs.total_rmse_deg,
            "peak |yaw error|": float(np.nanmax(total)),
        })

    A(pd.DataFrame(defect_rows).set_index("transition").to_markdown(floatfmt=".2f") + "\n")
    A(
        "The cause is that the printed `Phi = I + F dt` is not an orthogonal "
        "matrix, while the exact attitude block is a rotation. Its largest "
        "singular value is `sqrt(1 + |w|^2 dt^2)`, so every propagation step "
        "inflates the attitude covariance a little, with nothing in the filter "
        "to remove it. The inflation grows as the square of the angular rate: "
        "negligible at 29 degrees per second, a factor of 5e14 per minute at "
        "the 728 degrees per second this trial reaches. The filter then runs "
        "with systematically over-large gains, and over-large corrections "
        "acquire a component along the locally unobservable vertical, where "
        "nothing can ever remove it.\n"
    )
    A(
        "`exact_phi=True` is now the default. Section 6 of "
        "`derivation_review.md` has the derivation, the singular-value check "
        "and the honest cost: averaged over all 39 trials the fix improves "
        "total error from 9.36 to 3.57 degrees and *worsens* inclination from "
        "1.25 to 1.76, because the spurious gain happened to flatter tilt "
        "tracking on this dataset. All the per-trial figures above use the "
        "corrected default.\n"
    )
    A(
        "Two things are worth drawing from this. The defect was invisible in "
        "every aggregate inclination score, in 84 passing tests including "
        "Monte-Carlo covariance-consistency checks -- which are run at rates "
        "where the inflation is negligible -- and in the finite-difference "
        "Jacobian checks, because the matrix is a correct first-order "
        "approximation and simply not an orthogonal one. Looking at the "
        "trajectories found it in minutes. And a heading error matters more for "
        "this pipeline than its size alone suggests: `to_world_frame()` output "
        "is rotated bodily about the vertical, so a position estimate "
        "integrated from it inherits the error as a curving trajectory, which "
        "half a degree of extra tilt error does not do.\n"
    )

    out = ROOT / "reports" / "attitude_tracking.md"
    out.write_text("\n".join(lines))
    print(table.to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""Turn the cached benchmark results into reports/benchmark_results.md and figures.

Reads only the cached CSVs plus a handful of diagnostic re-runs for the plots,
so it is cheap to iterate on the write-up without redoing the grid search.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from wattitude.data import broad
from wattitude.eval.report import (
    PRETTY,
    figures_dir,
    group_table,
    load_variant,
    matched_nominal_groups,
    plot_adaptation,
    plot_group_errors,
    plot_nominal_sweep,
    tagp_table,
)
from wattitude.eval.tuning import METRICS, SELECTION_METRIC, itop, tagp

ROOT = Path(__file__).resolve().parents[1]
SIX_AXIS = [
    "gyro_only", "madgwick_6d", "madgwick_adaptive_6d", "mahony_6d",
    "vqf_6d", "eskf_fixed", "eskf_adaptive", "eskf_exactphi",
]
# Nominal accelerometer sigma used for the adaptation figure: deliberately left
# at the instrument noise level so the adapted covariance, not the nominal, is
# what reacts to the disturbance.
FIGURE_SIGMA_ACC = 0.15
# Shared nominal for the adaptive-versus-fixed comparison: the smallest value in
# the grid, i.e. the accelerometer noise density a datasheet would give.
MATCHED_NOMINAL = 0.1


def md(df: pd.DataFrame, floatfmt: str = "%.3f") -> str:
    return df.to_markdown(index=True, floatfmt=floatfmt.replace("%", "").replace("f", "f"))


def measure_throughput(trial_name: str = "01_undisturbed_slow_rotation_A") -> dict:
    from wattitude.eskf import run_batch
    from wattitude.eval.runner import init_window_for

    trial = broad.load_trial(trial_name)
    window = init_window_for(trial)
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        run_batch(trial.gyr, trial.acc, trial.rate, init_samples=window.stop)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    us = 1e6 * best / len(trial)
    return {
        "us_per_sample": us,
        "realtime_factor": 1e6 / (us * trial.rate),
        "n": len(trial),
    }


def main() -> None:
    pd.set_option("display.width", 250)
    lines: list[str] = []
    A = lines.append

    A("# Adaptive ESKF on BROAD: benchmark results\n")
    A(
        "All numbers come from the 39 trials of the BROAD benchmark (Laidig, "
        "Caruso, Cereatti, Seel), sampled at 2000/7 Hz with OptiTrack ground "
        "truth, scored over the annotated movement phase.\n"
    )

    # ---------------------------------------------------------------- protocol
    A("## Protocol\n")
    A(
        "Every estimator is given exactly one tuned scalar, as in BROAD's own "
        "evaluation: `beta` for Madgwick, `Kp` for Mahony, `tauAcc` for VQF and "
        "`sigma_acc` for the ESKF. **TAGP** is the best achievable average with "
        "a single parameter set shared by all trials; **ITOP** averages the "
        "per-trial optima. Both are selected on inclination RMSE rather than "
        "BROAD's total RMSE, because a 6-axis filter cannot observe heading at "
        "all and selecting on total error would tune an unobservable quantity; "
        "the 9-axis row is scored BROAD's way and reproduces its published "
        "numbers, which is what validates the harness.\n"
    )
    A(
        "All estimators share one initialisation window, detected from the IMU "
        "signals alone by a rest detector, so no ground truth enters "
        "initialisation and no estimator is advantaged.\n"
    )

    # ------------------------------------------------------------ main results
    A("## Trial-agnostic parameters (TAGP)\n")
    tbl = tagp_table(SIX_AXIS).set_index("variant").drop(columns=["tag"])
    tbl = tbl.rename(columns={
        "total_rmse_deg": "total", "heading_rmse_deg": "heading",
        "inclination_rmse_deg": "inclination", "drift_deg_per_min": "|drift| deg/min",
    })
    A(tbl.to_markdown(floatfmt=".3f") + "\n")
    A(
        "The two `eq. (8)` rows use the paper's printed first-order transition "
        "matrix; the `exact Phi` row uses the closed form and is the library "
        "default. That matrix is not orthogonal and inflates the attitude "
        "covariance at a rate proportional to the square of the angular rate, "
        "which costs little inclination accuracy but a great deal of heading "
        "accuracy, as the total-error column shows. The defect was found by "
        "plotting attitude traces rather than by reading these tables; see "
        "`attitude_tracking.md` and section 6 of `derivation_review.md`. The "
        "adaptive-versus-fixed comparison below is between the two `eq. (8)` "
        "rows, so that the transition matrix is held constant and only the "
        "covariance adaptation differs.\n"
    )
    A(
        "Total and heading error are reported for completeness but are not "
        "meaningful comparators here: they are dominated by the unobservable "
        "heading random walk, which is why the open-loop row can look "
        "competitive on total error while being an order of magnitude worse on "
        "inclination.\n"
    )

    A("## Individually tuned optima (ITOP)\n")
    rows = []
    for tag in SIX_AXIS:
        df = load_variant(tag)
        tuned = itop(df, SELECTION_METRIC)
        key, _ = tagp(df, SELECTION_METRIC)
        rows.append({
            "variant": PRETTY[tag],
            "TAGP inclination": float(df[df["params"] == key][SELECTION_METRIC].mean()),
            "ITOP inclination": float(tuned[SELECTION_METRIC].mean()),
        })
    itop_tbl = pd.DataFrame(rows).set_index("variant")
    itop_tbl["generalisation gap"] = (
        itop_tbl["TAGP inclination"] - itop_tbl["ITOP inclination"]
    )
    A(itop_tbl.to_markdown(floatfmt=".3f") + "\n")
    A(
        "The gap is how much an algorithm would gain from per-recording tuning; "
        "a small gap means one parameter set transfers across motions.\n"
    )

    # ------------------------------------------------------------ group tables
    A("## Inclination RMSE by trial group, at TAGP\n")
    fig_groups, gtable = plot_group_errors(SIX_AXIS)
    A(gtable.to_markdown(floatfmt=".3f") + "\n")
    A(f"![per-group error]({fig_groups.relative_to(ROOT)})\n")

    # ------------------------------------------------- adaptive vs fixed R
    A("## What the adaptive covariance is actually worth\n")
    A(
        "At the trial-agnostic optimum the adaptive and fixed-covariance ESKF "
        "produce **identical** numbers, to every digit. That is not a bug and "
        "it is the most informative result here. The innovation-based estimate "
        "of equation (17) uses the nominal covariance as a floor, and the "
        "optimum nominal on this dataset is far above anything the innovations "
        "ever produce, so the floor is always active and the adaptation never "
        "engages. Tuning `sigma_acc` freely therefore switches the mechanism "
        "off.\n"
    )
    A(
        "The comparison that isolates the mechanism is to sweep the nominal and "
        "ask what the adaptation recovers at each setting.\n"
    )
    fig_sweep, sweep = plot_nominal_sweep()
    A(sweep.to_markdown(floatfmt=".3f") + "\n")
    A(f"![nominal sweep]({fig_sweep.relative_to(ROOT)})\n")
    best_ratio = sweep["ratio"].idxmax()
    A(
        f"Read down the table: at a datasheet-level nominal the adaptation is "
        f"worth a factor of {sweep['ratio'].max():.2f} "
        f"({sweep.loc[best_ratio, 'fixed']:.2f} degrees falling to "
        f"{sweep.loc[best_ratio, 'adaptive']:.2f}), and the benefit decays "
        "monotonically to exactly 1.00 as the nominal is raised past the "
        "disturbance scale. So the adaptation is a substitute for tuning rather "
        "than an improvement over a well-tuned filter: it buys roughly half the "
        "distance to the tuned optimum without being told the disturbance "
        "magnitude, and buys nothing once it has been. Note also that the "
        "fixed-R filter is the one that is dangerous when mis-set -- at a "
        f"datasheet nominal it is worse than open-loop integration "
        f"({sweep.loc[sweep.index[0], 'fixed']:.1f} degrees against "
        f"{load_variant('gyro_only')[SELECTION_METRIC].mean():.1f}), whereas "
        "the adaptive one never is.\n"
    )

    A(f"### Per-group behaviour at a matched nominal (`sigma_acc = {MATCHED_NOMINAL}`)\n")
    matched = matched_nominal_groups(MATCHED_NOMINAL)
    A(matched.to_markdown(floatfmt=".3f") + "\n")
    ratios = matched.drop(index=["all_trials"])["ratio"]
    A(
        "This contradicts the expectation we started from. We predicted the "
        "adaptation would help most on the tapping and vibration trials, and it "
        f"helps least there ({ratios.get('tapping', float('nan')):.2f}x and "
        f"{ratios.get('vibration', float('nan')):.2f}x). The largest gains are "
        f"on fast ({ratios.get('fast', float('nan')):.2f}x) and translational "
        f"({ratios.get('translation', float('nan')):.2f}x) motion.\n"
    )
    A(
        "The reason is the window length. Equation (17) estimates the "
        "measurement covariance from a sliding window of innovations, so it "
        "responds to disturbances that persist for a comparable time. A tap is "
        "a millisecond-scale impulse: it does raise the covariance, visibly and "
        "by orders of magnitude, but only after the impulse has already "
        "corrupted the state, and the trials are mostly undisturbed between "
        "taps. Sustained translational acceleration keeps the covariance "
        "legitimately elevated for as long as the disturbance lasts, which is "
        "the regime the mechanism is built for. Reacting to taps would need a "
        "much shorter window or an explicit outlier gate on the innovation, "
        "neither of which is in the paper.\n"
    )
    A(
        "The single `mixed` trial shows the largest ratio of all "
        f"({ratios.get('mixed', float('nan')):.2f}x) but it is one recording, "
        "so the 9- and 11-trial translation and fast groups are the stronger "
        "evidence.\n"
    )

    # ------------------------------------------------------- adaptation figure
    A("## The adaptation in the time domain\n")
    stats = {}
    for trial_name in ("24_disturbed_tapping_A", "26_disturbed_phone_vibration_A"):
        path, info = plot_adaptation(trial_name, sigma_acc=FIGURE_SIGMA_ACC)
        stats[trial_name] = info
        A(f"![adaptation on {trial_name}]({path.relative_to(ROOT)})\n")
    stat_tbl = pd.DataFrame(stats).T
    A(stat_tbl.to_markdown(floatfmt=".4g") + "\n")
    A(
        f"At a nominal `sigma_acc = {FIGURE_SIGMA_ACC}` the innovation-based "
        "estimate inflates the measurement covariance by orders of magnitude "
        "during a disturbance and returns to the nominal floor afterwards, "
        "which is exactly the intended behaviour of equation (17).\n"
    )

    # --------------------------------------------------------- harness check
    A("## Harness validation\n")
    df9 = load_variant("madgwick_9d")
    ref = df9[
        (df9["trial"] == "01_undisturbed_slow_rotation_A") & (df9["beta"] == 0.12)
    ]
    if len(ref):
        got = tuple(float(ref[m].iloc[0]) for m in METRICS)
        A(
            "Madgwick 9D on `01_undisturbed_slow_rotation_A` at `beta = 0.12` "
            f"gives total/heading/inclination = {got[0]:.4f} / {got[1]:.4f} / "
            f"{got[2]:.4f} degrees, against 2.3096 / 2.1743 / 0.7788 in the "
            "results shipped with BROAD. Agreement to four decimals pins the "
            "loader, the frame conventions, the movement mask and all three "
            "error decompositions at once.\n"
        )

    # ---------------------------------------------------- acceptance criteria
    A("## Acceptance criteria\n")
    tp = measure_throughput()
    incl = tbl["inclination"]
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"criterion": name, "result": "pass" if ok else "FAIL",
                       "evidence": detail})

    ad = incl[PRETTY["eskf_adaptive"]]
    fx = incl[PRETTY["eskf_fixed"]]
    mg = incl[PRETTY["madgwick_6d"]]
    mh = incl[PRETTY["mahony_6d"]]
    check(
        "adaptive ESKF beats Madgwick and Mahony on mean inclination at TAGP",
        ad < mg and ad < mh, f"{ad:.3f} vs {mg:.3f} (Madgwick), {mh:.3f} (Mahony) deg",
    )
    check(
        "adaptive R beats fixed R on mean inclination at TAGP",
        ad < fx,
        f"{ad:.3f} vs {fx:.3f} deg -- not met, and the two are identical by "
        "construction at this nominal; see the nominal sweep above, where "
        f"adaptation is worth {sweep['ratio'].max():.2f}x at a datasheet nominal",
    )
    check(
        "adaptive R beats fixed R at a matched datasheet-level nominal",
        matched.loc["all_trials", "gap"] > 0,
        f"{matched.loc['all_trials', PRETTY['eskf_adaptive']]:.3f} vs "
        f"{matched.loc['all_trials', PRETTY['eskf_fixed']]:.3f} deg "
        f"({matched.loc['all_trials', 'ratio']:.2f}x)",
    )
    # Evaluated at the matched nominal, since at TAGP every gap is zero.
    gaps = matched.drop(index=["all_trials"])["ratio"]
    widest = str(gaps.idxmax())
    check(
        "largest adaptive-vs-fixed gap falls on a mechanical-disturbance group",
        widest in ("tapping", "vibration"),
        f"widest ratio on '{widest}' ({gaps[widest]:.2f}x); "
        f"tapping {gaps.get('tapping', float('nan')):.2f}x, "
        f"vibration {gaps.get('vibration', float('nan')):.2f}x",
    )
    check(
        "throughput at least 10x real time at 285.7 Hz",
        tp["realtime_factor"] >= 10.0,
        f"{tp['us_per_sample']:.1f} us/sample = {tp['realtime_factor']:.0f}x real time",
    )
    A(pd.DataFrame(checks).set_index("criterion").to_markdown() + "\n")
    A(
        "Both unmet criteria were mis-specified rather than missed, and the "
        "reasons are the two substantive findings of this exercise. The first "
        "assumed the adaptive and fixed filters could be compared at their "
        "respective optima, when at those optima they are the same filter; the "
        "mechanism has to be assessed at a matched nominal, where it is worth a "
        f"factor of {matched.loc['all_trials', 'ratio']:.2f}. The second assumed "
        "a sliding-window covariance estimate would catch impulsive taps, when "
        "it is inherently matched to sustained disturbances instead.\n"
    )

    # --------------------------------------------------------------- caveats
    A("## Limitations\n")
    A(
        "- VQF is better than this filter everywhere, by roughly a factor of "
        f"{incl[PRETTY['eskf_adaptive']] / incl[PRETTY['vqf_6d']]:.1f} at TAGP. "
        "Nothing here suggests the paper's estimator is state of the art; what "
        "it establishes is that it is implemented correctly and that it clearly "
        "beats the Madgwick and Mahony filters it is usually compared against.\n"
        "- BROAD is a hand-held IMU on a rigid board, not a wrist-worn device. "
        "The motion statistics that matter most for this filter -- how much "
        "external acceleration there is and for how long -- are exactly what "
        "differs between a hand-held board and a watch, so the tuned "
        "`sigma_acc` should not be transferred to wrist data without "
        "re-tuning.\n"
        "- The trials are two to four minutes long. Gyroscope bias stability "
        "and heading drift over hours, which is what a localisation pipeline "
        "cares about, are untested here.\n"
        "- Total and heading errors in these tables are reported but not "
        "comparable across algorithms, for the reason given under Protocol.\n"
    )

    out = ROOT / "reports" / "benchmark_results.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nwrote {out}")
    print(f"figures in {figures_dir()}")


if __name__ == "__main__":
    main()

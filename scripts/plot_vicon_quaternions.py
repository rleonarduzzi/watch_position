"""Quaternion components on the short IMU/Vicon prefixes.

Same filters and heading alignment as ``scripts/run_vicon_attitude.py``, drawn
as scalar-first ``w, x, y, z`` instead of Euler angles. One HTML page; the
sequence is chosen from a menu.
"""

from __future__ import annotations

from wattitude.data import vicon
from wattitude.eval.report import figures_dir, plot_quat_dropdown, quat_tracking

# Trial-agnostic optima from reports/benchmark_results.md, the same pair used
# by scripts/run_vicon_attitude.py.  Not retuned on this dataset.
ESKF_PARAMS = {"sigma_acc": 200.0}
VQF_PARAMS = {"tauAcc": 3.0}


def main() -> None:
    root = vicon.short_root()
    if not any(root.glob("v3_*.csv")):
        written = vicon.write_prefix_dataset()
        print(f"wrote {len(written)} prefixes of {vicon.PREFIX_S:.0f} s to {root}")

    plotted = []
    for name in vicon.sequence_names(root):
        trial = vicon.load_trial(name, root=root)
        data = quat_tracking(
            name,
            trial=trial,
            eskf_params=ESKF_PARAMS,
            vqf_params=VQF_PARAMS,
        )
        plotted.append((name, data))
        incl = data["series"]
        print(
            f"{name:6s}  "
            f"ESKF incl {incl['eskf']['errors'].inclination_rmse_deg:6.2f}  "
            f"VQF incl {incl['vqf']['errors'].inclination_rmse_deg:6.2f}"
        )

    out = figures_dir() / "vicon_quat.html"
    plot_quat_dropdown(plotted, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

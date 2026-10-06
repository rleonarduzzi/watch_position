"""Coarse parameter exploration for the adaptive ESKF on a BROAD subset.

Used to choose sensible ranges before committing to the full 39-trial grid
search, which is roughly forty times more expensive.  The subset spans one trial
from each of the motion, speed and disturbance categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wattitude.eval.runner import run_grid
from wattitude.eval.tuning import SELECTION_METRIC

SUBSET = [
    "01_undisturbed_slow_rotation_A",
    "15_undisturbed_fast_translation_A",
    "21_undisturbed_fast_combined",
    "24_disturbed_tapping_A",
    "26_disturbed_phone_vibration_A",
]

# The accelerometer disturbance on the fast trials reaches 50 m/s^2, so the
# useful range of sigma_acc is set by the disturbance scale rather than by the
# instrument noise scale -- hence values far above any datasheet figure.
GRID = {
    "sigma_gyr": [1e-3, 5e-3],
    "sigma_acc": [0.5, 2.0, 10.0, 50.0, 200.0],
    "sigma_bias_gyr": [0.0, 1e-6, 1e-5],
    "window": [100, 400],
}

AXES = ["sigma_gyr", "sigma_acc", "sigma_bias_gyr", "window"]


def main() -> None:
    pd.set_option("display.width", 250)
    for label, extra in (("adaptive", {}), ("fixed", {"adaptive": [False]})):
        grid = {**GRID, **extra}
        df = run_grid(
            "eskf", grid, trials=SUBSET, tag=f"explore_eskf_{label}", force=True, workers=8
        )
        for metric in (SELECTION_METRIC, "total_rmse_deg"):
            means = df.groupby(AXES)[metric].mean().sort_values()
            print(f"\n=== {label}: mean {metric} (deg) over {len(SUBSET)} trials ===")
            print(means.head(8).to_string())
            print(f"  worst: {means.max():.3f}, best: {means.min():.3f}")
        best = df.groupby(AXES)[SELECTION_METRIC].mean().idxmin()
        sel = df.set_index(AXES).loc[best]
        print(f"\n  per-trial at best {dict(zip(AXES, best))}:")
        print(sel.set_index("trial")[[SELECTION_METRIC, "total_rmse_deg"]].round(3).to_string())


if __name__ == "__main__":
    main()

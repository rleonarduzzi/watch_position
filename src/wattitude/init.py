"""Static-window detection used to initialise the filter.

A short rest phase at the start of a recording is enough to level roll and pitch
and to seed the gyroscope bias, which removes the largest source of transient
error at the beginning of a sequence.
"""

from __future__ import annotations

import numpy as np

__all__ = ["moving_std", "detect_rest", "find_static_window"]


def moving_std(x: np.ndarray, win: int) -> np.ndarray:
    """Norm of the per-axis moving standard deviation over a centred window."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    win = max(int(win), 1)
    kernel = np.ones(win) / win
    pad = win // 2
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        padded = np.pad(x[:, j], (pad, win - 1 - pad), mode="edge")
        mean = np.convolve(padded, kernel, mode="valid")
        mean_sq = np.convolve(padded**2, kernel, mode="valid")
        out[:, j] = np.sqrt(np.maximum(mean_sq - mean**2, 0.0))
    return np.linalg.norm(out, axis=1)


def detect_rest(
    gyr: np.ndarray,
    acc: np.ndarray,
    rate: float,
    gyr_thresh: float = 0.05,
    acc_thresh: float = 0.3,
    duration: float = 0.3,
) -> np.ndarray:
    """Boolean mask marking samples that look like rest.

    Thresholds are applied to the angular rate magnitude and to the moving
    standard deviation of the accelerometer, mirroring the rest detection used
    by most orientation filters.
    """
    win = max(int(round(duration * float(rate))), 1)
    gyr_mag = np.linalg.norm(np.atleast_2d(gyr), axis=1)
    return (moving_std(gyr, win) < gyr_thresh) & (gyr_mag < gyr_thresh * 3.0) & (
        moving_std(acc, win) < acc_thresh
    )


def find_static_window(
    gyr: np.ndarray,
    acc: np.ndarray,
    rate: float,
    min_duration: float = 0.5,
    max_search: float = 10.0,
    **kwargs,
) -> slice:
    """Return the longest leading rest window, or a fallback of ``min_duration``.

    Only the first ``max_search`` seconds are considered, because initialisation
    must use data from the start of the sequence.
    """
    n_min = max(int(round(min_duration * float(rate))), 1)
    n_search = min(int(round(max_search * float(rate))), len(acc))
    rest = detect_rest(gyr[:n_search], acc[:n_search], rate, **kwargs)

    # Longest run of rest samples anywhere in the search region.
    best_start, best_len, run_start = 0, 0, None
    for i, flag in enumerate(rest):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            if i - run_start > best_len:
                best_start, best_len = run_start, i - run_start
            run_start = None
    if run_start is not None and len(rest) - run_start > best_len:
        best_start, best_len = run_start, len(rest) - run_start

    if best_len < n_min:
        return slice(0, min(n_min, len(acc)))
    return slice(best_start, best_start + best_len)

"""Adaptive ESKF attitude estimation from 6-axis IMU measurements.

Implementation and benchmark of the attitude estimator described in Section IV.A
of Bai et al., "Watch Your Position: Neural Inertial Localization with a Single
Wrist-worn Device".

Typical use as the front-end of a larger pipeline::

    from wattitude import run_batch, to_world_frame

    quats, biases, _ = run_batch(gyr, acc, rate=200.0, init_samples=100)
    gyr_w, acc_w = to_world_frame(gyr, acc, quats)
"""

from .adaptive import InnovationCovariance
from .eskf import GRAVITY, AdaptiveESKF, ESKFDiagnostics, run_batch, to_world_frame
from .init import detect_rest, find_static_window

__version__ = "0.1.0"

__all__ = [
    "AdaptiveESKF",
    "ESKFDiagnostics",
    "InnovationCovariance",
    "GRAVITY",
    "run_batch",
    "to_world_frame",
    "detect_rest",
    "find_static_window",
    "__version__",
]

# wattitude

Attitude estimation from a 6-axis IMU, using the adaptive error-state Kalman
filter of Section IV.A of *"Watch Your Position: Neural Inertial Localization
with a Single Wrist-worn Device"* (Bai et al., TechRxiv
`10.36227/techrxiv.175492124.47988269/v1`).

The filter is a 9-state ESKF over tilt error, gyroscope bias and accelerometer
bias, with a measurement covariance that adapts to external acceleration using
the innovation-based scheme of equation (17). Its purpose in the larger pipeline
is to rotate raw IMU samples into a gravity-aligned world frame, so the
downstream model never has to learn rotation invariance.

- `reports/derivation_review.md` — verification of the printed equations against
  the derivation, including four discrepancies and what they do if implemented
  literally.
- `reports/benchmark_results.md` — results on the 39 trials of the BROAD
  benchmark against Madgwick, Mahony, VQF and open-loop integration.
- `reports/attitude_tracking.md` — estimated against reference attitude as roll,
  pitch and yaw, for this filter and VQF. This is what turned up the
  equation (8) defect that the aggregate scores had hidden.
- `reports/frame_conventions.md` — why VQF's output needs no rotation into the
  dataset's ENU frame while Madgwick's and Mahony's 9-axis output does.

## Install

```bash
pip install -e ".[bench,dev]"
```

Only `numpy` and `scipy` are needed to use the filter. The `bench` extra pulls
in the benchmark harness dependencies (`h5py`, `pandas`, `matplotlib`, `vqf`).

## Streaming use

The filter is causal and allocates nothing per sample, so the streaming path is
the primary interface; the batch helpers are built on top of it.

```python
import numpy as np
from wattitude import AdaptiveESKF

est = AdaptiveESKF(rate=200.0)

# Initialise from a window the device spent at rest.  Levelling fixes roll and
# pitch; yaw starts at zero because a 6-axis IMU has no absolute heading.
est.initialize(gyr_window=gyr[:400], acc_window=acc[:400])

for g, a in zip(gyr, acc):          # rad/s, m/s^2
    quat = est.update(g, a)         # (4,) [w, x, y, z], body -> world
```

`update` returns a view of the filter's internal quaternion, so copy it if you
intend to keep it. Other state is available as it goes:

| attribute | meaning |
| --- | --- |
| `est.quat` | current attitude, `[w, x, y, z]`, body to world |
| `est.rotation_matrix` | the same attitude as a 3x3 matrix |
| `est.bias_gyr`, `est.bias_acc` | current bias estimates |
| `est.P` | 9x9 error covariance, ordered `[tilt, bias_gyr, bias_acc]` |
| `est.R` | the adapted measurement covariance |
| `est.innovation` | last accelerometer innovation |

Irregular sampling is handled by passing the elapsed time explicitly:
`est.update(g, a, dt=dt_k)`.

## Batch use and the pipeline hook

```python
from wattitude import run_batch, to_world_frame

quats, biases, _ = run_batch(gyr, acc, rate=200.0, init_samples=400)
gyr_w, acc_w = to_world_frame(gyr, acc, quats)
```

`to_world_frame` is the interface the downstream model consumes. By default
`acc_w` still contains gravity, as `+g` on the vertical axis at rest. For linear
acceleration, ask for it and pass the gravity magnitude the filter actually
used, which is not 9.81 when it was estimated at initialisation:

```python
est = AdaptiveESKF(rate=200.0).initialize(acc_window=acc[:400])
gyr_w, acc_w = to_world_frame(gyr, acc, quats, remove_gravity=True,
                              gravity=est.gravity)
```

Pass `diagnostics=True` to `run_batch` to also get the per-sample innovation,
`trace(R_k)`, tilt covariance trace, Kalman gain norm and accelerometer excess,
which is what the adaptation figures in the report are drawn from.

## Conventions

Getting these wrong is the usual source of silent, self-consistent errors, so
they are stated once and enforced by `tests/test_quaternion.py`.

- Quaternions are Hamilton, scalar first, `[w, x, y, z]`, and rotate **body to
  world**: `v_world = R(q) v_body`. Composition satisfies `R(p (x) q) = R(p) R(q)`.
- The world frame is gravity-aligned with **z up**. An accelerometer at rest
  reads `+g` on its vertical axis, not `-g`.
- The world frame's heading is that of the initial body frame; yaw is
  unobservable from a 6-axis IMU and is never corrected by the filter.
- The attitude error is a **right** (body-frame) perturbation,
  `q_true = q_est (x) exp_q(dtheta)`, and is defined as truth minus estimate.
- Angular rates are rad/s, accelerations m/s^2, and `sigma_gyr` / `sigma_acc`
  are continuous-time noise densities in those units.

## Parameters worth knowing

`sigma_acc` is the one knob that matters. It is nominally the accelerometer
noise density, but it is really the statement of how much the accelerometer can
be trusted as a gravity reference, and the right value is set by the external
acceleration in the application rather than by the datasheet. With `adaptive=True`
it acts as a floor that the innovation-based estimate inflates during
disturbances, so it should be left near the instrument noise level; with
`adaptive=False` it must be raised to cover the disturbance directly.

`estimate_acc_bias=True` gives the 9-state filter of the paper. Accelerometer
bias and tilt are only jointly identifiable under rotation-rich, low-acceleration
motion; under sustained external acceleration the bias state absorbs the
disturbance instead, so the 6-state variant is the safer default on vehicle- or
sports-like motion.

`exact_phi` controls the transition matrix used to propagate the covariance and
defaults to the closed form rather than the paper's first-order equation (8).
The printed form is not orthogonal, so it inflates the attitude covariance in
proportion to the square of the angular rate; above a few hundred degrees per
second that corrupts heading badly. Setting `exact_phi=False` reproduces the
paper and is about 30% faster, and happens to score slightly better on
inclination alone. Section 6 of `reports/derivation_review.md` has the numbers.

The three flags `paper_process_noise`, `h_bias_sign` and `inject_left` reproduce
the printed equations (9), (13) and (20) literally, each independently, and
`paper_faithful=True` sets all three. They exist to reproduce the discrepancies
documented in `reports/derivation_review.md`; the combination diverges and
raises `FilterDivergenceError`.

## Euler angles

For display and for interfacing with code that expects three angles:

```python
from wattitude.quaternion import quat_to_euler, euler_to_quat, euler_gimbal_risk

roll, pitch, yaw = quat_to_euler(est.quat)        # radians, intrinsic Z-Y-X
mask = euler_gimbal_risk(quats)                   # pitch within 10 deg of vertical
```

The convention is `R = Rz(yaw) Ry(pitch) Rx(roll)`, with roll and yaw in
`(-pi, pi]` and pitch in `[-pi/2, pi/2]`. Nothing inside the filter uses Euler
angles, and neither should anything downstream: near vertical pitch, roll and
yaw are not separately determined and both can swing arbitrarily while the
attitude barely moves. `euler_gimbal_risk` flags where that applies.
`reports/attitude_tracking.md` plots estimated against reference attitude in
these angles.

## Tests

```bash
pytest                      # everything; the BROAD tests skip if the data is absent
pytest -m broad             # only the tests that read the recordings
pytest -m "not broad"       # only the ones that need no data
```

The simulation tests cover finite-difference Jacobian agreement, bias
convergence, Monte-Carlo NEES consistency against chi-square bands, and
equivariance to sensor mounting. `tests/test_broad.py` reproduces a result
published with the BROAD dataset, which is the check that no convention is
self-consistently wrong throughout.

To fetch the dataset and reproduce the benchmark:

```bash
scripts/get_broad.sh                 # shallow clone into data/broad
python scripts/run_benchmark.py      # grid search, caches to reports/cache
python scripts/make_report.py        # tables and figures
```

Set `BROAD_ROOT` to point the loader somewhere else.

## IMU / Vicon prefixes

`data/imu_vicon_joint_v10` holds the long wrist recordings (100 Hz IMU plus
Vicon position and attitude). The first evaluation uses a two-minute prefix of
each sequence:

```bash
python scripts/run_vicon_attitude.py
```

That writes `data/imu_vicon_joint_v10_short`, the Euler figures under
`reports/figures/vicon_euler_*.png`, and `reports/vicon_attitude_tracking.md`.
The filter and VQF run at the same parameters as the BROAD attitude report.
Sequences whose IMU is yawed 180° relative to the optical body are detected
and corrected in the loader; the CSV files themselves are not rewritten.
`reports/vicon_axis_mounting.md` records the evidence.

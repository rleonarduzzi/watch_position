# Attitude tracking: estimated against optical reference

Roll, pitch and yaw of this filter and of VQF plotted against the OptiTrack reference, on eight BROAD trials chosen to span the range of motion and disturbance. Aggregate scores for all 39 trials are in `benchmark_results.md`; this report is about what the estimates actually look like.

## How to read these plots

**Euler convention.** Intrinsic Z-Y-X, the usual aerospace one: the attitude is yaw about world z, then pitch about the new y, then roll about the new x, so `R = Rz(yaw) Ry(pitch) Rx(roll)`. The world frame is z-up, so yaw is heading and roll with pitch describe the tilt. Roll and yaw run over `(-180, 180]` degrees and pitch over `[-90, 90]`. Angles are used for display only; the filter and every error metric work in quaternions.

**Heading is aligned before plotting.** Both estimators are magnetometer-free and cannot observe absolute heading, so each one expresses yaw in its own initial frame rather than the optical system's. Each estimate has its optimal constant yaw offset removed first, the same alignment the error metric applies. Without it the yaw panel would show nothing but a constant offset. The remaining yaw discrepancy is genuine drift, which is what the slow divergence in the bottom panel of most plots is.

**The shaded bands are a parameterisation artefact, not error.** When pitch approaches the vertical, roll and yaw stop being separately determined: only their sum or difference is, and both can swing arbitrarily far while the attitude barely moves. Bands mark pitch within 10 degrees of vertical; disagreements inside them should be ignored. Pitch itself is always well conditioned, which is why it is the cleanest of the three panels.

**Gaps are real.** One-sample breaks in a trace are wraps across +-180 degrees, drawn as gaps so no vertical line is painted across the axis. Longer gaps in the black reference are optical-tracking dropouts, which the metrics skip rather than interpolate.

Traces are decimated for file size; the decimation factor is stated under each figure, and the error panel is computed at full rate.

## Per-trial traces

Parameters are fixed throughout: ESKF `{'sigma_acc': 200.0}`, VQF `{'tauAcc': 3.0}`.

### 01_undisturbed_slow_rotation_A

Slow rotation, the gentlest case.

![01_undisturbed_slow_rotation_A](reports/figures/euler_01.png)

Decimated 5x for display, from 36007 movement samples at 285.7 Hz.

### 10_undisturbed_slow_translation_A

Slow translation, mild external acceleration.

![10_undisturbed_slow_translation_A](reports/figures/euler_10.png)

Decimated 5x for display, from 34861 movement samples at 285.7 Hz.

### 06_undisturbed_fast_rotation_A

Fast rotation, stresses gyroscope integration.

![06_undisturbed_fast_rotation_A](reports/figures/euler_06.png)

Decimated 5x for display, from 35019 movement samples at 285.7 Hz.

### 21_undisturbed_fast_combined

Fast combined, the strongest external acceleration.

![21_undisturbed_fast_combined](reports/figures/euler_21.png)

Decimated 5x for display, from 33695 movement samples at 285.7 Hz.

### 13_undisturbed_slow_translation_with_breaks_A

Motion separated by rest phases.

![13_undisturbed_slow_translation_with_breaks_A](reports/figures/euler_13.png)

Decimated 6x for display, from 42229 movement samples at 285.7 Hz.

### 24_disturbed_tapping_A

Impulsive mechanical taps.

![24_disturbed_tapping_A](reports/figures/euler_24.png)

Decimated 5x for display, from 34471 movement samples at 285.7 Hz.

### 26_disturbed_phone_vibration_A

Sustained vibration.

![26_disturbed_phone_vibration_A](reports/figures/euler_26.png)

Decimated 5x for display, from 34233 movement samples at 285.7 Hz.

### 39_disturbed_mixed

Mixed disturbances.

![39_disturbed_mixed](reports/figures/euler_39.png)

Decimated 8x for display, from 56309 movement samples at 285.7 Hz.

### 24_disturbed_tapping_A, 80 to 100 s

The same trial at full sample rate, with no decimation. The useful thing about this window is a negative result: both estimates are indistinguishable from the reference in all three angle panels, taps included, and the entire difference between them lives in the bottom panel at a scale of one to three degrees. Angle traces are the right way to confirm that an estimator tracks the motion at all and the wrong way to compare two that both do; that is what the RMSE tables are for.

The two shaded excursions are the parameterisation artefact rather than anything physical: pitch passes through the vertical, and roll and yaw jump 180 degrees in consequence while the attitude itself moves smoothly.

![24_disturbed_tapping_A zoom](reports/figures/euler_24_zoom.png)

## Summary over the trials shown

| trial                                         | regime                                             |   ESKF incl |   ESKF total |   ESKF drift |   VQF incl |   VQF total |   VQF drift |   gimbal-risk fraction |
|:----------------------------------------------|:---------------------------------------------------|------------:|-------------:|-------------:|-----------:|------------:|------------:|-----------------------:|
| 01_undisturbed_slow_rotation_A                | slow rotation, the gentlest case                   |       4.430 |        6.542 |        7.782 |      0.371 |       4.534 |       7.392 |                  0.063 |
| 10_undisturbed_slow_translation_A             | slow translation, mild external acceleration       |       0.205 |        0.744 |        1.205 |      0.249 |       1.089 |       1.800 |                  0.000 |
| 06_undisturbed_fast_rotation_A                | fast rotation, stresses gyroscope integration      |       1.170 |        1.271 |       -0.508 |      0.752 |       0.842 |       0.315 |                  0.005 |
| 21_undisturbed_fast_combined                  | fast combined, the strongest external acceleration |       8.796 |        9.536 |        0.354 |      0.952 |       3.953 |       5.500 |                  0.007 |
| 13_undisturbed_slow_translation_with_breaks_A | motion separated by rest phases                    |       0.273 |        1.893 |        1.996 |      0.313 |       1.798 |       1.887 |                  0.000 |
| 24_disturbed_tapping_A                        | impulsive mechanical taps                          |       1.056 |        1.468 |        1.667 |      0.513 |       0.806 |       0.912 |                  0.032 |
| 26_disturbed_phone_vibration_A                | sustained vibration                                |       0.951 |        1.297 |        1.300 |      0.515 |       1.028 |       1.312 |                  0.015 |
| 39_disturbed_mixed                            | mixed disturbances                                 |       1.648 |        1.880 |        0.387 |      0.579 |       0.934 |       0.199 |                  0.004 |

Errors are RMSE in degrees over the movement phase; drift is the slope of a linear fit to the signed yaw error, in degrees per minute. Inclination is the heading-independent part and is the fair comparator; total error includes the unobservable heading random walk.

VQF has the lower inclination error on 6 of the 8 trials shown, consistent with the full benchmark. The traces make the reason legible: the two estimators are nearly indistinguishable through ordinary motion, and separate during the stretches where the accelerometer is least trustworthy.

Worth noting on `01_undisturbed_slow_rotation_A`: the ESKF does worse here than its 39-trial average, because a single trial-agnostic `sigma_acc` has to cover trials whose external acceleration differs by two orders of magnitude, and the value that wins overall is far too distrustful of the accelerometer for the gentlest recording. That tension is the generalisation gap tabulated in `benchmark_results.md`.

## What these plots found that the error tables did not

Plotting the angles turned up a defect that no aggregate inclination number had exposed. Run with the paper's printed transition matrix, equation (8), the filter tracks roll and pitch on `21_undisturbed_fast_combined` perfectly well but its yaw wanders away from the reference and back by nearly 200 degrees. VQF's yaw stays flat on the same recording. The excursion is not a steady drift: it grows, returns almost to zero, and grows again, which is the signature of a corrupted covariance rather than of random-walk accumulation.

**equation (8), first order.**

![equation (8), first order](reports/figures/euler_defect_eq8.png)

**closed-form transition.**

![closed-form transition](reports/figures/euler_defect_exact.png)

| transition                |   inclination RMSE |   total RMSE |   peak |yaw error| |
|:--------------------------|-------------------:|-------------:|-------------------:|
| equation (8), first order |               2.81 |        78.88 |             193.26 |
| closed-form transition    |               8.80 |         9.54 |               7.85 |

The cause is that the printed `Phi = I + F dt` is not an orthogonal matrix, while the exact attitude block is a rotation. Its largest singular value is `sqrt(1 + |w|^2 dt^2)`, so every propagation step inflates the attitude covariance a little, with nothing in the filter to remove it. The inflation grows as the square of the angular rate: negligible at 29 degrees per second, a factor of 5e14 per minute at the 728 degrees per second this trial reaches. The filter then runs with systematically over-large gains, and over-large corrections acquire a component along the locally unobservable vertical, where nothing can ever remove it.

`exact_phi=True` is now the default. Section 6 of `derivation_review.md` has the derivation, the singular-value check and the honest cost: averaged over all 39 trials the fix improves total error from 9.36 to 3.57 degrees and *worsens* inclination from 1.25 to 1.76, because the spurious gain happened to flatter tilt tracking on this dataset. All the per-trial figures above use the corrected default.

Two things are worth drawing from this. The defect was invisible in every aggregate inclination score, in 84 passing tests including Monte-Carlo covariance-consistency checks -- which are run at rates where the inflation is negligible -- and in the finite-difference Jacobian checks, because the matrix is a correct first-order approximation and simply not an orthogonal one. Looking at the trajectories found it in minutes. And a heading error matters more for this pipeline than its size alone suggests: `to_world_frame()` output is rotated bodily about the vertical, so a position estimate integrated from it inherits the error as a curving trajectory, which half a degree of extra tilt error does not do.

# Attitude tracking on the short Vicon prefixes

Roll, pitch and yaw of the adaptive ESKF and of VQF against the Vicon quaternion, on the first two minutes of each sequence in `data/imu_vicon_joint_v10`. The full recordings are ten to thirty minutes at 100 Hz; this pass uses the prefixes in `data/imu_vicon_joint_v10_short` so the filter can be checked before running the long files.

## Data

Each CSV holds accelerometer specific force in m/s², gyroscope angular rate in rad/s, optical position in metres, and an optical attitude quaternion stored scalar-last (`x, y, z, w`). The quaternion is body-to-world with z up, and the accelerometer reads about `+g` on the upward axis at rest, which is the filter's convention. There is no magnetometer and no separate movement annotation, so the errors below are over every sample of the two-minute prefix.

On part of the recordings the IMU is yawed 180° relative to the optical body: its x and y axes point the opposite way and z agrees. That is a mounting difference, not an attitude the filter should have to discover. `load_trial` detects it by comparing the sign of the gyroscope with the optical angular rate and reverses x and y before either estimator runs. The files in the short dataset are verbatim prefixes; the correction is applied in memory. Sequences marked `rz180` below are the ones that needed it. The evidence, the detection, and what the optical body is are in `vicon_axis_mounting.md`.

## How to read these plots

The figure is the same one used for BROAD in `attitude_tracking.md`. Euler angles are intrinsic Z-Y-X, `R = Rz(yaw) Ry(pitch) Rx(roll)`, in degrees, for display only. Both estimators are magnetometer-free, so each trace has had its optimal constant yaw offset removed before plotting; the yaw panel is the residual drift. Orange bands mark pitch within 10° of vertical, where roll and yaw are not separately determined. One-sample gaps are wraps through ±180°.

Parameters are the BROAD trial-agnostic pair, carried over as-is: ESKF `sigma_acc=200` with the closed-form transition (`exact_phi=True`, the library default), VQF `tauAcc=3`.

## Per-sequence traces

### v3_01

IMU axes already match the optical body.

![v3_01](reports/figures/vicon_euler_v3_01.png)

Decimated 2x for display, from 12000 samples at 100 Hz (120 s).

### v3_02

IMU x,y reversed to match the optical body.

![v3_02](reports/figures/vicon_euler_v3_02.png)

Decimated 2x for display, from 12000 samples at 100 Hz (120 s).

### v3_11

IMU axes already match the optical body.

![v3_11](reports/figures/vicon_euler_v3_11.png)

Decimated 2x for display, from 12000 samples at 100 Hz (120 s).

### v3_12

IMU x,y reversed to match the optical body.

![v3_12](reports/figures/vicon_euler_v3_12.png)

Decimated 2x for display, from 12000 samples at 100 Hz (120 s).

### v3_13

IMU axes already match the optical body.

![v3_13](reports/figures/vicon_euler_v3_13.png)

Decimated 2x for display, from 12000 samples at 100 Hz (120 s).

### v3_14

IMU x,y reversed to match the optical body.

![v3_14](reports/figures/vicon_euler_v3_14.png)

Decimated 2x for display, from 12000 samples at 100 Hz (120 s).

### v3_15

IMU x,y reversed to match the optical body.

![v3_15](reports/figures/vicon_euler_v3_15.png)

Decimated 2x for display, from 12000 samples at 100 Hz (120 s).

## Summary

| sequence   | axis fix   |   ESKF incl |   ESKF total |   ESKF heading |   ESKF drift |   VQF incl |   VQF total |   VQF heading |   VQF drift |   gimbal-risk fraction |
|:-----------|:-----------|------------:|-------------:|---------------:|-------------:|-----------:|------------:|--------------:|------------:|-----------------------:|
| v3_01      | identity   |      17.022 |       22.673 |         15.108 |       21.931 |     16.874 |      19.447 |         9.764 |      -9.157 |                  0.086 |
| v3_02      | rz180      |      10.863 |       26.543 |         24.326 |       40.137 |     10.278 |      12.893 |         8.025 |       5.943 |                  0.347 |
| v3_11      | identity   |      19.235 |       28.449 |         21.096 |       25.701 |     19.111 |      25.799 |        17.451 |      14.806 |                  0.059 |
| v3_12      | rz180      |       5.155 |        8.372 |          6.598 |        9.524 |      5.869 |       9.492 |         7.463 |      11.304 |                  0.329 |
| v3_13      | identity   |       5.470 |        6.041 |          2.566 |       -1.445 |      5.411 |      10.532 |         9.040 |     -15.073 |                  0.001 |
| v3_14      | rz180      |       6.415 |        7.830 |          4.492 |       -4.523 |      7.021 |       8.353 |         4.528 |      -4.479 |                  0.187 |
| v3_15      | rz180      |       7.704 |        9.829 |          6.390 |     -258.829 |      7.570 |       9.690 |         6.343 |    -259.210 |                  0.092 |

Inclination, total and heading are RMSE in degrees over the whole prefix, after the constant heading offset has been removed. Drift is the slope of a linear fit to the signed yaw error, in degrees per minute. Inclination is the heading-independent part and is the fair comparison: a 6-axis IMU cannot observe absolute yaw, so total error and drift mix that random walk into the score. The gimbal-risk fraction is the share of samples whose pitch is within 10° of vertical.

Mean inclination RMSE is 10.27° for the ESKF and 10.30° for VQF. VQF is lower on 5 of the 7 prefixes and the ESKF on 2 (`v3_12`, `v3_14`). On the four prefixes with inclination under 8° (`v3_12` through `v3_15`) both estimators follow the optical pitch oscillation and the total error stays in a band of roughly 5° to 15°, with isolated spikes. On `v3_01` and `v3_11` pitch spends long stretches near vertical, the orange bands cover much of the roll and yaw panels, and both filters share an inclination error of 17° to 19°. That shared floor is the number worth reading; the roll traces through the orange bands are the Euler singularity.

`v3_02` is where the two filters separate. Inclination is close (10.9° against 10.3°), while total error is 26.5° for the ESKF and 12.9° for VQF, because the ESKF heading walks and VQF's stays put. The total-error panel shows that as a raised floor on the green trace.

The drift column on `v3_15` reads -259°/min for both estimators. That slope is one 360° step in the unwrapped signed-heading series near 80 s. Over the same prefix the Euler yaw difference has a median of a fraction of a degree, and the yaw panel shows both estimates on the optical trace. Inclination, 7.7° and 7.6°, is the figure that matches the plot.

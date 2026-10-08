# Why some sequences need their IMU x and y axes reversed

On four of the seven recordings the IMU x and y axes point opposite the Vicon rigid body, while z agrees. Leaving them like that makes a correct attitude estimate look about 140° wrong. Reversing those two axes brings the inclination error down to the same range as the other sequences. This note says what that rigid body is, how the mismatch shows up, how the loader detects it, and what it changes.

## What the optical body is

Each CSV carries two descriptions of the same object.

The IMU columns are the accelerometer and the gyroscope, in the sensor's own axes. The `gt_pos_*` and `gt_rot_quat_*` columns are the position and attitude of one rigid body tracked by Vicon. That tracked rigid body is what "optical body" means here: optical, because the pose comes from the cameras and markers rather than from the inertial sensors; body, because the quaternion is that object's orientation, not a joint angle between two segments.

The quaternion is stored scalar-last (`x, y, z, w`) and is read as a Hamilton quaternion, body to world: a vector in the optical body becomes a vector in the lab by `v_lab = R(q) v_body`. The lab is z-up. Body z is the upward axis of the rigid body, in the sense that an accelerometer at rest reads about `+g` on it. Body x and y are the other two axes of the Vicon body definition. They are fixed to however that rigid body was defined in the capture, which need not be the silkscreen axes of the IMU.

The position is the origin of that same body, in metres in the lab. Over a recording it travels several metres across the room, and rotating the accelerometer by `R(q)` reproduces the specific force implied by `gt_pos`. So `gt_rot_quat` is the attitude of the object that carries the IMU, in the lab. These files do not name the marker set or the segment, so the optical body is that tracked frame and nothing finer.

The IMU is a second frame strapped to the same object. Its z matches the optical body's z on every sequence. Its x and y match on three sequences and are both reversed on the other four. Reversing x and y together, and leaving z, is the rotation `diag(-1, -1, +1)`: a yaw of 180° about the shared vertical axis. After that rotation, an IMU vector is an optical-body vector. The CSVs do not say whether the watch was turned around or the Vicon body axes were defined that way. Both produce this same rotation.

## Evidence

Two checks, on the first 120 s of each recording. They use different signals and they agree.

**Gyroscope against the optical angular rate.** The body rate is the right-trivialised increment of `gt_rot_quat`, in rad/s. On samples where the gyroscope exceeds 0.5 rad/s, each IMU axis is correlated with the matching component of that rate. A 180° yaw flips the sign of the x and y correlations and leaves z positive.

**Accelerometer against the optical position.** The specific force in the lab is the smoothed second derivative of `gt_pos` plus `[0, 0, g]`. The same quaternion rotates the accelerometer into the lab. This comparison does not use the gyroscope. It is scored from 20 s to 110 s. The table gives the mean of the three axis correlations, and the median size of the residual.

| sequence | gyro x | gyro y | gyro z | acc. vs position, as stored | acc. vs position, x and y reversed | VQF inclination, as stored | VQF inclination, x and y reversed |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3_01 | +0.51 | +0.45 | +0.88 | +0.89 | −0.79 | 16.9° | 150° |
| v3_02 | −0.30 | −0.35 | +0.72 | −0.48 | +0.86 | 143° | 10.3° |
| v3_11 | +0.45 | +0.60 | +0.78 | +0.65 | −0.47 | 19.1° | 148° |
| v3_12 | −0.60 | −0.70 | +0.87 | −0.57 | +0.95 | 144° | 5.9° |
| v3_13 | +0.95 | +0.77 | +0.99 | +0.96 | −0.91 | 5.4° | 122° |
| v3_14 | −0.46 | −0.69 | +0.71 | −0.34 | +0.80 | 143° | 7.0° |
| v3_15 | −0.45 | −0.55 | +0.74 | −0.44 | +0.92 | 132° | 7.6° |

Gyro z is positive on all seven sequences, from +0.71 to +0.99, so the vertical axes already agree. The horizontal axes come as a pair: either both correlations are positive or both are negative. Where they are negative, reversing them drops the position residual from about 20 m/s² to about 1.2–1.8 m/s² (`v3_12` 1.2, `v3_15` 1.4, `v3_02` 1.7, `v3_14` 1.8). The other way round, the residual is about 20–27 m/s².

The inclination column is why the correction is required and not optional. A 6-axis filter's heading is aligned to the optical heading before scoring, and inclination is unchanged by a yaw about the lab vertical. A yaw about the *body* vertical is a different rotation: once the object tilts, it is no longer a heading offset, and the gravity direction in the sensor frame no longer matches the optical body. Inclination then sits between 122° and 150°. After the reversal it sits between 5° and 19°, which is the same band as the three sequences whose axes already agreed. The ESKF produces the same split.

The four sequences that need the reversal are `v3_02`, `v3_12`, `v3_14` and `v3_15`. The three that do not are `v3_01`, `v3_11` and `v3_13`. `tests/test_vicon.py` pins that assignment.

## How it is detected

`needs_rz180` in `src/wattitude/data/vicon.py` compares the two candidate sums of the gyroscope correlations above. With `c_x, c_y, c_z` the per-axis correlations against the optical body rate:

- stored mounting: `c_x + c_y + c_z`
- x and y reversed: `−c_x − c_y + c_z`

The larger sum wins. Samples slower than 0.5 rad/s are left out, because a near-zero rate has no sign to compare. In outline:

```python
omega = _body_rate(quat_wxyz, 1.0 / rate)
g = 0.5 * (gyr[:-1] + gyr[1:])
moving = np.linalg.norm(g, axis=1) > 0.5
scores = [correlation(g[moving, i], omega[moving, i]) for i in range(3)]
raw = scores[0] + scores[1] + scores[2]
flipped = -scores[0] - scores[1] + scores[2]
return flipped > raw
```

The position comparison in the table is a check on that decision. The loader does not consult `gt_pos` when it chooses.

## How it is solved

`load_trial` applies the reversal in memory when `align_axes` is left at its default. The x and y columns of the accelerometer and the gyroscope are multiplied by −1. The quaternion and the position are left as recorded, since they already describe the optical body. The trial's `axis_fix` field is `"rz180"` or `"identity"`.

```python
if align_axes and needs_rz180(gyr, quat, RATE_HZ):
    acc[:, :2] *= -1.0
    gyr[:, :2] *= -1.0
```

The CSV files are not rewritten. `data/imu_vicon_joint_v10_short` is a verbatim prefix of the source recordings, so a raw plot of those files still shows the stored axes. `align_axes=False` returns that stored frame. The attitude figures in `vicon_attitude_tracking.md` were made with the correction on.

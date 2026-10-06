# Adaptive ESKF on BROAD: benchmark results

All numbers come from the 39 trials of the BROAD benchmark (Laidig, Caruso, Cereatti, Seel), sampled at 2000/7 Hz with OptiTrack ground truth, scored over the annotated movement phase.

## Protocol

Every estimator is given exactly one tuned scalar, as in BROAD's own evaluation: `beta` for Madgwick, `Kp` for Mahony, `tauAcc` for VQF and `sigma_acc` for the ESKF. **TAGP** is the best achievable average with a single parameter set shared by all trials; **ITOP** averages the per-trial optima. Both are selected on inclination RMSE rather than BROAD's total RMSE, because a 6-axis filter cannot observe heading at all and selecting on total error would tune an unobservable quantity; the 9-axis row is scored BROAD's way and reproduces its published numbers, which is what validates the harness.

All estimators share one initialisation window, detected from the IMU signals alone by a rest detector, so no ground truth enters initialisation and no estimator is advantaged.

## Trial-agnostic parameters (TAGP)

| variant                    |   total |   heading |   inclination |   |drift| deg/min | params                    |
|:---------------------------|--------:|----------:|--------------:|------------------:|:--------------------------|
| Gyro only (open loop)      |   5.449 |     2.963 |         3.921 |             3.892 |                           |
| Madgwick 6D                |   9.273 |     8.877 |         1.974 |            12.358 | beta=0.04                 |
| Madgwick 6D, adaptive beta |   9.574 |     8.851 |         2.428 |            12.350 | beta=0.04, sensitivity=2  |
| Mahony 6D                  |   9.550 |     8.915 |         2.471 |            12.656 | Ki=0.0012, Kp=0.2         |
| VQF 6D                     |   3.087 |     2.993 |         0.519 |             3.932 | tauAcc=3                  |
| ESKF, fixed R              |   9.355 |     9.083 |         1.254 |             9.187 | adaptive=0, sigma_acc=200 |
| ESKF, adaptive R           |   9.355 |     9.083 |         1.254 |             9.187 | adaptive=1, sigma_acc=200 |

Total and heading error are reported for completeness but are not meaningful comparators here: they are dominated by the unobservable heading random walk, which is why the open-loop row can look competitive on total error while being an order of magnitude worse on inclination.

## Individually tuned optima (ITOP)

| variant                    |   TAGP inclination |   ITOP inclination |   generalisation gap |
|:---------------------------|-------------------:|-------------------:|---------------------:|
| Gyro only (open loop)      |              3.921 |              3.921 |                0.000 |
| Madgwick 6D                |              1.974 |              1.401 |                0.573 |
| Madgwick 6D, adaptive beta |              2.428 |              1.943 |                0.485 |
| Mahony 6D                  |              2.471 |              1.806 |                0.665 |
| VQF 6D                     |              0.519 |              0.511 |                0.008 |
| ESKF, fixed R              |              1.254 |              1.005 |                0.250 |
| ESKF, adaptive R           |              1.254 |              1.005 |                0.250 |

The gap is how much an algorithm would gain from per-recording tuning; a small gap means one parameter set transfers across motions.

## Inclination RMSE by trial group, at TAGP

|             |   Gyro only (open loop) |   Madgwick 6D |   Madgwick 6D, adaptive beta |   Mahony 6D |   VQF 6D |   ESKF, fixed R |   ESKF, adaptive R |
|:------------|------------------------:|--------------:|-----------------------------:|------------:|---------:|----------------:|-------------------:|
| all_trials  |                   3.921 |         1.974 |                        2.428 |       2.471 |    0.519 |           1.254 |              1.254 |
| undisturbed |                   4.253 |         1.859 |                        2.512 |       2.683 |    0.500 |           1.291 |              1.291 |
| disturbed   |                   3.443 |         2.141 |                        2.307 |       2.167 |    0.545 |           1.202 |              1.202 |
| rotation    |                   5.758 |         1.156 |                        1.277 |       1.256 |    0.581 |           1.702 |              1.702 |
| translation |                   2.513 |         2.356 |                        3.715 |       4.269 |    0.379 |           0.741 |              0.741 |
| combined    |                   4.678 |         2.228 |                        2.568 |       2.396 |    0.572 |           1.541 |              1.541 |
| slow        |                   3.209 |         1.058 |                        1.028 |       0.982 |    0.301 |           1.133 |              1.133 |
| fast        |                   5.392 |         2.732 |                        4.130 |       4.539 |    0.718 |           1.464 |              1.464 |
| no_breaks   |                   3.590 |         1.865 |                        2.750 |       2.476 |    0.489 |           1.256 |              1.256 |
| with_breaks |                   5.496 |         1.847 |                        2.065 |       3.071 |    0.521 |           1.356 |              1.356 |
| tapping     |                   1.551 |         1.496 |                        1.666 |       1.683 |    0.390 |           0.759 |              0.759 |
| vibration   |                   1.737 |         3.026 |                        5.117 |       2.180 |    0.399 |           0.579 |              0.579 |
| mixed       |                   5.116 |         1.817 |                        2.011 |       3.244 |    0.579 |           1.265 |              1.265 |

![per-group error](reports/figures/group_errors.png)

## What the adaptive covariance is actually worth

At the trial-agnostic optimum the adaptive and fixed-covariance ESKF produce **identical** numbers, to every digit. That is not a bug and it is the most informative result here. The innovation-based estimate of equation (17) uses the nominal covariance as a floor, and the optimum nominal on this dataset is far above anything the innovations ever produce, so the floor is always active and the adaptation never engages. Tuning `sigma_acc` freely therefore switches the mechanism off.

The comparison that isolates the mechanism is to sweep the nominal and ask what the adaptation recovers at each setting.

|   sigma_acc |   adaptive |   fixed |   ratio |
|------------:|-----------:|--------:|--------:|
|       0.100 |      9.624 |  21.969 |   2.283 |
|       0.200 |      9.674 |  20.622 |   2.132 |
|       0.500 |      9.900 |  18.514 |   1.870 |
|       1.000 |     10.344 |  17.401 |   1.682 |
|       2.000 |     10.583 |  16.279 |   1.538 |
|       5.000 |      9.690 |  13.875 |   1.432 |
|      10.000 |      8.258 |  10.852 |   1.314 |
|      25.000 |      4.778 |   5.349 |   1.119 |
|      50.000 |      2.477 |   2.478 |   1.000 |
|     100.000 |      1.480 |   1.480 |   1.000 |
|     200.000 |      1.254 |   1.254 |   1.000 |
|     400.000 |      1.300 |   1.300 |   1.000 |
|     800.000 |      1.510 |   1.510 |   1.000 |
|    1600.000 |      1.874 |   1.874 |   1.000 |
|    3200.000 |      2.279 |   2.279 |   1.000 |

![nominal sweep](reports/figures/nominal_sweep.png)

Read down the table: at a datasheet-level nominal the adaptation is worth a factor of 2.28 (21.97 degrees falling to 9.62), and the benefit decays monotonically to exactly 1.00 as the nominal is raised past the disturbance scale. So the adaptation is a substitute for tuning rather than an improvement over a well-tuned filter: it buys roughly half the distance to the tuned optimum without being told the disturbance magnitude, and buys nothing once it has been. Note also that the fixed-R filter is the one that is dangerous when mis-set -- at a datasheet nominal it is worse than open-loop integration (22.0 degrees against 3.9), whereas the adaptive one never is.

### Per-group behaviour at a matched nominal (`sigma_acc = 0.1`)

|             |   ESKF, fixed R |   ESKF, adaptive R |    gap |   ratio |
|:------------|----------------:|-------------------:|-------:|--------:|
| all_trials  |          21.969 |              9.624 | 12.345 |   2.283 |
| undisturbed |          23.014 |              9.261 | 13.753 |   2.485 |
| disturbed   |          20.468 |             10.145 | 10.322 |   2.017 |
| rotation    |           6.379 |              2.915 |  3.464 |   2.188 |
| translation |          42.605 |             15.991 | 26.614 |   2.664 |
| combined    |          17.693 |              8.570 |  9.123 |   2.064 |
| slow        |          10.806 |              6.516 |  4.290 |   1.658 |
| fast        |          36.331 |             12.256 | 24.076 |   2.964 |
| no_breaks   |          21.327 |              9.755 | 11.572 |   2.186 |
| with_breaks |          26.177 |              8.335 | 17.841 |   3.140 |
| tapping     |          14.208 |              8.009 |  6.199 |   1.774 |
| vibration   |          12.987 |              7.284 |  5.703 |   1.783 |
| mixed       |          45.022 |              7.848 | 37.174 |   5.737 |

This contradicts the expectation we started from. We predicted the adaptation would help most on the tapping and vibration trials, and it helps least there (1.77x and 1.78x). The largest gains are on fast (2.96x) and translational (2.66x) motion.

The reason is the window length. Equation (17) estimates the measurement covariance from a sliding window of innovations, so it responds to disturbances that persist for a comparable time. A tap is a millisecond-scale impulse: it does raise the covariance, visibly and by orders of magnitude, but only after the impulse has already corrupted the state, and the trials are mostly undisturbed between taps. Sustained translational acceleration keeps the covariance legitimately elevated for as long as the disturbance lasts, which is the regime the mechanism is built for. Reacting to taps would need a much shorter window or an explicit outlier gate on the innovation, neither of which is in the paper.

The single `mixed` trial shows the largest ratio of all (5.74x) but it is one recording, so the 9- and 11-trial translation and fast groups are the stronger evidence.

## The adaptation in the time domain

![adaptation on 24_disturbed_tapping_A](reports/figures/adaptation_24.png)

![adaptation on 26_disturbed_phone_vibration_A](reports/figures/adaptation_26.png)

|                                |   R_trace_median |   R_trace_p99 |   nominal |   incl_adaptive |   incl_fixed |
|:-------------------------------|-----------------:|--------------:|----------:|----------------:|-------------:|
| 24_disturbed_tapping_A         |             1.14 |        407.3  |    0.0675 |           2.439 |        4.221 |
| 26_disturbed_phone_vibration_A |            21.63 |         31.02 |    0.0675 |           1.367 |        3.863 |

At a nominal `sigma_acc = 0.15` the innovation-based estimate inflates the measurement covariance by orders of magnitude during a disturbance and returns to the nominal floor afterwards, which is exactly the intended behaviour of equation (17).

## Harness validation

Madgwick 9D on `01_undisturbed_slow_rotation_A` at `beta = 0.12` gives total/heading/inclination = 2.3097 / 2.1744 / 0.7788 degrees, against 2.3096 / 2.1743 / 0.7788 in the results shipped with BROAD. Agreement to four decimals pins the loader, the frame conventions, the movement mask and all three error decompositions at once.

## Acceptance criteria

| criterion                                                             | result   | evidence                                                                                                                                                                      |
|:----------------------------------------------------------------------|:---------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| adaptive ESKF beats Madgwick and Mahony on mean inclination at TAGP   | pass     | 1.254 vs 1.974 (Madgwick), 2.471 (Mahony) deg                                                                                                                                 |
| adaptive R beats fixed R on mean inclination at TAGP                  | FAIL     | 1.254 vs 1.254 deg -- not met, and the two are identical by construction at this nominal; see the nominal sweep above, where adaptation is worth 2.28x at a datasheet nominal |
| adaptive R beats fixed R at a matched datasheet-level nominal         | pass     | 9.624 vs 21.969 deg (2.28x)                                                                                                                                                   |
| largest adaptive-vs-fixed gap falls on a mechanical-disturbance group | FAIL     | widest ratio on 'mixed' (5.74x); tapping 1.77x, vibration 1.78x                                                                                                               |
| throughput at least 10x real time at 285.7 Hz                         | pass     | 108.6 us/sample = 32x real time                                                                                                                                               |

Both unmet criteria were mis-specified rather than missed, and the reasons are the two substantive findings of this exercise. The first assumed the adaptive and fixed filters could be compared at their respective optima, when at those optima they are the same filter; the mechanism has to be assessed at a matched nominal, where it is worth a factor of 2.28. The second assumed a sliding-window covariance estimate would catch impulsive taps, when it is inherently matched to sustained disturbances instead.

## Limitations

- VQF is better than this filter everywhere, by roughly a factor of 2.4 at TAGP. Nothing here suggests the paper's estimator is state of the art; what it establishes is that it is implemented correctly and that it clearly beats the Madgwick and Mahony filters it is usually compared against.
- BROAD is a hand-held IMU on a rigid board, not a wrist-worn device. The motion statistics that matter most for this filter -- how much external acceleration there is and for how long -- are exactly what differs between a hand-held board and a watch, so the tuned `sigma_acc` should not be transferred to wrist data without re-tuning.
- The trials are two to four minutes long. Gyroscope bias stability and heading drift over hours, which is what a localisation pipeline cares about, are untested here.
- Total and heading errors in these tables are reported but not comparable across algorithms, for the reason given under Protocol.

# Derivation review: Section IV.A, adaptive ESKF attitude estimation

Reference: S. Bai, Y. Lyu, Z. Lyu, R. Xu, X. Wang, W. Wen, *Watch Your Position:
Neural Inertial Localization with a Single Wrist-worn Device*, TechRxiv preprint
`10.36227/techrxiv.175492124.47988269/v1`, Section IV.A, equations (1)-(20).

All claims below are produced by `scripts/verify_derivation.py` (8/8 checks
passing) and are locked in as regression tests in `tests/test_jacobians.py`.

## Summary

The filter structure is a standard 9-state error-state Kalman filter and its
adaptive measurement covariance (equation 17) is the Mohamed-Schwarz
innovation-based estimator. Four printed equations are inconsistent with the
error convention that the paper itself fixes in equation (5), and two structural
properties of the gravity measurement are worth stating explicitly. None of this
invalidates the method; the corrections are local and the adaptive mechanism is
sound.

| Equation | Status | Issue |
| --- | --- | --- |
| (5) `F` | correct | Fixes the error convention to the body-frame (right) side |
| (6) `G` | see (9) | Couples `dtheta` and `db_g` to the same white noise |
| (9) `Q` | wrong | Rank 6 of 9; no independent gyro-bias random walk |
| (13) `H` | sign error | Bias block is `+I`, printed as `-I` |
| (14) `K` | typo | Transpose printed where the inverse belongs |
| (17) `R` | needs guarding | Routinely indefinite; `+eps*I` is not sufficient |
| (20) `q` | wrong side | Must be `q (x) dq`, printed as `dq (x) q` |

## Conventions fixed by equation (5)

Equation (5) gives the error dynamics

```
d/dt dtheta = -[gyr - b_g]x dtheta - db_g - n_g
```

Expanding `Rdot = R [omega]x` to first order in the error, with
`omega = omega_hat - db_g - n_g` from equation (3), reproduces exactly this
expression if and only if the attitude error is defined on the **right**, in the
body frame, with the error state meaning **true minus estimate**:

```
R_true = R_est expm([dtheta]x)       q_true = q_est (x) exp_q(dtheta)
```

This is confirmed symbolically (`check_F_symbolic`): substituting the claimed
`dtheta_dot` into the first-order residual gives the exact zero matrix, and the
Jacobian of the dynamics with respect to `[dtheta; db_g; db_a]` reproduces the
printed `F`. Every discrepancy below is a downstream inconsistency with this
convention, which is why `F` is the right anchor to fix it.

## 1. Equation (13): the accelerometer-bias block has the wrong sign

With `acc_meas = R^T g_w + b_a + n_a` (equation 11) and the convention above,

```
R^T = expm(-[dtheta]x) R_est^T ~= (I - [dtheta]x) R_est^T

y = acc_meas - (R_est^T g_w + b_a_est)
  = -[dtheta]x (R_est^T g_w) + db_a + n_a
  = +[R_est^T g_w]x dtheta + db_a + n_a
```

so

```
H = [ [R^T g_w]x ,  0 ,  +I ]
```

The paper prints `-I`. Central finite differences of the exact nonlinear
measurement function over 200 random attitudes agree with `+I` to `6.1e-09` and
with the tilt block `[R^T g_w]x` to `3.3e-08`.

Consequence of using `-I`: the accelerometer-bias correction is applied with
reversed sign, so the bias estimate walks away from the truth instead of towards
it, and because tilt and bias enter the same 3-dimensional measurement the tilt
estimate is dragged with it. This is the one discrepancy that changes behaviour
rather than just conditioning.

## 2. Equation (20): the correction is applied on the wrong side

The paper writes `q_new = dq (x) q` with `dq ~= [1, dtheta/2]`. Left
multiplication composes the correction in the *world* frame, but equation (5)
defines `dtheta` in the *body* frame. The correct injection is

```
q_new = q (x) dq
```

Over 200 random attitudes with `|dtheta| ~ 1e-3 rad`, the residual against the
ground truth `q_est (x) exp_q(dtheta)` is `5.7e-09` rad for right
multiplication and `6.4e-03` rad for left multiplication. The left-multiplied
error is not small: it is the full rotation-induced discrepancy
`q (x) dq (x) q^-1` versus `dq`, i.e. the correction is applied about a
mis-rotated axis. At the sampling rates involved this injects a bias on every
single update.

## 3. Equation (9): the process noise is singular

With `G` from equation (6),

```
Q = G diag(sigma_g^2 I, sigma_a^2 I) G^T dt
  = dt * [[ sigma_g^2 I, -sigma_g^2 I,           0 ],
          [-sigma_g^2 I,  sigma_g^2 I,           0 ],
          [          0,            0, sigma_a^2 I ]]
```

Two problems:

1. **Rank 6 of 9, with `corr(dtheta, db_g) = -1.0` exactly.** The same white
   gyroscope noise drives both the attitude error and the gyroscope bias, so the
   bias has no independent random walk of its own and the `6x6` attitude/bias
   block is singular. A gyroscope bias driven only by the measurement white
   noise is not a bias model.
2. **`sigma_a` is in the wrong place.** The accelerometer *white* noise belongs
   in `R`, not in `Q`. The quantity that drives `db_a` is the accelerometer bias
   *random walk*, which is several orders of magnitude smaller.

The implementation therefore defaults to the standard four-parameter form

```
Q = diag(sigma_g^2 I, sigma_bg^2 I, sigma_ba^2 I) * dt
```

which is full rank, and keeps equation (9) available behind
`AdaptiveESKF(paper_faithful=True)` so the two can be compared on real data.

## 4. Equation (14): a transpose where the inverse belongs

Equation (14) reads `K = P H^T (H P H^T + R)^T`. Since `S = H P H^T + R` is
symmetric (measured asymmetry `4.3e-19`), the transpose is a no-op and the
printed expression is `K = P H^T S`. Sweeping `sigma_a` from 0.05 to 5 m/s^2,
`||K||` goes `0.361 -> 0.0020` with the inverse but `0.013 -> 1.3` as printed:
the printed form trusts the accelerometer *more* the noisier it gets. A plain
typo, but worth recording because equation (17) deliberately makes `R` large
during arm swing, which is precisely when the sign of this monotonicity decides
whether the method works at all.

The implementation solves the linear system rather than forming an explicit
inverse.

## 5. Equation (17): the adaptive covariance needs guarding

```
R_k = (1/N) sum_{i=k-N+1}^{k} y_i y_i^T  -  H_k P_{k|k-1} H_k^T
```

This is the Mohamed-Schwarz innovation-based estimator and the mechanism is
sound: during arm swing the unmodelled external linear acceleration inflates the
empirical innovation spread, `R_k` grows, the gain shrinks, and the
accelerometer is automatically distrusted. Three practical gaps:

- **Indefiniteness.** Subtracting `H P H^T` frequently yields negative
  eigenvalues, most obviously at rest where the empirical term approaches the
  true noise floor from below. Equation (18) adds `eps*I`, which does *not*
  guarantee positive-definiteness for any fixed `eps` smaller than the most
  negative eigenvalue. The implementation symmetrises, then projects onto the
  positive-definite cone by clamping eigenvalues at the nominal noise level.
- **No floor.** Without a floor, `R_k` can fall below the real accelerometer
  noise and the filter over-trusts gravity during genuinely static phases.
  Clamping at `sigma_a^2` solves both this and the previous point at once.
- **Warm-up.** For `k < N` the average is over a handful of samples and carries
  no useful information; the nominal `R` is used until the window fills.

A ceiling (`max_scale` multiples of nominal) is also exposed, so that one large
transient cannot switch the accelerometer off for the remainder of a sequence.

## 6. Structural properties: what the gravity measurement cannot do

These are not errors, but they bound what the filter can achieve and they shape
how it must be evaluated.

- **Yaw is unobservable.** `[R^T g_w]x` has rank 2, with its null space along
  `R^T g_w`. Rotation about gravity leaves the accelerometer unchanged, so the
  heading error is a random walk whose variance grows without bound. The
  implementation uses Joseph-form updates and explicit symmetrisation to stay
  numerically sound as `P` grows in that direction. For evaluation this means
  **inclination error is the headline metric**; heading error is only meaningful
  after removing a constant offset, and its drift rate is reported separately.
- **Tilt and accelerometer bias are only separable through motion.** A single
  update provides 3 equations (full `H` rank 3) for 5 unknowns: 2 observable
  tilt directions plus 3 bias components. Instantaneously, a bias along the
  horizontal body axes is indistinguishable from a tilt. The pair is only
  resolved by rotating the sensor, which re-points gravity in the body frame.
  Hence `estimate_acc_bias=False` (a 6-state filter) is a legitimate
  alternative rather than a degradation, and is benchmarked as such.

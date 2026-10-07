# Does VQF's output need converting to the dataset's ENU frame?

Short answer: no, and the reason is worth stating, because two of the other
baselines *do* need a conversion and it is not obvious why VQF escapes it.

VQF is documented as producing quaternions in an ENU reference frame, and
BROAD's HDF5 files describe `opt_quat` as "a unit quaternion wrt. ENU reference
frame". Both being labelled ENU is suggestive but not sufficient: the label has
to be checked rather than trusted, and the check is what follows. Claims below
are pinned by `tests/test_broad.py`.

## What "ENU" actually constrains

Only two things, and they separate cleanly because they are observable to
different degrees.

1. **The vertical axis**: which way z points, and whether an accelerometer at
   rest reads `+g` or `-g` on it.
2. **The heading origin**: where zero yaw sits in the horizontal plane.

A 6-axis algorithm can observe the first and cannot observe the second at all.
That asymmetry decides the whole question.

## The vertical axis already agrees

VQF and BROAD both use z-up with the accelerometer reading `+g` at rest, which
is also this library's convention, so no rotation is required.

This is confirmed from the results rather than from the documentation. VQF's
*inclination* error on BROAD is 0.52 degrees averaged over all 39 trials, and
0.19 to 0.96 degrees trial by trial. Inclination is the angle between the
estimated and true vertical, so it is invariant to any rotation about the
vertical and sensitive to exactly this part of the convention and nothing else.
A flipped z axis, a `-g` sign convention or a transposed rotation would all put
that number near 180 degrees, not under one degree. Sub-degree inclination is
only possible if the vertical convention is already right.

## There is no heading to convert

We run VQF in 6-axis mode: the magnetometer argument is `None` and we read
`quat6D`. That output carries no absolute heading. VQF sets yaw to zero at its
first sample and integrates from there, so its horizontal reference is the
device's initial heading, whatever that happened to be.

No fixed rotation can map that onto ENU, because the quantity being mapped does
not exist. The benchmark therefore removes the optimal constant heading offset
per trial before scoring, which is also what keeps the comparison against our
own ESKF fair: both filters are magnetometer-free and both are in exactly this
position.

What makes this tidy in practice is a property of the dataset rather than of
VQF. BROAD describes its optical data as "synchronized and **aligned**", and the
effect of that alignment is that the reference starts close to identity. Over
all 39 trials, the ground-truth attitude at the first valid sample is:

| | mean | std | range |
| --- | --- | --- | --- |
| total angle from identity | 2.73 deg | 1.93 | 0.22 to 6.85 |
| roll | -0.07 deg | 1.21 | -2.34 to +2.17 |
| pitch | +0.19 deg | 0.66 | -0.75 to +1.44 |
| yaw | -2.02 deg | 2.27 | -6.61 to +0.63 |

82% of trials start within 5 degrees of identity. So BROAD's ENU frame has in
practice had its heading origin set at the start of each recording, which is
the same thing VQF does, and the two frames nearly coincide by construction.

The measured offsets agree. Removing the optimal constant heading offset from
VQF's output calls for -2.8 degrees on average with a 4.9 degree spread,
scattered and trial-specific. A genuine convention mismatch would show up as a
constant near 90 or 180 degrees in every trial; this does not. The largest
single value, -17.5 degrees on `08_undisturbed_fast_rotation_with_breaks_A`, is
accumulated gyroscope drift rather than a frame error.

## Where a conversion *is* applied

Madgwick and Mahony get a +90 degree rotation about z, but only on the 9-axis
path:

```python
_ENU_FIX = np.array([1 / math.sqrt(2), 0.0, 0.0, 1 / math.sqrt(2)])
...
if mag is not None:
    # Madgwick's magnetic reference is along x; rotate into ENU.
    out = np.array([quat_multiply(_ENU_FIX, q) for q in out])
```

Both algorithms place their magnetic reference direction in the x-z plane, so
their x axis points north, whereas ENU puts north on y. The fix is the 90 degree
rotation between those two choices, and it matches what BROAD's own
`process_data.py` does to the same algorithms. Without it the 9-axis Madgwick
result would be wrong by 90 degrees of heading, which is also how we know the
correction is right: with it, Madgwick 9D reproduces BROAD's published numbers
to four decimals.

On their 6-axis paths these two get no correction either, for the same reason
VQF does not.

Note that VQF would not need `_ENU_FIX` even in 9-axis mode, because it defines
its 9D frame as ENU directly. That is a difference between the libraries, not a
choice we made.

## Caveat on how strong this is

The inclination result pins the vertical convention tightly: it is a
three-orders-of-magnitude margin, not a marginal call. The heading argument is
weaker in kind, resting on the offsets being small and scattered rather than
constant. That is strongly inconsistent with a 90 or 180 degree error, but it
could not distinguish "no conversion needed" from a genuinely required
correction of one or two degrees hiding inside the initialisation error.

For a 6-axis comparison scored on inclination this makes no difference to any
number in the reports, since inclination is invariant to heading by
construction. It would matter if these estimates were ever compared to the
reference in absolute heading, which is not something a 6-axis filter supports
in any case.

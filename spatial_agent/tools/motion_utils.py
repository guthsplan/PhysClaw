"""Motion fitting tool: a stable numerical-differentiation layer over noisy
position samples.

Does NOT do physics reasoning — it does not choose the measurements, the
time window, or the motion model. It only owns the numerically fragile part
(velocity/acceleration from a handful of noisy samples), where hand-written
`np.diff(np.diff(...))` tends to blow up.
"""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from spatial_agent.tools.base import CPUTool


class TrajectoryFit:
    """A degree-1 or degree-2 polynomial fit through (t, position) samples.

    Position may be scalar or a vector (2D pixel, 3D world); each dimension
    is fit independently. A degree-2 fit is a constant-acceleration model
    over the whole sample window — that is why ``acceleration()`` takes no
    ``t``: it is one number for the window, not a time-local value.
    """

    def __init__(self, times: np.ndarray, values: np.ndarray, degree: int):
        self._scalar = values.ndim == 1
        if self._scalar:
            values = values[:, None]
        self._dim = values.shape[1]
        self._degree = degree
        self.t_range = (float(times.min()), float(times.max()))
        self._pos = [np.poly1d(np.polyfit(times, values[:, d], degree)) for d in range(self._dim)]
        self._vel = [p.deriv(1) for p in self._pos]
        self._acc = [p.deriv(2) for p in self._pos]
        fitted = np.stack([p(times) for p in self._pos], axis=1)
        #: RMS distance between samples and the fitted curve, in the samples'
        #: units. Large relative to the motion means the model does not fit —
        #: e.g. the window spans a change in motion.
        self.residual = float(np.sqrt(np.mean(np.sum((fitted - values) ** 2, axis=1))))

    def _eval(self, polys, t: float):
        lo, hi = self.t_range
        if not (lo - 1e-6 <= t <= hi + 1e-6):
            print(f"[Motion] t={t} is outside the sampled range [{lo}, {hi}] — extrapolating; treat as unreliable.")
        vals = np.array([float(p(t)) for p in polys])
        return float(vals[0]) if self._scalar else vals

    def position(self, t: float) -> Union[float, np.ndarray]:
        """Fitted (smoothed) position at time ``t``."""
        return self._eval(self._pos, t)

    def velocity(self, t: float) -> Union[float, np.ndarray]:
        """Velocity at time ``t`` — a vector if positions were vectors."""
        return self._eval(self._vel, t)

    def speed(self, t: float) -> float:
        """``norm(velocity(t))`` — scalar."""
        return float(np.linalg.norm(np.atleast_1d(self._eval(self._vel, t))))

    def acceleration(self) -> Union[float, np.ndarray]:
        """Constant acceleration over the window (degree=2). Exactly 0 for degree=1."""
        vals = np.array([float(p.coeffs[0]) if p.order >= 0 and len(p.coeffs) else 0.0 for p in self._acc])
        return float(vals[0]) if self._scalar else vals

    def __repr__(self) -> str:
        return (
            f"TrajectoryFit(degree={self._degree}, t={self.t_range}, dim={self._dim}, "
            f"residual={self.residual:.4g})"
        )


class MotionUtils(CPUTool):
    """Numerically stable velocity/acceleration from noisy position samples."""

    TOOL_ABLATION_PREFIX = "tool_motion"

    TOOL_PROMPT_DESCRIPTION = """
### tools.Motion — Trajectory Fitting (CPU)

`tools.Motion.fit(samples, degree=2)` fits a polynomial through noisy
`(t, position)` samples and returns a `TrajectoryFit`. Use it instead of
hand-written finite differences (`np.diff`), which are unstable with few or
noisy samples. It does NOT choose what to measure or over which window —
that is still your decision.

| `TrajectoryFit` | Returns |
|---|---|
| `position(t)` | smoothed position at `t` (same shape as your samples) |
| `velocity(t)` | velocity at `t` — a **vector** if positions were vectors |
| `speed(t)` | scalar `norm(velocity(t))` |
| `acceleration()` | **one constant** for the whole window (`degree=2` model); exactly `0` for `degree=1` |
| `.residual` | RMS fit error in your units — if it is large relative to the motion, the window spans a change in motion: narrow it |
| `.t_range` | `(t_min, t_max)` of the samples |

`degree=1` (constant velocity) needs >= 2 samples; `degree=2` (constant
acceleration) needs >= 3, and more is better. Only 1 and 2 are accepted.

A fit is only as good as its samples: for acceleration, prefer many smooth
samples (pixel-space tracks scaled by a known size) over a few noisy 3D
centroids, and check `.residual` — if it is comparable to the motion over the
window, or two windows give accelerations of opposite sign, report that the
measurement failed rather than averaging.

```python
fit = tools.Motion.fit(track.xy_with_time(), degree=2)      # from tools.TAP.track (pixel space)
v = fit.velocity(3.0);  a = fit.acceleration();  print(fit.residual)

samples = []                                                 # from 3D centroids
for fr in window:
    xyz = recon.centroid(fr, track.mask(fr))
    if xyz is not None:
        samples.append((tools.Video.time_of(fr), xyz))
fit = tools.Motion.fit(samples, degree=2)
speed = fit.speed(3.0)      # m/s
```
"""

    @staticmethod
    def fit(samples: Sequence[Tuple[float, object]], degree: int = 2) -> TrajectoryFit:
        """Fit a degree-1 or degree-2 polynomial through `(t, position)` samples."""
        if degree not in (1, 2):
            raise ValueError(f"`degree` must be 1 (constant velocity) or 2 (constant acceleration), got {degree}.")
        if not isinstance(samples, (list, tuple)) or len(samples) == 0:
            raise ValueError(
                "`samples` must be a non-empty list of (t, position) pairs, "
                "e.g. track.xy_with_time() or [(t0, p0), (t1, p1), ...]."
            )
        times, values = zip(*samples)
        times = np.asarray(times, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim > 2:
            raise ValueError(f"Each position must be a scalar or a 1-D vector, got shape {values.shape[1:]}.")
        n = len(times)
        if n < degree + 1:
            raise ValueError(
                f"fit needs at least {degree + 1} samples for degree={degree} (got {n}). "
                f"Use more frames, or degree=1 (then acceleration() is 0 by definition)."
            )
        if len(set(times.tolist())) < 2:
            raise ValueError("`samples` must span more than one distinct timestamp.")
        return TrajectoryFit(times, values, degree)

    def __repr__(self) -> str:
        return "MotionUtils(methods: fit)"

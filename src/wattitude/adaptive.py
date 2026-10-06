"""Innovation-based adaptive measurement covariance (paper equations 15-18).

The estimator keeps a sliding window of the last ``N`` a-priori innovations and
forms

    R_k = (1/N) * sum y_i y_i^T  -  H_k P_{k|k-1} H_k^T

Subtracting the predicted innovation covariance leaves the part of the observed
spread that the filter cannot explain, which during arm swing is dominated by
external linear acceleration.  The raw expression is frequently indefinite, so
the result is regularised before use.
"""

from __future__ import annotations

from collections import deque

import numpy as np

__all__ = ["InnovationCovariance"]


class InnovationCovariance:
    """Sliding-window innovation covariance estimator.

    Parameters
    ----------
    dim
        Measurement dimension (3 for the accelerometer update).
    window
        Number of innovations averaged, ``N`` in equation (17).
    nominal
        Nominal measurement covariance, used during warm-up and as the floor.
    eps
        Diagonal loading from equation (18).
    psd_mode
        ``"eps"`` reproduces the paper exactly: add ``eps * I`` and nothing else.
        ``"clip"`` additionally projects onto the positive-definite cone by
        clamping the eigenvalues at the nominal noise level, which is what makes
        the estimator usable in practice.
    max_scale
        Upper bound on the adapted covariance expressed as a multiple of
        ``nominal``.  Prevents a single large transient from switching the
        accelerometer off for the rest of the sequence.
    """

    def __init__(
        self,
        dim: int = 3,
        window: int = 100,
        nominal: np.ndarray | float = 1.0,
        eps: float = 1e-6,
        psd_mode: str = "clip",
        max_scale: float = 1e6,
    ) -> None:
        if psd_mode not in ("eps", "clip"):
            raise ValueError(f"unknown psd_mode {psd_mode!r}")
        self.dim = int(dim)
        self.window = int(window)
        self.eps = float(eps)
        self.psd_mode = psd_mode
        self.max_scale = float(max_scale)

        nominal = np.asarray(nominal, dtype=float)
        if nominal.ndim == 0:
            nominal = float(nominal) * np.eye(self.dim)
        self.nominal = nominal
        self._floor = float(np.min(np.linalg.eigvalsh(self.nominal)))
        self._ceiling = self.max_scale * float(np.max(np.linalg.eigvalsh(self.nominal)))

        self._buffer: deque[np.ndarray] = deque(maxlen=self.window)
        self._outer_sum = np.zeros((self.dim, self.dim))
        self._eye = np.eye(self.dim)
        self._floor_matrix = self._floor * self._eye

    def reset(self) -> None:
        self._buffer.clear()
        self._outer_sum = np.zeros((self.dim, self.dim))

    @property
    def ready(self) -> bool:
        """True once the window is full and the estimate can be trusted."""
        return len(self._buffer) >= self.window

    def push(self, y: np.ndarray) -> None:
        """Append an a-priori innovation to the window."""
        y = np.asarray(y, dtype=float).reshape(self.dim)
        if len(self._buffer) == self.window:
            old = self._buffer[0]
            self._outer_sum -= old[:, None] * old[None, :]
        self._buffer.append(y.copy())
        self._outer_sum += y[:, None] * y[None, :]

    def estimate(self, hph: np.ndarray) -> np.ndarray:
        """Return ``R_k`` given the predicted innovation covariance ``H P H^T``.

        Falls back to the nominal covariance until the window has filled, since
        an average over a handful of samples carries no useful information.
        """
        if not self.ready:
            return self.nominal.copy()

        R = self._outer_sum / len(self._buffer) - hph
        R += R.T
        R *= 0.5
        R[np.diag_indices(self.dim)] += self.eps

        if self.psd_mode == "clip":
            R = self._project(R)
        return R

    def _project(self, R: np.ndarray) -> np.ndarray:
        """Clamp the eigenvalues of ``R`` into ``[floor, ceiling]``.

        Two cheap Cholesky probes cover the two dominant regimes -- during rest
        every eigenvalue sits below the floor, during vigorous motion every one
        sits above it -- so the eigendecomposition is only needed for the mixed
        case.  At a few hundred Hz over hours of data this matters: NumPy's
        ``eigh`` costs roughly ten times a Cholesky attempt on a 3x3.
        """
        try:
            np.linalg.cholesky(R - self._floor_matrix)
        except np.linalg.LinAlgError:
            pass
        else:
            # All eigenvalues already at or above the floor.
            if np.trace(R) <= self.dim * self._ceiling:
                return R
            return self._eigh_clip(R)

        try:
            np.linalg.cholesky(self._floor_matrix - R)
        except np.linalg.LinAlgError:
            return self._eigh_clip(R)
        # All eigenvalues at or below the floor.
        return self._floor_matrix.copy()

    def _eigh_clip(self, R: np.ndarray) -> np.ndarray:
        evals, evecs = np.linalg.eigh(R)
        np.clip(evals, self._floor, self._ceiling, out=evals)
        out = (evecs * evals) @ evecs.T
        out += out.T
        out *= 0.5
        return out

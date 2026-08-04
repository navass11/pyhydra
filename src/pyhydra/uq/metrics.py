"""Error measures, optionally weighted by the population a design represents.

A MaxDiss design over-samples the tails on purpose. Averaging error over its
points therefore answers a question about the design, not about the ensemble the
design stands in for. Every function here takes an optional weight vector --
normally the Voronoi counts from :func:`~pyhydra.uq.voronoi_weights` -- so the
weighted and unweighted numbers can both be reported, and the difference between
them can be seen rather than argued about.
"""

from __future__ import annotations

import numpy as np

__all__ = ["error_metrics", "weighted_r2", "weighted_rmse"]


def _prep(y_true, y_pred, weights):
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    if y.shape != p.shape:
        raise ValueError(f"shapes differ: {y.shape} vs {p.shape}")
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)
    if w.shape != y.shape:
        raise ValueError(f"weights have shape {w.shape}, expected {y.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if w.sum() <= 0:
        raise ValueError("weights sum to zero")
    return y, p, w


def weighted_rmse(y_true, y_pred, weights=None) -> float:
    """Root mean squared error, weighted by *weights*."""
    y, p, w = _prep(y_true, y_pred, weights)
    return float(np.sqrt(np.sum(w * (p - y) ** 2) / w.sum()))


def weighted_r2(y_true, y_pred, weights=None) -> float:
    """Coefficient of determination against the weighted mean.

    The reference is the *weighted* mean, so that the null model against which
    skill is measured is the same population the error is measured over.
    """
    y, p, w = _prep(y_true, y_pred, weights)
    mu = float(np.sum(w * y) / w.sum())
    ss_tot = float(np.sum(w * (y - mu) ** 2))
    if ss_tot <= 0:
        return float("-inf")
    return float(1 - np.sum(w * (p - y) ** 2) / ss_tot)


def error_metrics(y_true, y_pred, weights=None) -> dict:
    """RMSE, MAE, bias and :func:`weighted_r2` in one dict.

    Examples
    --------
    >>> import numpy as np
    >>> from pyhydra.uq import error_metrics
    >>> m = error_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    >>> m["rmse"], m["r2"]
    (0.0, 1.0)
    """
    y, p, w = _prep(y_true, y_pred, weights)
    e = p - y
    return {
        "rmse": weighted_rmse(y, p, w),
        "mae": float(np.sum(w * np.abs(e)) / w.sum()),
        "bias": float(np.sum(w * e) / w.sum()),
        "r2": weighted_r2(y, p, w),
        "n": int(len(y)),
        "weighted": weights is not None,
    }

"""The emulator families the decision layer is allowed to choose between.

Emulability is not a property of a problem on its own; it is a property of a
problem *together with* a declared set of candidate emulators and a design size.
Fixing that set here, in one place, is what makes the number reported by
:func:`pyhydra.uq.emulability` reproducible, and what stops the family set from
being quietly widened until something fits.

The default set is deliberately small and cheap: radial basis interpolants and
distance-weighted nearest neighbours, both fitted in seconds at the design sizes
this layer is meant for. Supply your own through ``families=`` when the problem
calls for something else -- anything with scikit-learn's ``fit`` and ``predict``
will do -- but record which set you used, because an emulability value means
nothing without it.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "emulator_families",
    "DEFAULT_FAMILIES",
    "RBF_SMOOTHING",
    "RBF",
    "EnsembleMean",
    "PeakOnly1D",
]

#: Families the decision rule chooses between unless told otherwise. Kept to
#: three so that cross-validated selection on a 30-point pilot stays affordable.
DEFAULT_FAMILIES = ("rbf_linear", "rbf_cubic", "knn_5")

#: Fixed so the emulator does not change character with design size along a
#: learning curve. Not tuned per problem: tuning it would make the emulability
#: value depend on a search the pilot has not paid for.
RBF_SMOOTHING = 1.0


class RBF(BaseEstimator, RegressorMixin):
    """scipy's RBF interpolator behind the scikit-learn estimator interface."""

    def __init__(self, kernel: str = "linear", smoothing: float = RBF_SMOOTHING):
        self.kernel = kernel
        self.smoothing = smoothing

    def fit(self, X, y):
        self.model_ = RBFInterpolator(
            np.asarray(X, float),
            np.asarray(y, float),
            kernel=self.kernel,
            smoothing=self.smoothing,
        )
        return self

    def predict(self, X):
        return self.model_(np.asarray(X, float))


class EnsembleMean(BaseEstimator, RegressorMixin):
    """Predict the design mean everywhere. The null model.

    An emulator that cannot beat this scores ``R^2 <= 0`` and has learned
    nothing. Keeping it available makes that comparison explicit rather than
    implied.
    """

    def fit(self, X, y):
        self.mu_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(np.asarray(X)), self.mu_)


class PeakOnly1D(BaseEstimator, RegressorMixin):
    """Monotone 1-D interpolation on the first feature: the classical view.

    Conventional flood practice treats the response as a function of peak flow
    alone. Provided as a baseline so the cost of that assumption can be measured
    on a given catchment instead of asserted.
    """

    def fit(self, X, y):
        x = np.asarray(X, float)[:, 0]
        o = np.argsort(x)
        xs, ys = x[o], np.asarray(y, float)[o]
        self.xs_, idx = np.unique(xs, return_index=True)
        self.ys_ = ys[idx]
        return self

    def predict(self, X):
        return np.interp(np.asarray(X, float)[:, 0], self.xs_, self.ys_)


def emulator_families(names=None) -> dict:
    """Build the candidate emulators, each as a standardizing pipeline.

    Parameters
    ----------
    names : sequence of str, optional
        Which families to return, in order. Default :data:`DEFAULT_FAMILIES`.
        Available: ``rbf_linear``, ``rbf_cubic``, ``rbf_thin_plate_spline``,
        ``knn_2``, ``knn_3``, ``knn_5``, ``knn_7``, ``knn_10``,
        ``baseline_mean``, ``baseline_peak_only``.

    Returns
    -------
    dict
        Name to unfitted scikit-learn pipeline.

    Examples
    --------
    >>> from pyhydra.uq import emulator_families
    >>> sorted(emulator_families())
    ['knn_5', 'rbf_cubic', 'rbf_linear']
    """
    d = {
        f"rbf_{k}": Pipeline([("sc", StandardScaler()), ("m", RBF(kernel=k))])
        for k in ("linear", "cubic", "thin_plate_spline")
    }
    for k in (2, 3, 5, 7, 10):
        d[f"knn_{k}"] = Pipeline(
            [("sc", StandardScaler()), ("m", KNeighborsRegressor(k, weights="distance"))]
        )
    d["baseline_mean"] = Pipeline([("sc", StandardScaler()), ("m", EnsembleMean())])
    d["baseline_peak_only"] = Pipeline([("sc", StandardScaler()), ("m", PeakOnly1D())])
    if names is None:
        names = DEFAULT_FAMILIES
    missing = set(names) - set(d)
    if missing:
        raise KeyError(f"unknown emulator families: {sorted(missing)}")
    return {k: d[k] for k in names}

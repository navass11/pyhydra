"""Design reduction and population weighting.

Two operations are needed before any emulator is fitted. The first chooses
which members of a large synthetic ensemble to send to the expensive solver;
the second records how much of the ensemble each chosen member stands for, so
that error can be weighted by the population rather than by the design.

Both act on the *inputs alone*. Neither consumes a solver evaluation, which is
what makes the pilot-based decision of :mod:`pyhydra.uq.strategy` affordable:
the ensemble exists before anything is simulated.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["maxdiss_order", "voronoi_weights", "pilot_design", "standardize"]


def standardize(X: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Z-score *X*, using the moments of *reference* when given.

    Distances between ensemble members are only meaningful once the columns are
    on a common scale, and the scale must come from the population rather than
    from the subset being transformed, or two calls will not be comparable.
    """
    X = np.asarray(X, float)
    R = X if reference is None else np.asarray(reference, float)
    sd = R.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - R.mean(0)) / sd


def maxdiss_order(X: np.ndarray, n: int) -> list[int]:
    """Maximum-dissimilarity selection (Camus et al., 2011) on standardized *X*.

    Returns the row indices of *X* in selection order. The algorithm is greedy
    and therefore **nested**: the first ``m`` entries of the returned order are
    themselves a valid size-``m`` MaxDiss design for any ``m <= n``. That is
    what allows a pilot to be paid for once and then reused as the opening
    segment of a larger design (see :mod:`pyhydra.uq.strategy`).

    It is deterministic given *X*, with no random seed to record.

    .. note::
       Not the same function as
       :func:`pyhydra.climate.hybrid_downscaling.reconstruction.maxdiss`, which
       is specific to synthetic flood events: that one treats one column as
       circular (hydrograph shape angle) and pre-seeds one case per shape type,
       so the two return different orders on the same data. This one is generic
       -- any input matrix, purely greedy -- and is the one the decision layer
       of :mod:`pyhydra.uq.strategy` assumes, because the pilot must be a prefix
       of the final design.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        The candidate ensemble, in physical units; it is standardized here.
    n : int
        How many members to select.

    Returns
    -------
    list of int
        Indices into *X*, in the order they were selected.

    Examples
    --------
    >>> import numpy as np
    >>> from pyhydra.uq import maxdiss_order
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(500, 3))
    >>> order = maxdiss_order(X, 50)
    >>> order[:10] == maxdiss_order(X, 10)          # nested
    True
    """
    X = np.asarray(X, float)
    if n < 1:
        raise ValueError("n must be at least 1")
    if n > len(X):
        raise ValueError(f"cannot select {n} members from an ensemble of {len(X)}")
    Z = standardize(X)
    order = [int(np.argmax(np.linalg.norm(Z, axis=1)))]
    dmin = np.linalg.norm(Z - Z[order[0]], axis=1)
    while len(order) < n:
        j = int(np.argmax(dmin))
        order.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(Z - Z[j], axis=1))
    return order


def voronoi_weights(design: np.ndarray, population: np.ndarray) -> np.ndarray:
    """How many population members each design point is responsible for.

    A MaxDiss design deliberately over-samples the edges of the input space, so
    it is *not* a random sample of the population. An unweighted error over the
    design therefore answers a question nobody asked. Assigning each population
    member to its nearest design point in standardized space and counting gives
    the weight that restores the population as the object of inference.

    Parameters
    ----------
    design : array of shape (n_design, n_features)
    population : array of shape (n_population, n_features)
        Moments of this array set the standardization for both.

    Returns
    -------
    array of shape (n_design,)
        Non-negative counts summing to ``len(population)``. A weight of zero
        means that design point represents no population member, which is
        informative and is not suppressed.
    """
    design = np.asarray(design, float)
    population = np.asarray(population, float)
    tree = cKDTree(standardize(design, population))
    _, idx = tree.query(standardize(population, population))
    return np.bincount(idx, minlength=len(design)).astype(float)


def pilot_design(X: np.ndarray, n_max: int, n_pilot: int | None = None) -> dict:
    """The opening segment of the design, and the budget it leaves behind.

    The pilot is charged against the same budget as everything else. Because
    :func:`maxdiss` is nested, if the run goes on to reduce and emulate then the
    pilot points are part of the final design and cost nothing extra; if it
    falls back to Monte Carlo they are spent, and only ``n_max - n_pilot``
    evaluations remain. Both branches are held to ``n_max`` in total.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        The full synthetic ensemble.
    n_max : int
        Total solver evaluations available.
    n_pilot : int, optional
        Pilot size. Default ``min(30, n_max // 2)`` -- small enough to leave a
        usable budget, large enough for a five-fold cross-validation.

    Returns
    -------
    dict with keys ``order`` (the full length-``n_max`` MaxDiss order),
    ``pilot`` (its first ``n_pilot`` entries), ``n_pilot`` and
    ``n_remaining`` (``n_max - n_pilot``).
    """
    if n_pilot is None:
        n_pilot = min(30, n_max // 2)
    if n_pilot < 2:
        raise ValueError(f"pilot of {n_pilot} points cannot be cross-validated")
    order = maxdiss_order(X, n_max)
    return {
        "order": order,
        "pilot": order[:n_pilot],
        "n_pilot": int(n_pilot),
        "n_remaining": int(n_max - n_pilot),
    }

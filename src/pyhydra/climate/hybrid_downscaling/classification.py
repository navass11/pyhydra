"""
Hydrograph shape classification via PCA + K-means.

Takes extracted flood events (from ``pyhydra.climate.time_series.events``) and
classifies them into ``n_types`` shape clusters by:

1. Normalising and resampling each hydrograph to a fixed number of points.
2. Reducing dimensionality with PCA (retaining 95 % of variance).
3. Clustering the PCA scores with K-means.
4. Arranging the cluster centroids on a 2-D grid using a minimum-distance
   spatial permutation (so adjacent types are morphologically similar).
"""

from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances


# ---------------------------------------------------------------------------
# Spatial-arrangement helpers (D4 / D8 neighbourhood + permutation search)
# ---------------------------------------------------------------------------

def _d4(i, j):
    return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]


def _d8(i, j):
    return [
        (i - 1, j - 1), (i, j - 1), (i + 1, j - 1),
        (i - 1, j),                  (i + 1, j),
        (i - 1, j + 1), (i, j + 1), (i + 1, j + 1),
    ]


class _GridNeighbours:
    """Bounded grid neighbour iterator (D4 or D8 connectivity)."""

    def __init__(self, n_rows, n_cols, method="D8"):
        self.n = n_rows
        self.m = n_cols
        self.f = _d4 if method == "D4" else _d8 if method == "D8" else None
        if self.f is None:
            raise ValueError(f"Unknown neighbour method '{method}'. Use 'D4' or 'D8'.")

    def __call__(self, i, j):
        return [(ii, jj) for ii, jj in self.f(i, j) if 0 <= ii < self.n and 0 <= jj < self.m]


def _combined_distance(centers):
    return np.sqrt(
        pairwise_distances(centers, metric="correlation") ** 2
        + pairwise_distances(centers, metric="euclidean") ** 2
    )


def find_spatial_arrangement(n_rows, n_cols, centers, iters=200_000, method="D8"):
    """
    Find a low-cost spatial arrangement of K-means cluster centres on an
    (n_rows × n_cols) grid via randomised permutation search.

    Adjacent cells in the grid should be morphologically similar types;
    this minimises the total combined (correlation + Euclidean) distance
    between neighbours.

    Args:
        n_rows, n_cols: Grid dimensions (n_rows * n_cols == number of clusters).
        centers:        K-means cluster centres (n_clusters × n_features).
        iters:          Number of random permutations to evaluate.
        method:         Neighbour connectivity — ``'D4'`` or ``'D8'``.

    Returns:
        best_perm: list of length n_rows * n_cols mapping grid positions to
                   cluster indices.
    """
    visitor = _GridNeighbours(n_rows, n_cols, method)
    dist_mat = _combined_distance(centers)

    best_dist = np.inf
    best_perm = []

    for _ in range(iters):
        perm = np.random.permutation(n_rows * n_cols)
        D = 0.0
        for idx in range(n_rows * n_cols):
            jj, ii = idx % n_cols, idx // n_cols
            for nn in visitor(ii, jj):
                D += dist_mat[perm[idx], perm[nn[0] * n_cols + nn[1]]]
        if D < best_dist:
            best_dist = D
            best_perm = list(perm)

    return best_perm


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class HydrographClassifier:
    """
    Classify flood hydrograph shapes using PCA dimensionality reduction and
    K-means clustering.

    Parameters
    ----------
    discharge : pd.Series
        Daily discharge time series (DatetimeIndex, values in m³/s).
    events_bounds : pd.DataFrame
        Event start/end positions as integer indices into *discharge*, with
        columns ``Inicio_evento`` and ``Fin_evento`` (output of
        ``extract_discharge_events``).
    n_types : int
        Number of shape clusters (must be a perfect square, e.g. 25 → 5×5).
    n_points : int
        Number of points to which each hydrograph is resampled before PCA
        (default 100).
    """

    def __init__(self, discharge, events_bounds, n_types, n_points=100):
        self.discharge = discharge
        self.events_bounds = events_bounds.reset_index(drop=True)
        self.n_types = n_types
        self.n_points = n_points
        self._pca = None
        self._kmeans = None
        self._arrangement = None

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _resample_hydrographs(self):
        """Normalise and resample each event hydrograph to ``n_points``."""
        Q = self.discharge.values
        t_grid = np.linspace(0, 1, self.n_points)
        matrix = np.zeros((len(self.events_bounds), self.n_points))

        for n, (_, row) in enumerate(self.events_bounds.iterrows()):
            x1, x2 = int(row["Inicio_evento"]), int(row["Fin_evento"])
            segment = Q[x1: x2 + 1]
            if segment.max() == 0:
                continue
            t_orig = np.linspace(0, 1, len(segment))
            matrix[n] = np.interp(t_grid, t_orig, segment / segment.max())

        return pd.DataFrame(matrix)

    def fit_pca(self):
        """Fit PCA on resampled hydrographs (95 % variance retained)."""
        shapes = self._resample_hydrographs()
        self._pca = PCA(n_components=0.95)
        self._pca.fit(shapes)
        return self._pca.transform(shapes)

    def fit_kmeans(self, pca_scores):
        """
        Fit K-means on PCA scores and arrange clusters spatially.

        Returns
        -------
        labels : ndarray (n_events,) — cluster label for each event.
        arrangement : list — spatial permutation of cluster indices.
        """
        import math
        side = int(math.ceil(self.n_types ** 0.5))
        if side * side != self.n_types:
            import warnings
            warnings.warn(
                f"n_types={self.n_types} is not a perfect square — "
                f"rounding up to {side * side} for spatial arrangement."
            )
            self.n_types = side * side
        self._kmeans = KMeans(n_clusters=self.n_types, n_init="auto")
        self._kmeans.fit(pca_scores)
        self._arrangement = find_spatial_arrangement(
            side, side, self._kmeans.cluster_centers_
        )
        return self._kmeans.labels_, self._arrangement

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        """
        Run the full classification pipeline.

        Returns
        -------
        classified : pd.DataFrame
            Copy of *events_bounds* augmented with columns ``Qmax``,
            ``Qmed``, ``Duracion``, and ``shape_type`` (cluster label).
        """
        Q = self.discharge.values
        pca_scores = self.fit_pca()
        labels, _ = self.fit_kmeans(pca_scores)

        qmax, qmed, dur = [], [], []
        for _, row in self.events_bounds.iterrows():
            x1, x2 = int(row["Inicio_evento"]), int(row["Fin_evento"])
            seg = Q[x1: x2 + 1]
            qmax.append(seg.max())
            qmed.append(seg.mean())
            dur.append(x2 - x1)

        classified = self.events_bounds.copy()
        classified["Qmax"] = qmax
        classified["Qmed"] = qmed
        classified["Duracion"] = dur
        classified["shape_type"] = labels
        return classified

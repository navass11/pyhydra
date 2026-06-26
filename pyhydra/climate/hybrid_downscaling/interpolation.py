"""
KNN distance-weighted flood map interpolation (Hybrid Downscaling).

Reconstructs flood depth maps for every synthetic event in the ensemble from
a small set of hydraulic model simulations, using K-Nearest Neighbours in the
(Qmax, Qmed, Duration) feature space.

Workflow
--------
1. For each synthetic event, find the k nearest simulated centroids.
2. Weight each neighbour by the inverse of its distance.
3. Compute the weighted average of the k simulated flood maps → reconstructed map.
4. Sort the full ensemble pixel-by-pixel → derive T-year return-period maps.

Two variants are provided:
- ``FloodMapInterpolator``   — historical period only.
- ``FloodMapInterpolatorCC`` — combines historical + climate-change simulations.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from pyhydra.climate.hybrid_downscaling.return_period import (
    DEFAULT_RETURN_PERIODS,
    pixel_return_period,
    save_return_period_geotiffs,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_tif_array(path):
    """Read a GeoTIFF band as a float array; replace nodata with 0."""
    import rasterio

    with rasterio.open(str(path)) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = 0.0
    arr[arr < 0.0] = 0.0
    return arr


def _block_split(data, n_blocks_row, n_blocks_col):
    """Split a 2-D array into a (n_blocks_row × n_blocks_col) grid of sub-arrays."""
    row_splits = np.array_split(data, n_blocks_row, axis=0)
    blocks = []
    for row in row_splits:
        col_splits = np.array_split(row, n_blocks_col, axis=1)
        blocks.append(col_splits)
    return blocks  # list of lists: blocks[r][c] → 2-D array


def _block_bounds(nrows, ncols, n_blocks, r, c):
    """Return pixel (r0, r1, c0, c1) for block (r, c) in an n_blocks×n_blocks grid."""
    row_edges = np.array_split(np.arange(nrows), n_blocks)
    col_edges = np.array_split(np.arange(ncols), n_blocks)
    r0, r1 = int(row_edges[r][0]), int(row_edges[r][-1]) + 1
    c0, c1 = int(col_edges[c][0]), int(col_edges[c][-1]) + 1
    return r0, r1, c0, c1


def _read_tif_block(path, r0, r1, c0, c1, total_nrows, total_ncols):
    """
    Read block [r0:r1, c0:c1] from a GeoTIFF, returning an (r1-r0, c1-c0) array.

    - Same resolution as (total_nrows × total_ncols): direct windowed read.
    - Different resolution: compute proportional source window, read, zoom to block size.

    Args:
        path:         Path to GeoTIFF.
        r0, r1:       Row pixel range in the target raster space.
        c0, c1:       Col pixel range in the target raster space.
        total_nrows:  Height of the full target raster.
        total_ncols:  Width of the full target raster.

    Returns:
        ndarray (r1-r0, c1-c0), float, nodata replaced with 0.
    """
    import rasterio
    from rasterio.windows import Window

    out_h = r1 - r0
    out_w = c1 - c0

    with rasterio.open(str(path)) as src:
        src_h, src_w = src.height, src.width
        nodata = src.nodata

        if src_h == total_nrows and src_w == total_ncols:
            win = Window(col_off=c0, row_off=r0, width=out_w, height=out_h)
            arr = src.read(1, window=win).astype(float)
            if nodata is not None:
                arr[arr == nodata] = 0.0
            arr[arr < 0.0] = 0.0
            return arr

        # Map target-space window to source-space proportionally
        sr0 = int(round(r0 * src_h / total_nrows))
        sr1 = int(round(r1 * src_h / total_nrows))
        sc0 = int(round(c0 * src_w / total_ncols))
        sc1 = int(round(c1 * src_w / total_ncols))
        # Clamp and ensure at least 1 pixel
        sr0 = max(0, min(sr0, src_h - 1))
        sr1 = max(sr0 + 1, min(sr1, src_h))
        sc0 = max(0, min(sc0, src_w - 1))
        sc1 = max(sc0 + 1, min(sc1, src_w))
        win = Window(col_off=sc0, row_off=sr0, width=sc1 - sc0, height=sr1 - sr0)
        patch = src.read(1, window=win).astype(float)

    if nodata is not None:
        patch[patch == nodata] = 0.0
    patch[patch < 0.0] = 0.0
    if patch.shape != (out_h, out_w):
        from scipy.ndimage import zoom as _zoom
        patch = _zoom(patch, (out_h / patch.shape[0], out_w / patch.shape[1]), order=1)
    return patch


def _rbf_weights(synthetic_matrix, centroids_sim, sigma=None):
    """
    Gaussian RBF kernel weights over all simulated centroids.

    Unlike KNN (k nearest neighbours only), RBF uses ALL simulated centroids
    with Gaussian-decaying weights.  The bandwidth σ defaults to the median
    pairwise distance between centroids (median heuristic / Silverman's rule).

    Args:
        synthetic_matrix: DataFrame with columns Qmax, Qmed, Duracion.
        centroids_sim:    DataFrame of simulated centroid features.
        sigma:            RBF bandwidth (km in feature space). None → auto.

    Returns:
        weights: ndarray (n_synthetic, n_centroids), rows sum to 1.
    """
    cols = ["Qmax", "Qmed", "Duracion"]
    cs = centroids_sim[cols].values.astype(float)
    ms = synthetic_matrix[cols].values.astype(float)

    if sigma is None:
        # Median pairwise distance between centroids as bandwidth
        diff = cs[:, None, :] - cs[None, :, :]           # (k, k, 3)
        pairwise_d = np.sqrt((diff ** 2).sum(axis=2))     # (k, k)
        flat = pairwise_d[pairwise_d > 0]
        sigma = float(np.median(flat)) if len(flat) > 0 else 1.0

    n = len(ms)
    k = len(cs)
    weights = np.empty((n, k))
    inv_2sig2 = 1.0 / (2.0 * sigma ** 2)

    for i in range(n):
        d = np.sqrt(((cs - ms[i]) ** 2).sum(axis=1))
        # Use log-sum-exp trick (subtract max) to prevent underflow for tail events
        x = -d ** 2 * inv_2sig2
        w = np.exp(x - np.max(x))
        weights[i] = w / w.sum()

    return weights


def _knn_weights(synthetic_matrix, centroids_sim, k):
    """
    Compute KNN indices and distance weights for every synthetic event.

    Args:
        synthetic_matrix: DataFrame with columns Qmax, Qmed, Duracion.
        centroids_sim:    DataFrame (subset of centroids) that correspond to
                          available hydraulic simulations.
        k:                Number of nearest neighbours.

    Returns:
        positions: ndarray (n_synthetic, k) — row indices into centroids_sim.
        weights:   ndarray (n_synthetic, k) — inverse-distance weights (sum to 1).
    """
    cols = ["Qmax", "Qmed", "Duracion"]
    n = len(synthetic_matrix)
    cs = centroids_sim[cols].values.astype(float)
    ms = synthetic_matrix[cols].values.astype(float)

    k_eff = min(k, len(cs))
    positions = np.empty((n, k_eff), dtype=int)
    weights = np.empty((n, k_eff))

    for i in range(n):
        d = np.sqrt(((cs - ms[i]) ** 2).sum(axis=1))
        if k_eff < len(d):
            idx = np.argpartition(d, k_eff)[:k_eff]
        else:
            idx = np.arange(len(d))
        d_k = d[idx]
        # avoid division by zero when an event matches a centroid exactly
        d_k = np.where(d_k == 0, 1e-12, d_k)
        w = 1.0 / d_k
        positions[i] = idx
        weights[i] = w / w.sum()

    return positions, weights


# ---------------------------------------------------------------------------
# Public classes
# ---------------------------------------------------------------------------

class FloodMapInterpolator:
    """
    KNN reconstruction of flood maps for all synthetic events.

    Parameters
    ----------
    synthetic_matrix : pd.DataFrame
        One row per synthetic event; must contain columns
        ``Qmax``, ``Qmed``, ``Duracion``.
    centroids : pd.DataFrame
        Representative events selected by MaxDiss / K-means. The first
        ``n_simulations`` rows correspond to simulated centroids.
    simulations_dir : str or Path
        Directory containing ``Simul_0.tif``, ``Simul_1.tif``, …
    n_simulations : int
        Number of available hydraulic simulation results.
    k_neighbors : int
        Number of nearest neighbours used in the weighted average (default 6).
    landa : float
        Annual event rate (events/year) from the historical record.
    output_dir : str or Path or None
        Where to write the return-period GeoTIFFs. If None, maps are returned
        but not saved.
    """

    def __init__(
        self,
        synthetic_matrix,
        centroids,
        simulations_dir,
        n_simulations,
        k_neighbors=6,
        landa=4.943,
        output_dir=None,
        method='knn',
        rbf_sigma=None,
    ):
        """
        Parameters
        ----------
        method : str
            Reconstruction method: ``'knn'`` (inverse-distance K-nearest
            neighbours, default) or ``'rbf'`` (Gaussian radial basis functions
            over all simulated centroids).
        rbf_sigma : float or None
            RBF bandwidth in feature space.  ``None`` → median pairwise
            distance between simulated centroids (recommended).
        """
        self.synthetic_matrix = synthetic_matrix.reset_index(drop=True)
        self.centroids = centroids
        self.simulations_dir = Path(simulations_dir)
        self.n_simulations = n_simulations
        self.k = k_neighbors
        self.landa = landa
        self.output_dir = Path(output_dir) if output_dir else None
        self.method = method
        self.rbf_sigma = rbf_sigma

    def _centroids_sim(self):
        return self.centroids[self.centroids.index < self.n_simulations]

    def _sim_path(self, j):
        return self.simulations_dir / f"Simul_{j}.tif"

    def _template_tif(self):
        return self._sim_path(0)

    def _load_sim_array(self, j, nrows, ncols):
        """Load simulation TIF j, resampling to (nrows, ncols) if needed."""
        arr = _open_tif_array(self._sim_path(j))
        if arr.shape != (nrows, ncols):
            from scipy.ndimage import zoom as _zoom
            arr = _zoom(arr, (nrows / arr.shape[0], ncols / arr.shape[1]), order=1)
        return arr

    def _load_sim_block(self, j, r0, r1, c0, c1, nrows, ncols):
        """Read only the block [r0:r1, c0:c1] of simulation TIF j.

        Uses rasterio windowed reads so the full raster is never loaded into
        memory.  If the TIF has a different native resolution the proportional
        source window is read and resampled via scipy zoom.
        """
        return _read_tif_block(self._sim_path(j), r0, r1, c0, c1, nrows, ncols)

    def compute_return_period_maps(
        self,
        return_periods=DEFAULT_RETURN_PERIODS,
        n_blocks=10,
        max_block_mb=1500,
    ):
        """
        Interpolate all synthetic events and compute pixel-based return-period maps.

        The raster is processed in ``n_blocks × n_blocks`` tiles to keep memory
        usage bounded.  ``n_blocks`` is automatically increased when the default
        would cause the per-block ``events`` array to exceed ``max_block_mb`` MB.
        Float32 is used throughout to halve memory relative to float64.

        Args:
            return_periods: Iterable of T values in years.
            n_blocks:       Minimum number of blocks per spatial dimension.
                            Auto-increased for large rasters.
            max_block_mb:   Memory budget in MB for the per-block ``events`` array.
                            Controls the auto-adjustment of ``n_blocks``.

        Returns:
            calados: dict {T: ndarray(nrows, ncols)} — flood depth per return period.
            paths:   list of Path objects (written GeoTIFFs) or empty list if
                     ``output_dir`` is None.
        """
        import math as _math

        try:
            import rasterio  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "FloodMapInterpolator requires rasterio. "
                "Install it with: pip install rasterio"
            ) from exc

        if self.method == 'rbf':
            weights = _rbf_weights(
                self.synthetic_matrix, self._centroids_sim(), sigma=self.rbf_sigma
            )
            # RBF uses all centroids — positions are just sequential indices
            n_sim = len(self._centroids_sim())
            positions = np.tile(np.arange(n_sim), (len(self.synthetic_matrix), 1))
        else:
            positions, weights = _knn_weights(
                self.synthetic_matrix, self._centroids_sim(), self.k
            )
        weights = weights.astype(np.float32)

        import rasterio as _rio
        with _rio.open(str(self._template_tif())) as _src:
            nrows, ncols = _src.height, _src.width

        # Auto-adjust n_blocks so the per-block events array stays under max_block_mb.
        n_events = len(self.synthetic_matrix)
        max_pixels = int(max_block_mb * 1e6 / (n_events * 4))  # float32 = 4 bytes
        n_blocks = max(
            n_blocks,
            _math.ceil(_math.sqrt(nrows * ncols / max(max_pixels, 1))),
        )

        # Build weight_matrix (once per call): weight_matrix[j, e] accumulates
        # the total KNN weight contributed by simulation j to synthetic event e.
        # Shape (n_sim, n_events) — only K non-zeros per column.
        # The block inner loop becomes hiper_2d @ weight_matrix (BLAS SGEMM),
        # which uses all CPU cores and avoids any Python event-loop overhead.
        K_eff = positions.shape[1]
        weight_matrix = np.zeros((self.n_simulations, n_events), dtype=np.float32)
        _col_idx = np.arange(n_events)
        for _k in range(K_eff):
            weight_matrix[positions[:, _k], _col_idx] += weights[:, _k]

        calados = {T: np.zeros((nrows, ncols), dtype=np.float32) for T in return_periods}

        for r in tqdm.tqdm(range(n_blocks), desc="Processing blocks"):
            for c in range(n_blocks):
                r0, r1, c0, c1 = _block_bounds(nrows, ncols, n_blocks, r, c)
                nb_r, nb_c = r1 - r0, c1 - c0

                # Load block portion of every simulated map (windowed read)
                hiper = np.full((nb_r, nb_c, self.n_simulations), np.nan, dtype=np.float32)
                for j in range(self.n_simulations):
                    hiper[:, :, j] = self._load_sim_block(
                        j, r0, r1, c0, c1, nrows, ncols
                    )

                # events[pixel, event] = hiper_2d @ weight_matrix — single BLAS call.
                # No Python loop over events; uses multi-threaded OpenBLAS internally.
                events = (
                    hiper.reshape(-1, self.n_simulations) @ weight_matrix
                ).reshape(nb_r, nb_c, n_events)

                events.sort(axis=2)
                rp = pixel_return_period(events, self.landa, return_periods)
                for T in return_periods:
                    calados[T][r0:r1, c0:c1] = rp[T]

        paths = []
        if self.output_dir is not None:
            paths = save_return_period_geotiffs(
                calados, self._template_tif(), self.output_dir
            )

        return calados, paths


class FloodMapInterpolatorCC(FloodMapInterpolator):
    """
    KNN flood map reconstruction combining historical and climate-change simulations.

    Extends :class:`FloodMapInterpolator` to merge two pools of hydraulic
    simulations — historical (baseline) and future (climate change) — when
    deriving return-period maps for a given climate scenario.

    Parameters
    ----------
    simulations_dir_hist : str or Path
        Directory with historical simulation GeoTIFFs.
    simulations_dir_cc : str or Path
        Directory with climate-change simulation GeoTIFFs.
    n_simulations_hist : int
        Number of available historical simulations.
    n_simulations_cc : int
        Number of available climate-change simulations.

    All other parameters are inherited from :class:`FloodMapInterpolator`.
    """

    def __init__(
        self,
        synthetic_matrix,
        centroids,
        simulations_dir_hist,
        simulations_dir_cc,
        n_simulations_hist,
        n_simulations_cc,
        k_neighbors=6,
        landa=4.943,
        output_dir=None,
        method="knn",
        rbf_sigma=None,
    ):
        self.path_hist = Path(simulations_dir_hist)
        self.path_cc = Path(simulations_dir_cc)
        self.n_sim_hist = n_simulations_hist
        self.n_sim_cc = n_simulations_cc

        super().__init__(
            synthetic_matrix=synthetic_matrix,
            centroids=centroids,
            simulations_dir=simulations_dir_cc,   # template taken from CC dir
            n_simulations=n_simulations_hist + n_simulations_cc,
            k_neighbors=k_neighbors,
            landa=landa,
            output_dir=output_dir,
            method=method,
            rbf_sigma=rbf_sigma,
        )

    def _sim_path(self, j):
        if j < self.n_sim_hist:
            return self.path_hist / f"Simul_{j}.tif"
        return self.path_cc / f"Simul_{j - self.n_sim_hist}.tif"

    def _template_tif(self):
        # Use CC dir as reference grid (higher resolution than hist reference maps)
        return self.path_cc / "Simul_0.tif"

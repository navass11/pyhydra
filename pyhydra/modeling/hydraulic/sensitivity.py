"""
Manning roughness sensitivity analysis for 2-D flood models.

Provides tools to:
  - Generate Monte Carlo combinations of Manning n values per land use type.
  - Load flood ensemble results from a directory of GeoTIFFs with correct
    numeric ordering (fixes the lexicographic glob bug).
  - Build a matching Manning roughness ensemble from per-simulation CSV files.
  - Compute correctly weighted spatial statistics and flooded area.
  - Pair Manning and flood metrics per simulation for regression analysis.

Typical usage
-------------
>>> from pyhydra.modeling.hydraulic.sensitivity import (
...     generate_manning_combinations,
...     load_flood_ensemble,
...     build_manning_ensemble,
...     flooded_area,
...     spatial_stats,
...     manning_flood_regression,
... )
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ── Monte Carlo generation ────────────────────────────────────────────────────

def generate_manning_combinations_correlated(
    manning_dist_csv: str,
    n_samples: int = 1000,
    rho: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Gaussian-copula correlated Manning n sampler.

    Fits the same marginal distributions as :func:`generate_manning_combinations`
    and samples from a Gaussian copula with an equi-correlation matrix using
    :class:`pyhydra.climate.spatial_analysis.copulas.GaussianCopulaSampler`.
    ``rho=0`` recovers independent sampling; ``rho=1`` places all classes at
    the same quantile simultaneously.

    Args:
        manning_dist_csv: Same CSV as :func:`generate_manning_combinations`.
        n_samples: Number of combined samples to return.
        rho: Pearson correlation between all class pairs in the Gaussian copula.
             Must be in [0, 1].
        seed: Random seed for reproducibility.

    Returns:
        DataFrame of shape (n_samples, n_land_use_types), same format as
        :func:`generate_manning_combinations`.
    """
    from scipy import stats
    from pyhydra.climate.spatial_analysis.copulas import GaussianCopulaSampler

    df = pd.read_csv(manning_dist_csv, index_col=0)

    class_names: list[str] = []
    frozen_dists: list = []

    for _, row in df.iterrows():
        if str(row["N"]) == "-999":
            continue
        values = np.array([float(v) for v in str(row["N"]).split(",")])
        dist_name, params = _best_distribution(values)
        dist_cls = getattr(stats, dist_name)
        frozen_dists.append(
            dist_cls(*params[:-2], loc=params[-2], scale=params[-1])
        )
        class_names.append(row["Descripción"])

    sampler = GaussianCopulaSampler(frozen_dists, class_names)
    return sampler.sample(n_samples, rho=rho, seed=seed)


def generate_manning_combinations(
    manning_dist_csv: str,
    n_samples: int = 1000,
    mc_size: int = 10_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Fit distributions to Manning n values and draw Monte Carlo samples.

    For each land use type in *manning_dist_csv*, fits the best of normal,
    lognormal and gamma distributions (selected by Kolmogorov-Smirnov p-value),
    draws *mc_size* samples, then picks *n_samples* without replacement.

    Args:
        manning_dist_csv: CSV with columns 'Descripción' and 'N' (comma-
                          separated n values as a string) plus a row filter
                          'N' == '-999' to skip NoData rows.
        n_samples: Number of combinations to return (rows of output).
        mc_size: Internal Monte Carlo pool size per land use.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame of shape (n_samples, n_land_use_types) with one Manning n
        value per cell, indexed 0 … n_samples-1.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    df = pd.read_csv(manning_dist_csv, index_col=0)
    combinations: dict[str, np.ndarray] = {}

    for _, row in df.iterrows():
        if str(row["N"]) == "-999":
            continue
        values = np.array([float(v) for v in str(row["N"]).split(",")])
        best_dist, best_params = _best_distribution(values)
        dist = getattr(stats, best_dist)
        pool = dist.rvs(*best_params[:-2], loc=best_params[-2],
                        scale=best_params[-1], size=mc_size,
                        random_state=rng.integers(1 << 31))
        pool = pool[pool >= 0]
        selected = rng.choice(pool, size=n_samples, replace=False)
        combinations[row["Descripción"]] = selected

    return pd.DataFrame(combinations)


def _best_distribution(data: np.ndarray) -> tuple[str, tuple]:
    """Return the best-fitting distribution name and parameters via KS test."""
    from scipy import stats

    candidates = ["norm", "lognorm", "gamma"]
    best_p, best_name, best_params = -1.0, "norm", ()
    for name in candidates:
        dist = getattr(stats, name)
        params = dist.fit(data)
        _, p = stats.kstest(data, name, args=params)
        if p > best_p:
            best_p, best_name, best_params = p, name, params
    return best_name, best_params


# ── Ensemble loading ──────────────────────────────────────────────────────────

def load_flood_ensemble(
    results_dir: str,
    pattern: str = "hamax_sim_*.tif",
    threshold: float = 0.05,
    chunks: dict | None = None,
) -> "xr.DataArray":
    """Load flood map GeoTIFFs into a lazy dask-backed DataArray ordered numerically.

    By default each TIF is opened with dask chunks equal to its full spatial
    extent (``chunks={'x': -1, 'y': -1}``), so the resulting (n_sims, y, x)
    DataArray is lazy: no data is read from disk until an operation like
    ``.values`` or ``.compute()`` forces it.  This keeps peak RAM near
    one-TIF-at-a-time instead of loading the whole ensemble at once.

    Pass ``chunks={}`` to disable dask and load eagerly (legacy behaviour).

    Args:
        results_dir: Directory containing the result GeoTIFFs.
        pattern: Glob pattern for result files.
        threshold: Minimum depth (m) to consider a cell flooded; shallower
                   values are set to NaN.
        chunks: Dask chunk spec forwarded to ``rioxarray.open_rasterio``.
                Defaults to ``{'x': -1, 'y': -1}`` (one chunk per TIF, lazy).
                Pass ``{}`` or ``None`` explicitly for eager loading.

    Returns:
        DataArray of shape (n_sims, y, x) with dimension ``simulation``.
    """
    try:
        import rioxarray
        import xarray as xr
    except ImportError as exc:
        raise ImportError("rioxarray and xarray are required") from exc

    files = glob.glob(str(Path(results_dir) / pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {results_dir}")

    files = sorted(
        files,
        key=lambda f: int(re.search(r"(\d+)", Path(f).stem).group(1)),
    )
    sim_numbers = [
        int(re.search(r"(\d+)", Path(f).stem).group(1)) for f in files
    ]

    _chunks = {"x": -1, "y": -1} if chunks is None else chunks

    arrays = []
    for f in files:
        open_kw = {"lock": False, "chunks": _chunks} if _chunks else {}
        da = rioxarray.open_rasterio(f, **open_kw).squeeze(drop=True)
        arrays.append(da)

    ensemble = xr.concat(arrays, dim="simulation")
    ensemble = ensemble.assign_coords(simulation=("simulation", sim_numbers))
    ensemble = ensemble.where(ensemble >= threshold)
    return ensemble


# ── Manning ensemble ──────────────────────────────────────────────────────────

def build_manning_ensemble(
    raster_path: str,
    combinations_dir: str,
    simulation_numbers: list[int] | None = None,
    pattern: str = "combinacion_{n}.csv",
) -> "xr.DataArray":
    """Build a Manning roughness DataArray aligned to *simulation_numbers*.

    Reclassifies the land use raster for each simulation using the
    corresponding combination CSV.  The result has a ``simulation`` coordinate
    matching *simulation_numbers* (same order as :func:`load_flood_ensemble`).

    Args:
        raster_path: Path to the land use GeoTIFF (integer codes per cell).
        combinations_dir: Directory with per-simulation CSV files.
        simulation_numbers: List of simulation indices to load, in the order
                            they appear in the flood ensemble.  If None, loads
                            all CSVs found in *combinations_dir* sorted
                            numerically.
        pattern: Filename pattern; ``{n}`` is replaced by ``sim_number + 1``.

    Returns:
        DataArray of shape (n_sims, y, x) with dimension ``simulation`` and
        NaN where the land use raster had NoData (value 0 or negative).
    """
    try:
        import rioxarray
        import xarray as xr
    except ImportError as exc:
        raise ImportError("rioxarray and xarray are required") from exc

    template = rioxarray.open_rasterio(raster_path).squeeze(drop=True)
    luse_np = template.values.astype(np.int32)  # (y, x) — loaded once, never copied
    valid_mask = luse_np > 0
    max_code = int(luse_np[valid_mask].max()) if valid_mask.any() else 0

    if simulation_numbers is None:
        csvs = glob.glob(str(Path(combinations_dir) / "combinacion_*.csv"))
        simulation_numbers = sorted(
            [int(re.search(r"(\d+)", Path(f).stem).group(1)) - 1 for f in csvs]
        )

    # Pre-allocate one contiguous block — avoids accumulating 1000 separate arrays
    # and eliminates the intermediate xarray copies from chained .where() calls.
    n_sims = len(simulation_numbers)
    output = np.full((n_sims, *luse_np.shape), np.nan, dtype=np.float32)

    for i, sim_n in enumerate(simulation_numbers):
        csv_path = Path(combinations_dir) / pattern.format(n=sim_n + 1)
        table = pd.read_csv(csv_path)
        reclass = dict(zip(table["landuse"], table["N"]))

        lut = np.full(max_code + 1, np.nan, dtype=np.float32)
        for code, val in reclass.items():
            c = int(code)
            if 0 < c <= max_code:
                try:
                    lut[c] = float(val)
                except (ValueError, TypeError):
                    pass

        output[i] = np.where(valid_mask, lut[luse_np.clip(0, max_code)], np.nan)

    ensemble = xr.DataArray(
        output,
        dims=["simulation", "y", "x"],
        coords={"simulation": simulation_numbers, "y": template.y, "x": template.x},
    )
    ensemble = ensemble.rio.write_crs(template.rio.crs)
    return ensemble


# ── Spatial statistics ────────────────────────────────────────────────────────

def flooded_area(
    ensemble: "xr.DataArray",
    cell_area_m2: float = 25.0,
    threshold: float = 0.05,
) -> np.ndarray:
    """Compute flooded area (m²) per simulation.

    Iterates simulation by simulation to avoid loading the full ensemble
    into memory simultaneously.

    Args:
        ensemble: DataArray of shape (n_sims, y, x).  NaN = dry cell.
        cell_area_m2: Area of one raster cell in m².
        threshold: Minimum depth to count as flooded (applied again here in
                   case the ensemble was loaded without filtering).

    Returns:
        1-D array of length n_sims with flooded area in m².
    """
    areas = []
    for i in range(ensemble.sizes["simulation"]):
        vals = ensemble.isel(simulation=i).values
        areas.append(float(np.sum(vals >= threshold)) * cell_area_m2)
    return np.array(areas)


def spatial_stats(ensemble: "xr.DataArray") -> pd.DataFrame:
    """Compute spatially correct mean and median per simulation.

    Iterates simulation by simulation so that only one 2-D raster is in
    RAM at a time.  This is safe with both eager and dask-backed DataArrays
    and avoids the >1 GB peak that arises from vectorised reductions over
    the full (n_sims, y, x) stack.

    Args:
        ensemble: DataArray of shape (n_sims, y, x).

    Returns:
        DataFrame with index = simulation numbers and columns
        ['mean', 'median', 'std', 'max'].
    """
    sim_coord = (
        ensemble.coords["simulation"].values
        if "simulation" in ensemble.coords
        else range(ensemble.sizes["simulation"])
    )
    records = []
    for i in range(ensemble.sizes["simulation"]):
        vals = ensemble.isel(simulation=i).values.ravel()
        valid = vals[~np.isnan(vals)]
        if valid.size:
            records.append({
                "mean": float(valid.mean()),
                "median": float(np.median(valid)),
                "std": float(valid.std()),
                "max": float(valid.max()),
            })
        else:
            records.append({"mean": np.nan, "median": np.nan, "std": np.nan, "max": np.nan})

    return pd.DataFrame(records, index=sim_coord)


# ── Regression helper ─────────────────────────────────────────────────────────

def manning_flood_regression(
    flood_ensemble: "xr.DataArray",
    manning_ensemble: "xr.DataArray",
    cell_area_m2: float = 25.0,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """Compute per-simulation Manning–flood metric pairs for regression.

    For each simulation, computes the mean and median Manning n restricted to
    wet cells (depth ≥ threshold), together with mean flood depth and flooded
    area.

    Both ensembles must have a ``simulation`` coordinate with matching values.

    Args:
        flood_ensemble: DataArray (n_sims, y, x) with flood depths.
        manning_ensemble: DataArray (n_sims, y, x) with Manning n values.
        cell_area_m2: Cell area in m².
        threshold: Minimum depth for a cell to be considered wet.

    Returns:
        DataFrame indexed by simulation number with columns:
        ``manning_mean``, ``manning_median``, ``depth_mean``, ``depth_median``,
        ``flooded_area_m2``.
    """
    import xarray as xr

    # Align both ensembles on the simulation coordinate only
    flood_ensemble, manning_ensemble = xr.align(
        flood_ensemble, manning_ensemble, join="inner", exclude=["x", "y"]
    )

    # Interpolate Manning to flood grid if resolutions differ
    if not (
        np.array_equal(manning_ensemble.x.values, flood_ensemble.x.values)
        and np.array_equal(manning_ensemble.y.values, flood_ensemble.y.values)
    ):
        manning_ensemble = manning_ensemble.interp(
            x=flood_ensemble.x, y=flood_ensemble.y, method="nearest"
        )

    records = []
    sim_coords = flood_ensemble.coords["simulation"].values

    for i, sim_n in enumerate(sim_coords):
        depth_arr = flood_ensemble.isel(simulation=i).values.ravel().astype(np.float32)
        mann_arr  = manning_ensemble.isel(simulation=i).values.ravel()

        wet = depth_arr >= threshold
        valid = wet & ~np.isnan(depth_arr) & ~np.isnan(mann_arr)

        records.append({
            "simulation": sim_n,
            "manning_mean":   float(np.mean(mann_arr[valid]))   if valid.any() else np.nan,
            "manning_median": float(np.median(mann_arr[valid])) if valid.any() else np.nan,
            "depth_mean":     float(np.mean(depth_arr[valid]))  if valid.any() else np.nan,
            "depth_median":   float(np.median(depth_arr[valid]))if valid.any() else np.nan,
            "flooded_area_m2": float(valid.sum() * cell_area_m2),
        })

    return pd.DataFrame(records).set_index("simulation")


# ── Memory-efficient combined Manning+flood stats ─────────────────────────────

def compute_manning_stats(
    raster_path: str,
    combinations_dir: str,
    flood_ensemble: "xr.DataArray",
    simulation_numbers: list[int] | None = None,
    pattern: str = "combinacion_{n}.csv",
    cell_area_m2: float = 25.0,
    threshold: float = 0.05,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Compute Manning spatial stats and flood-regression pairs in a single pass.

    Processes one simulation at a time — the Manning raster is never
    materialised as a full ``(n_sims, y, x)`` array, so peak RAM is roughly
    ``2 × (y × x × 4 bytes)`` instead of ``n_sims × y × x × 4 bytes``.

    This replaces the three-call sequence::

        manning = build_manning_ensemble(...)
        manning_stats = spatial_stats(manning)
        reg = manning_flood_regression(flood, manning, ...)

    with a single call that returns both outputs.

    Args:
        raster_path: Path to the land-use GeoTIFF (integer codes per cell).
        combinations_dir: Directory with per-simulation CSV files.
        flood_ensemble: DataArray ``(n_sims, y, x)`` with flood depths.
        simulation_numbers: Simulation indices to process.  If *None*, uses
            the ``simulation`` coordinate of *flood_ensemble*.
        pattern: Filename pattern; ``{n}`` is replaced by ``sim_number + 1``.
        cell_area_m2: Area of one raster cell in m².
        threshold: Minimum depth for a cell to be considered wet.

    Returns:
        Tuple ``(manning_spatial_stats, regression_df)``:

        * **manning_spatial_stats** — DataFrame indexed by simulation number
          with columns ``['mean', 'median', 'std', 'max']`` of the Manning n
          values over all valid (non-NaN) cells.
        * **regression_df** — DataFrame indexed by simulation number with
          columns ``['manning_mean', 'manning_median', 'depth_mean',
          'depth_median', 'flooded_area_m2']`` restricted to wet cells.
    """
    try:
        import rioxarray
        import xarray as xr
    except ImportError as exc:
        raise ImportError("rioxarray and xarray are required") from exc

    template = rioxarray.open_rasterio(raster_path).squeeze(drop=True)
    luse_np = template.values.astype(np.int32)
    valid_mask = luse_np > 0
    max_code = int(luse_np[valid_mask].max()) if valid_mask.any() else 0

    flood_sims = set(flood_ensemble.coords["simulation"].values.tolist())
    if simulation_numbers is None:
        simulation_numbers = sorted(flood_sims)
    else:
        simulation_numbers = [s for s in simulation_numbers if s in flood_sims]

    spatial_records: list[dict] = []
    reg_records: list[dict] = []

    for sim_n in simulation_numbers:
        csv_path = Path(combinations_dir) / pattern.format(n=sim_n + 1)
        table = pd.read_csv(csv_path)
        reclass = dict(zip(table["landuse"], table["N"]))

        lut = np.full(max_code + 1, np.nan, dtype=np.float32)
        for code, val in reclass.items():
            c = int(code)
            if 0 < c <= max_code:
                try:
                    lut[c] = float(val)
                except (ValueError, TypeError):
                    pass

        mann_2d = np.where(valid_mask, lut[luse_np.clip(0, max_code)], np.nan)
        mann_arr = mann_2d.ravel()

        depth_arr = flood_ensemble.sel(simulation=sim_n).values.ravel().astype(np.float32)

        # Spatial stats — all valid Manning cells
        mann_valid = mann_arr[~np.isnan(mann_arr)]
        spatial_records.append({
            "mean":   float(mann_valid.mean())        if mann_valid.size else np.nan,
            "median": float(np.median(mann_valid))    if mann_valid.size else np.nan,
            "std":    float(mann_valid.std())         if mann_valid.size else np.nan,
            "max":    float(mann_valid.max())         if mann_valid.size else np.nan,
        })

        # Regression pairs — wet cells only
        wet = depth_arr >= threshold
        valid = wet & ~np.isnan(depth_arr) & ~np.isnan(mann_arr)
        reg_records.append({
            "simulation":      sim_n,
            "manning_mean":    float(np.mean(mann_arr[valid]))    if valid.any() else np.nan,
            "manning_median":  float(np.median(mann_arr[valid]))  if valid.any() else np.nan,
            "depth_mean":      float(np.mean(depth_arr[valid]))   if valid.any() else np.nan,
            "depth_median":    float(np.median(depth_arr[valid])) if valid.any() else np.nan,
            "flooded_area_m2": float(valid.sum() * cell_area_m2),
        })

    manning_spatial_stats = pd.DataFrame(spatial_records, index=simulation_numbers)
    regression_df = pd.DataFrame(reg_records).set_index("simulation")
    return manning_spatial_stats, regression_df


# ── SFINCS NetCDF ensemble loader ────────────────────────────────────────────

def load_sfincs_ensemble(
    results_dir: str,
    subdir_pattern: str = "tmp_sfincs_compound_{n}",
    nc_filename: str = "sfincs_map.nc",
    hmax_var: str = "hmax",
    hmax_tidx: int = 0,
    threshold: float = 0.05,
    crs: int = 32630,
) -> "xr.DataArray":
    """Load a SFINCS flood ensemble from raw model output directories.

    Reads ``hmax`` (max water depth) from each ``sfincs_map.nc`` produced by
    SFINCS, georeferences the result using the x/y cell-centre coordinates
    stored in the NetCDF, and returns a DataArray equivalent to the one
    produced by :func:`load_flood_ensemble` for TIF-based workflows.

    The expected directory layout is::

        results_dir/
            tmp_sfincs_compound_1/sfincs_map.nc
            tmp_sfincs_compound_2/sfincs_map.nc
            …

    Args:
        results_dir: Root directory containing one sub-folder per simulation.
        subdir_pattern: Sub-folder name pattern; ``{n}`` is replaced by the
            simulation number.  The function auto-discovers all matching
            subdirectories and extracts ``n`` from their names.
        nc_filename: Name of the NetCDF output file inside each sub-folder.
        hmax_var: Variable name for maximum water depth in the NetCDF.
        hmax_tidx: Index along the ``timemax`` dimension to read (0 = event
            maximum, which is what SFINCS writes at index 0).
        threshold: Minimum depth (m) to consider a cell flooded; shallower
            values are set to NaN.
        crs: EPSG code of the model CRS (default 32630 = UTM 30N).

    Returns:
        DataArray of shape ``(n_sims, y, x)`` with a ``simulation`` coordinate
        equal to the integer ``n`` extracted from each sub-folder name and
        spatial coordinates ``x`` / ``y`` in the model CRS.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required: pip install xarray") from exc

    root = Path(results_dir)
    prefix = subdir_pattern.split("{n}")[0]

    # Find all nc_filename files recursively; filter to dirs whose name matches prefix
    all_nc = [p for p in root.rglob(nc_filename) if p.parent.name.startswith(prefix)]
    if not all_nc:
        raise FileNotFoundError(
            f"No '{nc_filename}' files under dirs matching '{subdir_pattern}' in {results_dir}"
        )
    all_nc = sorted(
        all_nc,
        key=lambda p: int(re.search(r"(\d+)", p.parent.name[len(prefix):]).group(1)),
    )

    arrays, sim_numbers = [], []
    for nc_path in all_nc:
        n = int(re.search(r"(\d+)", nc_path.parent.name[len(prefix):]).group(1))
        with xr.open_dataset(str(nc_path)) as ds:
            hmax = ds[hmax_var].isel(timemax=hmax_tidx)   # (n_rows, n_cols)
            x2d = ds["x"].values                           # cell-centre x coords
            y2d = ds["y"].values                           # cell-centre y coords

        x1d = x2d[0, :] if x2d.ndim == 2 else x2d
        y1d = y2d[:, 0] if y2d.ndim == 2 else y2d

        da = xr.DataArray(
            hmax.values,
            dims=["y", "x"],
            coords={"x": ("x", x1d), "y": ("y", y1d)},
        )
        da = da.rio.write_crs(f"EPSG:{crs}", inplace=True)
        arrays.append(da)
        sim_numbers.append(n)

    if not arrays:
        raise FileNotFoundError(f"No valid '{nc_filename}' files found under {results_dir}")

    ensemble = xr.concat(arrays, dim="simulation")
    ensemble = ensemble.assign_coords(simulation=("simulation", sim_numbers))
    ensemble = ensemble.where(ensemble >= threshold)
    return ensemble


# ── Outlier filtering ─────────────────────────────────────────────────────────

def filter_anomalous_simulations(
    *results: pd.DataFrame,
    metrics: list[str] | None = None,
    z_threshold: float = 3.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Remove simulations with extreme values in any metric from any model.

    Uses a two-step strategy:
    1. Z-score > *z_threshold* on each metric of each DataFrame (catches
       clear numerical non-convergences — extreme isolated values).
    2. The union of flagged indices is removed from every DataFrame so that
       all returned DataFrames share the same simulation set.

    The z-score is computed on a MAD-normalised scale
    (z_i = |x_i - median| / (1.4826 × MAD)) which is robust against the very
    outliers being detected.

    Args:
        *results: One or more DataFrames indexed by simulation number, each
                  returned by :func:`manning_flood_regression`.
        metrics: Column names to evaluate. Defaults to
                 ``['depth_mean', 'flooded_area_m2', 'flooded_area_km2']``
                 (whichever are present in each DataFrame).
        z_threshold: Robust z-score threshold above which a simulation is
                     flagged. Default 3.0 (≈ 3σ).

    Returns:
        Tuple of:
        - ``flagged``: boolean Series (union index) — True where anomalous.
        - ``report``: DataFrame with one row per flagged simulation showing
          which model/metric triggered the flag and the robust z-score.
    """
    if metrics is None:
        metrics = ["depth_mean", "flooded_area_m2", "flooded_area_km2"]

    all_indices = results[0].index
    for df in results[1:]:
        all_indices = all_indices.union(df.index)

    flagged = pd.Series(False, index=all_indices)
    report_rows: list[dict] = []

    for model_idx, df in enumerate(results):
        for col in metrics:
            if col not in df.columns:
                continue
            s = df[col].dropna()
            med = s.median()
            mad = (s - med).abs().median()
            if mad == 0:
                continue
            robust_z = (s - med).abs() / (1.4826 * mad)
            bad = robust_z[robust_z > z_threshold].index
            for sim in bad:
                if not flagged.loc[sim]:
                    report_rows.append({
                        "simulation": sim,
                        "model": model_idx,
                        "metric": col,
                        "value": float(df.loc[sim, col]),
                        "robust_z": float(robust_z.loc[sim]),
                    })
                flagged.loc[sim] = True

    report = (
        pd.DataFrame(report_rows).set_index("simulation")
        if report_rows
        else pd.DataFrame(columns=["model", "metric", "value", "robust_z"])
    )
    return flagged, report

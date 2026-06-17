"""
SFINCS hydraulic model automation via hydromt_sfincs.

SFINCS (Super-Fast INundation of CoastS) is a reduced-complexity flood model
developed by Deltares. This module provides helpers to set up and run SFINCS
using the hydromt_sfincs Python interface.

Requires:
    - hydromt_sfincs: ``pip install hydromt_sfincs``
    - geopandas, xarray, numpy.
    - SFINCS executable (CPU or GPU binary).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# ── pandas 2.x compatibility shims for hydromt_sfincs ────────────────────────
# Index.is_integer() and .is_numeric() were removed in pandas 2.0
if not hasattr(pd.Index, "is_integer"):
    pd.Index.is_integer = lambda self: pd.api.types.is_integer_dtype(self)
if not hasattr(pd.Index, "is_numeric"):
    pd.Index.is_numeric = lambda self: pd.api.types.is_numeric_dtype(self)

# pd.read_csv(delim_whitespace=True) was removed in pandas 2.x
_orig_read_csv = pd.read_csv


def _patched_read_csv(*args, **kwargs):
    if kwargs.pop("delim_whitespace", False):
        kwargs.setdefault("sep", r"\s+")
    return _orig_read_csv(*args, **kwargs)


pd.read_csv = _patched_read_csv


def _write_sfincs_discharge(mod, src_gdf, discharge_series, output_dir, tref):
    """Write sfincs.src and sfincs.dis files directly, bypassing hydromt internals.

    This avoids pandas 2.x incompatibilities inside hydromt_sfincs when
    calling setup_discharge_forcing.
    """
    from datetime import datetime

    out = Path(output_dir)
    tref_dt = datetime.strptime(tref, "%Y%m%d %H%M%S")

    # sfincs.src — one x y pair per source point (projected coordinates)
    src_path = out / "sfincs.src"
    with open(src_path, "w") as f:
        for _, row in src_gdf.iterrows():
            f.write(f"  {row.geometry.x:.3f}  {row.geometry.y:.3f}\n")

    # sfincs.dis — time(s) + one Q column per source point
    dis_path = out / "sfincs.dis"
    n_src = len(src_gdf)
    # align column count to number of source points
    ts = discharge_series.iloc[:, :n_src].copy()
    ts.columns = range(n_src)
    with open(dis_path, "w") as f:
        for t, row in ts.iterrows():
            dt_s = int((t - tref_dt).total_seconds())
            vals = "  ".join(f"{v:.4f}" for v in row.values)
            f.write(f"  {dt_s}  {vals}\n")

    # Register in config so sfincs.inp includes these entries after mod.write()
    mod.set_config("srcfile", "sfincs.src")
    mod.set_config("disfile", "sfincs.dis")
    print(f"  src: {len(src_gdf)} points, dis: {len(ts)} time steps")


def setup_sfincs_model(
    basin_geom,
    dem_path: str,
    output_dir: str,
    discharge_series: pd.DataFrame,
    src_points,
    crs: int = 32630,
    resolution: float = 100.0,
    manning: float = 0.04,
    tref: str = "20200101 000000",
    tstart: str = "20200101 000000",
    tstop: str = "20200110 000000",
    dt_max: float = 10.0,
    z_min: float = -5.0,
) -> object:
    """Set up a SFINCS model for a single basin.

    Builds the model grid from a DEM clipped to the basin polygon, sets up
    discharge source points, and configures the simulation period.

    Args:
        basin_geom: GeoDataFrame (single row) with the basin polygon.
        dem_path: Path to the input DEM GeoTIFF.
        output_dir: Root directory for SFINCS model files.
        discharge_series: DataFrame with a datetime index and one column per
                          source point (indexed by source point ID).
        src_points: GeoDataFrame with discharge source point geometries.
        crs: EPSG code for the model CRS.
        resolution: Grid resolution in metres.
        manning: Manning roughness coefficient (uniform).
        tref: Reference time 'YYYYMMDD HHMMSS'.
        tstart: Simulation start 'YYYYMMDD HHMMSS'.
        tstop: Simulation end 'YYYYMMDD HHMMSS'.
        dt_max: Maximum internal time step (seconds).
        z_min: Minimum elevation threshold for active cells (m).

    Returns:
        Configured SfincsModel instance (not yet written to disk — call
        ``mod.write()`` to persist).
    """
    try:
        from hydromt_sfincs import SfincsModel
        import geopandas as gpd
        import rasterio
        from rasterio.mask import mask as rio_mask
    except ImportError as exc:
        raise ImportError(
            "hydromt_sfincs, geopandas and rasterio are required: "
            "pip install hydromt_sfincs rasterio geopandas"
        ) from exc

    crs_str = f"EPSG:{crs}" if isinstance(crs, int) else crs
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    gis_dir = Path(output_dir) / "gis"
    gis_dir.mkdir(exist_ok=True)

    # Write basin and source-point shapefiles
    if hasattr(basin_geom, "to_file"):
        basin_gdf = basin_geom
    else:
        basin_gdf = gpd.GeoDataFrame(geometry=[basin_geom], crs=crs_str)
    if basin_gdf.crs is None:
        basin_gdf = basin_gdf.set_crs(crs_str)
    basin_gdf.to_file(str(gis_dir / "basin.shp"))

    pts_within = src_points[src_points.within(basin_gdf.geometry.iloc[0])]
    pts_within.to_file(str(gis_dir / "src.shp"))

    # Clip DEM to basin extent using rasterio
    clipped_dem = str(gis_dir / "dem_clipped.tif")
    shapes = [basin_gdf.geometry.iloc[0].__geo_interface__]
    with rasterio.open(dem_path) as src:
        out_image, out_transform = rio_mask(src, shapes, crop=True, nodata=np.nan)
        out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": np.nan,
    })
    with rasterio.open(clipped_dem, "w", **out_meta) as dst:
        dst.write(out_image)

    # Build SFINCS model
    mod = SfincsModel(root=output_dir, mode="w+")
    mod.setup_grid_from_region(
        region={"geom": str(gis_dir / "basin.shp")},
        res=resolution,
        crs=crs_str,
    )
    mod.setup_dep([{"elevtn": clipped_dem, "zmin": z_min}])
    mod.setup_mask_active(include_mask=str(gis_dir / "basin.shp"))
    mod.setup_mask_bounds(btype="outflow", zmin=z_min, reset_bounds=True)
    # Uniform Manning's n — set as a global config value (no roughness raster needed)
    mod.set_config("manning", manning)

    # Simulation period
    mod.set_config("tref", tref)
    mod.set_config("tstart", tstart)
    mod.set_config("tstop", tstop)
    mod.set_config("dtmax", dt_max)

    # Discharge source time series — written directly (avoids hydromt pandas 2.x issues)
    src_gdf = gpd.read_file(str(gis_dir / "src.shp"))
    if len(src_gdf) > 0:
        _write_sfincs_discharge(
            mod=mod,
            src_gdf=src_gdf.reset_index(drop=True),
            discharge_series=discharge_series,
            output_dir=output_dir,
            tref=tref,
        )

    print(f"✓ SFINCS model set up → {output_dir}")
    return mod


def write_manning_wl_boundary(
    model_dir: str,
    discharge_series: pd.DataFrame,
    ch_w: float,
    ch_zb: float,
    mann_n: float = 0.035,
    slope: float = 2e-4,
    tref: str = "20240101 000000",
) -> None:
    """Add a downstream Manning WL boundary to an existing SFINCS run directory.

    Modifies ``sfincs.msk`` to set the downstream edge cells to msk=2 (water
    level boundary), writes ``sfincs.bnd`` (boundary point coordinates) and
    ``sfincs.bzs`` (Manning normal-depth WL time series), and updates
    ``sfincs.inp`` to reference these files.

    Call this **after** ``SfincsModel.write()`` so that the msk modification
    takes effect.  The downstream boundary WL equals Manning's normal depth for
    each Q(t) in *discharge_series*, giving a rating-curve exit condition.

    Args:
        model_dir: Root directory of the SFINCS run (contains ``sfincs.inp``).
        discharge_series: DataFrame with datetime index and one Q column (m³/s).
        ch_w: Channel width (m).
        ch_zb: Channel bottom elevation at the downstream end (m).
        mann_n: Manning's n for the channel (default 0.035).
        slope: Longitudinal bed slope (m/m, default 2e-4).
        tref: Reference time string 'YYYYMMDD HHMMSS'.
    """
    from datetime import datetime
    from scipy.optimize import brentq

    out = Path(model_dir)
    msk_path = out / "sfincs.msk"
    inp_path = out / "sfincs.inp"

    if not msk_path.exists():
        raise FileNotFoundError(f"sfincs.msk not found in {model_dir}")

    # ── Read grid dimensions from sfincs.inp ─────────────────────────────────
    nmax = mmax = dy = dx = None
    for line in inp_path.read_text().splitlines():
        k, _, v = (line.partition("="))
        k, v = k.strip().lower(), v.strip()
        if k == "nmax":
            nmax = int(v)
        elif k == "mmax":
            mmax = int(v)
        elif k == "dy":
            dy = float(v)
        elif k == "dx":
            dx = float(v)
    if None in (nmax, mmax, dy, dx):
        raise ValueError("Could not read nmax/mmax/dy/dx from sfincs.inp")

    # ── Manning normal depth ──────────────────────────────────────────────────
    def _channel_Q(yn: float) -> float:
        A = ch_w * yn
        R = A / (ch_w + 2 * yn)
        return (A / mann_n) * R ** (2 / 3) * slope ** 0.5

    def _yn(Q: float) -> float:
        if Q < 1.0:
            return 0.05
        return brentq(lambda yn: _channel_Q(yn) - Q, 0.01, 80)

    tref_dt = datetime.strptime(tref, "%Y%m%d %H%M%S")
    Q_col = discharge_series.iloc[:, 0].values
    wl_series = np.array([ch_zb + _yn(float(q)) for q in Q_col])

    # ── Modify MSK: downstream edge → msk=2 (WL boundary) ────────────────────
    msk = np.fromfile(msk_path, dtype=np.int8).reshape(nmax, mmax)
    ds_col = msk[:, mmax - 1].copy()
    ds_col[(ds_col == 1) | (ds_col == 3)] = 2
    msk[:, mmax - 1] = ds_col
    msk.tofile(msk_path)

    # ── Write BND file: centre of each downstream-edge WL cell ───────────────
    bnd_rows = np.where(ds_col == 2)[0]
    bnd_x = (mmax - 0.5) * dx
    with open(out / "sfincs.bnd", "w") as f:
        for r in bnd_rows:
            f.write(f"  {bnd_x:.3f}  {(r + 0.5) * dy:.3f}\n")

    # ── Write BZS file: time(s from tref) + WL columns ───────────────────────
    n_pts = len(bnd_rows)
    with open(out / "sfincs.bzs", "w") as f:
        for t, wl in zip(discharge_series.index, wl_series):
            dt_s = int((t - tref_dt).total_seconds())
            cols = "  ".join(f"{wl:.4f}" for _ in range(n_pts))
            f.write(f"{dt_s}  {cols}\n")

    # ── Update sfincs.inp ─────────────────────────────────────────────────────
    lines = [
        ln for ln in inp_path.read_text().splitlines()
        if not ln.startswith("bndfile") and not ln.startswith("bzsfile")
    ]
    lines.append("bndfile              = sfincs.bnd")
    lines.append("bzsfile              = sfincs.bzs")
    inp_path.write_text("\n".join(lines) + "\n")
    print(f"  WL boundary: {n_pts} pts at x={bnd_x:.0f} m · WL_max={wl_series.max():.2f} m")


def run_sfincs(
    model_dir: str,
    sfincs_exe: str,
    timeout: int = 7200,
) -> int:
    """Execute the SFINCS binary.

    Args:
        model_dir: Directory containing the SFINCS input files (sfincs.inp).
        sfincs_exe: Path to the SFINCS executable.
        timeout: Maximum run time in seconds (default 7200 s = 2 h).

    Returns:
        Process return code (0 = success).
    """
    result = subprocess.run(
        [sfincs_exe],
        cwd=model_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode == 0:
        print(f"✓ SFINCS finished successfully in {model_dir}.")
    else:
        print(f"✗ SFINCS failed (returncode={result.returncode}).")
        print(result.stderr[-2000:])
    return result.returncode

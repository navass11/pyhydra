"""
Run 100 SFINCS simulations with correlated Manning roughness (Gaussian copula, rho=0.5).
Mac-adapted version using deltares/sfincs-cpu:latest Docker image.

Usage:
    python run_correlated_sims_mac.py [--n_sims 100] [--rho 0.5] [--start_idx 0]

Outputs:
    results_correlated/hamax_corr_{rho}_{i:04d}.tif  — max flood depth GeoTIFF
    results_correlated/summary_corr_{rho}.csv         — flooded area, mean depth per sim
"""
import argparse
import shutil
import subprocess
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_MODEL  = Path("/Volumes/My Passport 2/COPIA_IH/E/Rugosidades_UCLM/Ejemplo_Besaya/corrales2")
COMB_DIR    = Path("/Volumes/My Passport 2/COPIA_IH/E/Rugosidades_UCLM/Ejemplo_Besaya")
WORK_DIR    = Path("/tmp/sfincs_corr")           # Docker-accessible temp dir
RESULTS_DIR = BASE_MODEL / "results_correlated"  # output GeoTIFFs

# ── SFINCS grid parameters ─────────────────────────────────────────────────
MMAX, NMAX   = 679, 1201  # cols, rows
X0, Y0       = 411727.0, 4788004.0
DX, DY       = 5.0, 5.0
EPSG         = 25830
DEFAULT_N    = 0.04       # sfincs.inp default (used for class-0 nodata cells)
H_THRESH     = 0.05       # minimum depth (m) to count a cell as wet

# Manning CSV column order → land-use class code
# CSV columns: Trees, Dense vegetation, Urban vegetation, Infrastructure,
#              Sparse vegetation, Residential, Industrial, River, Brushland
CLASS_TO_COL = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}

DOCKER_IMAGE = "deltares/sfincs-cpu:latest"


def precompute_active_classes():
    """Return int16 array (n_active,) of land-use class per active SFINCS cell."""
    cache = Path("/tmp/active_classes.npy")
    if cache.exists():
        return np.load(cache)

    print("Pre-computing class per active cell …")
    ind_raw = np.fromfile(BASE_MODEL / "sfincs.ind", dtype=np.int32)
    ind = ind_raw[1:]                         # strip leading count value
    idx_0     = ind - 1                       # 0-based
    col_idx   = idx_0 // NMAX                 # x direction (Fortran col-major)
    row_idx   = idx_0 % NMAX                  # y direction

    x_cells = X0 + (col_idx + 0.5) * DX
    y_cells = Y0 + (row_idx + 0.5) * DY

    with rasterio.open(COMB_DIR / "manning.tif") as src:
        cls_data  = src.read(1).astype(np.int16)
        c_left, c_top = src.bounds.left, src.bounds.top
        c_res     = src.res[0]

    col_c = np.clip(((x_cells - c_left) / c_res).astype(int), 0, cls_data.shape[1] - 1)
    row_c = np.clip(((c_top  - y_cells) / c_res).astype(int), 0, cls_data.shape[0] - 1)
    active_classes = cls_data[row_c, col_c]

    np.save(cache, active_classes)
    print(f"  Saved {len(active_classes)} class codes → {cache}")
    return active_classes


def write_sfincs_man(active_classes: np.ndarray, n_values: np.ndarray,
                     output_path: Path) -> None:
    """Write sfincs.man binary float32 file for given Manning n_values (9-element array)."""
    man = np.full(len(active_classes), DEFAULT_N, dtype=np.float32)
    for cls, col in CLASS_TO_COL.items():
        man[active_classes == cls] = n_values[col]
    man.tofile(output_path)


def extract_hamax(nc_path: Path) -> np.ndarray:
    """Read sfincs_map.nc and return 2-D max-depth array (NMAX×MMAX, float32, nan=dry)."""
    with nc.Dataset(nc_path) as ds:
        hmax2d = np.array(ds["hmax"][-1], dtype=np.float32)  # last time step = cumul max
        msk    = np.array(ds["msk"][:],   dtype=np.int8)
    hmax2d = np.where((hmax2d > H_THRESH) & (msk > 0), hmax2d, np.nan)
    return hmax2d


def save_geotiff(hmax2d: np.ndarray, out_path: Path) -> None:
    """Save (NMAX, MMAX) float32 array as GeoTIFF in EPSG:25830."""
    transform = from_origin(X0, Y0 + NMAX * DY, DX, DY)
    with rasterio.open(
        out_path, "w",
        driver="GTiff", width=MMAX, height=NMAX,
        count=1, dtype="float32", crs=f"EPSG:{EPSG}",
        transform=transform, nodata=float("nan"),
        compress="lzw",
    ) as dst:
        dst.write(hmax2d, 1)


def run_sfincs_docker(work_dir: Path) -> float:
    """Run SFINCS Docker container; return wall-clock time in seconds."""
    t0 = time.time()
    result = subprocess.run(
        ["docker", "run", "--rm",
         "-v", f"{work_dir}:/data",
         DOCKER_IMAGE],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=7200,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"SFINCS failed (rc={result.returncode}):\n{result.stderr[-500:]}")
    return elapsed


def setup_work_dir(work_dir: Path) -> None:
    """Create/reset work dir with base model files (except sfincs.man)."""
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    for fname in ["sfincs.inp", "sfincs.dep", "sfincs.msk", "sfincs.ind",
                  "sfincs.src", "sfincs.dis"]:
        shutil.copy(BASE_MODEL / fname, work_dir / fname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sims",    type=int,   default=100)
    parser.add_argument("--rho",       type=float, default=0.5)
    parser.add_argument("--start_idx", type=int,   default=0)
    args = parser.parse_args()

    rho_tag = f"{args.rho:.2f}".replace(".", "")

    # Load correlated combinations
    csv_file = COMB_DIR / f"combinaciones_rho{rho_tag}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    combos = pd.read_csv(csv_file).values   # shape (1000, 9)
    end_idx = args.start_idx + args.n_sims
    combos  = combos[args.start_idx:end_idx]
    print(f"Loaded {len(combos)} correlated Manning combinations (rho={args.rho})")

    active_classes = precompute_active_classes()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / f"summary_corr_{rho_tag}.csv"

    records = []
    setup_work_dir(WORK_DIR)

    for i, n_values in enumerate(combos):
        sim_idx  = args.start_idx + i
        out_tif  = RESULTS_DIR / f"hamax_corr_{rho_tag}_{sim_idx:04d}.tif"

        if out_tif.exists():
            print(f"  [{sim_idx:04d}] already exists — skip")
            continue

        # Write updated Manning file
        write_sfincs_man(active_classes, n_values, WORK_DIR / "sfincs.man")

        # Remove previous output
        nc_out = WORK_DIR / "sfincs_map.nc"
        nc_out.unlink(missing_ok=True)

        # Run SFINCS
        print(f"  [{sim_idx:04d}] running SFINCS …", end="", flush=True)
        t0 = time.time()
        try:
            elapsed = run_sfincs_docker(WORK_DIR)
        except Exception as e:
            print(f" FAILED: {e}")
            continue

        # Extract results
        hmax2d  = extract_hamax(nc_out)
        wet_n   = int(np.sum(~np.isnan(hmax2d)))
        area_km2 = wet_n * DX * DY / 1e6
        depth_mean = float(np.nanmean(hmax2d)) if wet_n > 0 else 0.0

        save_geotiff(hmax2d, out_tif)

        records.append({
            "sim_idx":       sim_idx,
            "rho":           args.rho,
            "elapsed_s":     round(elapsed, 1),
            "area_km2":      round(area_km2, 6),
            "depth_mean_m":  round(depth_mean, 4),
            **{col: float(v) for col, v in zip(
                ["n_trees","n_dense_veg","n_urban_veg","n_infra",
                 "n_sparse_veg","n_residential","n_industrial",
                 "n_river","n_brushland"], n_values)},
        })

        eta = elapsed * (len(combos) - i - 1)
        print(f" done in {elapsed/60:.1f} min | area={area_km2:.4f} km² | "
              f"ETA {eta/3600:.1f} h remaining")

        # Save incremental CSV
        pd.DataFrame(records).to_csv(summary_path, index=False)

    print(f"\nCompleted {len(records)} simulations → {summary_path}")


if __name__ == "__main__":
    main()

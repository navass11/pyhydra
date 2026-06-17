"""
CMIP6 downloader via the Copernicus Climate Data Store (CDS) API.

Downloads projections-cmip6 datasets with parallel execution per year,
automatic retry on transient failures, and early stop on critical errors.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


def download_CDS_CMIP6(
    url,
    api_key,
    start_date,
    end_date,
    temporal_resolution,
    model,
    experiments,
    variables,
    download_base_dir,
    area,
    combinations_csv=None,
    max_workers=5,
    max_retries=3,
):
    """
    Download CMIP6 projections from the Copernicus CDS.

    Args:
        url: CDS API URL
        api_key: CDS API key
        start_date: Start year (e.g. '1950-01-01')
        end_date: End year (e.g. '2100-12-31')
        temporal_resolution: 'daily' or 'monthly'
        model: Model name or 'All' for all available models
        experiments: List of experiments (e.g. ['historical', 'ssp245'])
        variables: List of CDS variable names
        download_base_dir: Root directory for downloaded files
        area: Bounding box [N, W, S, E] in degrees
        combinations_csv: Path to CSV listing valid model/experiment/variable combinations.
                          If None, looks for a default file inside download_base_dir.
        max_workers: Parallel threads per model (default 5)
        max_retries: Download retries per year (default 3)
    """
    import cdsapi

    c = cdsapi.Client(url=url, key=api_key)

    if combinations_csv is None:
        fname = f"combinaciones_validas_{temporal_resolution}.csv"
        combinations_csv = os.path.join(download_base_dir, fname)

    if temporal_resolution not in ("daily", "monthly"):
        raise ValueError("temporal_resolution must be 'daily' or 'monthly'.")

    if os.path.exists(combinations_csv):
        all_combinations = pd.read_csv(combinations_csv)
    else:
        # No combinations CSV available — build one from the requested parameters
        _models = [model] if model != "All" else [
            "ACCESS-CM2", "ACCESS-ESM1-5", "BCC-CSM2-MR", "CanESM5",
            "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1", "EC-Earth3",
            "EC-Earth3-Veg", "FGOALS-g3", "GFDL-ESM4", "INM-CM5-0",
            "IPSL-CM6A-LR", "KACE-1-0-G", "MIROC-ES2L", "MIROC6",
            "MPI-ESM1-2-HR", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM",
            "NorESM2-MM", "TaiESM1", "UKESM1-0-LL",
        ]
        rows = [
            {"model": m, "experiment": e, "variable": v}
            for m in _models for e in experiments for v in variables
        ]
        all_combinations = pd.DataFrame(rows)
        print(f"[CDS] No combinations CSV found — using all requested model/experiment/variable combinations ({len(rows)} rows).")

    filtered = all_combinations[
        all_combinations["experiment"].isin(experiments)
        & all_combinations["variable"].isin(variables)
    ]
    if model != "All":
        filtered = filtered[filtered["model"] == model]

    if filtered.empty:
        print("No valid combinations found for the requested parameters.")
        return

    periods = pd.date_range(start=start_date, end=end_date, freq="YE")
    _download_model(c, filtered, periods, temporal_resolution, area, download_base_dir, max_workers, max_retries)


def _download_period(c, model, per, variable, experiment, temporal_resolution, area, download_dir, max_retries=3):
    path = os.path.join(
        download_dir,
        f"{model}_{variable}_{experiment}_{temporal_resolution}_{per.year}.zip",
    )
    if os.path.exists(path):
        print(f"Already exists: {path}")
        return True

    request = {
        "temporal_resolution": temporal_resolution,
        "experiment": experiment,
        "variable": variable,
        "model": model,
        "year": [str(per.year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "area": area,
    }

    for attempt in range(max_retries):
        try:
            print(f"Downloading {model}/{variable}/{experiment}/{per.year} (attempt {attempt + 1})...")
            c.retrieve("projections-cmip6", request, path)
            print(f"Done: {path}")
            return True
        except Exception as exc:
            if "RoocsValueError" in str(exc):
                raise RuntimeError(f"Critical error (RoocsValueError) for {model}/{per.year}")
            print(f"Error attempt {attempt + 1}/{max_retries}: {exc}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))

    print(f"Permanent failure: {model}/{per.year}")
    return False


def _cancel_pending(tasks):
    for f in tasks:
        if not f.done():
            f.cancel()


def _download_model(c, combinations, periods, temporal_resolution, area, download_dir, max_workers, max_retries):
    for _, row in combinations.iterrows():
        model, variable, experiment = row["model"], row["variable"], row["experiment"]
        consecutive_failures = 0
        tasks = []
        critical_error = False

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for per in periods:
                    if critical_error:
                        break
                    tasks.append(executor.submit(
                        _download_period,
                        c, model, per, variable, experiment, temporal_resolution, area, download_dir, max_retries,
                    ))

                for task in as_completed(tasks):
                    try:
                        ok = task.result()
                        if not ok:
                            consecutive_failures += 1
                            if consecutive_failures > 2:
                                print(f"Stopping {model}: too many consecutive failures.")
                                _cancel_pending(tasks)
                                break
                        else:
                            consecutive_failures = 0
                    except RuntimeError as exc:
                        if "RoocsValueError" in str(exc):
                            print(f"Critical error for {model}: {exc}")
                            critical_error = True
                            _cancel_pending(tasks)
                            break
                        raise
        except Exception as exc:
            print(f"Unexpected error for {model}: {exc}")

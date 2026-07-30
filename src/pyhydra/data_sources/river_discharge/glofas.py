"""
GloFAS (Global Flood Awareness System) river discharge downloader.

Downloads river discharge reanalysis or forecast data via the
Copernicus Early Warning Data Store (EWDS) API.

Requires ``cdsapi`` and a valid ``~/.cdsapirc`` configuration file
pointing to the EWDS endpoint (https://ewds.climate.copernicus.eu/api).
Portal and terms acceptance: https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical

Available datasets:
  - Historical reanalysis : ``'cems-glofas-historical'``
  - Seasonal reforecast   : ``'cems-glofas-reforecast'``
  - Operational forecast  : ``'cems-glofas-forecast'``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_glofas(
    area: list[float],
    years: Union[list[int], range],
    output_dir: str,
    dataset: str = "cems-glofas-historical",
    system_version: str = "version_4_0",
    hydrological_model: str = "lisflood",
    product_type: str = "consolidated",
    months: Optional[list[int]] = None,
    variable: str = "river_discharge_in_the_last_24_hours",
    file_format: str = "netcdf4",
    api_key: Optional[str] = None,
    api_url: str = "https://ewds.climate.copernicus.eu/api",
) -> list[str]:
    """
    Download GloFAS river discharge data from the Copernicus CDS.

    Args:
        area: Bounding box as ``[North, West, South, East]`` in decimal degrees.
        years: List or range of years to download.
        output_dir: Directory where downloaded files will be saved.
        dataset: CDS dataset name. Options:

            - ``'cems-glofas-historical'`` (default) — reanalysis 1979-present
            - ``'cems-glofas-reforecast'`` — seasonal reforecast
            - ``'cems-glofas-forecast'`` — operational forecast

        system_version: GloFAS system version (default ``'version_4_0'``).
        hydrological_model: ``'lisflood'`` (default) or ``'htessel_lisflood'``.
        product_type: ``'consolidated'`` (default) or ``'intermediate'``.
        months: List of months to download (1–12). Defaults to all 12 months.
        variable: CDS variable name (default: river discharge last 24 h).
        file_format: ``'netcdf4'`` (default) or ``'grib2'``.
        api_key: Copernicus CDS API key. If ``None``, reads from ``~/.cdsapirc``.
        api_url: CDS API endpoint URL.

    Returns:
        List of paths to downloaded files.

    Raises:
        ImportError: If ``cdsapi`` is not installed.
        Exception: On CDS API or network errors.

    Example:
        >>> paths = download_glofas(
        ...     area=[44, -10, 35, 5],   # Iberian Peninsula
        ...     years=[2000, 2001],
        ...     output_dir="glofas_data",
        ... )
    """
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "cdsapi is required to download GloFAS data. "
            "Install it with: pip install cdsapi"
        ) from exc

    if months is None:
        months = list(range(1, 13))

    os.makedirs(output_dir, exist_ok=True)

    client_kwargs: dict = {"url": api_url}
    if api_key is not None:
        client_kwargs["key"] = api_key

    client = cdsapi.Client(**client_kwargs)
    downloaded: list[str] = []

    ext = "nc" if file_format == "netcdf4" else "grib2"
    month_strs = [f"{m:02d}" for m in months]
    year_strs  = [str(y) for y in years]

    request: dict = {
        "system_version":    system_version,
        "hydrological_model": hydrological_model,
        "product_type":      product_type,
        "variable":          variable,
        "year":              year_strs,
        "month":             month_strs,
        "area":              area,
        "format":            file_format,
    }

    tag = f"{dataset}_{year_strs[0]}_{year_strs[-1]}"
    out_path = str(Path(output_dir) / f"glofas_{tag}.{ext}")

    client.retrieve(dataset, request, out_path)
    downloaded.append(out_path)

    return downloaded


def download_glofas_by_year(
    area: list[float],
    years: Union[list[int], range],
    output_dir: str,
    dataset: str = "cems-glofas-historical",
    system_version: str = "version_4_0",
    hydrological_model: str = "lisflood",
    product_type: str = "consolidated",
    months: Optional[list[int]] = None,
    variable: str = "river_discharge_in_the_last_24_hours",
    file_format: str = "netcdf4",
    api_key: Optional[str] = None,
    api_url: str = "https://ewds.climate.copernicus.eu/api",
) -> list[str]:
    """
    Download GloFAS data one file per year (avoids large single requests).

    Same parameters as :func:`download_glofas` but sends one CDS request per year.

    Returns:
        List of paths to all downloaded files (one per year).
    """
    try:
        import cdsapi
    except ImportError as exc:
        raise ImportError(
            "cdsapi is required to download GloFAS data. "
            "Install it with: pip install cdsapi"
        ) from exc

    if months is None:
        months = list(range(1, 13))

    os.makedirs(output_dir, exist_ok=True)

    client_kwargs: dict = {"url": api_url}
    if api_key is not None:
        client_kwargs["key"] = api_key

    client = cdsapi.Client(**client_kwargs)
    downloaded: list[str] = []
    ext = "nc" if file_format == "netcdf4" else "grib2"
    month_strs = [f"{m:02d}" for m in months]

    for year in years:
        request: dict = {
            "system_version":     system_version,
            "hydrological_model": hydrological_model,
            "product_type":       product_type,
            "variable":           variable,
            "year":               str(year),
            "month":              month_strs,
            "area":               area,
            "format":             file_format,
        }
        out_path = str(Path(output_dir) / f"glofas_{dataset}_{year}.{ext}")
        print(f"Downloading GloFAS {year} → {out_path}")
        client.retrieve(dataset, request, out_path)
        downloaded.append(out_path)

    return downloaded


def read_glofas_nc(filepath, lat: float, lon: float,
                   variable: str = "dis24") -> "pd.DataFrame":
    """
    Extract a discharge time series from a GloFAS NetCDF file at a given point.

    Args:
        filepath: Path to a GloFAS ``.nc`` file downloaded via CDS.
        lat: Latitude of the target point (decimal degrees).
        lon: Longitude of the target point (decimal degrees).
        variable: NetCDF variable name for discharge (default ``'dis24'``).

    Returns:
        pd.DataFrame with a DatetimeIndex (``date``) and a ``discharge`` column (m³/s).

    Raises:
        ImportError: If ``xarray`` is not installed.
        KeyError: If the variable is not found in the NetCDF file.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "xarray is required to read GloFAS NetCDF files. "
            "Install it with: pip install xarray netcdf4"
        ) from exc

    import pandas as pd

    ds = xr.open_dataset(filepath)
    if variable not in ds:
        available = list(ds.data_vars)
        raise KeyError(
            f"Variable '{variable}' not found in {filepath}. "
            f"Available variables: {available}"
        )

    lat_dim = "latitude" if "latitude" in ds.coords else "lat"
    lon_dim = "longitude" if "longitude" in ds.coords else "lon"
    da = ds[variable].sel({lat_dim: lat, lon_dim: lon}, method="nearest")
    df = da.to_dataframe().reset_index()[["time", variable]]
    df = df.rename(columns={"time": "date", variable: "discharge"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    ds.close()
    return df

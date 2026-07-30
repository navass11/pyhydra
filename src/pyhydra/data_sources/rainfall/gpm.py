"""
GPM IMERG downloader via NASA Earthdata (earthaccess).

Supported products: GPM_3IMERGHH (30-min), GPM_3IMERGDF (daily), GPM_3IMERGM (monthly).
"""

from datetime import datetime, timedelta
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


class GPMDownloader:
    """
    Download and extract GPM IMERG data from NASA Earthdata.

    Usage::

        gpm = GPMDownloader()
        gpm.set_region(lat_bounds=(36, 44), lon_bounds=(-10, 4))
        results = gpm.search("2020-01-01", "2020-01-31", resolution="daily")
        gpm.open_dataset(results, output_folder="outputs/gpm")
    """

    _PRODUCTS = {
        "hourly": "GPM_3IMERGHH",
        "daily": "GPM_3IMERGDF",
        "monthly": "GPM_3IMERGM",
    }

    def __init__(self):
        import earthaccess
        # Try silent login (netrc / env vars) first; fall back to interactive only in a terminal
        try:
            self.auth = earthaccess.login(strategy="netrc")
        except Exception:
            try:
                self.auth = earthaccess.login(strategy="environment")
            except Exception:
                import sys
                if sys.stdin.isatty():
                    self.auth = earthaccess.login()
                else:
                    print(
                        "[GPMDownloader] NASA Earthdata credentials not found.\n"
                        "  Set EARTHDATA_USERNAME / EARTHDATA_PASSWORD env vars,\n"
                        "  or add credentials to ~/.netrc (see https://urs.earthdata.nasa.gov/)."
                    )
                    self.auth = None
        self.points = None
        self.lat_bounds = None
        self.lon_bounds = None

    def set_region(self, points=None, lat_bounds=None, lon_bounds=None):
        """
        Define spatial region.

        Args:
            points: list of (lat, lon) tuples for point extraction
            lat_bounds: (min_lat, max_lat) for rectangular region
            lon_bounds: (min_lon, max_lon) for rectangular region
        """
        self.points = points
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds

    def search(self, start_date, end_date, resolution="hourly",
               points=None, lat_bounds=None, lon_bounds=None):
        """
        Search for available granules.

        Args:
            start_date: start of date range (YYYY-MM-DD or datetime)
            end_date: end of date range (YYYY-MM-DD or datetime)
            resolution: 'hourly', 'daily', or 'monthly'

        Returns:
            list of matching earthaccess granules
        """
        import earthaccess

        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        now = datetime.utcnow()
        if end > now:
            print("Warning: requested range includes future dates. Adjusting to current UTC date.")
            end = now

        short_name = self._PRODUCTS.get(resolution)
        if short_name is None:
            raise ValueError(f"Invalid resolution '{resolution}'. Choose from: {list(self._PRODUCTS)}")

        points = points or self.points
        lat_bounds = lat_bounds or self.lat_bounds
        lon_bounds = lon_bounds or self.lon_bounds

        if points and (lat_bounds is None or lon_bounds is None):
            lats, lons = zip(*points)
            lat_bounds = (min(lats) - 0.05, max(lats) + 0.05)
            lon_bounds = (min(lons) - 0.05, max(lons) + 0.05)

        if lat_bounds is None or lon_bounds is None:
            raise ValueError("Define a region via set_region() or pass lat_bounds/lon_bounds.")

        bbox = (lon_bounds[0], lat_bounds[0], lon_bounds[1], lat_bounds[1])
        return earthaccess.search_data(
            short_name=short_name,
            version="07",
            temporal=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
            bounding_box=bbox,
        )

    def open_dataset(self, results, variable="precipitation",
                     points=None, lat_bounds=None, lon_bounds=None,
                     chunk_size=48, flip_lat=True, max_retries=3,
                     retry_delay=5, output_folder="outputs"):
        """
        Download, process, and save GPM granules.

        Args:
            results: granule list from search()
            variable: variable name to extract (default 'precipitation')
            points: list of (lat, lon) for point extraction
            lat_bounds/lon_bounds: bounding box for spatial subset
            chunk_size: granules per download batch
            flip_lat: sort latitude ascending
            max_retries: retries on download failure
            retry_delay: seconds between retries
            output_folder: directory for output files
        """
        import earthaccess

        points = points or self.points
        lat_bounds = lat_bounds or self.lat_bounds
        lon_bounds = lon_bounds or self.lon_bounds

        if points is None and (lat_bounds is None or lon_bounds is None):
            raise ValueError("Define a region via set_region() before calling open_dataset().")

        tempdir = Path(tempfile.mkdtemp())
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        aggregated_dfs = []

        for idx in tqdm(range(0, len(results), chunk_size), desc="Overall progress"):
            batch = results[idx:idx + chunk_size]
            if not batch:
                continue

            files = []
            for attempt in range(max_retries):
                try:
                    files = earthaccess.download(batch, local_path=tempdir, pqdm_kwargs={"disable": True})
                    if files:
                        break
                except Exception as exc:
                    print(f"Retry {attempt + 1}/{max_retries}: {exc}")
                    time.sleep(retry_delay)

            if not files:
                print(f"Warning: no files downloaded for batch {idx}-{idx + chunk_size}")
                continue

            for file in files:
                try:
                    ds = xr.open_dataset(file, group="/Grid")
                    var = ds[variable] if variable in ds else list(ds.data_vars.values())[0]

                    if lat_bounds and lon_bounds:
                        var = var.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
                        var["time"] = pd.to_datetime([x.strftime("%Y-%m-%d %H:%M:%S") for x in var["time"].values])

                    if points:
                        stacked = []
                        for lat, lon in points:
                            sel = var.sel(lat=lat, lon=lon, method="nearest")
                            sel = sel.expand_dims(dim={"point": [f"({lat:.2f},{lon:.2f})"]})
                            stacked.append(sel)
                        var = xr.concat(stacked, dim="point")
                        var["time"] = pd.to_datetime([x.strftime("%Y-%m-%d %H:%M:%S") for x in var["time"].values])

                    if flip_lat and "lat" in var.coords and var.lat.size > 1 and var.lat[0] > var.lat[-1]:
                        var = var.sortby("lat")

                    if "point" in var.dims:
                        df = var.to_dataframe(name=variable).reset_index()
                        df = df.pivot(index="time", columns="point", values=variable)
                        aggregated_dfs.append(df)
                    else:
                        timestamp = var.time.values[0].strftime("%Y%m%dT%H%M%S")
                        var.to_netcdf(output_path / f"{variable}_{timestamp}.nc")
                    ds.close()
                except Exception as exc:
                    print(f"Error processing {file}: {exc}")

        try:
            shutil.rmtree(tempdir, ignore_errors=True)
        except Exception:
            pass

        if aggregated_dfs:
            final_df = pd.concat(aggregated_dfs).sort_index()
            final_df.to_csv(output_path / f"{variable}_points.csv")
            return final_df
        return None

    @staticmethod
    def _parse_date(date):
        return datetime.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date

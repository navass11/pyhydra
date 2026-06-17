"""
PERSIANN-CCS downloader via FTP (University of California, Irvine).

Supports hourly and daily precipitation at 0.04° resolution, global 60°S–60°N.
"""

import concurrent.futures
import gzip
import logging
import os
import shutil
import threading
import time
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_FTP_BASE = "ftp://persiann.eng.uci.edu/CHRSdata/PERSIANN-CCS"
_GRID_X = 9000
_GRID_Y = 3000


class PERSSIANDownloader:
    """
    Download PERSIANN-CCS data from the UCI FTP server.

    Initialise with either point coordinates or a bounding box::

        # Point mode
        dl = PERSSIANDownloader(lon=[-3.7, -4.0], lat=[40.4, 41.2])

        # Area mode
        dl = PERSSIANDownloader(lon_min=-10, lon_max=4, lat_min=36, lat_max=44)
    """

    def __init__(self, lon=None, lat=None, lon_min=None, lon_max=None,
                 lat_min=None, lat_max=None, path_output="output", max_workers=2):
        self.max_workers = max_workers
        self.path_output = path_output
        os.makedirs(path_output, exist_ok=True)

        if lon is not None and lat is not None:
            if isinstance(lon, list) and isinstance(lat, list):
                self.points = list(zip(lon, lat))
                self.is_point = True
            else:
                buffer = 0.1
                self.lon_min, self.lon_max = lon - buffer, lon + buffer
                self.lat_min, self.lat_max = lat - buffer, lat + buffer
                self.points = [(lon, lat)]
                self.is_point = True
        elif None not in (lon_min, lon_max, lat_min, lat_max):
            self.lon_min, self.lon_max = lon_min, lon_max
            self.lat_min, self.lat_max = lat_min, lat_max
            self.is_point = False
        else:
            raise ValueError("Provide either (lon, lat) or (lon_min, lon_max, lat_min, lat_max).")

    def download_daily(self, start_date, end_date):
        """Download daily accumulation files."""
        dates = pd.date_range(f"{start_date} 00:00", f"{end_date} 23:00", freq="1D")
        if self.is_point:
            series = self._download_point_series(dates, "1d")
            series.to_csv(os.path.join(self.path_output, "point_timeseries_daily.csv"))
            return series
        self._parallel_download(dates, "1d", "daily")

    def download_hourly(self, start_date, end_date):
        """Download hourly files."""
        dates = pd.date_range(f"{start_date} 00:00", f"{end_date} 23:00", freq="1h")
        if self.is_point:
            series = self._download_point_series(dates, "1h")
            series.to_csv(os.path.join(self.path_output, "point_timeseries_hourly.csv"))
            return series
        self._parallel_download(dates, "1h", "hrly")

    def _build_url(self, dt, time_step):
        doy = dt.dayofyear
        yr = str(dt.year)[2:]
        if time_step == "1d":
            return f"{_FTP_BASE}/daily/rgccs{time_step}{yr}{doy:03d}.bin.gz"
        return f"{_FTP_BASE}/hrly/{dt.year}/rgccs{time_step}{yr}{doy:03d}{dt.hour:02d}.bin.gz"

    def _download_point_series(self, dates, time_step):
        def fetch(dt):
            url = self._build_url(dt, time_step)
            retries = 5
            temp_path = os.path.join(self.path_output, f"temp_{threading.get_ident()}.bin")
            for attempt in range(retries):
                try:
                    with urllib.request.urlopen(url) as resp, gzip.GzipFile(fileobj=resp) as gz:
                        with open(temp_path, "wb") as f:
                            f.write(gz.read())
                    da = self._parse_binary(temp_path, time_step)
                    os.remove(temp_path)
                    values = {}
                    for lon_pt, lat_pt in self.points:
                        key = f"({lon_pt},{lat_pt})"
                        try:
                            values[key] = da.sel(lon=lon_pt, lat=lat_pt, method="nearest").values.item()
                        except Exception:
                            values[key] = np.nan
                    del da  # release the full-grid array immediately
                    return dt, values
                except Exception as exc:
                    logging.warning(f"Attempt {attempt + 1}/{retries} failed for {dt}: {exc}")
                    time.sleep(2 ** attempt)
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except Exception:
                            pass
                    if attempt + 1 == retries:
                        return dt, {f"({lon},{lat})": np.nan for lon, lat in self.points}

        # max_workers=1 for point mode: each file loads the full global grid (~216 MB as
        # float64). Running serially prevents OOM in memory-constrained environments.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            results = list(tqdm(ex.map(fetch, dates), total=len(dates), desc="Overall progress"))

        times, values_dicts = zip(*results)
        return pd.DataFrame(values_dicts, index=pd.to_datetime(times))

    def _parallel_download(self, dates, time_step, path_type):
        def task(dt):
            doy = dt.dayofyear
            yr = str(dt.year)[2:]
            if time_step == "1d":
                filename = f"rgccs_1d_{dt.year}_{dt.month:02d}_{dt.day:02d}_00h"
            else:
                filename = f"rgccs_1h_{dt.year}_{dt.month:02d}_{dt.day:02d}_{dt.hour:02d}h"
            url = self._build_url(dt, time_step)
            out = os.path.join(self.path_output, filename)
            if not os.path.exists(out + ".nc"):
                self._download_and_process(url, out, dt.strftime("%Y-%m-%d %H:%M:%S"), time_step)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            list(tqdm(ex.map(task, dates), total=len(dates), desc="Overall progress"))

    def _download_and_process(self, url, output_path, date_str, time_step):
        retries = 5
        for attempt in range(retries):
            gzip_file = f"{output_path}.bin.gz"
            bin_file = f"{output_path}.bin"
            try:
                urllib.request.urlretrieve(url, gzip_file)
                with gzip.open(gzip_file, "rb") as f_in, open(bin_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gzip_file)
                da = self._parse_binary(bin_file, time_step)
                self._create_netcdf(da, date_str, time_step)
                os.remove(bin_file)
                return
            except Exception as exc:
                logging.warning(f"[{attempt + 1}/{retries}] Error: {url}: {exc}")
                time.sleep(2 ** attempt)
                for f in (gzip_file, bin_file):
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except Exception:
                            pass
                if attempt + 1 == retries:
                    logging.error(f"Failed after {retries} attempts: {url}")

    def _parse_binary(self, binary_file, time_step):
        dtype = ">f" if time_step == "1d" else ">H"
        with open(binary_file, "rb") as fid:
            data = np.fromfile(fid, dtype=dtype)

        matriz = np.transpose(data.reshape((_GRID_X, _GRID_Y), order="F")).astype(float)

        xx_raw = np.linspace(0.02, 359.98, _GRID_X)
        yy = np.linspace(59.98, -59.98, _GRID_Y)
        xx = (xx_raw + 180) % 360 - 180
        sort_idx = np.argsort(xx)
        xx = xx[sort_idx]
        matriz = matriz[:, sort_idx]

        if matriz.shape != (len(yy), len(xx)):
            raise ValueError("Dimension mismatch in parsed binary data")

        matriz[matriz < 0] = np.nan
        matriz[matriz == 555.37] = np.nan
        if time_step == "1h":
            matriz /= 100

        return xr.DataArray(matriz, coords={"lat": yy, "lon": xx}, dims=["lat", "lon"], name="prcp")

    def _create_netcdf(self, data_array, time_str, time_step):
        timestamp = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        filename = os.path.join(
            self.path_output,
            f"rgccs_{time_step}_{timestamp.strftime('%Y_%m_%d_%Hh')}.nc"
        )
        da = data_array.sel(
            lon=slice(self.lon_min, self.lon_max),
            lat=slice(self.lat_max, self.lat_min),
        )
        da.attrs.update({
            "description": "PERSIANN-CCS precipitation",
            "units": "mm",
            "missing_value": -9999,
            "long_name": "Daily precipitation" if time_step == "1d" else "Hourly precipitation",
        })
        da["lat"].attrs.update({"units": "degrees_north", "long_name": "latitude"})
        da["lon"].attrs.update({"units": "degrees_east", "long_name": "longitude"})
        da = da.expand_dims(time=[timestamp])
        try:
            da.to_netcdf(filename, format="NETCDF4")
        except Exception as exc:
            logging.error(f"Error saving {filename}: {exc}")

    def plot_extent(self):
        """Plot the spatial extent on a map (requires cartopy)."""
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
        ax.set_extent([self.lon_min, self.lon_max, self.lat_min, self.lat_max])
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.add_feature(cfeature.OCEAN)
        ax.gridlines(draw_labels=True)
        ax.set_title("Download extent for PERSIANN-CCS")
        plt.show()

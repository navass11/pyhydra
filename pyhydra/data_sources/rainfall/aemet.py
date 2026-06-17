"""
AEMET OpenData downloader and Jupyter widget interface.

Two entry points:
- download_aemet_daily_data(): programmatic download of all Spanish stations
- AEMETDownloader(): interactive Jupyter widget for station selection + download
- AemetCSVLoader: load and quality-check downloaded series
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import xarray as xr


def download_aemet_daily_data(path_output, api_key, start_date_str, end_date_str, interval_days=15):
    """
    Download daily climatological data for all AEMET stations.

    Args:
        path_output: Directory to store output NetCDF files
        api_key: AEMET OpenData API key
        start_date_str: Start date in '%Y-%m-%dT%H:%M:%S[UTC]' format
        end_date_str: End date in '%Y-%m-%dT%H:%M:%S[UTC]' format
        interval_days: Download chunk size in days (default 15, AEMET limit)
    """
    start_date = datetime.strptime(start_date_str[:19], "%Y-%m-%dT%H:%M:%S")
    end_date = datetime.strptime(end_date_str[:19], "%Y-%m-%dT%H:%M:%S")
    interval = timedelta(days=interval_days)
    base_url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos"
    date_ranges = [start_date + i * interval for i in range((end_date - start_date) // interval + 1)]

    def _to_float(value):
        try:
            return float(str(value).replace(",", "."))
        except (ValueError, AttributeError):
            return np.nan

    station_metadata = {}

    for current_start in date_ranges:
        current_end = min(current_start + interval, end_date)
        cs_str = current_start.strftime("%Y-%m-%dT%H:%M:%SUTC")
        ce_str = current_end.strftime("%Y-%m-%dT%H:%M:%SUTC")
        filename = (
            f"{path_output}observations_{current_start.strftime('%Y-%m-%d')}"
            f"_to_{current_end.strftime('%Y-%m-%d')}.nc"
        )

        if os.path.exists(filename):
            print(f"Already exists: {filename}")
            continue

        url = f"{base_url}/fechaini/{cs_str}/fechafin/{ce_str}/todasestaciones?api_key={api_key}"
        records = None

        for attempt in range(1, 4):
            try:
                r = requests.get(url)
                if r.status_code == 200:
                    data_url = r.json().get("datos")
                    if data_url:
                        dr = requests.get(data_url)
                        if dr.status_code == 200 and isinstance(dr.json(), list):
                            records = dr.json()
                            break
                print(f"Attempt {attempt}/3: HTTP {r.status_code}")
                time.sleep(20)
            except requests.RequestException as exc:
                print(f"Attempt {attempt}/3: {exc}")
                time.sleep(20)

        if records is None:
            print(f"Failed to download {cs_str} to {ce_str}")
            continue

        fields = {
            "prec": [], "tmed": [], "tmin": [], "tmax": [], "horatmin": [], "horatmax": [],
            "dir": [], "velmedia": [], "racha": [], "horaracha": [], "presMax": [], "horaPresMax": [],
            "presMin": [], "horaPresMin": [], "hrMedia": [], "hrMax": [], "horaHrMax": [],
            "hrMin": [], "horaHrMin": [], "sol": [],
        }
        times, idemas = [], []

        for rec in records:
            idema = rec.get("indicativo")
            if idema not in station_metadata:
                station_metadata[idema] = {
                    "lon": _to_float(rec.get("lon")),
                    "lat": _to_float(rec.get("lat")),
                    "alt": _to_float(rec.get("alt")),
                    "ubi": rec.get("nombre", np.nan),
                    "provincia": rec.get("provincia", np.nan),
                }
            times.append(pd.to_datetime(rec["fecha"], format="%Y-%m-%d").to_datetime64())
            idemas.append(idema)
            for key in fields:
                fields[key].append(_to_float(rec.get(key, np.nan)))

        df = pd.DataFrame({"time": times, "idema": idemas, **fields})
        df.sort_values("time", inplace=True)
        ds = df.set_index(["time", "idema"]).to_xarray()
        for coord in ("lon", "lat", "alt", "ubi", "provincia"):
            ds.coords[coord] = ("idema", [station_metadata[id_][coord] for id_ in ds["idema"].values])

        ds.to_netcdf(filename)
        print(f"Saved: {filename}")


def AEMETDownloader(netcdf_folder, stations_shapefile=None):
    """
    Interactive Jupyter widget for selecting and downloading AEMET station series.

    Requires: ipyleaflet, ipywidgets, ipyfilechooser (Jupyter environment).
    geopandas is only needed when stations_shapefile is provided.

    Args:
        netcdf_folder: Folder containing NetCDF files from download_aemet_daily_data()
        stations_shapefile: Optional path to AEMET stations shapefile (EPSG:4326).
                            If omitted, station positions are extracted from the NetCDF files.
    """
    import threading

    from IPython.display import display
    from ipyfilechooser import FileChooser
    from ipyleaflet import DrawControl, LayersControl, Map, Marker, MarkerCluster, Popup
    from ipywidgets import Button, DatePicker, HBox, HTML, IntProgress, Output, SelectMultiple, VBox

    output = Output()
    cancel_flag = {"stop": False}
    selected_ids = set()

    start_picker = DatePicker(description="Start", value=pd.Timestamp("2012-01-01"))
    end_picker = DatePicker(description="End", value=pd.Timestamp(datetime.today()))
    folder_chooser = FileChooser(".", title="Output folder", select_dirs=True)
    cancel_button = Button(description="Cancel Download", button_style="danger")

    cancel_button.on_click(lambda _: cancel_flag.update({"stop": True}))

    try:
        nc_files_available = sorted(f for f in os.listdir(netcdf_folder) if f.endswith(".nc"))
        if not nc_files_available:
            raise FileNotFoundError(f"No .nc files found in {netcdf_folder}")
        ds_sample = xr.open_dataset(os.path.join(netcdf_folder, nc_files_available[0]))
        variable_selector = SelectMultiple(options=list(ds_sample.data_vars), description="Variables")
    except Exception as exc:
        with output:
            print(f"Error loading NetCDF sample: {exc}")
        return

    # Build station metadata DataFrame from shapefile or from NetCDF coordinates
    stations_meta = None
    if stations_shapefile is not None:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(stations_shapefile).to_crs("EPSG:4326")
            stations_meta = pd.DataFrame({
                "idema": gdf["idema"].values,
                "nombre": gdf["NOMBRE"].values if "NOMBRE" in gdf.columns else gdf["idema"].values,
                "lat": gdf.geometry.y.values,
                "lon": gdf.geometry.x.values,
            })
        except Exception as exc:
            with output:
                print(f"Warning: could not load shapefile ({exc}). Extracting stations from NetCDF.")

    if stations_meta is None:
        try:
            idemas = ds_sample["idema"].values
            lats = ds_sample.coords["lat"].values if "lat" in ds_sample.coords else np.zeros(len(idemas))
            lons = ds_sample.coords["lon"].values if "lon" in ds_sample.coords else np.zeros(len(idemas))
            names = (ds_sample.coords["ubi"].values.astype(str)
                     if "ubi" in ds_sample.coords else idemas.astype(str))
            stations_meta = pd.DataFrame({"idema": idemas, "nombre": names, "lat": lats, "lon": lons})
            stations_meta = stations_meta.dropna(subset=["lat", "lon"])
            stations_meta = stations_meta[(stations_meta["lat"] != 0) | (stations_meta["lon"] != 0)]
        except Exception as exc:
            with output:
                print(f"Warning: could not extract station positions from NetCDF ({exc}). Markers disabled.")
            stations_meta = pd.DataFrame(columns=["idema", "nombre", "lat", "lon"])

    def extract_series_bulk(station_ids, variables, out_folder, start, end, prog_total, prog_station, prog_label):
        files = sorted(f for f in os.listdir(netcdf_folder) if f.endswith(".nc"))
        for i, sid in enumerate(station_ids, 1):
            if cancel_flag["stop"]:
                break
            prog_label.value = f"<b>Processing {sid} ({i}/{len(station_ids)})</b>"
            prog_total.value = i
            all_series = {}
            for var in variables:
                for f in files:
                    ds = xr.open_dataset(os.path.join(netcdf_folder, f)).sel(time=slice(start, end))
                    prog_station.value = min(prog_station.value + 1, prog_station.max)
                    if sid not in ds["idema"].values:
                        continue
                    try:
                        s = ds[var].sel(idema=sid).to_dataframe().reset_index()[["time", var]]
                        s = s.set_index("time").rename(columns={var: f"{sid}_{var}"})
                        key = (sid, var)
                        all_series[key] = pd.concat([all_series[key], s]) if key in all_series else s
                    except Exception as exc:
                        with output:
                            print(f"Error {sid}/{var}/{f}: {exc}")

            if all_series:
                df = pd.concat(all_series.values(), axis=1)
                os.makedirs(out_folder, exist_ok=True)
                df.to_csv(os.path.join(out_folder, f"AEMET_{sid}_series.csv"))

    markers = []
    for _, row in stations_meta.iterrows():
        marker = Marker(location=(float(row["lat"]), float(row["lon"])))
        html = HTML(value=f"<b>{row['idema']}</b><br>{row['nombre']}")
        btn = Button(description="Download", button_style="info", layout={"width": "auto"})

        def on_click(_, sid=row["idema"]):
            folder = folder_chooser.selected_path
            if not folder or not variable_selector.value:
                return
            cancel_flag["stop"] = False
            prog = IntProgress(value=0, min=0, max=1, description="Chunks:")
            prog_label = HTML(value="Starting...")
            display(VBox([prog_label, prog, cancel_button]))
            extract_series_bulk([sid], variable_selector.value, folder,
                                 start_picker.value, end_picker.value,
                                 IntProgress(min=0, max=1), prog, prog_label)

        btn.on_click(on_click)
        marker.popup = Popup(location=marker.location, child=VBox([html, btn]),
                             close_button=False, auto_close=False)
        markers.append(marker)

    m = Map(center=(40.0, -3.5), zoom=5, scroll_wheel_zoom=True)
    m.add_layer(MarkerCluster(markers=markers))
    m.add_control(LayersControl())

    draw_control = DrawControl(circlemarker={}, marker={}, polygon={}, polyline={})
    draw_control.rectangle = {"shapeOptions": {"color": "#3388ff", "fillOpacity": 0.2}}
    m.add_control(draw_control)

    def _on_draw(_, action, geo_json):
        if action != "created":
            return
        try:
            coords = geo_json["geometry"]["coordinates"][0]
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)
            in_bounds = stations_meta[
                (stations_meta["lat"] >= lat_min) & (stations_meta["lat"] <= lat_max) &
                (stations_meta["lon"] >= lon_min) & (stations_meta["lon"] <= lon_max)
            ]
            selected_ids.clear()
            selected_ids.update(in_bounds["idema"].tolist())
            with output:
                print(f"Selected {len(selected_ids)} stations in drawn area.")
        except Exception as exc:
            with output:
                print(f"Draw callback error: {exc}")

    draw_control.on_draw(_on_draw)

    download_btn = Button(description="Download Selected Area", button_style="success")
    download_thread = None

    def start_download(_):
        nonlocal download_thread
        if download_thread and download_thread.is_alive():
            return
        if not selected_ids or not variable_selector.value or not folder_chooser.selected_path:
            with output:
                print("Select a region on the map, choose variables, and set an output folder first.")
            return
        cancel_flag["stop"] = False
        prog_total = IntProgress(value=0, min=0, max=len(selected_ids), description="Total:")
        prog_station = IntProgress(value=0, min=0, max=1, description="Chunks:")
        prog_label = HTML(value="Starting...")
        display(VBox([prog_label, prog_total, prog_station, cancel_button]))
        download_thread = threading.Thread(
            target=extract_series_bulk,
            args=(list(selected_ids), variable_selector.value, folder_chooser.selected_path,
                  start_picker.value, end_picker.value, prog_total, prog_station, prog_label),
        )
        download_thread.start()

    download_btn.on_click(start_download)
    display(VBox([m, HBox([start_picker, end_picker]), variable_selector, folder_chooser, download_btn, output]))


class AemetCSVLoader:
    """Load and quality-check AEMET series downloaded with AEMETDownloader."""

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.station_df = None
        self.series_df = None

    def load_station_data(self):
        """Load station metadata from selected_stations.csv."""
        path = os.path.join(self.folder_path, "selected_stations.csv")
        self.station_df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        return self.station_df

    def load_series_data(self, variable):
        """
        Load time series for a given variable from all AEMET_*_series.csv files.

        Args:
            variable: Variable name as stored in the CSV (e.g. 'prec', 'tmed')
        """
        if variable is None:
            raise ValueError("Specify a variable to load.")

        series_files = [f for f in os.listdir(self.folder_path)
                        if f.startswith("AEMET_") and f.endswith("_series.csv")]
        dfs = []
        for file in series_files:
            path = os.path.join(self.folder_path, file)
            try:
                df = pd.read_csv(path, parse_dates=["time"])
                sid = file.replace("AEMET_", "").replace("_series.csv", "")
                col = f"{sid}_{variable}"
                if col in df.columns:
                    df = df[["time", col]].drop_duplicates("time").rename(columns={col: sid}).set_index("time")
                    dfs.append(df)
            except Exception as exc:
                print(f"Error reading {file}: {exc}")

        self.series_df = pd.concat(dfs, axis=1).sort_index() if dfs else pd.DataFrame()
        return self.series_df

    def analyze_series_quality(self):
        """Return a DataFrame with data quality metrics per station."""
        if self.series_df is None or self.series_df.empty:
            raise ValueError("Load data first with load_series_data().")

        rows = []
        for station in self.series_df.columns:
            s = self.series_df[station]
            total = len(s)
            missing_pct = 100 * (1 - s.notna().sum() / total) if total > 0 else 100.0
            try:
                full_years = sum(len(g) >= 365 for _, g in s.dropna().groupby(s.dropna().index.year))
                full_months = sum(len(g) >= 28 for _, g in s.dropna().groupby(s.dropna().index.to_period("M")))
            except Exception:
                full_years = full_months = 0
            rows.append({
                "station_id": station,
                "start_year": s.index.year.min(),
                "end_year": s.index.year.max(),
                "missing_percent": round(missing_pct, 2),
                "full_years": full_years,
                "full_months": full_months,
                "min_value": s.min(skipna=True),
                "max_value": s.max(skipna=True),
            })
        return pd.DataFrame(rows)

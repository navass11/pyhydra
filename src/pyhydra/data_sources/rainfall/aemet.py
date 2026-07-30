"""
AEMET OpenData downloader and Jupyter widget interface.

Public API:
- fetch_station_inventory(api_key): fetch all station metadata from AEMET API
- download_aemet_daily_data(): bulk download all stations to NetCDF (for spatial analyses)
- AEMETDownloader(): interactive Jupyter widget — two modes:
    * Direct mode (api_key, no pre-downloaded NetCDF): select on map → download only chosen stations
    * Bulk mode (NetCDF files available): select on map → extract from pre-downloaded files
- AemetCSVLoader: load and quality-check downloaded series
"""

import os
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import xarray as xr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(value):
    try:
        return float(str(value).replace(",", "."))
    except (ValueError, AttributeError):
        return np.nan


def _parse_dms_coord(s):
    """Parse AEMET DMS coordinate string (e.g. '415135N', '0011224O') or decimal string."""
    s = str(s).strip()
    try:
        return float(s.replace(",", "."))
    except ValueError:
        pass
    try:
        hemi = s[-1].upper()
        digits = s[:-1]
        if len(digits) == 6:                                         # lat: DDMMSS
            d, m, sec = int(digits[:2]), int(digits[2:4]), int(digits[4:6])
        elif len(digits) == 7:                                       # lon: DDDMMSS
            d, m, sec = int(digits[:3]), int(digits[3:5]), int(digits[5:7])
        else:
            return np.nan
        val = d + m / 60 + sec / 3600
        return -val if hemi in ("S", "W", "O") else val
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Public: station inventory
# ---------------------------------------------------------------------------

def fetch_station_inventory(api_key):
    """
    Fetch the full AEMET station inventory from the API (~900 stations, <1 s).

    Returns a DataFrame with columns: idema, nombre, lat, lon, alt, provincia.
    """
    url = (
        "https://opendata.aemet.es/opendata/api/valores/climatologicos/"
        f"inventarioestaciones/todasestaciones?api_key={api_key}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data_url = r.json().get("datos")
    dr = requests.get(data_url, timeout=30)
    dr.raise_for_status()

    rows = [{
        "idema":     rec.get("indicativo", ""),
        "nombre":    rec.get("nombre", ""),
        "lat":       _parse_dms_coord(rec.get("latitud", "")),
        "lon":       _parse_dms_coord(rec.get("longitud", "")),
        "alt":       _to_float(rec.get("altitud", "")),
        "provincia": rec.get("provincia", ""),
    } for rec in dr.json()]

    return pd.DataFrame(rows).dropna(subset=["lat", "lon"])


# ---------------------------------------------------------------------------
# Public: bulk download (all stations → NetCDF)
# ---------------------------------------------------------------------------

def download_aemet_daily_data(path_output, api_key, start_date_str, end_date_str, interval_days=15):
    """
    Download daily climatological data for ALL AEMET stations to NetCDF files.

    Use this when you need data for the full Spanish network or plan spatial analyses.
    For a few specific stations, use AEMETDownloader in direct mode instead.

    Args:
        path_output: Directory to store output NetCDF files (created if absent).
        api_key: AEMET OpenData API key.
        start_date_str: Start date in '%Y-%m-%dT%H:%M:%S[UTC]' format.
        end_date_str: End date in '%Y-%m-%dT%H:%M:%S[UTC]' format.
        interval_days: Download chunk size in days (default 15, AEMET API limit).
    """
    os.makedirs(path_output, exist_ok=True)
    start_date = datetime.strptime(start_date_str[:19], "%Y-%m-%dT%H:%M:%S")
    end_date = datetime.strptime(end_date_str[:19], "%Y-%m-%dT%H:%M:%S")
    interval = timedelta(days=interval_days)
    base_url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos"
    date_ranges = [start_date + i * interval for i in range((end_date - start_date) // interval + 1)]

    station_metadata = {}

    for current_start in date_ranges:
        current_end = min(current_start + interval, end_date)
        cs_str = current_start.strftime("%Y-%m-%dT%H:%M:%SUTC")
        ce_str = current_end.strftime("%Y-%m-%dT%H:%M:%SUTC")
        filename = os.path.join(
            path_output,
            f"observations_{current_start.strftime('%Y-%m-%d')}_to_{current_end.strftime('%Y-%m-%d')}.nc",
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
                    "lon": _to_float(rec.get("lon")), "lat": _to_float(rec.get("lat")),
                    "alt": _to_float(rec.get("alt")), "ubi": rec.get("nombre", np.nan),
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


# ---------------------------------------------------------------------------
# Private: per-station direct download
# ---------------------------------------------------------------------------

def _download_stations_direct(api_key, station_ids, variables, out_folder,
                               start_date, end_date, interval_days=15,
                               cancel_flag=None, progress_cb=None):
    """Download daily data for specific stations directly from AEMET API."""
    base_url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos"
    interval = timedelta(days=interval_days)
    all_fields = ["prec", "tmed", "tmin", "tmax", "velmedia", "racha", "sol",
                  "presMax", "presMin", "hrMedia"]
    os.makedirs(out_folder, exist_ok=True)
    saved = []

    for idx, sid in enumerate(station_ids):
        if cancel_flag and cancel_flag.get("stop"):
            break
        if progress_cb:
            progress_cb(idx + 1, len(station_ids), sid)

        all_records = []
        current = start_date
        while current < end_date:
            chunk_end = min(current + interval, end_date)
            cs = current.strftime("%Y-%m-%dT%H:%M:%SUTC")
            ce = chunk_end.strftime("%Y-%m-%dT%H:%M:%SUTC")
            url = f"{base_url}/fechaini/{cs}/fechafin/{ce}/estacion/{sid}?api_key={api_key}"
            for attempt in range(1, 4):
                try:
                    r = requests.get(url, timeout=30)
                    if r.status_code == 200:
                        data_url = r.json().get("datos")
                        if data_url:
                            dr = requests.get(data_url, timeout=30)
                            if dr.status_code == 200 and isinstance(dr.json(), list):
                                all_records.extend(dr.json())
                                break
                    time.sleep(5)
                except requests.RequestException:
                    time.sleep(10)
            current = chunk_end
            time.sleep(1)

        if not all_records:
            continue

        rows = {"time": []}
        for f in all_fields:
            rows[f] = []
        for rec in all_records:
            rows["time"].append(pd.to_datetime(rec.get("fecha", ""), format="%Y-%m-%d", errors="coerce"))
            for f in all_fields:
                rows[f].append(_to_float(rec.get(f, np.nan)))

        df = pd.DataFrame(rows).dropna(subset=["time"]).set_index("time")
        df = df[~df.index.duplicated(keep="first")].sort_index()
        if variables:
            keep = [v for v in variables if v in df.columns]
            if keep:
                df = df[keep]
        df.columns = [f"{sid}_{col}" for col in df.columns]
        df.to_csv(os.path.join(out_folder, f"AEMET_{sid}_series.csv"))
        saved.append(sid)

    return saved


# ---------------------------------------------------------------------------
# Private: bulk extraction from NetCDF
# ---------------------------------------------------------------------------

def _extract_series_bulk(station_ids, variables, out_folder, netcdf_folder,
                          start, end, prog_total, prog_station, prog_label,
                          cancel_flag, output):
    files = sorted(f for f in os.listdir(netcdf_folder) if f.endswith(".nc"))
    prog_station.max = max(1, len(files))
    os.makedirs(out_folder, exist_ok=True)

    for i, sid in enumerate(station_ids, 1):
        if cancel_flag.get("stop"):
            break
        prog_label.value = f"<b>Processing {sid} ({i}/{len(station_ids)})</b>"
        prog_total.value = i
        prog_station.value = 0
        all_series = {}

        for f in files:
            ds = xr.open_dataset(os.path.join(netcdf_folder, f)).sel(time=slice(start, end))
            prog_station.value = min(prog_station.value + 1, prog_station.max)
            if sid not in ds["idema"].values:
                continue
            for var in variables:
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
            df.to_csv(os.path.join(out_folder, f"AEMET_{sid}_series.csv"))


# ---------------------------------------------------------------------------
# Public: interactive widget
# ---------------------------------------------------------------------------

def AEMETDownloader(netcdf_folder, api_key=None, stations_shapefile=None):
    """
    Interactive Jupyter widget for selecting AEMET stations on a map and downloading series.

    Works in two modes (auto-detected):
    - **Direct mode** (api_key provided, no NetCDF in folder): fetches the station
      inventory from AEMET (~900 stations, <1 s) and downloads only the stations
      selected on the map directly to CSV. No bulk pre-download needed.
    - **Bulk mode** (NetCDF files present in netcdf_folder): extracts series from
      files previously downloaded with download_aemet_daily_data().

    Args:
        netcdf_folder: Folder that may contain pre-downloaded NetCDF files.
                       Also used as default output path in direct mode.
        api_key: AEMET OpenData API key. Required in direct mode.
        stations_shapefile: Unused. Kept for backwards compatibility.

    Requires: ipyleaflet, ipywidgets, ipyfilechooser (Jupyter environment).
    """
    from IPython.display import display
    from ipyfilechooser import FileChooser
    from ipyleaflet import DrawControl, LayersControl, Map, Marker, MarkerCluster, Popup
    from ipywidgets import Button, DatePicker, HBox, HTML, IntProgress, Output, SelectMultiple, VBox

    output = Output()
    cancel_flag = {"stop": False}
    selected_ids = set()

    # --- Mode detection ---
    nc_files = []
    if os.path.isdir(netcdf_folder):
        nc_files = sorted(f for f in os.listdir(netcdf_folder) if f.endswith(".nc"))
    direct_mode = len(nc_files) == 0

    if direct_mode and not api_key:
        with output:
            print("[AEMET] No NetCDF files found and no api_key provided.")
            print("  → Direct mode: call AEMETDownloader(folder, api_key='YOUR_KEY')")
            print("  → Bulk mode:   run download_aemet_daily_data() first, then retry")
        display(output)
        return

    # --- Shared widgets ---
    mode_label = HTML(value=(
        "<b style='color:#2a7'>▶ Direct mode</b> — downloads only selected stations via API"
        if direct_mode else
        "<b style='color:#27a'>▶ Bulk mode</b> — extracts from pre-downloaded NetCDF files"
    ))
    start_picker = DatePicker(description="Start", value=pd.Timestamp("2012-01-01"))
    end_picker = DatePicker(description="End", value=pd.Timestamp(datetime.today()))
    folder_chooser = FileChooser(netcdf_folder if os.path.isdir(netcdf_folder) else ".",
                                 title="Output folder", select_dirs=True)
    cancel_button = Button(description="Cancel", button_style="danger")
    cancel_button.on_click(lambda _: cancel_flag.update({"stop": True}))

    all_variables = ["prec", "tmed", "tmin", "tmax", "velmedia", "racha",
                     "sol", "presMax", "presMin", "hrMedia"]
    variable_selector = SelectMultiple(
        options=all_variables, value=["prec"], description="Variables", rows=5,
    )

    # --- Load station metadata ---
    stations_meta = pd.DataFrame(columns=["idema", "nombre", "lat", "lon"])
    status_html = HTML(value="")

    if direct_mode:
        status_html.value = "<i>Fetching station inventory from AEMET API...</i>"
        display(status_html)
        try:
            stations_meta = fetch_station_inventory(api_key)
            status_html.value = (
                f"<span style='color:#2a7'>✓ {len(stations_meta)} stations loaded from AEMET inventory.</span>"
            )
        except Exception as exc:
            status_html.value = f"<b style='color:red'>✗ Could not fetch inventory: {exc}</b>"
            display(output)
            return
    else:
        try:
            ds_sample = xr.open_dataset(os.path.join(netcdf_folder, nc_files[0]))
            variable_selector.options = list(ds_sample.data_vars)
            variable_selector.value = ["prec"] if "prec" in ds_sample.data_vars else []
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
                print(f"Error loading NetCDF: {exc}")
            display(output)
            return

    # --- Build map markers ---
    markers = []
    for _, row in stations_meta.iterrows():
        marker = Marker(location=(float(row["lat"]), float(row["lon"])))
        info_html = HTML(value=f"<b>{row['idema']}</b><br>{row.get('nombre', '')}")
        btn = Button(description="Select / Deselect", button_style="info", layout={"width": "auto"})

        def _on_marker_click(_, sid=row["idema"]):
            if sid in selected_ids:
                selected_ids.discard(sid)
                with output:
                    print(f"Removed: {sid} — total: {len(selected_ids)}")
            else:
                selected_ids.add(sid)
                with output:
                    print(f"Added: {sid} — total: {len(selected_ids)}")

        btn.on_click(_on_marker_click)
        marker.popup = Popup(
            location=marker.location,
            child=VBox([info_html, btn]),
            close_button=False, auto_close=False,
        )
        markers.append(marker)

    # --- Map ---
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
            lat_min = min(c[1] for c in coords)
            lat_max = max(c[1] for c in coords)
            lon_min = min(c[0] for c in coords)
            lon_max = max(c[0] for c in coords)
            in_bounds = stations_meta[
                (stations_meta["lat"] >= lat_min) & (stations_meta["lat"] <= lat_max) &
                (stations_meta["lon"] >= lon_min) & (stations_meta["lon"] <= lon_max)
            ]
            selected_ids.update(in_bounds["idema"].tolist())
            with output:
                print(f"Added {len(in_bounds)} stations in drawn area — total: {len(selected_ids)}")
        except Exception as exc:
            with output:
                print(f"Draw error: {exc}")

    draw_control.on_draw(_on_draw)

    # --- Action buttons ---
    download_btn = Button(description="Download Selected Stations", button_style="success")
    clear_btn = Button(description="Clear Selection", button_style="warning")
    download_thread = None

    def _clear(_):
        selected_ids.clear()
        with output:
            print("Selection cleared.")
    clear_btn.on_click(_clear)

    def _start_download(_):
        nonlocal download_thread
        if download_thread and download_thread.is_alive():
            return
        folder = folder_chooser.selected_path
        if not selected_ids or not folder:
            with output:
                print("Select at least one station on the map and set an output folder.")
            return

        cancel_flag["stop"] = False
        prog = IntProgress(value=0, min=0, max=len(selected_ids), description="Stations:")
        prog_label = HTML(value="Starting…")
        display(VBox([prog_label, prog, cancel_button]))

        if direct_mode:
            def _run():
                def _cb(i, total, sid):
                    prog.value = i
                    prog_label.value = f"<b>Downloading {sid} ({i}/{total})</b>"

                saved = _download_stations_direct(
                    api_key=api_key,
                    station_ids=list(selected_ids),
                    variables=list(variable_selector.value) or None,
                    out_folder=folder,
                    start_date=pd.Timestamp(start_picker.value).to_pydatetime(),
                    end_date=pd.Timestamp(end_picker.value).to_pydatetime(),
                    cancel_flag=cancel_flag,
                    progress_cb=_cb,
                )
                meta_subset = stations_meta[stations_meta["idema"].isin(saved)]
                meta_subset.to_csv(os.path.join(folder, "selected_stations.csv"), index=False)
                prog_label.value = f"<b>Done — {len(saved)} stations saved to {folder}</b>"
                with output:
                    print(f"✓ {len(saved)} CSVs + selected_stations.csv → {folder}")

            download_thread = threading.Thread(target=_run, daemon=True)
            download_thread.start()

        else:
            prog_chunks = IntProgress(value=0, min=0, max=1, description="Chunks:")

            def _run():
                _extract_series_bulk(
                    station_ids=list(selected_ids),
                    variables=list(variable_selector.value),
                    out_folder=folder,
                    netcdf_folder=netcdf_folder,
                    start=start_picker.value,
                    end=end_picker.value,
                    prog_total=prog,
                    prog_station=prog_chunks,
                    prog_label=prog_label,
                    cancel_flag=cancel_flag,
                    output=output,
                )
                meta_subset = stations_meta[stations_meta["idema"].isin(selected_ids)]
                meta_subset.to_csv(os.path.join(folder, "selected_stations.csv"), index=False)
                prog_label.value = f"<b>Done — saved to {folder}</b>"
                with output:
                    print(f"✓ CSVs + selected_stations.csv → {folder}")

            download_thread = threading.Thread(target=_run, daemon=True)
            download_thread.start()

    download_btn.on_click(_start_download)

    display(VBox([
        mode_label,
        status_html,
        m,
        HBox([start_picker, end_picker]),
        variable_selector,
        folder_chooser,
        HBox([download_btn, clear_btn]),
        output,
    ]))


# ---------------------------------------------------------------------------
# Public: CSV loader
# ---------------------------------------------------------------------------

class AemetCSVLoader:
    """Load and quality-check AEMET series downloaded with AEMETDownloader."""

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.station_df = None
        self.series_df = None

    def load_station_data(self):
        """Load station metadata saved by AEMETDownloader as selected_stations.csv."""
        path = os.path.join(self.folder_path, "selected_stations.csv")
        if not os.path.exists(path):
            print(f"[AEMET] selected_stations.csv not found in {self.folder_path}")
            self.station_df = pd.DataFrame()
        else:
            self.station_df = pd.read_csv(path)
        return self.station_df

    def load_series_data(self, variable):
        """
        Load time series for a given variable from all AEMET_*_series.csv files.

        Args:
            variable: Variable name (e.g. 'prec', 'tmed').
        """
        if not os.path.isdir(self.folder_path):
            raise FileNotFoundError(self.folder_path)

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
                    df = (df[["time", col]]
                          .drop_duplicates("time")
                          .rename(columns={col: sid})
                          .set_index("time"))
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
                full_months = sum(len(g) >= 28
                                  for _, g in s.dropna().groupby(s.dropna().index.to_period("M")))
            except Exception:
                full_years = full_months = 0
            rows.append({
                "station_id":      station,
                "start_year":      s.index.year.min(),
                "end_year":        s.index.year.max(),
                "missing_percent": round(missing_pct, 2),
                "full_years":      full_years,
                "full_months":     full_months,
                "min_value":       s.min(skipna=True),
                "max_value":       s.max(skipna=True),
            })
        return pd.DataFrame(rows)

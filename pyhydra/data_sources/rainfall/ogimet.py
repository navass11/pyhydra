"""
OGIMET SYNOP downloader and Jupyter widget interface.

Downloads daily meteorological observations from global SYNOP stations via ogimet.com.
"""

import os
import re
import threading
import time
import unicodedata
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


def _resolve_output_dir() -> Path:
    if os.environ.get("HYDRA_OGIMET_DIR"):
        return Path(os.environ["HYDRA_OGIMET_DIR"])
    if Path("/workspace").exists():
        return Path("/workspace/data/ogimet")
    return Path.home() / "hydra_data" / "ogimet"


DEFAULT_OGIMET_OUTPUT_DIR = _resolve_output_dir()


def _candidate_stations_csv_paths():
    """Return likely locations for the bundled OGIMET station catalogue."""
    module_path = Path(__file__).resolve()
    repo_root = module_path.parents[3]
    return [
        Path(os.environ["HYDRA_OGIMET_STATIONS_CSV"]) if os.environ.get("HYDRA_OGIMET_STATIONS_CSV") else None,
        repo_root / "Data_Sources" / "Rainfall" / "OGIMET" / "data" / "estaciones_ogimet_all.csv",
        Path("/workspace/Data_Sources/Rainfall/OGIMET/data/estaciones_ogimet_all.csv"),
        Path.cwd() / "Data_Sources" / "Rainfall" / "OGIMET" / "data" / "estaciones_ogimet_all.csv",
    ]


def get_default_ogimet_stations_csv():
    """Locate the default OGIMET station metadata CSV."""
    for path in _candidate_stations_csv_paths():
        if path and path.exists():
            return str(path)
    checked = "\n".join(f"- {p}" for p in _candidate_stations_csv_paths() if p)
    raise FileNotFoundError(
        "Could not find the OGIMET station catalogue. Pass stations_csv explicitly "
        "or set HYDRA_OGIMET_STATIONS_CSV. Checked:\n"
        f"{checked}"
    )


def normalize_filename(name):
    """Normalize a station name to a safe ASCII filename."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("utf-8")
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.ogimet.com/synops.phtml",
}


def download_synop(station_id, start_date, end_date, progress=None, cancel_flag=None):
    """
    Download SYNOP daily data for a single station from ogimet.com.

    Args:
        station_id: Numeric SYNOP station code
        start_date: Start date (str or datetime)
        end_date: End date (str or datetime)
        progress: Optional ipywidgets IntProgress for Jupyter display
        cancel_flag: Optional dict with key 'parar' to interrupt download

    Returns:
        DataFrame with multi-index columns, or None on failure
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    date = end_date
    all_data = []

    while date >= start_date:
        if cancel_flag and cancel_flag.get("parar", False):
            print(f"Canceled while downloading {station_id}")
            return None

        ndays = min(30, (date - start_date).days + 1)
        url = (
            f"https://www.ogimet.com/cgi-bin/gsynres?"
            f"ind={station_id}&ndays={ndays}&ano={date.year}&mes={date.month}"
            f"&day={date.day}&hora=0&ord=REV&enviar=Ver"
        )

        try:
            from bs4 import BeautifulSoup
            r = requests.get(url, headers=_BROWSER_HEADERS, timeout=30)
            if r.status_code != 200:
                print(f"  [OGIMET] HTTP {r.status_code} for {url}")
                date -= pd.Timedelta(days=ndays)
                if progress:
                    progress.value += 1
                time.sleep(1.0)
                continue
            soup = BeautifulSoup(r.content, "html.parser")
            tables = pd.read_html(StringIO(str(soup)))
            if len(tables) > 2:
                df = tables[2]
                date_col = next((c for c in df.columns if "Fecha" in str(c)), None)
                if date_col:
                    df[date_col] = (df[date_col].astype(str).str.strip() + f"/{date.year}")
                    df[date_col] = pd.to_datetime(df[date_col], format="%d/%m/%Y", errors="coerce")
                    df = df.sort_values(by=date_col)
                    all_data.append(df)
                else:
                    print(f"  [OGIMET] No 'Fecha' column found in table for station {station_id} ({date.date()})")
            else:
                print(
                    f"  [OGIMET] Expected >2 tables but got {len(tables)} for station {station_id} "
                    f"({date.date()}). OGIMET may be blocking this request or returning an error page."
                )
        except Exception as exc:
            print(f"  [OGIMET] Error for station {station_id} ({date.date()}): {exc}")

        date -= pd.Timedelta(days=ndays)
        time.sleep(1.0)
        if progress:
            progress.value += 1

    return pd.concat(all_data, ignore_index=True) if all_data else None


def process_all_meteorological_variables(df):
    """
    Process a raw OGIMET DataFrame into a clean daily time series.

    Converts wind directions to degrees, handles trace precipitation ('ip'),
    and aggregates 3-hourly observations to daily values.

    Args:
        df: Raw DataFrame with MultiIndex columns from download_synop()

    Returns:
        DataFrame with clean column names and one row per day
    """
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected a DataFrame with MultiIndex columns.")

    df = df[[c for c in df.columns if c[0] != "Diario meteorológico"]].copy()
    date_col = next(c for c in df.columns if "Fecha" in c[0])
    df.loc[:, date_col] = pd.to_datetime(df[date_col], errors="coerce")

    direction_to_degrees = {
        "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
        "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
        "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
        "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
    }

    numeric_cols = [c for c in df.columns if c != date_col]
    for col in numeric_cols:
        if "viento" in col[0].lower() and "dir" in col[1].lower():
            df.loc[:, col] = df[col].map(lambda x: direction_to_degrees.get(str(x).strip().upper()))
        elif "prec" in col[0].lower() or "prec" in col[1].lower():
            df.loc[:, col] = df[col].apply(
                lambda x: 0.1 if isinstance(x, str) and x.strip().lower() == "ip" else x
            )
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

    agg_dict = {
        col: (lambda x: x.sum(min_count=1)) if ("prec" in col[0].lower() or "prec" in col[1].lower()) else "mean"
        for col in numeric_cols
    }
    grouped = df.groupby(df[date_col].dt.date).agg(agg_dict)

    def _clean_name(col):
        parts = [
            p.strip().lower().replace(" ", "_").replace(".", "").replace("%", "pct")
            .replace("(", "").replace(")", "").replace("-", "_").replace("/", "_")
            for p in filter(None, map(str, col))
        ]
        unique = []
        for p in parts:
            if p not in unique:
                unique.append(p)
        return "_".join(unique)

    grouped.columns = [_clean_name(c) for c in grouped.columns]
    grouped = grouped.reset_index()
    grouped.rename(columns={grouped.columns[0]: "date"}, inplace=True)

    rename_map = {
        "temperatura_c_max": "tmax_celsius",
        "temperatura_c_min": "tmin_celsius",
        "temperatura_c_med": "tmed_celsius",
        "td_med_c": "dewpoint_mean_celsius",
        "hr_med_pct": "humidity_mean_percent",
        "viento_km_h_dir": "wind_direction_deg",
        "viento_km_h_vel": "wind_speed_kmh",
        "pres_n_mar_hp": "pressure_msl_hpa",
        "prec_mm": "precipitation_mm",
        "nub_tot_oct": "cloud_total_oktas",
        "nub_baj_oct": "cloud_low_oktas",
        "sol_d_1_h": "sun_hours_day_before",
        "vis_km": "visibility_km",
    }
    grouped.rename(columns={k: v for k, v in rename_map.items() if k in grouped.columns}, inplace=True)
    return grouped


def OGIMETDownloader(
    stations_csv=None,
    zoom=5,
    center=(40.0, -3.5),
    max_markers=None,
    output_folder=None,
):
    """
    Interactive Jupyter widget for selecting and downloading OGIMET station series.

    Requires: ipyleaflet, ipywidgets, beautifulsoup4 (Jupyter environment).

    Args:
        stations_csv: Path to CSV with OGIMET station metadata. If omitted, HYDRA
                      uses the bundled estaciones_ogimet_all.csv catalogue.
                      (columns: Nombre, WIGOS ID, OACI, Latitud_decimal, Longitud_decimal, Altitud, Estado)
        zoom: Initial map zoom level
        center: Initial map center (lat, lon)
        max_markers: Limit number of markers shown (for performance; None = all)
        output_folder: Initial download folder. Defaults to HYDRA_OGIMET_DIR or /workspace/data/ogimet.
    """
    from tqdm.auto import tqdm as _tqdm
    from IPython.display import display
    from ipyleaflet import DrawControl, LayersControl, Map, Marker, MarkerCluster, Popup
    from ipywidgets import Button, DatePicker, HBox, HTML, IntProgress, Output, Text, VBox

    stations_csv = stations_csv or get_default_ogimet_stations_csv()
    output_folder = Path(output_folder or DEFAULT_OGIMET_OUTPUT_DIR)
    output_folder.mkdir(parents=True, exist_ok=True)

    stations_df = pd.read_csv(stations_csv)
    stations_df = stations_df[~stations_df["WIGOS ID"].str.contains("MISSING", case=False, na=False)]
    if max_markers:
        stations_df = stations_df.sample(max_markers, random_state=0)

    output = Output()
    m = Map(center=center, zoom=zoom, scroll_wheel_zoom=True)
    m.layout.height = "480px"
    selected = set()
    cancel_flag = {"parar": False}

    start_picker = DatePicker(description="Start Date", value=pd.Timestamp("2020-01-01"))
    end_picker = DatePicker(description="End Date", value=pd.Timestamp("2020-12-31"))
    folder_input = Text(
        value=str(output_folder),
        description="Save to:",
        layout={"width": "480px"},
        style={"description_width": "60px"},
    )
    cancel_button = Button(description="Cancel Download", button_style="danger")
    cancel_button.on_click(lambda _: cancel_flag.update({"parar": True}))

    markers = []
    for _, row in _tqdm(stations_df.iterrows(), total=len(stations_df), desc="Loading stations"):
        marker = Marker(location=(row["Latitud_decimal"], row["Longitud_decimal"]), draggable=False)

        def on_click(marker=marker, row=row, **kwargs):
            if marker.popup:
                return
            info = HTML(value=(
                f"<b>{row['Nombre']}</b><br>"
                f"WIGOS: {row['WIGOS ID']}<br>ICAO: {row['OACI']}<br>"
                f"Lat/Lon: {row['Latitud_decimal']}, {row['Longitud_decimal']}"
            ))
            btn = Button(description="Download", button_style="info", layout={"width": "auto"})

            def download_single(_, row=row):
                folder = folder_input.value.strip() or str(output_folder)
                base = normalize_filename(row["Nombre"])
                parts = str(row["WIGOS ID"]).split("-")
                code = parts[-1].strip() if len(parts) >= 4 and parts[-1].strip().isdigit() else None
                if not code:
                    with output:
                        print(f"Cannot parse station code from WIGOS ID '{row['WIGOS ID']}'")
                    return
                total_days = (pd.to_datetime(end_picker.value) - pd.to_datetime(start_picker.value)).days + 1
                prog = IntProgress(value=0, min=0, max=(total_days + 29) // 30, description="Download:")
                status = HTML(value=f"<b>Downloading {row['Nombre']}</b> → <code>{folder}</code>")
                display(VBox([status, prog]))

                def _do():
                    try:
                        os.makedirs(folder, exist_ok=True)
                        with output:
                            df_raw = download_synop(code, start_picker.value, end_picker.value,
                                                    progress=prog, cancel_flag=cancel_flag)
                        if df_raw is not None:
                            df_proc = process_all_meteorological_variables(df_raw)
                            df_proc.to_csv(os.path.join(folder, f"series_{base}.csv"), index=False)
                            pd.DataFrame([row]).to_csv(os.path.join(folder, f"station_{base}.csv"), index=False)
                            status.value = f"Saved → <code>{folder}/series_{base}.csv</code>"
                        else:
                            status.value = f"<span style='color:red'>No data for {row['Nombre']} (code {code})</span>"
                    except Exception as exc:
                        with output:
                            print(f"Error downloading {base}: {exc}")

                threading.Thread(target=_do).start()

            btn.on_click(download_single)
            marker.popup = Popup(location=marker.location, child=VBox([info, btn]),
                                 close_button=True, auto_close=False)

        marker.on_click(on_click)
        markers.append(marker)

    m.add_layer(MarkerCluster(markers=markers))
    m.add_control(LayersControl())

    draw_control = DrawControl(circle={}, polygon={}, polyline={}, circlemarker={})
    draw_control.rectangle = {"shapeOptions": {"color": "#6bc2e5", "fillOpacity": 0.3}}
    m.add_control(draw_control)

    selection_label = HTML(value="<i>Draw a rectangle on the map to select stations.</i>")

    def _on_draw(_, action, geo_json):
        if action != "created":
            return
        try:
            coords = geo_json["geometry"]["coordinates"][0]
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)
            in_bounds = stations_df[
                (pd.to_numeric(stations_df["Latitud_decimal"], errors="coerce") >= lat_min) &
                (pd.to_numeric(stations_df["Latitud_decimal"], errors="coerce") <= lat_max) &
                (pd.to_numeric(stations_df["Longitud_decimal"], errors="coerce") >= lon_min) &
                (pd.to_numeric(stations_df["Longitud_decimal"], errors="coerce") <= lon_max)
            ]
            selected.clear()
            selected.update(in_bounds["Nombre"].tolist())
            selection_label.value = f"<b>{len(selected)} stations selected.</b> Click 'Download Selected Area' to start."
        except Exception as exc:
            selection_label.value = f"<span style='color:red'>Draw error: {exc}</span>"

    draw_control.on_draw(_on_draw)

    download_area_btn = Button(description="Download Selected Area", button_style="success")
    download_thread = None

    def start_download(_):
        nonlocal download_thread
        if download_thread and download_thread.is_alive():
            return
        if not selected:
            selection_label.value = "<i>Draw a rectangle on the map to select stations first.</i>"
            return
        folder = folder_input.value.strip() or str(output_folder)
        cancel_flag["parar"] = False
        selected_df = stations_df[stations_df["Nombre"].isin(selected)]

        prog_total = IntProgress(value=0, min=0, max=len(selected_df), description="Total:")
        prog_station = IntProgress(value=0, min=0, max=1, description="Station:")
        prog_label = HTML(value=f"Starting download → <code>{folder}</code>")
        display(VBox([prog_label, prog_total, prog_station, cancel_button, output]))

        def _run():
            try:
                os.makedirs(folder, exist_ok=True)
                selected_df.to_csv(os.path.join(folder, "selected_stations.csv"), index=False)
                for i, (_, row) in enumerate(selected_df.iterrows(), 1):
                    if cancel_flag["parar"]:
                        break
                    prog_label.value = (
                        f"<b>Station {i}/{len(selected_df)}: {row['Nombre']}</b> "
                        f"→ <code>{folder}</code>"
                    )
                    base = normalize_filename(row["Nombre"])
                    parts = str(row["WIGOS ID"]).split("-")
                    code = parts[-1].strip() if len(parts) >= 4 and parts[-1].strip().isdigit() else None
                    if not code:
                        with output:
                            print(f"Skipping {row['Nombre']}: bad WIGOS ID '{row['WIGOS ID']}'")
                        prog_total.value = i
                        continue
                    total_days = (pd.to_datetime(end_picker.value) - pd.to_datetime(start_picker.value)).days + 1
                    prog_station.max = (total_days + 29) // 30
                    prog_station.value = 0
                    with output:
                        df_raw = download_synop(code, start_picker.value, end_picker.value,
                                                progress=prog_station, cancel_flag=cancel_flag)
                    if df_raw is not None:
                        try:
                            df_proc = process_all_meteorological_variables(df_raw)
                            df_proc.to_csv(os.path.join(folder, f"series_{base}.csv"), index=False)
                            pd.DataFrame([row]).to_csv(os.path.join(folder, f"station_{base}.csv"), index=False)
                        except Exception as exc:
                            with output:
                                print(f"Processing error {base}: {exc}")
                    else:
                        with output:
                            print(f"No data for {row['Nombre']} (code {code})")
                    prog_total.value = i
                prog_label.value = f"<b>Download complete.</b> Files saved to <code>{folder}</code>"
            except Exception as exc:
                with output:
                    print(f"Fatal download error: {exc}")
                prog_label.value = f"<span style='color:red'>Error: {exc}</span>"

        download_thread = threading.Thread(target=_run, daemon=True)
        download_thread.start()

    download_area_btn.on_click(start_download)
    display(VBox([m, HBox([start_picker, end_picker]), folder_input,
                  HBox([download_area_btn, selection_label]), output]))


class OgimetCSVLoader:
    """Load and quality-check OGIMET series downloaded with OGIMETDownloader."""

    def __init__(self, folder_path=None, create=True):
        self.folder_path = str(Path(folder_path) if folder_path else DEFAULT_OGIMET_OUTPUT_DIR)
        if create:
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        self.station_df = None
        self.series_df = None

    def load_station_data(self):
        """Load all station metadata CSVs into a single DataFrame."""
        if not os.path.isdir(self.folder_path):
            raise FileNotFoundError(
                f"OGIMET folder does not exist: {self.folder_path}. "
                "Run OGIMETDownloader(output_folder=...) first or pass create=True."
            )
        files = [f for f in os.listdir(self.folder_path) if f.startswith("station_") and f.endswith(".csv")]
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_csv(os.path.join(self.folder_path, f)))
            except Exception as exc:
                print(f"Error reading {f}: {exc}")
        self.station_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return self.station_df

    def load_series_data(self, variable):
        """
        Load a specific variable from all series_*.csv files.

        Args:
            variable: Column name to extract (e.g. 'precipitation_mm')
        """
        if variable is None:
            raise ValueError("Specify a variable to load.")
        if not os.path.isdir(self.folder_path):
            raise FileNotFoundError(
                f"OGIMET folder does not exist: {self.folder_path}. "
                "Run OGIMETDownloader(output_folder=...) first or pass create=True."
            )
        files = [f for f in os.listdir(self.folder_path) if f.startswith("series_") and f.endswith(".csv")]
        dfs = []
        for f in files:
            path = os.path.join(self.folder_path, f)
            try:
                df = pd.read_csv(path, parse_dates=["date"])
                sid = f.replace("series_", "").replace(".csv", "")
                if variable in df.columns:
                    dfs.append(df[["date", variable]].rename(columns={variable: sid}).set_index("date"))
            except Exception as exc:
                print(f"Error reading {f}: {exc}")
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

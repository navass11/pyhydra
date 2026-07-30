"""
Meteostat downloader and Jupyter widget interface.

This module is designed as a drop-in style alternative to an OGIMET downloader,
but using the Meteostat Python package instead of scraping ogimet.com.

Main features
-------------
- Search nearby Meteostat stations from a map center.
- Draw a rectangle in Jupyter to select stations.
- Download daily or hourly meteorological series.
- Score stations by data coverage for a requested date range.
- Save one station metadata CSV and one series CSV per station.
- Load downloaded CSVs and analyze basic quality metrics.

Tested with Meteostat 2.1.x API style:
    from meteostat import stations, daily, hourly, Point

Install
-------
    pip install meteostat pandas ipyleaflet ipywidgets tqdm
"""

from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Paths and small utilities
# -----------------------------------------------------------------------------


def _resolve_output_dir() -> Path:
    """Resolve the default output folder for downloaded Meteostat data."""
    if os.environ.get("HYDRA_METEOSTAT_DIR"):
        return Path(os.environ["HYDRA_METEOSTAT_DIR"])
    if Path("/workspace").exists():
        return Path("/workspace/data/meteostat")
    return Path.home() / "hydra_data" / "meteostat"


DEFAULT_METEOSTAT_OUTPUT_DIR = _resolve_output_dir()


DAILY_VARIABLES = [
    "temp",  # mean air temperature, °C
    "tmin",  # minimum air temperature, °C
    "tmax",  # maximum air temperature, °C
    "rhum",  # relative humidity, %
    "prcp",  # precipitation, mm
    "snwd",  # snow depth, mm
    "wspd",  # wind speed, km/h
    "wpgt",  # peak wind gust, km/h
    "pres",  # sea-level pressure, hPa
    "tsun",  # sunshine duration, minutes
    "cldc",  # cloud cover, %
]

HOURLY_VARIABLES = [
    "temp",
    "dwpt",
    "rhum",
    "prcp",
    "snow",
    "wdir",
    "wspd",
    "wpgt",
    "pres",
    "tsun",
    "coco",
]


NORMALIZED_DAILY_RENAME = {
    "time": "date",
    "temp": "tmed_celsius",
    "tmin": "tmin_celsius",
    "tmax": "tmax_celsius",
    "rhum": "humidity_mean_percent",
    "prcp": "precipitation_mm",
    "snwd": "snow_depth_mm",
    "wspd": "wind_speed_kmh",
    "wpgt": "wind_gust_kmh",
    "pres": "pressure_msl_hpa",
    "tsun": "sunshine_minutes",
    "cldc": "cloud_cover_percent",
}

NORMALIZED_HOURLY_RENAME = {
    "time": "datetime",
    "temp": "temperature_celsius",
    "dwpt": "dewpoint_celsius",
    "rhum": "humidity_percent",
    "prcp": "precipitation_mm",
    "snow": "snow_depth_mm",
    "wdir": "wind_direction_deg",
    "wspd": "wind_speed_kmh",
    "wpgt": "wind_gust_kmh",
    "pres": "pressure_msl_hpa",
    "tsun": "sunshine_minutes",
    "coco": "weather_condition_code",
}


def normalize_filename(name: object) -> str:
    """Normalize a station name/code to a safe ASCII filename."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("utf-8")
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "station"


def _to_datetime(value) -> pd.Timestamp:
    if value is None:
        raise ValueError("Date value cannot be None")
    return pd.to_datetime(value)


def _safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Meteostat search and download functions
# -----------------------------------------------------------------------------


def get_meteostat_stations_nearby(
    lat: float,
    lon: float,
    radius: int = 150_000,
    limit: int = 100,
) -> pd.DataFrame:
    """
    Return Meteostat stations near a point as a DataFrame.

    Meteostat 2.1.x returns a DataFrame directly from stations.nearby(), so this
    function intentionally does not call .fetch() after nearby().
    """
    from meteostat import Point, stations

    point = Point(float(lat), float(lon))
    df = stations.nearby(point, radius=int(radius), limit=int(limit))
    if df is None:
        return pd.DataFrame()
    return df.copy()


def get_meteostat_station_catalog(limit: int = 100_000) -> pd.DataFrame:
    """Return the full Meteostat station catalog as a DataFrame."""
    from meteostat import stations

    try:
        df = stations.query(f"SELECT * FROM stations LIMIT {int(limit)}")
    except Exception:
        return pd.DataFrame()
    if df is None:
        return pd.DataFrame()
    return df.copy()


def find_station_by_wmo(wmo_code: str) -> pd.DataFrame:
    """Find stations by WMO code, e.g. '08021'."""
    from meteostat import stations

    wanted = str(wmo_code).zfill(5)
    try:
        df = stations.query(
            "SELECT * FROM stations WHERE wmo = ?", params=(wanted,)
        )
        if df is not None and not df.empty:
            return df.copy()
    except Exception:
        pass
    # Fallback: scan with LIKE
    try:
        df = stations.query("SELECT * FROM stations WHERE id = ?", params=(wanted,))
        if df is not None and not df.empty:
            return df.copy()
    except Exception:
        pass
    return pd.DataFrame()


def normalize_meteostat_series(
    df: pd.DataFrame,
    station_id: Optional[str] = None,
    station_name: Optional[str] = None,
    frequency: str = "daily",
) -> pd.DataFrame:
    """
    Normalize Meteostat output column names to stable HYDRA-style names.

    Args:
        df: DataFrame returned by meteostat.daily(...).fetch() or hourly(...).fetch().
        station_id: Meteostat station ID.
        station_name: Human station name.
        frequency: 'daily' or 'hourly'.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy().reset_index()

    if frequency == "hourly":
        rename = NORMALIZED_HOURLY_RENAME
        if "time" in out.columns:
            out["time"] = pd.to_datetime(out["time"], errors="coerce")
    else:
        rename = NORMALIZED_DAILY_RENAME
        if "time" in out.columns:
            out["time"] = pd.to_datetime(out["time"], errors="coerce").dt.date

    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    out["source"] = "meteostat"
    out["frequency"] = frequency
    out["station_id"] = station_id
    out["station_name"] = station_name
    return out


def download_meteostat_series(
    station_id: str,
    start_date,
    end_date,
    frequency: str = "daily",
    normalize: bool = True,
    station_name: Optional[str] = None,
):
    """
    Download one Meteostat station time series.

    Args:
        station_id: Meteostat station ID, e.g. '08221'.
        start_date: Start date.
        end_date: End date.
        frequency: 'daily' or 'hourly'.
        normalize: Whether to normalize column names.
        station_name: Optional station name for metadata columns.

    Returns:
        DataFrame or None if no data is available.
    """
    from meteostat import daily, hourly, config as mconfig

    start = _to_datetime(start_date).to_pydatetime()
    end = _to_datetime(end_date).to_pydatetime()

    if frequency not in {"daily", "hourly"}:
        raise ValueError("frequency must be 'daily' or 'hourly'")

    # Meteostat blocks requests longer than 30 years by default
    _old_block = getattr(mconfig, "block_large_requests", True)
    if (end - start).days > 30 * 365:
        mconfig.block_large_requests = False

    try:
        if frequency == "hourly":
            df = hourly(station_id, start, end).fetch()
        else:
            df = daily(station_id, start, end).fetch()
    except Exception as exc:
        print(f"[METEOSTAT] Error downloading {station_id}: {exc}")
        return None
    finally:
        mconfig.block_large_requests = _old_block

    if df is None or df.empty:
        return None

    if normalize:
        return normalize_meteostat_series(df, station_id=station_id, station_name=station_name, frequency=frequency)
    return df


def score_meteostat_stations(
    stations_df: pd.DataFrame,
    start_date,
    end_date,
    frequency: str = "daily",
    variables: Optional[Sequence[str]] = None,
    distance_weight: float = 0.10,
) -> pd.DataFrame:
    """
    Score candidate stations by variable coverage and distance.

    The score is primarily based on non-null coverage. A small distance penalty is
    applied so that a very far station only wins when it has much better coverage.
    """
    if stations_df is None or stations_df.empty:
        return pd.DataFrame()

    raw_vars = list(variables or (HOURLY_VARIABLES if frequency == "hourly" else DAILY_VARIABLES))
    rows = []

    for station_id, row in stations_df.iterrows():
        station_name = row.get("name", str(station_id))
        df = download_meteostat_series(
            station_id=str(station_id),
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            normalize=False,
            station_name=station_name,
        )

        if df is None or df.empty:
            continue

        total = len(df)
        non_null = df.notna().sum()
        result = {
            "station_id": str(station_id),
            "name": station_name,
            "country": row.get("country"),
            "region": row.get("region"),
            "latitude": row.get("latitude"),
            "longitude": row.get("longitude"),
            "elevation": row.get("elevation"),
            "timezone": row.get("timezone"),
            "distance_km": _safe_float(row.get("distance"), 0.0) / 1000.0,
            "n_records": total,
        }

        coverage_sum = 0.0
        for var in raw_vars:
            pct = 100.0 * non_null.get(var, 0) / total if total else 0.0
            result[f"{var}_pct"] = pct
            coverage_sum += pct

        distance_km = result["distance_km"] or 0.0
        result["coverage_score"] = coverage_sum
        result["score"] = coverage_sum - distance_weight * distance_km
        rows.append(result)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "distance_km"], ascending=[False, True]).reset_index(drop=True)


def select_best_meteostat_station(
    lat: float,
    lon: float,
    start_date,
    end_date,
    radius: int = 150_000,
    limit: int = 30,
    frequency: str = "daily",
    variables: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], pd.DataFrame, pd.DataFrame]:
    """Search nearby stations, score them and return the best station ID."""
    candidates = get_meteostat_stations_nearby(lat, lon, radius=radius, limit=limit)
    ranking = score_meteostat_stations(candidates, start_date, end_date, frequency=frequency, variables=variables)
    best = None if ranking.empty else ranking.iloc[0]["station_id"]
    return best, ranking, candidates


def download_best_meteostat_series(
    lat: float,
    lon: float,
    start_date,
    end_date,
    radius: int = 150_000,
    limit: int = 30,
    frequency: str = "daily",
    variables: Optional[Sequence[str]] = None,
    normalize: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[str], pd.DataFrame]:
    """Download the series from the best nearby station by coverage score."""
    best, ranking, _ = select_best_meteostat_station(
        lat=lat,
        lon=lon,
        start_date=start_date,
        end_date=end_date,
        radius=radius,
        limit=limit,
        frequency=frequency,
        variables=variables,
    )
    if best is None:
        return None, None, ranking

    station_name = ranking.loc[ranking["station_id"] == best, "name"].iloc[0]
    df = download_meteostat_series(
        station_id=best,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        normalize=normalize,
        station_name=station_name,
    )
    return df, best, ranking


# -----------------------------------------------------------------------------
# Jupyter widget
# -----------------------------------------------------------------------------


def MeteostatDownloader(
    zoom: int = 5,
    center: Tuple[float, float] = (40.0, -3.5),
    radius: int = 250_000,
    max_markers: int = 300,
    output_folder: Optional[str] = None,
):
    """
    Interactive Jupyter widget for selecting and downloading Meteostat stations.

    The widget loads stations near the initial map center. To change the station
    domain, modify the center/radius controls and click 'Reload stations'.

    Requires: ipyleaflet, ipywidgets, meteostat.
    """
    from IPython.display import display
    from ipyleaflet import DrawControl, LayersControl, Map, Marker, MarkerCluster, Popup
    from ipywidgets import (
        Button,
        Checkbox,
        DatePicker,
        Dropdown,
        FloatText,
        HBox,
        HTML,
        IntProgress,
        IntText,
        Output,
        Text,
        VBox,
    )

    output_folder = Path(output_folder or DEFAULT_METEOSTAT_OUTPUT_DIR)
    output_folder.mkdir(parents=True, exist_ok=True)

    output = Output()
    selected = set()
    cancel_flag = {"parar": False}
    state = {"stations_df": pd.DataFrame(), "download_thread": None}

    m = Map(center=center, zoom=zoom, scroll_wheel_zoom=True)
    m.layout.height = "520px"
    m.add_control(LayersControl())

    lat_input = FloatText(value=float(center[0]), description="Lat:", layout={"width": "170px"})
    lon_input = FloatText(value=float(center[1]), description="Lon:", layout={"width": "170px"})
    radius_input = IntText(value=int(radius), description="Radius m:", layout={"width": "190px"})
    limit_input = IntText(value=int(max_markers), description="Limit:", layout={"width": "150px"})

    start_picker = DatePicker(description="Start", value=pd.Timestamp("2020-01-01"))
    end_picker = DatePicker(description="End", value=pd.Timestamp("2020-12-31"))
    frequency_dropdown = Dropdown(
        options=[("Daily", "daily"), ("Hourly", "hourly")],
        value="daily",
        description="Freq:",
        layout={"width": "180px"},
    )
    normalize_checkbox = Checkbox(value=True, description="Normalize columns")

    folder_input = Text(
        value=str(output_folder),
        description="Save to:",
        layout={"width": "620px"},
        style={"description_width": "60px"},
    )

    reload_btn = Button(description="Reload stations", button_style="primary")
    score_btn = Button(description="Score selected", button_style="warning")
    download_area_btn = Button(description="Download selected", button_style="success")
    cancel_button = Button(description="Cancel", button_style="danger")
    cancel_button.on_click(lambda _: cancel_flag.update({"parar": True}))

    selection_label = HTML(value="<i>Draw a rectangle to select stations.</i>")
    stations_label = HTML(value="")

    marker_cluster = None

    def _station_base(row, station_id):
        return normalize_filename(f"{station_id}_{row.get('name', station_id)}")

    def _save_station_and_series(row, station_id, folder, frequency, normalize, progress=None):
        station_name = row.get("name", str(station_id))
        df = download_meteostat_series(
            station_id=str(station_id),
            start_date=start_picker.value,
            end_date=end_picker.value,
            frequency=frequency,
            normalize=normalize,
            station_name=station_name,
        )
        if progress:
            progress.value += 1
        if df is None or df.empty:
            return False, f"No data for {station_id} - {station_name}"

        base = _station_base(row, station_id)
        Path(folder).mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).assign(station_id=str(station_id)).to_csv(
            os.path.join(folder, f"station_{base}.csv"), index=False
        )
        df.to_csv(os.path.join(folder, f"series_{frequency}_{base}.csv"), index=False)
        return True, f"Saved {station_id} - {station_name}"

    def _make_marker(station_id, row):
        marker = Marker(location=(row["latitude"], row["longitude"]), draggable=False)

        def on_click(marker=marker, row=row, station_id=station_id, **kwargs):
            info = HTML(value=(
                f"<b>{row.get('name', station_id)}</b><br>"
                f"ID: {station_id}<br>"
                f"Country/region: {row.get('country', '')}/{row.get('region', '')}<br>"
                f"Elevation: {row.get('elevation', '')} m<br>"
                f"Distance: {_safe_float(row.get('distance'), 0) / 1000:.1f} km<br>"
                f"Lat/Lon: {row.get('latitude')}, {row.get('longitude')}"
            ))
            btn = Button(description="Download", button_style="info", layout={"width": "auto"})

            def download_single(_):
                if getattr(btn, "_downloading", False):
                    return
                btn._downloading = True
                btn.disabled = True
                folder = folder_input.value.strip() or str(output_folder)
                prog = IntProgress(value=0, min=0, max=1, description="Download:")
                status = HTML(value=f"<b>Downloading {row.get('name', station_id)}</b> → <code>{folder}</code>")
                display(VBox([status, prog]))

                def _do():
                    try:
                        ok, msg = _save_station_and_series(
                            row=row,
                            station_id=station_id,
                            folder=folder,
                            frequency=frequency_dropdown.value,
                            normalize=normalize_checkbox.value,
                            progress=prog,
                        )
                        status.value = f"<b>{msg}</b>" if ok else f"<span style='color:red'>{msg}</span>"
                    except Exception as exc:
                        with output:
                            print(f"Error downloading {station_id}: {exc}")
                        status.value = f"<span style='color:red'>Error: {exc}</span>"
                    finally:
                        btn._downloading = False
                        btn.disabled = False

                threading.Thread(target=_do, daemon=True).start()

            btn.on_click(download_single)
            marker.popup = Popup(location=marker.location, child=VBox([info, btn]), close_button=True, auto_close=False)

        marker.on_click(on_click)
        return marker

    def _reload_markers(_=None):
        nonlocal marker_cluster
        try:
            df = get_meteostat_stations_nearby(
                lat=lat_input.value,
                lon=lon_input.value,
                radius=radius_input.value,
                limit=limit_input.value,
            )
            df = df.dropna(subset=["latitude", "longitude"])
            state["stations_df"] = df
            selected.clear()

            if marker_cluster is not None:
                try:
                    m.remove_layer(marker_cluster)
                except Exception:
                    pass

            markers = [_make_marker(station_id, row) for station_id, row in df.iterrows()]
            marker_cluster = MarkerCluster(markers=markers)
            m.add_layer(marker_cluster)
            m.center = (lat_input.value, lon_input.value)
            stations_label.value = f"<b>{len(df)} stations loaded.</b>"
            selection_label.value = "<i>Draw a rectangle to select stations.</i>"
        except Exception as exc:
            stations_label.value = f"<span style='color:red'>Error loading stations: {exc}</span>"

    reload_btn.on_click(_reload_markers)

    draw_control = DrawControl(circle={}, polygon={}, polyline={}, circlemarker={})
    draw_control.rectangle = {"shapeOptions": {"color": "#6bc2e5", "fillOpacity": 0.3}}
    m.add_control(draw_control)

    def _on_draw(_, action, geo_json):
        if action != "created":
            return
        df = state.get("stations_df", pd.DataFrame())
        if df.empty:
            selection_label.value = "<i>No stations loaded.</i>"
            return
        try:
            coords = geo_json["geometry"]["coordinates"][0]
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)
            in_bounds = df[
                (pd.to_numeric(df["latitude"], errors="coerce") >= lat_min) &
                (pd.to_numeric(df["latitude"], errors="coerce") <= lat_max) &
                (pd.to_numeric(df["longitude"], errors="coerce") >= lon_min) &
                (pd.to_numeric(df["longitude"], errors="coerce") <= lon_max)
            ]
            selected.clear()
            selected.update(map(str, in_bounds.index.tolist()))
            selection_label.value = f"<b>{len(selected)} stations selected.</b>"
        except Exception as exc:
            selection_label.value = f"<span style='color:red'>Draw error: {exc}</span>"

    draw_control.on_draw(_on_draw)

    def _score_selected(_):
        df = state.get("stations_df", pd.DataFrame())
        if df.empty or not selected:
            with output:
                print("No selected stations. Draw a rectangle first.")
            return
        selected_df = df.loc[df.index.astype(str).isin(selected)]
        with output:
            print("Scoring selected stations...")
        ranking = score_meteostat_stations(
            selected_df,
            start_date=start_picker.value,
            end_date=end_picker.value,
            frequency=frequency_dropdown.value,
        )
        folder = folder_input.value.strip() or str(output_folder)
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = os.path.join(folder, f"selected_station_ranking_{frequency_dropdown.value}.csv")
        ranking.to_csv(path, index=False)
        with output:
            print(ranking.head(20))
            print(f"Ranking saved to: {path}")

    score_btn.on_click(_score_selected)

    def _download_selected(_):
        if state["download_thread"] and state["download_thread"].is_alive():
            return
        df = state.get("stations_df", pd.DataFrame())
        if df.empty or not selected:
            selection_label.value = "<i>Draw a rectangle to select stations first.</i>"
            return

        folder = folder_input.value.strip() or str(output_folder)
        frequency = frequency_dropdown.value
        normalize = normalize_checkbox.value
        cancel_flag["parar"] = False
        selected_df = df.loc[df.index.astype(str).isin(selected)]

        prog_total = IntProgress(value=0, min=0, max=len(selected_df), description="Total:")
        status = HTML(value=f"Starting download → <code>{folder}</code>")
        display(VBox([status, prog_total, cancel_button, output]))

        def _run():
            try:
                Path(folder).mkdir(parents=True, exist_ok=True)
                selected_df.assign(station_id=selected_df.index.astype(str)).to_csv(
                    os.path.join(folder, "selected_stations.csv"), index=False
                )
                for i, (station_id, row) in enumerate(selected_df.iterrows(), 1):
                    if cancel_flag.get("parar", False):
                        status.value = "<b>Download canceled.</b>"
                        break
                    status.value = f"<b>Station {i}/{len(selected_df)}: {row.get('name', station_id)}</b>"
                    ok, msg = _save_station_and_series(
                        row=row,
                        station_id=str(station_id),
                        folder=folder,
                        frequency=frequency,
                        normalize=normalize,
                    )
                    with output:
                        print(msg)
                    prog_total.value = i
                    time.sleep(0.05)
                else:
                    status.value = f"<b>Download complete.</b> Files saved to <code>{folder}</code>"
            except Exception as exc:
                with output:
                    print(f"Fatal download error: {exc}")
                status.value = f"<span style='color:red'>Error: {exc}</span>"

        state["download_thread"] = threading.Thread(target=_run, daemon=True)
        state["download_thread"].start()

    download_area_btn.on_click(_download_selected)

    controls = VBox([
        HBox([lat_input, lon_input, radius_input, limit_input, reload_btn]),
        HBox([start_picker, end_picker, frequency_dropdown, normalize_checkbox]),
        folder_input,
        HBox([download_area_btn, score_btn, selection_label]),
        stations_label,
    ])

    display(VBox([controls, m, output]))
    _reload_markers()


# -----------------------------------------------------------------------------
# CSV loader and quality checker
# -----------------------------------------------------------------------------


class MeteostatCSVLoader:
    """Load and quality-check Meteostat series downloaded with MeteostatDownloader."""

    def __init__(self, folder_path=None, create=True):
        self.folder_path = str(Path(folder_path) if folder_path else DEFAULT_METEOSTAT_OUTPUT_DIR)
        if create:
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)
        self.station_df = None
        self.series_df = None

    def load_station_data(self) -> pd.DataFrame:
        """Load all station metadata CSVs into a single DataFrame."""
        if not os.path.isdir(self.folder_path):
            raise FileNotFoundError(f"Meteostat folder does not exist: {self.folder_path}")

        files = [f for f in os.listdir(self.folder_path) if f.startswith("station_") and f.endswith(".csv")]
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_csv(os.path.join(self.folder_path, f)))
            except Exception as exc:
                print(f"Error reading {f}: {exc}")
        self.station_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return self.station_df

    def load_series_data(self, variable: str, frequency: Optional[str] = None) -> pd.DataFrame:
        """
        Load one variable from all series CSV files.

        Args:
            variable: Column name to extract, e.g. 'precipitation_mm'.
            frequency: Optional 'daily' or 'hourly' filter.
        """
        if not variable:
            raise ValueError("Specify a variable to load.")
        if not os.path.isdir(self.folder_path):
            raise FileNotFoundError(f"Meteostat folder does not exist: {self.folder_path}")

        files = [f for f in os.listdir(self.folder_path) if f.startswith("series_") and f.endswith(".csv")]
        if frequency:
            files = [f for f in files if f.startswith(f"series_{frequency}_")]

        dfs = []
        for f in files:
            path = os.path.join(self.folder_path, f)
            try:
                df = pd.read_csv(path)
                date_col = "date" if "date" in df.columns else "datetime" if "datetime" in df.columns else None
                if date_col is None or variable not in df.columns:
                    continue
                station_label = f.replace("series_daily_", "").replace("series_hourly_", "").replace(".csv", "")
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                dfs.append(df[[date_col, variable]].rename(columns={variable: station_label}).set_index(date_col))
            except Exception as exc:
                print(f"Error reading {f}: {exc}")

        self.series_df = pd.concat(dfs, axis=1).sort_index() if dfs else pd.DataFrame()
        return self.series_df

    def analyze_series_quality(self) -> pd.DataFrame:
        """Return quality metrics per downloaded station series."""
        if self.series_df is None or self.series_df.empty:
            raise ValueError("Load data first with load_series_data().")

        rows = []
        for station in self.series_df.columns:
            s = self.series_df[station]
            total = len(s)
            valid = s.notna().sum()
            missing_pct = 100 * (1 - valid / total) if total > 0 else 100.0
            try:
                full_years = sum(len(g) >= 365 for _, g in s.dropna().groupby(s.dropna().index.year))
                full_months = sum(len(g) >= 28 for _, g in s.dropna().groupby(s.dropna().index.to_period("M")))
            except Exception:
                full_years = full_months = 0
            rows.append({
                "station_id": station,
                "start_year": int(s.index.year.min()) if total else None,
                "end_year": int(s.index.year.max()) if total else None,
                "records": int(total),
                "valid_records": int(valid),
                "missing_percent": round(float(missing_pct), 2),
                "full_years": int(full_years),
                "full_months": int(full_months),
                "min_value": s.min(skipna=True),
                "max_value": s.max(skipna=True),
            })
        return pd.DataFrame(rows)

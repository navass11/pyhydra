"""
Author: Salvador Navas
Date: 2025-06-27
"""

from ipyleaflet import Map, Marker, MarkerCluster, DrawControl, LayersControl, Popup
from ipywidgets import VBox, Output, Button, HTML, DatePicker, HBox, IntProgress
from ipyfilechooser import FileChooser
from IPython.display import display
import pandas as pd
from tqdm.auto import tqdm
import requests
from bs4 import BeautifulSoup
from io import StringIO
import time
from datetime import datetime, timedelta
import re
import unicodedata
import threading
import os

def normalize_filename(name):
    name = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode('utf-8')
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def download_synop(station_id, start_date, end_date, progress=None, cancel_flag=None):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    date = end_date
    all_data = []

    while date >= start_date:
        if cancel_flag and cancel_flag.get('parar', False):
            print(f"⛔ Canceled while downloading {station_id}")
            return None

        ndays = min(30, (date - start_date).days + 1)
        url = (
            f"https://www.ogimet.com/cgi-bin/gsynres?"
            f"ind={station_id}&ndays={ndays}&ano={date.year}&mes={date.month}&day={date.day}&hora=0&ord=REV&enviar=Ver"
        )

        try:
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            tables = pd.read_html(StringIO(str(soup)))

            if len(tables) > 2:
                df = tables[2]
                date_col = [col for col in df.columns if 'Fecha' in str(col)]
                if date_col:
                    col_date = date_col[0]
                    df[col_date] = df[col_date].astype(str).str.strip() + f"/{date.year}"
                    df[col_date] = pd.to_datetime(df[col_date], format="%d/%m/%Y", errors='coerce')
                    df = df.sort_values(by=col_date)
                    all_data.append(df)
        except Exception as e:
            print(f"❌ Error downloading {station_id}: {e}")

        date -= pd.Timedelta(days=ndays)
        time.sleep(0.5)

        if progress:
            progress.value += 1

    return pd.concat(all_data, ignore_index=True) if all_data else None

def process_all_meteorological_variables(df):
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected a DataFrame with MultiIndex columns.")

    df = df[[col for col in df.columns if col[0] != 'Diario meteorológico']].copy()
    date_col = [col for col in df.columns if 'Fecha' in col[0]][0]
    df.loc[:, date_col] = pd.to_datetime(df[date_col], errors='coerce')

    direction_to_degrees = {
        'N': 0.0, 'NNE': 22.5, 'NE': 45.0, 'ENE': 67.5, 'E': 90.0, 'ESE': 112.5,
        'SE': 135.0, 'SSE': 157.5, 'S': 180.0, 'SSW': 202.5, 'SW': 225.0, 'WSW': 247.5,
        'W': 270.0, 'WNW': 292.5, 'NW': 315.0, 'NNW': 337.5
    }

    numeric_cols = [col for col in df.columns if col != date_col]
    for col in numeric_cols:
        if 'viento' in col[0].lower() and 'dir' in col[1].lower():
            df.loc[:, col] = df[col].map(lambda x: direction_to_degrees.get(str(x).strip().upper(), None))
        elif 'prec' in col[0].lower() or 'prec' in col[1].lower():
            df.loc[:, col] = df[col].apply(lambda x: 0.1 if isinstance(x, str) and x.strip().lower() == 'ip' else x)
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    agg_dict = {
        col: (lambda x: x.sum(min_count=1)) if 'prec' in col[0].lower() or 'prec' in col[1].lower()
        else 'mean'
        for col in numeric_cols
    }

    grouped = df.groupby(df[date_col].dt.date).agg(agg_dict)

    def clean_column_name(col):
        parts = list(filter(None, map(str, col)))
        unique = []
        for p in parts:
            clean = (
                p.strip().lower().replace(' ', '_').replace('.', '').replace('%', 'pct')
                .replace('(', '').replace(')', '').replace('-', '_').replace('/', '_')
            )
            if clean not in unique:
                unique.append(clean)
        return '_'.join(unique)

    grouped.columns = [clean_column_name(col) for col in grouped.columns]
    grouped = grouped.reset_index()
    grouped.rename(columns={grouped.columns[0]: 'date'}, inplace=True)

    rename_dict = {
        'temperatura_c_max': 'tmax_celsius',
        'temperatura_c_min': 'tmin_celsius',
        'temperatura_c_med': 'tmed_celsius',
        'td_med_c_td_med_c': 'dewpoint_mean_celsius',
        'hr_med_pct_hr_med_pct': 'humidity_mean_percent',
        'viento_kmh_dir': 'wind_direction_deg',
        'viento_kmh_vel': 'wind_speed_kmh',
        'pres_n_mar_hp_pres_n_mar_hp': 'pressure_msl_hpa',
        'prec_mm_prec_mm': 'precipitation_mm',
        'nub_tot_oct_nub_tot_oct': 'cloud_total_oktas',
        'nub_baj_oct_nub_baj_oct': 'cloud_low_oktas',
        'sol_d_1_h_sol_d_1_h': 'sun_hours_day_before',
        'vis_km_vis_km': 'visibility_km'
    }

    grouped.rename(columns={k: v for k, v in rename_dict.items() if k in grouped.columns}, inplace=True)
    return grouped

def OGIMETDownloader(zoom=5, center=(40.0, -3.5), max_markers=None):
    stations_df = pd.read_csv('../HYDRA/data/estaciones_ogimet_all.csv')
    stations_df = stations_df[~stations_df['WIGOS ID'].str.contains('MISSING', case=False, na=False)]
    output = Output()
    map_widget = Map(center=center, zoom=zoom, scroll_wheel_zoom=True)
    selected = set()

    start_date_picker = DatePicker(description="Start Date", value=pd.Timestamp("2020-01-01"))
    end_date_picker = DatePicker(description="End Date", value=pd.Timestamp("2020-12-31"))
    folder_chooser = FileChooser('.', title='Select output folder', select_dirs=True)

    cancel_flag = {'parar': False}
    cancel_button = Button(description="❌ Cancel Download", button_style='danger')

    def cancel_download(b):
        cancel_flag['parar'] = True
        with output:
            print("🚫 Download canceled by user.")

    cancel_button.on_click(cancel_download)

    markers = []
    if max_markers:
        stations_df = stations_df.sample(max_markers, random_state=0)

    for _, row in tqdm(stations_df.iterrows(), total=len(stations_df), desc="Loading stations"):
        lat = row['Latitud_decimal']
        lon = row['Longitud_decimal']
        name = row['Nombre']
        marker = Marker(location=(lat, lon), draggable=False)

        def on_click_handler(marker=marker, row=row, **kwargs):
            if not marker.popup:
                info = f"""
                <b><u>{row['Nombre']}</u></b><br>
                <b>WIGOS ID:</b> {row['WIGOS ID']}<br>
                <b>ICAO:</b> {row['OACI']}<br>
                <b>Status:</b> {row['Estado']}<br>
                <b>Lat/Lon:</b> {row['Latitud_decimal']}, {row['Longitud_decimal']}<br>
                <b>Altitude:</b> {row['Altitud']} m<br>
                """
                html_widget = HTML(value=info)
                download_button = Button(description="⬇ Download", button_style='info', layout={'width': 'auto'})

                def download_single(b, row=row):
                    folder = folder_chooser.selected_path
                    if not folder:
                        with output:
                            output.clear_output()
                            print("⚠️ You must select an output folder.")
                        return

                    start = pd.to_datetime(start_date_picker.value)
                    end = pd.to_datetime(end_date_picker.value)

                    if pd.isna(start) or pd.isna(end):
                        with output:
                            output.clear_output()
                            print("⚠️ Please select a valid date range.")
                        return

                    name = row['Nombre']
                    base_name = normalize_filename(name)
                    meta_file = f"{folder}/station_{base_name}.csv"
                    data_file = f"{folder}/series_{base_name}.csv"
                    pd.DataFrame([row]).to_csv(meta_file, index=False)

                    wigos = str(row['WIGOS ID'])
                    parts = wigos.split('-')
                    station_code = parts[-1].strip() if len(parts) >= 4 and parts[-1].strip().isdigit() else None

                    if not station_code:
                        with output:
                            output.clear_output()
                            print(f"⚠️ Invalid WIGOS code for '{name}': '{wigos}'")
                        return

                    total_days = (end - start).days + 1
                    blocks = (total_days + 29) // 30
                    progress = IntProgress(value=0, min=0, max=blocks, description='Download:', bar_style='info')
                    display(VBox([progress]))

                    df_raw = download_synop(
                        station_code, start, end,
                        progress=progress,
                        cancel_flag=cancel_flag
                    )

                    if df_raw is not None:
                        try:
                            df_proc = process_all_meteorological_variables(df_raw)
                            df_proc.to_csv(data_file, index=False)
                            with output:
                                output.clear_output()
                                print(f"✅ Series saved as '{data_file}'")
                        except Exception as e:
                            with output:
                                output.clear_output()
                                print(f"❌ Error processing '{base_name}': {e}")
                    else:
                        with output:
                            output.clear_output()
                            print(f"⚠️ Failed to download data for {base_name}")

                download_button.on_click(lambda b, row=row: download_single(b, row=row))
                marker.popup = Popup(location=marker.location, child=VBox([html_widget, download_button]), close_button=True, auto_close=False)

        marker.on_click(on_click_handler)
        markers.append(marker)

    cluster = MarkerCluster(markers=markers)
    map_widget.add_layer(cluster)

    draw_control = DrawControl(circle={}, polygon={}, polyline={}, circlemarker={})
    draw_control.rectangle = {"shapeOptions": {"color": "#6bc2e5", "fillOpacity": 0.3}}

    def handle_draw(self, action, geo_json):
        if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
            with output:
                output.clear_output()
                coords = geo_json['geometry']['coordinates'][0]
                lons = [pt[0] for pt in coords]
                lats = [pt[1] for pt in coords]
                west, east = min(lons), max(lons)
                south, north = min(lats), max(lats)

                selected_area = stations_df[
                    (stations_df['Latitud_decimal'] >= south) & (stations_df['Latitud_decimal'] <= north) &
                    (stations_df['Longitud_decimal'] >= west) & (stations_df['Longitud_decimal'] <= east)
                ]
                new = 0
                for name in selected_area['Nombre']:
                    if name not in selected:
                        selected.add(name)
                        new += 1

                print(f"📦 Stations selected by area: {new} (total: {len(selected)})")

    draw_control.on_draw(handle_draw)
    map_widget.add_control(draw_control)
    map_widget.add_control(LayersControl())

    download_area_button = Button(description="⬇ Download Selected Area", button_style='success')

    download_thread = None

    def download_selected():
        cancel_flag['parar'] = False
        folder = folder_chooser.selected_path
        if not folder:
            with output:
                output.clear_output()
                print("⚠️ You must select an output folder.")
            return

        start = pd.to_datetime(start_date_picker.value)
        end = pd.to_datetime(end_date_picker.value)

        if pd.isna(start) or pd.isna(end):
            with output:
                output.clear_output()
                print("⚠️ Please select a valid date range.")
            return

        selected_df = stations_df[stations_df['Nombre'].isin(selected)]
        total_stations = len(selected_df)
        selected_df.to_csv(f'{folder}/selected_stations.csv', index=False)

        with output:
            output.clear_output()

        progress_label = HTML(value="📊 Starting download...")
        progress_total = IntProgress(value=0, min=0, max=total_stations, description='Total:', bar_style='info')
        progress_station = IntProgress(value=0, min=0, max=1, description='Station:', bar_style='info')
        display(VBox([progress_label, progress_total, progress_station, cancel_button]))

        for i, (_, row) in enumerate(selected_df.iterrows(), start=1):
            if cancel_flag['parar']:
                with output:
                    output.clear_output()
                    print("🛑 Download interrupted by user.")
                break

            name = row['Nombre']
            progress_label.value = f"<b>📊 Station {i} of {total_stations}:</b> {name}"

            base_name = normalize_filename(name)
            meta_file = f"{folder}/station_{base_name}.csv"
            data_file = f"{folder}/series_{base_name}.csv"
            pd.DataFrame([row]).to_csv(meta_file, index=False)

            wigos = str(row['WIGOS ID'])
            parts = wigos.split('-')
            station_code = parts[-1].strip() if len(parts) >= 4 and parts[-1].strip().isdigit() else None

            if not station_code:
                with output:
                    output.clear_output(wait=True)
                    print(f"⚠️ Invalid WIGOS code for '{name}': '{wigos}'")
                progress_total.value = i
                continue

            total_days = (end - start).days + 1
            blocks = (total_days + 29) // 30
            progress_station.max = blocks
            progress_station.value = 0

            df_raw = download_synop(
                station_code, start, end,
                progress=progress_station,
                cancel_flag=cancel_flag
            )

            if df_raw is not None:
                try:
                    df_proc = process_all_meteorological_variables(df_raw)
                    df_proc.to_csv(data_file, index=False)
                    with output:
                        output.clear_output(wait=True)
                        print(f"✅ Series saved as '{data_file}'")
                except Exception as e:
                    with output:
                        output.clear_output(wait=True)
                        print(f"❌ Error processing '{base_name}': {e}")
            else:
                with output:
                    output.clear_output(wait=True)
                    print(f"⚠️ Failed to download data for {base_name}")

            progress_total.value = i

    def start_download(b):
        nonlocal download_thread
        if download_thread and download_thread.is_alive():
            with output:
                print("⏳ A download is already in progress.")
            return

        download_thread = threading.Thread(target=download_selected)
        download_thread.start()

    download_area_button.on_click(start_download)

    display(VBox([
        map_widget,
        HBox([start_date_picker, end_date_picker]),
        folder_chooser,
        download_area_button,
        output
    ]))

class OgimetCSVLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.station_df = None
        self.series_df = None

    def load_station_data(self):
        station_files = [f for f in os.listdir(self.folder_path)
                         if f.startswith("station_") and f.endswith(".csv")]
        dfs = []
        for file in station_files:
            path = os.path.join(self.folder_path, file)
            try:
                df = pd.read_csv(path)
                df['source_file'] = file
                dfs.append(df)
            except Exception as e:
                print(f"❌ Error reading {file}: {e}")
        self.station_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return self.station_df

    def load_series_data(self, variable):
        if variable is None:
            raise ValueError("You must specify a variable to load from the series CSV files.")

        series_files = [f for f in os.listdir(self.folder_path)
                        if f.startswith("series_") and f.endswith(".csv")]
        dfs = []
        for file in series_files:
            path = os.path.join(self.folder_path, file)
            try:
                df = pd.read_csv(path, parse_dates=['date'])
                station_id = file.replace("series_", "").replace(".csv", "")
                if variable in df.columns:
                    df = df[['date', variable]]
                    df = df.rename(columns={variable: station_id})
                    df = df.set_index('date')
                    dfs.append(df)
                else:
                    print(f"⚠️ Variable '{variable}' not found in {file}. Skipping this file.")
            except Exception as e:
                print(f"❌ Error reading {file}: {e}")

        if dfs:
            combined = pd.concat(dfs, axis=1)
            combined.index = pd.to_datetime(combined.index)
            combined.sort_index(inplace=True)
            self.series_df = combined
        else:
            self.series_df = pd.DataFrame()

        return self.series_df

    def analyze_series_quality(self):
        if self.series_df is None or self.series_df.empty:
            raise ValueError("No series data available. Please load data first.")

        summary_list = []
        for station in self.series_df.columns:
            series = self.series_df[station]
            valid = series.notna()
            total = len(series)
            missing_pct = 100 * (1 - valid.sum() / total) if total > 0 else 100.0

            full_years = 0
            full_months = 0
            try:
                grouped_years = series.dropna().groupby(series.dropna().index.year)
                full_years = sum(len(g) >= 365 for _, g in grouped_years)

                grouped_months = series.dropna().groupby(series.dropna().index.to_period('M'))
                full_months = sum(len(g) >= 28 for _, g in grouped_months)
            except Exception as e:
                print(f"⚠️ Error in grouping series for station {station}: {e}")

            summary_list.append({
                'station_id': station,
                'start_year': series.index.year.min(),
                'end_year': series.index.year.max(),
                'missing_percent': round(missing_pct, 2),
                'full_years': full_years,
                'full_months': full_months,
                'min_value': series.min(skipna=True),
                'max_value': series.max(skipna=True),
            })

        return pd.DataFrame(summary_list)

    def summary(self):
        print(f"📂 Folder: {self.folder_path}")
        print(f"📊 Stations: {len(self.station_df) if self.station_df is not None else 'Not loaded'} rows")
        print(f"📈 Series: {self.series_df.shape if self.series_df is not None else 'Not loaded'}")


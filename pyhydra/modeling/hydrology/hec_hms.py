"""
HEC-HMS automation utilities.

Cross-platform support:
    - **Windows**: runs HEC-HMS via the embedded Jython API
      (``from hms.model import Project``).
    - **Linux / Docker**: calls ``xvfb-run <hms_dir>/hec-hms.sh -script <script>``
      for headless execution. Set the environment variable ``HEC_HMS_DIR``
      to the HEC-HMS installation directory (e.g. /workspace/data/hms/HEC-HMS-4.13),
      or pass ``hms_dir`` explicitly to :func:`run_hms_script` / :class:`HMSModel`.

Requires:
    - HEC-HMS 4.x installed (Windows or Linux binary).
    - hecdss: ``pip install hecdss``  (DSS read/write, numpy-version-agnostic)
    - xvfb (Linux only, for headless display): ``apt-get install xvfb``
    - spotpy (calibration only): ``pip install spotpy``
    - rasterio, rasterstats, geopandas (parameter extraction only).

Workflow overview:
    1. Read existing model components (read_* functions).
    2. Extract basin parameters (CN, Clark, SMA, routing) from GIS rasters.
    3. Generate / update input files (generate_*, fill_gage).
    4. Execute the model (generate_py → run_hms_script, or HMSModel).
    5. Extract results (generate_flow, HMSModel).
    6. Climate-change scenario loop (fill_gage → generate_run per scenario).
"""

from __future__ import annotations

import os
import inspect
import re
import shutil
import struct
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pure-Python DSS v6 initialiser — no pydsstools / Java required
# ---------------------------------------------------------------------------

def _init_dss6(path: str | Path) -> None:
    """Create a valid, empty HEC-DSS version-6 file.

    The binary layout is derived from a known-good empty DSS v6 file
    (Calibracion1.dss, 51 pages × 512 bytes = 26 112 bytes).  Includes
    block-size configuration fields that allow pathnames up to 127 chars.

    Args:
        path: Destination file path.  Any existing file is overwritten.
    """
    path = Path(path)
    data = bytearray(26112)

    # Non-zero bytes extracted from a valid empty DSS v6 file
    _PATCH = [
        (0, 0x5a), (1, 0x44), (2, 0x53), (3, 0x53),  # magic: ZDSS
        (12, 0x06),                                     # DSS version 6
        # Library version (pydsstools 3.x expects "6-YO")
        (16, 0x36), (17, 0x2d), (18, 0x59), (19, 0x4f),  # "6-YO"
        # Block-size configuration (controls max pathname length ~127)
        (57, 0x08), (60, 0x01),
        (64, 0x20), (68, 0x20), (72, 0x7f),
        (76, 0xef), (77, 0x08), (80, 0xef), (81, 0x08),
        (176, 0x64),
        (180, 0xc8), (184, 0xc8), (188, 0xc8), (192, 0xc8), (196, 0xc8),
        # Free-space sentinels
        (508, 0x62), (509, 0xda), (510, 0xff), (511, 0xff),
        (8708, 0x62), (8709, 0xda), (8710, 0xff), (8711, 0xff),
        (25600, 0x62), (25601, 0xda), (25602, 0xff), (25603, 0xff),
    ]
    for off, b in _PATCH:
        data[off] = b
    path.write_bytes(data)

warnings.filterwarnings("ignore")


def _ensure_writable_hms_model(path_model: str | Path) -> str:
    """Return a writable HMS project directory for generated files.

    Azure/Jupyter mounts shared datasets below /workspace/data as read-only.
    Older notebooks may still pass those shared paths to generate_* helpers.
    In that case, mirror the project into the user's runtime directory and
    update the caller notebook's PATH_MODEL global so subsequent cells follow.
    """
    original = Path(path_model)
    try:
        resolved = original.resolve()
    except OSError:
        resolved = original

    workspace_data = Path("/workspace/data")
    try:
        rel = resolved.relative_to(workspace_data)
    except ValueError:
        return str(original)

    runtime_root = Path(os.environ.get("HYDRA_RUNTIME_DIR", Path.cwd() / ".hydra_runtime"))
    target = runtime_root / rel

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(resolved, target)

    writable = str(target) + os.sep

    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame and frame.f_back and frame.f_back.f_back else None
    if caller is not None and caller.f_globals.get("PATH_MODEL") == str(path_model):
        caller.f_globals["PATH_MODEL"] = writable

    return writable

# ── Read helpers ──────────────────────────────────────────────────────────────

def read_gages(path_model: str, file_gage: str) -> list[str]:
    """Return gage names defined in a .gage file."""
    txt = Path(path_model, file_gage).read_text()
    return [re.split(r"[ ]", t)[1] for t in re.findall(r"Gage: [\w\d]+(?=\s)", txt)]


def read_met(path_model: str, file_hms: str) -> list[str]:
    """Return meteorological model names defined in a .hms file."""
    txt = Path(path_model, file_hms).read_text()
    return [re.split(r"[.]", t)[0] for t in re.findall(r"[\w]+\.met", txt)]


def read_basin(path_model: str, file_basin: str) -> list[str]:
    """Return basin names defined in a .basin file."""
    txt = Path(path_model, file_basin).read_text()
    return [re.split(r"[ ]", t)[1] for t in re.findall(r"Basin: [-\w]+", txt)]


def read_subbasin(path_model: str, file_basin: str) -> list[str]:
    """Return sub-basin names defined in a .basin file."""
    txt = Path(path_model, file_basin).read_text()
    return [re.split(r"[ ]", t)[1] for t in re.findall(r"Subbasin: [-\w]+", txt)]


def read_control(path_model: str, file_hms: str) -> list[str]:
    """Return control names defined in a .hms file."""
    txt = Path(path_model, file_hms).read_text()
    return [re.split(r"[.]", t)[0] for t in re.findall(r"[\w]+\.control", txt)]


def read_run(path_model: str, file_run: str) -> list[str]:
    """Return run names defined in a .run file."""
    txt = Path(path_model, file_run).read_text()
    return [re.split(r"[:]\s", t)[1] for t in re.findall(r"Run:[ \w]+", txt)]


# ── File generators ───────────────────────────────────────────────────────────

def generate_gage(
    name_model: str,
    names_stations: list[str],
    time_interval: str,
    path_model: str,
    start_time: str,
    end_time: str,
    file_dss: str,
    exists_gage: bool = False,
) -> None:
    """Write or append precipitation gage entries to a .gage file.

    Args:
        name_model: Project name (without extension).
        names_stations: List of gage/station names.
        time_interval: HEC-HMS time step string (e.g. '1HOUR', '1DAY', '5MIN').
        path_model: Directory containing the model files.
        start_time: Simulation start (e.g. '1 January 2010, 00:00').
        end_time: Simulation end.
        file_dss: Name of the DSS file storing precipitation data.
        exists_gage: If True, append to the existing .gage file.
    """
    path_model = _ensure_writable_hms_model(path_model)

    def _gage_block(station: str) -> list[str]:
        return [
            f"Gage: {station}\n",
            f"     Description: Precipitation series — {station}\n",
            "     Last Modified Date: 13 November 2020\n",
            "     Last Modified Time: 09:21:23\n",
            "     Reference Height Units: Meters\n",
            "     Reference Height: 10.0\n",
            "     Gage Type: Precipitation\n",
            "     Precipitation Type: Incremental\n",
            "     Units: MM\n",
            "     Data Type: PER-CUM\n",
            "     Data Source Type: External DSS\n",
            "     Variant: Variant-1\n",
            "       Last Variant Modified Date: 13 November 2020\n",
            "       Last Variant Modified Time: 09:07:39\n",
            "       Default Variant: Yes\n",
            f"       DSS File Name: {file_dss}\n",
            f"       DSS Pathname: //{station}/PRECIP-INC//{time_interval}/GAGE/\n",
            f"       Start Time: {start_time}\n",
            f"       End Time: {end_time}\n",
            "     End Variant: Variant-1\n",
            "End:\n",
            "\n",
        ]

    gage_path = Path(path_model, name_model + ".gage")
    header = (
        []
        if exists_gage
        else [
            f"Gage Manager:{name_model}\n",
            "     Version: 4.9\n",
            "     Filepath Separator: /\n",
            "End:\n",
            "\n",
        ]
    )
    blocks = [line for s in names_stations for line in _gage_block(s)]
    existing = gage_path.read_text().splitlines(keepends=True) if exists_gage and gage_path.exists() else []
    gage_path.write_text("".join(existing + header + blocks))
    print(f"✓ {gage_path.name} written ({len(names_stations)} gages).")


def _fmt_dss_time(s: str) -> str:
    """Convert '1 January 2000, 00:00' → '01JAN2000 0000' (DSS date format)."""
    _MON = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    dt = datetime.strptime(s.strip(), "%d %B %Y, %H:%M")
    return f"{dt.day:02d}{_MON[dt.month-1]}{dt.year} {dt.hour:02d}{dt.minute:02d}"


def _dss_epart(minutes: int) -> str:
    """Return DSS E-part string for a given timestep in minutes (e.g. '1HOUR', '30MIN')."""
    if minutes >= 60 and minutes % 60 == 0:
        return f"{minutes // 60}HOUR"
    return f"{minutes}MIN"


def _hecdss_interval(epart: str) -> str:
    """Convert a DSS E-part string to hecdss interval format ('1Hour', '30Min', etc.)."""
    import re as _re
    m = _re.fullmatch(r"(\d+)(HOUR|MIN|SEC|DAY|WEEK|MONTH|YEAR)", epart.upper())
    if not m:
        return epart
    n, unit = m.group(1), m.group(2)
    label = {"HOUR": "Hour", "MIN": "Min", "SEC": "Sec",
             "DAY": "Day", "WEEK": "Week", "MONTH": "Month", "YEAR": "Year"}[unit]
    return f"{n}{label}"


def _epart_to_timedelta(epart: str):
    """Return timedelta for a DSS E-part string (e.g. '1HOUR' → timedelta(hours=1))."""
    import re as _re
    from datetime import timedelta as _td
    m = _re.fullmatch(r"(\d+)(HOUR|MIN|SEC|DAY)", epart.upper())
    if not m:
        return _td(hours=1)
    n, unit = int(m.group(1)), m.group(2)
    return {"HOUR": _td(hours=n), "MIN": _td(minutes=n), "SEC": _td(seconds=n),
            "DAY": _td(days=n)}[unit]


def fill_gage(
    names_stations: list[str],
    path_rain: str,
    time_interval: str,
    path_model: str,
    file_dss: str,
    start_time: str,
    end_time: str,
) -> None:
    """Write precipitation time series into the DSS file.

    Uses hecdss (numpy-version-agnostic) to write the time series.

    Args:
        names_stations: Station names matching columns in the rainfall CSV.
        path_rain: Path to the CSV with a datetime index and one column per station.
        time_interval: HEC-HMS E-part interval string (e.g. '1HOUR', '30MIN').
        path_model: Directory containing the model and DSS file.
        file_dss: Name of the DSS file.
        start_time: Start datetime string (e.g. '1 January 2010, 00:00').
        end_time: End datetime string.
    """
    path_model = _ensure_writable_hms_model(path_model)

    try:
        from hecdss import HecDss, RegularTimeSeries
        from hecdss.native import _Native
        _Native()
    except (ImportError, OSError) as exc:
        raise ImportError(f"fill_gage requires working hecdss library: {exc}") from exc

    rain = pd.read_csv(path_rain, index_col=0, parse_dates=True)
    t0 = datetime.strptime(start_time, "%d %B %Y, %H:%M")
    t1 = datetime.strptime(end_time, "%d %B %Y, %H:%M")
    dss_path_str = str(Path(path_model, file_dss))
    dss_hec_start = _fmt_dss_time(start_time)
    dss_d_part = dss_hec_start[:9]
    hecdss_int = _hecdss_interval(time_interval)
    dt_step = _epart_to_timedelta(time_interval)

    for station in names_stations:
        series = rain.loc[t0:t1, station].values
        pathname = f"//{station}/PRECIP-INC/{dss_d_part}/{time_interval}/GAGE/"
        times = [t0 + dt_step * i for i in range(len(series))]
        rts = RegularTimeSeries.create(
            values=list(series),
            times=times,
            units="MM",
            data_type="PER-CUM",
            interval=hecdss_int,
            path=pathname,
        )
        with HecDss(dss_path_str) as f:
            f.put(rts)
        print(f"  ✓ {station} written to DSS.")

    print("✓ DSS file updated with precipitation data.")


def fill_gage_series(
    name_station: str,
    values,
    start_time: str,
    time_interval: int,
    path_model: str,
    file_dss: str,
) -> None:
    """Write a single precipitation series (numpy array or list) directly to DSS.

    Uses hecdss (numpy-version-agnostic). Creates the DSS file if it does not exist.

    Args:
        name_station: Gage name used in the DSS pathname.
        values: Precipitation values in mm (numpy array or list).
        start_time: Start datetime string (e.g. '1 January 2000, 00:00').
        time_interval: Time step in minutes (int, e.g. 60 for hourly).
        path_model: Directory containing the DSS file.
        file_dss: Name of the DSS file (e.g. 'Project_1.dss').
    """
    path_model = _ensure_writable_hms_model(path_model)

    try:
        from hecdss import HecDss, RegularTimeSeries
        from hecdss.native import _Native
        _Native()
    except (ImportError, OSError) as exc:
        raise ImportError(f"fill_gage_series requires working hecdss library: {exc}") from exc

    from datetime import timedelta as _td

    t0 = datetime.strptime(start_time, "%d %B %Y, %H:%M")
    dss_hec_start = _fmt_dss_time(start_time)
    dss_d_part = dss_hec_start[:9]
    epart = _dss_epart(time_interval)
    pathname = f"//{name_station}/PRECIP-INC/{dss_d_part}/{epart}/GAGE/"
    times = [t0 + _td(minutes=i * time_interval) for i in range(len(values))]

    rts = RegularTimeSeries.create(
        values=list(values),
        times=times,
        units="MM",
        data_type="PER-CUM",
        interval=_hecdss_interval(epart),
        path=pathname,
    )
    dss_str = str(Path(path_model) / file_dss)
    with HecDss(dss_str) as f:
        f.put(rts)
    print(f"  ✓ {name_station} written to DSS.")


def generate_met(
    name_met: str,
    names_sbasin: list[str],
    names_gage: list[str],
    path_model: str,
    name_basin: str,
    evapotranspiration: bool = False,
    et_table: pd.DataFrame | None = None,
) -> None:
    """Generate a HEC-HMS meteorological model (.met) file.

    Args:
        name_met: Name for the met model.
        names_sbasin: Sub-basin names.
        names_gage: Gage names assigned to each sub-basin (same order).
        path_model: Model directory.
        name_basin: Basin model name.
        evapotranspiration: Include monthly ET if True (requires et_table).
        et_table: DataFrame with sub-basins as index, columns '1'…'12' (pan evap)
                  and 'Factor_1'…'Factor_12' (crop coefficient). Required when
                  evapotranspiration=True.
    """
    path_model = _ensure_writable_hms_model(path_model)

    lines: list[str] = [
        f"Meteorology: {name_met.replace('_', ' ')}\n",
        "     Last Modified Date: 13 November 2020\n",
        "     Last Modified Time: 11:16:56\n",
        "     Version: 4.9\n",
        "     Unit System: Metric\n",
        "     Set Missing Data to Default: Yes\n",
        "     Precipitation Method: Specified Average\n",
        "     Short-Wave Radiation Method: None\n",
        "     Long-Wave Radiation Method: None\n",
        "     Snowmelt Method: None\n",
    ]

    if evapotranspiration:
        lines += [
            "     Evapotranspiration Method: Monthly Evaporation\n",
            f"     Use Basin Model: {name_basin}\n",
            "End:\n\n",
            "Precip Method Parameters: Specified Average\n",
            "     Last Modified Date: 6 August 2020\n",
            "     Last Modified Time: 08:05:04\n",
            "     Allow Depth Override: No\n",
            "End:\n\n",
            "Evapotranspiration Method Parameters: Monthly Evaporation\n",
            "     Last Modified Date: 18 December 2020\n",
            "     Last Modified Time: 13:11:43\n",
            "End:\n\n",
        ]
        for basin, gage in zip(names_sbasin, names_gage):
            lines += [f"Subbasin: {basin}\n", f"     Gage: {gage}\n", "     Begin Et:\n"]
            for m in range(1, 13):
                lines.append(f"     Pan Evaporation: {et_table.loc[basin, str(m)]}\n")
            for m in range(1, 13):
                lines.append(f"     Evapotranspiration Coefficient: {et_table.loc[basin, f'Factor_{m}']}\n")
            lines += ["     End Et:\n", "End:\n\n"]
    else:
        lines += [
            "     Evapotranspiration Method: No Evapotranspiration\n",
            f"     Use Basin Model: {name_basin}\n",
            "End:\n\n",
            "Precip Method Parameters: Specified Average\n",
            "     Last Modified Date: 6 August 2020\n",
            "     Last Modified Time: 08:05:04\n",
            "     Allow Depth Override: No\n",
            "End:\n\n",
        ]
        for basin, gage in zip(names_sbasin, names_gage):
            lines += [f"Subbasin: {basin}\n", f"     Gage: {gage}\n", "End:\n\n"]

    Path(path_model, name_met + ".met").write_text("".join(lines))
    print(f"✓ {name_met}.met written.")


def generate_met_freq_storm(
    name_met: str,
    names_sbasin: list[str],
    path_model: str,
    idf: pd.DataFrame,
    name_basin: str,
    storm_type: str = "Hydro-35/TP-40/TP-49",
    basin_area_km2: float = 0.0,
) -> None:
    """Generate a frequency-based hypothetical storm meteorological file (HEC-HMS 4.13).

    Writes the Hydro-35/TP-40/TP-49 frequency storm format required by HEC-HMS 4.13,
    with depths at all 10 standard TP-40 durations (5–1440 min).  Sub-30 min depths
    are extrapolated by power-law from the 30-min and 60-min IDF anchor points.

    Args:
        name_met: Met model name (e.g. ``'IDF_T100'``).
        names_sbasin: Sub-basin names.
        path_model: Model directory (must contain the ``.hms`` project file).
        idf: DataFrame indexed by duration (hours), columns = sub-basin names,
             values = precipitation depths (mm).  Must include 0.5 h and 1 h rows.
        name_basin: Basin model name.
        storm_type: HEC-HMS precipitation type string (default
            ``'Hydro-35/TP-40/TP-49'`` — SE United States standard).
        basin_area_km2: Drainage area in km² (used for depth-area reduction;
            0 disables reduction but HEC-HMS still requires the field).
    """
    path_model = _ensure_writable_hms_model(path_model)

    import math

    # Convert basin area km² → sq miles (HEC-HMS Storm Size field unit)
    basin_area_sqmi = basin_area_km2 / 2.58999

    # Standard TP-40 durations needed by HEC-HMS 4.13 (minutes)
    tp40_durs_min = [5, 10, 15, 30, 60, 120, 180, 360, 720, 1440]
    # TP-40 anchor durations from IDF (hours → minutes)
    anchor_durs_h  = [0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0]  # standard TP-40/NOAA Atlas 14
    anchor_durs_min = [int(d * 60) for d in anchor_durs_h]    # [30,60,120,180,360,720,1440]

    # Mean depth per anchor duration across all sub-basins
    anchor_depths: dict[int, float] = {}
    for h, m in zip(anchor_durs_h, anchor_durs_min):
        if h in idf.index:
            anchor_depths[m] = float(idf.loc[h].mean())
        else:
            # Find nearest available duration
            diffs = {abs(idx - h): idx for idx in idf.index}
            nearest = diffs[min(diffs)]
            anchor_depths[m] = float(idf.loc[nearest].mean())

    # Power-law extrapolation for sub-30 min durations
    d30 = anchor_depths.get(30, anchor_depths[min(anchor_depths)])
    d60 = anchor_depths.get(60, d30 * 1.4)
    if d60 > d30 > 0:
        n_exp = math.log(d60 / d30) / math.log(60.0 / 30.0)
    else:
        n_exp = 0.45  # typical SE US IDF exponent
    for m in [5, 10, 15]:
        anchor_depths[m] = d30 * (m / 30.0) ** n_exp

    # Build depth lines for all 10 TP-40 durations
    depth_lines = "".join(
        f"     Depth {m}: {anchor_depths[m]:.1f}\n" for m in tp40_durs_min
    )

    header = [
        f"Meteorology: {name_met.replace('_', ' ')}\n",
        "     Last Modified Date: 13 November 2020\n",
        "     Last Modified Time: 11:16:56\n",
        "     Version: 4.13\n",
        "     Unit System: Metric\n",
        "     Set Missing Data to Default: No\n",
        "     Precipitation Method: Frequency Based Hypothetical\n",
        "     Air Temperature Method: None\n",
        "     Atmospheric Pressure Method: None\n",
        "     Dew Point Method: None\n",
        "     Wind Speed Method: None\n",
        "     Shortwave Radiation Method: None\n",
        "     Longwave Radiation Method: None\n",
        "     Snowmelt Method: None\n",
        "     Evapotranspiration Method: No Evapotranspiration\n",
        f"     Use Basin Model: {name_basin}\n",
        "End:\n\n",
        "Precip Method Parameters: Frequency Based Hypothetical\n",
        "     Last Modified Date: 13 November 2020\n",
        "     Last Modified Time: 11:16:56\n",
        f"     Storm Type: {storm_type}\n",
        "     Single Hypothetical Storm Size: Yes\n",
        "     Uniform Depth Duration Curve: No\n",
        "     User Specified Storm Area: Yes\n",
        f"     Storm Size: {basin_area_sqmi:.3f}\n",
        "     Re-sort Storm Symmetrically: No\n",
        "     Total Duration: 1440\n",
        "     Time Interval: 5\n",
        "     Percent of Duration Before Peak Rainfall: 50\n",
        "     Depth-Area Reduction Method: No Reduction\n",
        depth_lines,
        "End:\n\n",
    ]

    basin_blocks: list[str] = []
    for basin in names_sbasin:
        basin_blocks += [
            f"Subbasin: {basin}\n",
            "     Last Modified Date: 13 November 2020\n",
            "     Last Modified Time: 11:16:56\n",
            depth_lines,
            "End:\n\n",
        ]

    met_path = Path(path_model, name_met + ".met")
    met_path.write_text("".join(header + basin_blocks))

    # Register the met model in the .hms project file so HEC-HMS can find it
    hms_files = list(Path(path_model).glob("*.hms"))
    if hms_files:
        hms_path = hms_files[0]
        hms_text = hms_path.read_text()
        entry = (
            f"Precipitation: {name_met.replace('_', ' ')}\n"
            f"     Filename: {name_met}.met\n"
            "     Description: Frequency storm\n"
            "End:\n\n"
        )
        if f"Precipitation: {name_met.replace('_', ' ')}" not in hms_text:
            with hms_path.open("a") as fh:
                fh.write(entry)
    print(f"✓ {name_met}.met (frequency storm) written.")


def generate_hms(
    name_model: str,
    path_model: str,
    names_met: list[str],
    file_dss: str,
    names_basin: list[str],
    names_control: list[str],
) -> None:
    """Rewrite the .hms project file with updated met/basin/control references.

    Args:
        name_model: Project name.
        path_model: Model directory.
        names_met: Met model names.
        file_dss: DSS file name.
        names_basin: Basin names.
        names_control: Control names.
    """
    path_model = _ensure_writable_hms_model(path_model)

    lines: list[str] = [
        f"Project: {name_model}\n",
        f"     Description: {name_model}\n",
        "     Version: 4.9\n",
        "     Filepath Separator: \\\n",
        f"     DSS File Name: {file_dss}\n",
        "     Time Zone ID: Europe/Paris\n",
        "End:\n\n",
    ]
    for met in names_met:
        lines += [
            f"Precipitation: {met.replace('_', ' ')}\n",
            f"     Filename: {met}.met\n",
            f"     Description: HMS generated met file for {name_model}\n",
            "     Last Modified Date: 13 November 2020\n",
            "     Last Modified Time: 11:18:28\n",
            "End:\n\n",
        ]
    for basin in names_basin:
        lines += [
            f"Basin: {basin}\n",
            f"     Filename: {basin}.basin\n",
            f"     Description: HMS generated basin file for {name_model}\n",
            "     Last Modified Date: 13 November 2020\n",
            "     Last Modified Time: 11:18:28\n",
            "End:\n\n",
        ]
    for control in names_control:
        lines += [
            f"Control: {control.replace('_', ' ')}\n",
            f"     FileName: {control}.control\n",
            f"     Description: HMS generated control file for {name_model}\n",
            "End:\n\n",
        ]
    Path(path_model, name_model + ".hms").write_text("".join(lines))
    print(f"✓ {name_model}.hms written.")


def generate_control(
    name_model: str,
    path_model: str,
    name_control: str,
    start_time: str,
    end_time: str,
    time_interval: str,
) -> None:
    """Create a .control file and append it to the .hms project file.

    Args:
        name_model: Project name.
        path_model: Model directory.
        name_control: Control name.
        start_time: Start date string (e.g. '01 January 2010').
        end_time: End date string.
        time_interval: Simulation time step in minutes (e.g. '60' for hourly).
    """
    path_model = _ensure_writable_hms_model(path_model)

    # Strip any trailing time portion (e.g. ', 00:00') — HEC-HMS date field is date-only
    def _date_only(s: str) -> str:
        return s.split(",")[0].strip()

    ctrl_lines = [
        f"Control: {name_control.replace('_', ' ')}\n",
        "     Last Modified Date: 13 November 2020\n",
        "     Last Modified Time: 11:18:20\n",
        "     Version: 4.13\n",
        f"     Start Date: {_date_only(start_time)}\n",
        "     Start Time: 00:00\n",
        f"     End Date: {_date_only(end_time)}\n",
        "     End Time: 00:00\n",
        f"     Time Interval: {time_interval}\n",
        "End:\n\n",
    ]
    Path(path_model, name_control + ".control").write_text("".join(ctrl_lines))

    hms_path = Path(path_model, name_model + ".hms")
    with hms_path.open("a") as f:
        f.writelines([
            f"Control: {name_control.replace('_', ' ')}\n",
            f"     Filename: {name_control}.control\n",
            "     Description: Control\n",
            "End:\n\n",
        ])
    print(f"✓ {name_control}.control created and registered in {name_model}.hms.")


def generate_run(
    path_model: str,
    name_model: str,
    name_run: str,
    name_met: str,
    name_basin: str,
    name_control: str,
    exists_run: bool = True,
) -> None:
    """Add a run to the .run file and create required .log and .dss stubs.

    Args:
        path_model: Model directory.
        name_model: Project name.
        name_run: Name for the new simulation run.
        name_met: Met model name.
        name_basin: Basin model name.
        name_control: Control name.
        exists_run: If True, append to the existing .run file.
    """
    path_model = _ensure_writable_hms_model(path_model)

    run_block = [
        f"Run: {name_run.replace('_', ' ')}\n",
        "     Default Description: Yes\n",
        f"     Log File: {name_run}.log\n",
        f"     DSS File: {name_run}.dss\n",
        "     Is Save Spatial Results: No\n",
        "     Last Modified Date: 17 November 2020\n",
        "     Last Modified Time: 09:31:48\n",
        f"     Basin: {name_basin}\n",
        f"     Precip: {name_met.replace('_', ' ')}\n",
        f"     Control: {name_control.replace('_', ' ')}\n",
        "     Time-Series Output: Save All\n",
        "End:\n\n",
    ]

    # Create empty .log
    Path(path_model, name_run + ".log").write_text("")
    # Create a valid empty DSS output file for HEC-HMS to write results into.
    # hecdss creates DSS v7; pure-Python _init_dss6 creates DSS v6 as fallback.
    dss_path = Path(path_model, name_run + ".dss")
    try:
        from hecdss import HecDss as _HecDss
        with _HecDss(str(dss_path)):
            pass
    except Exception:
        _init_dss6(dss_path)

    run_path = Path(path_model, name_model + ".run")
    existing = run_path.read_text().splitlines(keepends=True) if exists_run and run_path.exists() else []
    run_path.write_text("".join(existing + run_block))
    print(f"✓ Run '{name_run}' added to {name_model}.run.")


def generate_py(path_model: str, name_model: str, names_run: list[str]) -> None:
    """Generate the compute_current.py script used to run HEC-HMS from Python.

    Args:
        path_model: Model directory.
        name_model: Project name.
        names_run: List of run names to execute.
    """
    path_model = _ensure_writable_hms_model(path_model)

    scripts_dir = Path(path_model, "scripts")
    scripts_dir.mkdir(exist_ok=True)
    hms_path = str(Path(path_model, name_model + ".hms"))
    run_lines = "\n".join(f"myProject.computeRun('{r}')" for r in names_run)
    dss_path = str(Path(path_model, name_model + ".dss"))
    script = (
        "# -*- coding: utf-8 -*-\n"
        "from hec.heclib.dss import HecDss\n"
        "from hms.model import Project\n"
        "from hms import Hms\n"
        "import os\n\n"
        # Only create a fresh DSS 7 if the file is missing or is ZDSS8 (old format).
        # A ZDSS file (header bytes b'ZDSS\\x00' or b'ZDSS\\xc7') already contains gage
        # data and must not be deleted — HEC-HMS needs it for the historical simulation.
        f"_dss = r'{dss_path}'\n"
        "if os.path.exists(_dss):\n"
        "    with open(_dss, 'rb') as _f:\n"
        "        _hdr = _f.read(8)\n"
        "    if _hdr[:5] == b'ZDSS8':  # DSS 8 format — must convert\n"
        "        os.remove(_dss)\n"
        "        _d = HecDss.open(_dss)\n"
        "        _d.done()\n"
        "    # else: existing ZDSS (DSS 6) file has gage data — leave it intact\n"
        "else:\n"
        "    _d = HecDss.open(_dss)\n"
        "    _d.done()\n\n"
        f"myProject = Project.open(r'{hms_path}')\n"
        f"{run_lines}\n"
        "myProject.close()\n\n"
        "Hms.shutdownEngine()\n"
    )
    (scripts_dir / "compute_current.py").write_text(script)
    print(f"✓ compute_current.py written to {scripts_dir}.")


def run_hms_script(
    path_model: str,
    name_model: str,
    names_run: list[str],
    hms_dir: str | None = None,
    timeout: int = 3600,
    strict_logs: bool = True,
) -> int:
    """Execute HEC-HMS via a generated Jython script — cross-platform.

    On **Windows** the HEC-HMS Python API is called directly in-process.
    On **Linux / Docker** the HMS shell wrapper is invoked under Xvfb for
    headless display.

    Args:
        path_model: Model directory.
        name_model: Project name (without extension).
        names_run: Run names to compute.
        hms_dir: HEC-HMS installation directory. Falls back to the
                 ``HEC_HMS_DIR`` environment variable, then the Docker default
                 ``/workspace/data/hms/HEC-HMS-4.13``.
        timeout: Maximum execution time in seconds.
        strict_logs: If True, inspect each run log and treat HEC-HMS internal
                     errors/aborted runs as failures. The Linux wrapper may
                     return 0 even when a simulation is aborted.

    Returns:
        Process return code (0 = success). On Windows always returns 0 or
        raises on failure.
    """
    import platform
    path_model = _ensure_writable_hms_model(path_model)
    generate_py(path_model, name_model, names_run)
    script_path = str(Path(path_model, "scripts", "compute_current.py"))

    if platform.system() == "Darwin":
        raise OSError(
            "HEC-HMS execution is not supported natively on macOS. "
            "Please run HEC-HMS on Windows, Linux, or in a Docker container."
        )

    if platform.system() == "Windows":
        from hms.model import Project
        from hms import Hms
        hms_path = str(Path(path_model, name_model + ".hms"))
        project = Project.open(hms_path)
        for run in names_run:
            project.computeRun(run)
        project.close()
        Hms.shutdownEngine()
        print(f"✓ HEC-HMS runs completed (Windows API): {names_run}")
        return 0

    # Linux: use xvfb-run + hec-hms.sh
    _hms_dir = (
        hms_dir
        or os.environ.get("HEC_HMS_DIR")
        or "/workspace/data/hms/HEC-HMS-4.13"
    )
    hms_sh = str(Path(_hms_dir, "hec-hms.sh"))
    cmd = ["xvfb-run", "--auto-servernum", hms_sh, "-script", script_path]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"✗ HEC-HMS failed (returncode={result.returncode}).")
        print(result.stdout[-1000:])
        print(result.stderr[-1000:])
        return result.returncode

    failed_logs: list[Path] = []
    if strict_logs:
        run_text = Path(path_model, name_model + ".run").read_text(errors="replace")
        for run in names_run:
            log_name = None
            block_match = re.search(
                rf"^Run:\s+{re.escape(run)}\s*$.*?^End:\s*$",
                run_text,
                flags=re.MULTILINE | re.DOTALL,
            )
            if block_match:
                log_match = re.search(r"^\s+Log File:\s*(.+?)\s*$", block_match.group(0), flags=re.MULTILINE)
                if log_match:
                    log_name = log_match.group(1)
            if log_name is None:
                log_name = run.replace(" ", "_") + ".log"
            log_path = Path(path_model, log_name)
            if not log_path.exists():
                failed_logs.append(log_path)
                continue
            log = log_path.read_text(errors="replace")
            if re.search(r"\bERROR\b|Aborted run|aborted run", log):
                failed_logs.append(log_path)

    if failed_logs:
        print("✗ HEC-HMS reported errors in run logs:")
        for log_path in failed_logs:
            print(f"  - {log_path}")
            if log_path.exists():
                print(log_path.read_text(errors="replace")[-1200:])
        return 2

    print(f"✓ HEC-HMS runs completed (Linux xvfb): {names_run}")
    return 0


def read_dss6_timeseries(
    dss_path: str,
    pathname_prefix: str,
    n_months: int = 6,
    units: str = "CFS",
    latest: bool = True,
) -> pd.DataFrame:
    """Read a regular-interval time series from a HEC-DSS 6 binary file.

    Pure-Python, cross-platform — no pydsstools or Java required.
    Handles the DSS 6 alternating-page binary format (even pages = standard
    little-endian float32; odd pages = high-16/low-16-bit swapped float32).

    Each monthly block stores ``n_hours × 2`` float32 values: alternating
    (quality_code, value) pairs.  Only the value fields are returned.

    Parameters
    ----------
    dss_path:
        Path to the ``.dss`` file.
    pathname_prefix:
        Prefix shared by all monthly blocks, e.g.
        ``"//STATION I/FLOW-OBSERVED"`` or ``"//74006/FLOW"``.
        Monthly suffixes (``/01JAN1970/1HOUR/…``) are found automatically.
    n_months:
        Number of consecutive months to read (default 6).
    units:
        Expected unit string in the block header (default ``"CFS"``).
        Returned series values keep these units.
    latest:
        If ``True`` (default), read the latest matching monthly blocks. HEC-HMS
        appends repeated computations to DSS files, so calibration workflows
        normally need the last blocks rather than the first historical blocks.

    Returns
    -------
    pd.DataFrame
        Columns ``datetime`` (str, hourly) and ``value`` (float, in *units*).
        Missing / sentinel values are stored as ``NaN``.

    Examples
    --------
    >>> df = read_dss6_timeseries(
    ...     '/workspace/data/hms/Tifton/1970_simulation.dss',
    ...     '//74006/FLOW',
    ... )
    >>> df['value'].max()  # peak simulated flow in CFS
    927.84...
    """
    import struct as _struct

    PAGE = 512

    with open(dss_path, "rb") as _f:
        raw = _f.read()

    # Locate data blocks: look for pathname_prefix inside the file and identify
    # data blocks (not directory entries) by validating block signature (-9753)
    # and checking that the 8-byte sentinel following the padded path string is zero.
    search_key = pathname_prefix.encode("ascii")
    positions: list[int] = []
    pos = 0
    while True:
        pos = raw.find(search_key, pos)
        if pos == -1:
            break
        path_start = pos
        block_start = path_start - 12  # link(4) + type(4) + pathlen(4)
        if block_start < 0:
            pos += 1
            continue
        # Discriminate data block vs. directory entry:
        # Data blocks start with signature -9753 (0xFFFFD9E7) and have a zero sentinel.
        try:
            signature = _struct.unpack_from("<i", raw, block_start)[0]
            if signature == -9753:
                plen = _struct.unpack_from("<i", raw, block_start + 8)[0]
                sentinel_pos = ((12 + plen + 3) // 4) * 4
                sentinel = _struct.unpack_from("<q", raw, block_start + sentinel_pos)[0]
                if sentinel == 0:
                    if block_start not in positions:
                        positions.append(block_start)
        except Exception:
            pass
        pos += 1

    positions.sort()
    positions = positions[-n_months:] if latest else positions[:n_months]

    def _read_block(bs: int) -> tuple[np.ndarray, list] | None:
        """Return (flow_values, datetime_list) for one monthly block, or None if invalid."""
        try:
            plen = _struct.unpack_from("<i", raw, bs + 8)[0]
            sentinel_pos = ((12 + plen + 3) // 4) * 4
            header_start = sentinel_pos + 8
            
            n_floats = _struct.unpack_from("<i", raw, bs + header_start + 4)[0]
            n_interv = _struct.unpack_from("<i", raw, bs + header_start + 8)[0]
        except Exception:
            return None

        # Sanity-check: allow generous headroom for sub-hourly data.
        if not (0 < n_floats < 100_000):
            return None

        # Decode start date from pathname part D (e.g. "01JAN1970")
        try:
            path_str = raw[bs + 12 : bs + 12 + plen].decode("ascii", errors="replace")
            parts = path_str.strip("/").split("/")
            part_d = parts[2] if len(parts) > 2 else ""
            t0 = pd.to_datetime(part_d) + pd.Timedelta(hours=1)
        except Exception:
            return None

        data_start = bs + header_start + 196
        start_page = data_start // PAGE
        vals: list[float] = []
        p = data_start
        for _ in range(n_floats):
            if p + 4 > len(raw):
                break
            rel_page = p // PAGE - start_page
            b = raw[p : p + 4]
            if rel_page % 2 == 1:  # odd record page → swap high/low 16-bit words
                b = bytes([b[2], b[3], b[0], b[1]])
            vals.append(_struct.unpack("<f", b)[0])
            p += 4

        arr = np.array(vals, dtype="f4")
        # Values are at odd indices (quality at even, flow at odd)
        flow = arr[1::2]
        # Replace sentinels (±huge, NaN, inf, negative)
        bad = ~(np.isfinite(flow) & (flow >= 0.0) & (flow < 100_000.0))
        flow = flow.astype(float)
        flow[bad] = np.nan

        # Check time interval from path part E (e.g. "1DAY" vs "1HOUR")
        part_e = parts[3] if len(parts) > 3 else "1HOUR"
        is_daily = "DAY" in part_e.upper()

        times = []
        for i in range(len(flow)):
            try:
                if is_daily:
                    times.append((t0 + pd.Timedelta(days=i)).strftime("%Y-%m-%d %H:%M"))
                else:
                    times.append((t0 + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M"))
            except (OverflowError, pd.errors.OutOfBoundsDatetime):
                return None
        return flow, times

    rows: list[dict] = []
    for bs in positions:
        result = _read_block(bs)
        if result is None:
            continue
        flow_vals, dt_strs = result
        for dt, v in zip(dt_strs, flow_vals):
            rows.append({"datetime": dt, "value": v})

    df = pd.DataFrame(rows)
    return df


def read_hms_dss_timeseries(
    dss_path: str,
    pathname_prefix: str,
    run_name: str | None = None,
    n_months: int = 6,
    latest: bool = True,
) -> pd.DataFrame:
    """Read HMS output time series from DSS7 or DSS6.

    HEC-HMS 4.x usually writes DSS7 output files. Historical sample files may
    still be DSS6. This helper tries ``hecdss`` first and falls back to the
    bundled pure-Python DSS6 reader.

    Args:
        dss_path: Path to the DSS file.
        pathname_prefix: Prefix such as ``"//74006/FLOW"`` or
            ``"//Station I/FLOW"``.
        run_name: Optional HMS run name used to select matching F-part
            pathnames (``RUN:<run_name>``).
        n_months: Number of DSS6 blocks to read in fallback mode.
        latest: Read latest matching DSS6 blocks in fallback mode.

    Returns:
        DataFrame with ``datetime`` and ``value`` columns. Values are in the DSS
        units stored by HMS; most HMS flow outputs are CFS for English projects.
    """
    prefix_key = (pathname_prefix.rstrip("/") + "/").upper()
    run_key = f"/RUN:{run_name}/".upper() if run_name else None

    try:
        from hecdss import HecDss as _HecDss

        with _HecDss(str(dss_path)) as dss:
            catalog = dss.get_catalog()
            paths = []
            for path in catalog.uncondensed_paths:
                path_upper = path.upper()
                if not path_upper.startswith(prefix_key):
                    continue
                if run_key and run_key not in path_upper:
                    continue
                paths.append(path)
            if paths:
                ts = dss.get(sorted(paths)[-1])
                return pd.DataFrame({
                    "datetime": pd.to_datetime(ts.get_dates()),
                    "value": np.asarray(ts.get_values(), dtype=float),
                })
    except Exception:
        pass

    return read_dss6_timeseries(
        dss_path,
        pathname_prefix,
        n_months=n_months,
        latest=latest,
    )


def generate_flow(
    pathname: str,
    path_dss: str,
    dss_name: str,
    start_date: str,
    end_date: str,
    path_output: str,
    name_file_output: str,
) -> pd.DataFrame:
    """Extract a discharge time series from a DSS file and save it as CSV.

    Uses hecdss (numpy-version-agnostic) to read the time series.

    Args:
        pathname: DSS pathname (e.g. '//JUNCTION/FLOW/…/RUN:Sim_hist/').
        path_dss: Directory containing the DSS file.
        dss_name: DSS filename.
        start_date: Start date string (any pandas-parseable format).
        end_date: End date string.
        path_output: Output directory.
        name_file_output: Output CSV base name (without extension).

    Returns:
        DataFrame with a 'flow' column indexed by date.
    """
    dss_path = str(Path(path_dss, dss_name))
    try:
        from hecdss import HecDss
        from hecdss.native import _Native
        _Native()
        with HecDss(dss_path) as f:
            ts = f.get(pathname)
        values = ts.get_values()
        dates = ts.get_dates()
    except (ImportError, OSError, Exception):
        # Fallback to pure-Python DSS v6 reader
        parts = pathname.strip("/").split("/")
        if len(parts) >= 2:
            prefix = f"//{parts[1]}/{parts[2]}"
        else:
            prefix = pathname
        df = read_dss6_timeseries(dss_path, prefix)
        values = df["value"].values
        dates = df["datetime"].values

    q = pd.DataFrame({"flow": values}, index=pd.DatetimeIndex(dates))
    t0 = pd.Timestamp(start_date)
    t1 = pd.Timestamp(end_date)
    q = q.loc[(q.index >= t0) & (q.index <= t1)]
    os.makedirs(path_output, exist_ok=True)
    q.to_csv(str(Path(path_output, name_file_output + ".csv")))
    print(f"✓ Flow saved → {path_output}{name_file_output}.csv")
    return q


# ── Calibration helper ────────────────────────────────────────────────────────

class HMSModel:
    """Thin wrapper around a HEC-HMS project for automated calibration.

    Runs the model via the HEC-HMS Python API and returns simulated discharge
    at a target junction/reach as a numpy array.

    Args:
        path_model: Model directory.
        name_model: Project name.
        name_run: Simulation run name.
        name_basin: Basin name.
        name_control: Control name.
        time_interval: Time step in minutes.
        pathname: DSS pathname template for output discharge.
        name_precip: Met model name.
        start_date: Start date string.
        end_date: End date string.
        path_output: Directory for result CSVs.
        observed: Observed discharge Series for objective-function evaluation.
        parameter_names: Names of parameters passed to run_hms (for spotpy).
    """

    def __init__(
        self,
        path_model: str,
        name_model: str,
        name_run: str,
        name_basin: str,
        name_control: str,
        time_interval: str,
        pathname: str,
        name_precip: str,
        start_date: str,
        end_date: str,
        path_output: str,
        observed: pd.Series | None = None,
        parameter_names: list[str] | None = None,
        hms_dir: str | None = None,
    ):
        self.path_model = path_model
        self.name_model = name_model
        self.name_run = name_run
        self.name_basin = name_basin
        self.name_control = name_control
        self.time_interval = time_interval
        self.pathname = pathname
        self.name_precip = name_precip
        self.start_date = start_date
        self.end_date = end_date
        self.path_output = path_output
        self.observed = observed
        self.parameter_names = parameter_names or []
        self.hms_dir = hms_dir

    def run_hms(self, *params) -> np.ndarray:
        """Execute HEC-HMS and return simulated discharge array.

        Cross-platform: uses the Jython API on Windows and xvfb-run on Linux.
        Override :meth:`_write_params` to apply calibration parameters.

        Args:
            *params: Parameter values in the order given by parameter_names.

        Returns:
            1-D numpy array of simulated discharge values.
        """
        from hecdss import HecDss

        self._write_params(params)

        run_hms_script(
            self.path_model,
            self.name_model,
            [self.name_run],
            hms_dir=self.hms_dir,
        )

        dss_path = str(Path(self.path_model, self.name_run + ".dss"))
        with HecDss(dss_path) as f:
            ts = f.get(self.pathname)
        return np.array(ts.get_values())

    def _write_params(self, params):
        """Override in subclasses to write calibration parameters to .basin."""
        pass


@dataclass(frozen=True)
class HMSCalibrationParameter:
    """Definition of one scalar HEC-HMS calibration parameter.

    Attributes:
        name: Public calibration parameter name, e.g. ``"tc_mult"``.
        keyword: HEC-HMS ``.basin`` keyword to update.
        lower: Lower multiplier bound for SCE-UA.
        upper: Upper multiplier bound for SCE-UA.
        baseline: Optional absolute baseline value. If omitted, the value is
            read from the target ``.basin`` file.
    """

    name: str
    keyword: str
    lower: float
    upper: float
    baseline: float | None = None


def _read_hms_basin_keyword_value(
    basin_path: str | Path,
    subbasin: str,
    keyword: str,
) -> float:
    text = Path(basin_path).read_text()
    pattern = rf"Subbasin:\s+{re.escape(str(subbasin))}.*?^\s+{re.escape(keyword)}:\s*([\d.\-]+)"
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"Keyword {keyword!r} not found for subbasin {subbasin!r}")
    return float(match.group(1))


class TextBasinHMSCalibrationModel(HMSModel):
    """Calibratable HEC-HMS model backed by text ``.basin`` edits.

    This class implements the common HMS calibration loop used by HYDRA:
    write candidate basin parameters, execute HMS, read the latest output DSS
    flow series, and return a daily simulated hydrograph.

    It is intentionally conservative: parameters are multiplicative factors
    around the original basin values, and known HMS physical constraints are
    enforced before writing.
    """

    def __init__(
        self,
        *args,
        subbasin: str,
        parameters: list[HMSCalibrationParameter],
        output_pathname_prefix: str,
        flow_unit_factor: float = 0.028316846592,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.subbasin = str(subbasin)
        self.parameters = parameters
        self.output_pathname_prefix = output_pathname_prefix
        self.flow_unit_factor = flow_unit_factor
        self.basin_path = Path(self.path_model) / f"{self.name_basin}.basin"
        self.baseline = {
            p.name: p.baseline
            if p.baseline is not None
            else _read_hms_basin_keyword_value(self.basin_path, self.subbasin, p.keyword)
            for p in self.parameters
        }

    @property
    def parameter_bounds(self) -> list[tuple[str, float, float]]:
        return [(p.name, p.lower, p.upper) for p in self.parameters]

    def run_hms(self, *params) -> pd.Series:
        self._write_params(params)
        ret = run_hms_script(
            self.path_model,
            self.name_model,
            [self.name_run],
            hms_dir=self.hms_dir,
            strict_logs=True,
        )
        if ret != 0:
            raise RuntimeError(f"HEC-HMS failed or aborted internally; return code {ret}")

        dss_path = self._find_output_dss()
        df = read_hms_dss_timeseries(
            str(dss_path),
            self.output_pathname_prefix,
            run_name=self.name_run,
            latest=True,
        )
        if df.empty or "value" not in df.columns:
            raise RuntimeError(f"No HMS output values found in {dss_path} for {self.output_pathname_prefix}")

        return pd.Series(
            df["value"].to_numpy(dtype=float) * self.flow_unit_factor,
            index=pd.to_datetime(df["datetime"]),
            name="sim_m3s",
        ).sort_index().resample("D").mean().dropna()

    def _find_output_dss(self) -> Path:
        candidates = [
            Path(self.path_model, self.name_run.replace(" ", "_") + ".dss"),
            Path(self.path_model, self.name_run + ".dss"),
            Path(self.path_model, self.name_model + ".dss"),
        ]
        dss_path = next((p for p in candidates if p.exists()), None)
        if dss_path is None:
            raise FileNotFoundError(f"No HMS output DSS found in {self.path_model}")
        return dss_path

    def _write_params(self, params):
        if len(params) != len(self.parameters):
            raise ValueError(f"Expected {len(self.parameters)} HMS parameters, received {len(params)}")

        values = {
            p.name: self.baseline[p.name] * float(value)
            for p, value in zip(self.parameters, params)
        }
        self._apply_hms_constraints(values)
        df = pd.DataFrame([values], index=[self.subbasin])
        update_basin_file(
            str(self.basin_path),
            df,
            {p.name: p.keyword for p in self.parameters},
        )

    @staticmethod
    def _apply_hms_constraints(values: dict[str, float]) -> None:
        """Apply simple physical constraints known to abort HEC-HMS runs."""
        storage_keys = ["soil_storage", "soil_storage_mult"]
        tension_keys = ["soil_tension", "soil_tension_mult"]
        storage_key = next((k for k in storage_keys if k in values), None)
        tension_key = next((k for k in tension_keys if k in values), None)
        if storage_key and tension_key:
            values[tension_key] = max(0.001, min(values[tension_key], 0.95 * values[storage_key]))


def _align_hms_series(simulation, evaluation) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(simulation, pd.Series) and isinstance(evaluation, pd.Series):
        idx = simulation.dropna().index.intersection(evaluation.dropna().index)
        if len(idx):
            return simulation.loc[idx].to_numpy(dtype=float), evaluation.loc[idx].to_numpy(dtype=float)
    sim = np.asarray(simulation, dtype=float)
    obs = np.asarray(evaluation, dtype=float)
    n = min(len(sim), len(obs))
    return sim[:n], obs[:n]


def validate_hms_parameter_sensitivity(
    model: TextBasinHMSCalibrationModel,
    baseline: np.ndarray | None = None,
    perturbed: np.ndarray | None = None,
    min_delta: float = 1e-5,
) -> dict[str, float]:
    """Run baseline and perturbed HMS simulations and confirm parameters matter."""
    if baseline is None:
        baseline = np.ones(len(model.parameters), dtype=float)
    if perturbed is None:
        perturbed = np.array(
            [p.lower if i % 2 == 0 else p.upper for i, p in enumerate(model.parameters)],
            dtype=float,
        )
    q0 = model.run_hms(*baseline)
    q1 = model.run_hms(*perturbed)
    idx = q0.index.intersection(q1.index)
    if len(idx) == 0:
        raise RuntimeError("HMS sensitivity validation failed: runs do not overlap.")
    max_delta = float(np.nanmax(np.abs(q0.loc[idx].values - q1.loc[idx].values)))
    if not np.isfinite(max_delta) or max_delta < min_delta:
        raise RuntimeError("HMS parameter edits did not change the hydrograph; calibration stopped.")
    return {
        "baseline_peak": float(q0.loc[idx].max()),
        "perturbed_peak": float(q1.loc[idx].max()),
        "max_delta": max_delta,
    }


def calibrate_hms_sceua(
    model: TextBasinHMSCalibrationModel,
    observed: pd.Series,
    dbname: str,
    n_evals: int = 120,
    objective: str = "nse",
    ngs: int | None = None,
):
    """Calibrate a HEC-HMS model with SPOTPY SCE-UA.

    Args:
        model: Calibratable HMS model.
        observed: Observed discharge series, preferably daily with a
            ``DatetimeIndex`` and units matching ``model.run_hms`` output.
        dbname: SPOTPY database path without ``.csv``.
        n_evals: Number of SCE-UA evaluations.
        objective: ``"nse"`` (minimize ``-NSE``), ``"pbias_abs"``, or ``"rmse"``.
        ngs: Number of SCE-UA complexes. Defaults to ``n_parameters + 1``.

    Returns:
        The SPOTPY sampler after completion.
    """
    try:
        import spotpy
    except ImportError as exc:
        raise ImportError("calibrate_hms_sceua requires spotpy") from exc

    class _Setup:
        def __init__(self):
            self.observed = observed.dropna()
            self.params = [
                spotpy.parameter.Uniform(name, lower, upper)
                for name, lower, upper in model.parameter_bounds
            ]

        def parameters(self):
            return spotpy.parameter.generate(self.params)

        def simulation(self, vector):
            return model.run_hms(*vector)

        def evaluation(self):
            return self.observed

        def objectivefunction(self, simulation, evaluation):
            sim, obs = _align_hms_series(simulation, evaluation)
            if len(sim) == 0:
                return np.inf
            if objective == "nse":
                return -spotpy.objectivefunctions.nashsutcliffe(obs, sim)
            if objective == "pbias_abs":
                return abs(spotpy.objectivefunctions.pbias(obs, sim))
            return spotpy.objectivefunctions.rmse(obs, sim)

    sampler = spotpy.algorithms.sceua(_Setup(), dbname=dbname, dbformat="csv")
    sampler.sample(n_evals, ngs=ngs or len(model.parameters) + 1)
    return sampler


# ── Basin parameter extraction ────────────────────────────────────────────────

def extract_curve_number(
    subbasins_shp: str,
    cn_raster: str,
    land_use_raster: str | None = None,
    id_col: str = "Subbasin",
) -> pd.DataFrame:
    """Extract mean Curve Number (CN) per sub-basin from a raster.

    Requires rasterio, rasterstats, geopandas.

    Args:
        subbasins_shp: Path to sub-basins shapefile.
        cn_raster: Path to CN raster (GeoTIFF).
        land_use_raster: Optional land-use raster for impervious area (%).
        id_col: Column in the shapefile with sub-basin names.

    Returns:
        DataFrame with sub-basin names as index and columns:
        'CN', 'CN_Ia' (initial abstraction = 0.2·S), and optionally
        'Impervious_pct'.
    """
    try:
        import geopandas as gpd
        from rasterstats import zonal_stats
    except ImportError as exc:
        raise ImportError("geopandas and rasterstats are required: pip install geopandas rasterstats") from exc

    gdf = gpd.read_file(subbasins_shp)
    stats = zonal_stats(subbasins_shp, cn_raster, stats=["mean"], nodata=-9999)
    df = pd.DataFrame(index=gdf[id_col].values)
    df["CN"] = [s["mean"] for s in stats]
    # S (potential maximum retention, mm) and Ia (initial abstraction, mm)
    df["S_mm"]  = (25400 / df["CN"]) - 254
    df["Ia_mm"] = 0.2 * df["S_mm"]

    if land_use_raster:
        imp_stats = zonal_stats(subbasins_shp, land_use_raster, stats=["mean"], nodata=-9999)
        df["Impervious_pct"] = [s["mean"] for s in imp_stats]

    return df


def calculate_clark_parameters(
    subbasins_shp: str,
    flow_len_raster: str,
    slope_raster: str,
    id_col: str = "Subbasin",
    area_col: str = "Area_km2",
) -> pd.DataFrame:
    """Estimate Clark unit-hydrograph parameters (Tc, R) per sub-basin.

    Uses the Kirpich formula for Tc and the NRCS formula for the storage
    coefficient R = 0.6 · Tc.

    Args:
        subbasins_shp: Path to sub-basins shapefile (must have area column).
        flow_len_raster: Raster of maximum flow length (m).
        slope_raster: Raster of mean basin slope (m/m).
        id_col: Column with sub-basin names.
        area_col: Column with sub-basin area (km²).

    Returns:
        DataFrame with columns 'Tc_hr' (time of concentration, hours)
        and 'R_hr' (storage coefficient, hours).
    """
    try:
        import geopandas as gpd
        from rasterstats import zonal_stats
    except ImportError as exc:
        raise ImportError("geopandas and rasterstats are required") from exc

    gdf = gpd.read_file(subbasins_shp)
    len_stats   = zonal_stats(subbasins_shp, flow_len_raster, stats=["max"], nodata=-9999)
    slope_stats = zonal_stats(subbasins_shp, slope_raster,    stats=["mean"], nodata=-9999)

    df = pd.DataFrame(index=gdf[id_col].values)
    L_m = np.array([s["max"] or 0.0 for s in len_stats])     # max flow length (m)
    S   = np.array([s["mean"] or 0.001 for s in slope_stats]) # mean slope (m/m)

    # Kirpich formula: Tc (min) = 0.0195 · (L^0.77) / (S^0.385)
    Tc_min = 0.0195 * (L_m ** 0.77) / (S ** 0.385)
    df["Tc_hr"] = Tc_min / 60.0
    df["R_hr"]  = 0.6 * df["Tc_hr"]    # NRCS rule of thumb

    return df


def estimate_muskingum_k(
    L_km: float,
    slope: float,
    velocity_mps: float | None = None,
    method: str = "custom",
) -> tuple[float, float]:
    """Estimate Muskingum-Kunge routing parameters K and X.

    Args:
        L_km: Reach length (km).
        slope: Reach slope (m/m).
        velocity_mps: Mean flow velocity (m/s). If None, estimated from
                      slope using Manning's approximation.
        method: 'custom' uses the Viessman (1989) formula; 'usace' uses
                the USACE recommendation.

    Returns:
        Tuple (K_hr, X) where K is the travel time (hours) and X is the
        weighting factor (0–0.5).
    """
    L_m = L_km * 1000.0
    if velocity_mps is None:
        # Approximate V via Manning assuming n=0.04, hydraulic radius ≈ 1 m
        velocity_mps = max((1.0 ** (2 / 3) * slope ** 0.5) / 0.04, 0.1)

    K_hr = (L_m / velocity_mps) / 3600.0

    # X: Cunge formula — X = 0.5*(1 - q/(B*S*V*L))
    # Use empirical default range for natural channels
    X = 0.2 if slope < 0.001 else 0.3

    return round(K_hr, 3), round(X, 3)


def update_basin_file(
    basin_path: str,
    df: pd.DataFrame,
    parameter_map: dict[str, str],
) -> None:
    """Update parameter values in a HEC-HMS .basin file.

    Reads the .basin file, finds each parameter keyword for each sub-basin,
    and replaces its value in place.

    Args:
        basin_path: Path to the .basin file.
        df: DataFrame indexed by sub-basin name; columns are parameter names.
        parameter_map: Mapping from df column name → HEC-HMS keyword
                       (e.g. {'CN': 'Curve Number', 'Tc_hr': 'Time of Concentration'}).
    """
    text = Path(basin_path).read_text()
    for subbasin, row in df.iterrows():
        for col, keyword in parameter_map.items():
            if col not in row.index:
                continue
            val = row[col]
            pattern = rf"(Subbasin:\s+{re.escape(str(subbasin))}.*?)(^\s+{re.escape(keyword)}:\s*)[\d\.\-]+"
            replacement = rf"\g<1>\g<2>{val:.4f}"
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE | re.DOTALL)
    Path(basin_path).write_text(text)
    print(f"✓ {Path(basin_path).name} updated with {len(df)} sub-basins × {len(parameter_map)} parameters.")


# ── Climate-change scenario runner ────────────────────────────────────────────

def run_climate_change_scenarios(
    path_model: str,
    name_model: str,
    file_basin: str,
    file_gage: str,
    file_dss: str,
    time_interval: str,
    scenarios: list[dict],
    hms_python_api: bool = True,
) -> None:
    """Run HEC-HMS for a set of climate-change scenarios in batch.

    Each scenario generates its own gage data, met model, control, and run.
    Results are stored as individual DSS files inside the model folder.

    Args:
        path_model: Model directory.
        name_model: Project name.
        file_basin: .basin filename.
        file_gage: .gage filename.
        file_dss: DSS filename for precipitation.
        time_interval: HEC-HMS time step string.
        scenarios: List of dicts, each with keys:
            - 'name': scenario label (e.g. 'SSP245_near').
            - 'path_rain': CSV file with precipitation series.
            - 'start_time': HEC-HMS start string.
            - 'end_time': HEC-HMS end string.
            - 'start_ctrl': control start date string.
            - 'end_ctrl': control end date string.
            - 'time_interval_ctrl': control time step (minutes as str).
        hms_python_api: If True, attempt to run via HEC-HMS Python API.
            Set False to only generate input files.
    """
    names_stations = read_gages(path_model, file_gage)
    names_basin    = read_basin(path_model, file_basin)
    names_control  = read_control(path_model, name_model + ".hms")

    for sc in scenarios:
        name = sc["name"]
        print(f"\n── Scenario: {name} ──")

        # Write precipitation into DSS
        fill_gage(
            names_stations=names_stations,
            path_rain=sc["path_rain"],
            time_interval=time_interval,
            path_model=path_model,
            file_dss=file_dss,
            start_time=sc["start_time"],
            end_time=sc["end_time"],
        )

        # Meteorological model
        met_name = f"Met_{name}"
        generate_met(
            name_met=met_name,
            names_sbasin=read_subbasin(path_model, file_basin),
            names_gage=names_stations,
            path_model=path_model,
            name_basin=names_basin[0],
            evapotranspiration=False,
        )

        # Control
        ctrl_name = f"Control_{name}"
        generate_control(
            name_model=name_model,
            path_model=path_model,
            name_control=ctrl_name,
            start_time=sc["start_ctrl"],
            end_time=sc["end_ctrl"],
            time_interval=sc["time_interval_ctrl"],
        )

        # Update .hms with the new met
        all_mets = [f[:-4] for f in os.listdir(path_model) if f.endswith(".met")]
        generate_hms(name_model, path_model, all_mets, file_dss, names_basin, names_control)

        # Run entry
        run_name = f"Run_{name}"
        generate_run(
            path_model=path_model,
            name_model=name_model,
            name_run=run_name,
            name_met=met_name,
            name_basin=names_basin[0],
            name_control=ctrl_name,
        )

        if hms_python_api:
            try:
                from hms.model import Project
                from hms import Hms
                prj = Project.open(str(Path(path_model, name_model + ".hms")))
                prj.computeRun(run_name)
                prj.close()
                Hms.shutdownEngine()
                print(f"  ✓ Run completed → {run_name}.dss")
            except Exception as exc:
                print(f"  ✗ HEC-HMS API error: {exc}")
        else:
            generate_py(path_model, name_model, [run_name])
            print(f"  ✓ Input files ready. Run compute_current.py manually.")

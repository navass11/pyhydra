"""
SWAT+ automation utilities.

Handles climate input file generation and scenario execution for SWAT+.

Requires:
    - SWAT+ executable (Rev. 60.5.4 or compatible).
    - pandas, numpy.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def write_precipitation_file(
    df_coords: pd.DataFrame,
    df_series: pd.DataFrame,
    output_path: str,
    missing_value: float = -99.0,
) -> None:
    """Write a SWAT+ precipitation input file (.pcp format).

    Args:
        df_coords: DataFrame with columns ['Station', 'Lati', 'Long', 'Elev'],
                   index = station IDs.
        df_series: DataFrame with a datetime index and one column per station
                   (same order as df_coords). Values in mm/day.
        output_path: Output file path (e.g. 'TxtInOut/pcp1.pcp').
        missing_value: Value used for missing data (-99.0 by default).
    """
    stations = list(df_coords["Station"])
    lines: list[str] = [
        "Observed precipitation (mm)\n",
        "nbyr   tstep   lat        lon        elev\n",
    ]
    for _, row in df_coords.iterrows():
        lines.append(f"    0       0  {row['Lati']:9.4f}  {row['Long']:9.4f}  {row['Elev']:9.1f}\n")

    df_filled = df_series[stations].fillna(missing_value)
    for date, row in df_filled.iterrows():
        values = "".join(f"{v:8.2f}" for v in row)
        lines.append(f"{date.strftime('%Y%j')}{values}\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("".join(lines))
    print(f"✓ Precipitation file written → {output_path}")


def write_temperature_file(
    df_coords: pd.DataFrame,
    df_tmax: pd.DataFrame,
    df_tmin: pd.DataFrame,
    output_path: str,
    missing_value: float = -99.0,
) -> None:
    """Write a SWAT+ temperature input file (.tmp format).

    Args:
        df_coords: DataFrame with columns ['Station', 'Lati', 'Long', 'Elev'].
        df_tmax: Daily maximum temperature (°C), same structure as df_tmin.
        df_tmin: Daily minimum temperature (°C).
        output_path: Output file path (e.g. 'TxtInOut/tmp1.tmp').
        missing_value: Value used for missing data.
    """
    stations = list(df_coords["Station"])
    lines: list[str] = [
        "Observed temperature (°C)\n",
        "nbyr   tstep   lat        lon        elev\n",
    ]
    for _, row in df_coords.iterrows():
        lines.append(f"    0       0  {row['Lati']:9.4f}  {row['Long']:9.4f}  {row['Elev']:9.1f}\n")

    tmax = df_tmax[stations].fillna(missing_value)
    tmin = df_tmin[stations].fillna(missing_value)
    for date in tmax.index:
        row_max = tmax.loc[date]
        row_min = tmin.loc[date]
        values = "".join(f"{mx:8.2f}{mn:8.2f}" for mx, mn in zip(row_max, row_min))
        lines.append(f"{date.strftime('%Y%j')}{values}\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("".join(lines))
    print(f"✓ Temperature file written → {output_path}")


def edit_file_cio(file_cio_path: str, start_year: int, end_year: int) -> None:
    """Edit the SWAT+ file.cio to set the simulation period.

    Updates IYR (start year) and NBYR (number of years) in place.

    Args:
        file_cio_path: Path to the file.cio file.
        start_year: First year of the simulation.
        end_year: Last year of the simulation.
    """
    n_years = end_year - start_year + 1
    text = Path(file_cio_path).read_text()

    text = re.sub(r"(?m)^(\s*\d+\s+|\s*)IYR\b.*", f"   {start_year}    IYR", text)
    text = re.sub(r"(?m)^(\s*\d+\s+|\s*)NBYR\b.*", f"   {n_years}    NBYR", text)

    Path(file_cio_path).write_text(text)
    print(f"✓ file.cio updated: IYR={start_year}, NBYR={n_years}.")


def write_swatplus_precipitation_files(
    df_stations: pd.DataFrame,
    df_series: pd.DataFrame,
    txtinout_dir: str,
    missing_value: float = -99.0,
) -> None:
    """Write SWAT+ individual precipitation files (one .pcp file per station).

    SWAT+ uses one file per station (e.g. p1.pcp, p2.pcp) referenced from
    pcp.cli, instead of the legacy SWAT 2012 multi-station format.

    Args:
        df_stations: DataFrame with columns ['name', 'lat', 'lon', 'elev'].
                     Each row defines one precipitation station.
        df_series: DataFrame with a datetime index and one column per station
                   (column names must match df_stations['name']). Values in mm/day.
        txtinout_dir: Path to the SWAT+ TxtInOut directory.
        missing_value: Value written for missing data (-99.0 by default).
    """
    out_dir = Path(txtinout_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    for _, row in df_stations.iterrows():
        name = row["name"]
        fname = f"{name}.pcp"
        filenames.append(fname)
        series = df_series[name].fillna(missing_value)
        lines = [
            "Precipitation data - file written by pyhydra\n",
            f"nbyr     tstep       lat       lon      elev\n",
            f"   0         0  {float(row['lat']):9.3f}  {float(row['lon']):9.3f}  {float(row['elev']):9.3f}\n",
        ]
        for date, val in series.items():
            lines.append(f"{date.year:4d}  {date.dayofyear:3d}  {val:10.5f}\n")
        (out_dir / fname).write_text("".join(lines))

    cli_lines = ["pcp.cli: Precipitation station files - written by pyhydra\nfilename\n"]
    cli_lines += [f"{fn}\n" for fn in filenames]
    (out_dir / "pcp.cli").write_text("".join(cli_lines))
    print(f"✓ {len(filenames)} SWAT+ precipitation files written → {out_dir}")


def write_swatplus_temperature_files(
    df_stations: pd.DataFrame,
    df_tmax: pd.DataFrame,
    df_tmin: pd.DataFrame,
    txtinout_dir: str,
    missing_value: float = -99.0,
) -> None:
    """Write SWAT+ individual temperature files (one .tmp file per station).

    Args:
        df_stations: DataFrame with columns ['name', 'lat', 'lon', 'elev'].
        df_tmax: Daily max temperature (°C), columns matching df_stations['name'].
        df_tmin: Daily min temperature (°C), same structure.
        txtinout_dir: Path to the SWAT+ TxtInOut directory.
        missing_value: Value written for missing data.
    """
    out_dir = Path(txtinout_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    for _, row in df_stations.iterrows():
        name = row["name"]
        fname = f"{name}.tmp"
        filenames.append(fname)
        tmax = df_tmax[name].fillna(missing_value)
        tmin = df_tmin[name].fillna(missing_value)
        lines = [
            "Temperature data - file written by pyhydra\n",
            f"nbyr     tstep       lat       lon      elev\n",
            f"   0         0  {float(row['lat']):9.3f}  {float(row['lon']):9.3f}  {float(row['elev']):9.3f}\n",
        ]
        for date in tmax.index:
            lines.append(f"{date.year:4d}  {date.dayofyear:3d}  {tmax[date]:10.5f}  {tmin[date]:10.5f}\n")
        (out_dir / fname).write_text("".join(lines))

    cli_lines = ["tmp.cli: Temperature station files - written by pyhydra\nfilename\n"]
    cli_lines += [f"{fn}\n" for fn in filenames]
    (out_dir / "tmp.cli").write_text("".join(cli_lines))
    print(f"✓ {len(filenames)} SWAT+ temperature files written → {out_dir}")


def run_swat(model_dir: str, swat_exe: str, timeout: int = 3600) -> int:
    """Execute the SWAT+ model in the given directory.

    Args:
        model_dir: Path to the TxtInOut directory (where swat.exe is run from).
        swat_exe: Path to the SWAT+ executable.
        timeout: Maximum run time in seconds (default 3600 s = 1 h).

    Returns:
        Process return code (0 = success).
    """
    result = subprocess.run(
        [swat_exe],
        cwd=model_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode == 0:
        print(f"✓ SWAT+ finished successfully in {model_dir}.")
    else:
        print(f"✗ SWAT+ failed (returncode={result.returncode}).")
        print(result.stderr[-2000:])
    return result.returncode

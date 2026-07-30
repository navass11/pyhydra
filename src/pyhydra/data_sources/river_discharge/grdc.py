"""
GRDC (Global Runoff Data Centre) file reader.

GRDC does not expose a public download API; data must be requested and
downloaded manually from https://portal.grdc.bafg.de.
This module parses the GRDC daily/monthly text-file format into DataFrames.

File format reference:
  https://www.bafg.de/GRDC/EN/02_srvcs/21_tmsrs/211_AnnRep/annrep_node.html
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_grdc(filepath) -> pd.DataFrame:
    """
    Parse a GRDC daily or monthly discharge file into a tidy DataFrame.

    Args:
        filepath: Path to a GRDC ``.day`` or ``.mon`` text file.

    Returns:
        pd.DataFrame with columns:
            ``date``       — pd.Timestamp (DatetimeIndex after set_index)
            ``discharge``  — float (m³/s), NaN where missing/flagged
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"GRDC file not found: {path}")

    header, data_lines = _split_grdc_file(path)
    df = _parse_data_lines(data_lines)
    return df


def read_grdc_metadata(filepath) -> dict:
    """
    Extract station metadata from the GRDC file header.

    Args:
        filepath: Path to a GRDC ``.day`` or ``.mon`` file.

    Returns:
        Dict with keys such as 'station', 'river', 'country', 'latitude',
        'longitude', 'altitude', 'catchment_area', 'grdc_no'.
    """
    path = Path(filepath)
    header, _ = _split_grdc_file(path)
    return _parse_header(header)


def read_grdc_folder(folder, pattern="*.day") -> dict[str, pd.DataFrame]:
    """
    Read all GRDC files matching a glob pattern in a folder.

    Args:
        folder: Directory containing GRDC files.
        pattern: Glob pattern (default ``'*.day'``; use ``'*.mon'`` for monthly).

    Returns:
        Dict mapping filename stem → parsed DataFrame.
    """
    folder = Path(folder)
    result = {}
    for f in sorted(folder.glob(pattern)):
        try:
            result[f.stem] = read_grdc(f)
        except Exception as exc:
            print(f"Warning: could not read {f.name}: {exc}")
    return result


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------

def _split_grdc_file(path: Path):
    """Return (header_lines, data_lines) split at the data start marker."""
    with open(path, encoding="latin-1") as fh:
        lines = fh.readlines()

    # The data section begins with a line starting with '#' that contains
    # column headers (YYYY-MM-DD, hh, Original, Calculated, Qualifier)
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip()
        if re.match(r"YYYY-MM-DD", stripped, re.IGNORECASE):
            data_start = i + 1
            break

    if data_start is None:
        # Fallback: look for first non-comment line with a date-like token
        for i, line in enumerate(lines):
            if not line.startswith("#") and re.match(r"\d{4}-\d{2}-\d{2}", line.strip()):
                data_start = i
                break

    if data_start is None:
        raise ValueError(f"Could not locate data section in {path}")

    return lines[:data_start], lines[data_start:]


def _parse_data_lines(lines: list[str]) -> pd.DataFrame:
    """Parse the data rows of a GRDC file."""
    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Fields separated by semicolons: date; hh; original_value; calculated; qualifier
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 3:
            continue
        date_str = parts[0]
        raw_val  = parts[2] if len(parts) > 2 else parts[1]
        try:
            date = pd.to_datetime(date_str, format="%Y-%m-%d")
        except Exception:
            continue
        try:
            val = float(raw_val)
            if val < -990:      # GRDC missing-value sentinel
                val = float("nan")
        except (ValueError, TypeError):
            val = float("nan")
        records.append({"date": date, "discharge": val})

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _parse_header(lines: list[str]) -> dict:
    """Extract key-value pairs from GRDC comment lines."""
    meta = {}
    mapping = {
        "GRDC-No.":        "grdc_no",
        "River":           "river",
        "Station":         "station",
        "Country":         "country",
        "Latitude":        "latitude",
        "Longitude":       "longitude",
        "Altitude":        "altitude_m",
        "Catchment area":  "catchment_area_km2",
    }
    for line in lines:
        line = line.lstrip("#").strip()
        for key, field in mapping.items():
            if line.startswith(key):
                value = line.split(":", 1)[-1].strip()
                try:
                    meta[field] = float(value)
                except ValueError:
                    meta[field] = value
    return meta


# ---------------------------------------------------------------------------
# Quality check helper
# ---------------------------------------------------------------------------

def analyze_grdc_quality(df: pd.DataFrame) -> dict:
    """
    Return basic quality statistics for a parsed GRDC DataFrame.

    Args:
        df: Output of read_grdc().

    Returns:
        Dict with 'start', 'end', 'n_days', 'missing_pct', 'mean_m3s', 'max_m3s'.
    """
    if df.empty:
        return {}
    q = df["discharge"]
    return {
        "start":       df["date"].min(),
        "end":         df["date"].max(),
        "n_days":      len(df),
        "missing_pct": round(100 * q.isna().mean(), 2),
        "mean_m3s":    round(q.mean(skipna=True), 3),
        "max_m3s":     round(q.max(skipna=True), 3),
    }

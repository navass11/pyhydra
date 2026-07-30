"""
USGS NWIS (National Water Information System) downloader.

Downloads daily streamflow data for any US gauging station via the
USGS Water Services REST API (no registration required).

API reference: https://waterservices.usgs.gov/docs/daily-values/
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union

import pandas as pd
import requests


_BASE_URL = "https://waterservices.usgs.gov/nwis/dv/"
_PARAMETER_DISCHARGE = "00060"   # Discharge (ft³/s)
_STAT_MEAN = "00003"             # Daily mean


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_usgs(site_no: Union[str, list[str]],
                  start_date: str,
                  end_date: str,
                  parameter_cd: str = _PARAMETER_DISCHARGE,
                  units: str = "metric",
                  max_retries: int = 3) -> pd.DataFrame:
    """
    Download daily streamflow from one or more USGS gauging stations.

    Args:
        site_no: USGS site number(s) as string or list of strings
                 (e.g. ``'08279500'`` or ``['08279500', '01646500']``).
        start_date: First date in ``'YYYY-MM-DD'`` format.
        end_date: Last date in ``'YYYY-MM-DD'`` format.
        parameter_cd: NWIS parameter code (default ``'00060'`` = discharge).
        units: ``'metric'`` (m³/s, default) or ``'imperial'`` (ft³/s).
        max_retries: Number of retry attempts per request (default 3).

    Returns:
        pd.DataFrame with a DatetimeIndex (``date``) and one column per site
        named ``Q_<site_no>`` in the requested units.
        Missing / flagged values are set to NaN.
    """
    if isinstance(site_no, str):
        site_no = [site_no]

    dfs = []
    for site in site_no:
        df = _fetch_site(site, start_date, end_date, parameter_cd, max_retries)
        if df is not None and not df.empty:
            col_name = f"Q_{site}"
            df = df.rename(columns={"discharge": col_name})
            if units == "metric":
                df[col_name] = df[col_name] * 0.028316847   # ft³/s → m³/s
            dfs.append(df.set_index("date"))

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, axis=1).sort_index()
    result.index.name = "date"
    return result


def get_usgs_site_info(site_no: Union[str, list[str]]) -> pd.DataFrame:
    """
    Retrieve station metadata (name, latitude, longitude, drainage area) from USGS.

    Args:
        site_no: USGS site number(s).

    Returns:
        pd.DataFrame with one row per site and columns:
        ``site_no``, ``station_nm``, ``dec_lat_va``, ``dec_long_va``,
        ``drain_area_va`` (km², converted from mi²).
    """
    if isinstance(site_no, str):
        site_no = [site_no]

    params = {
        "format":   "rdb",
        "sites":    ",".join(site_no),
        "siteType": "ST",
        "siteOutput": "expanded",
    }
    url = "https://waterservices.usgs.gov/nwis/site/"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    lines = [ln for ln in r.text.splitlines()
             if not ln.startswith("#") and ln.strip()]
    if len(lines) < 3:
        return pd.DataFrame()

    headers = lines[0].split("\t")
    data_lines = lines[2:]   # line 1 = format row, skip
    rows = [dict(zip(headers, ln.split("\t"))) for ln in data_lines]
    df = pd.DataFrame(rows)

    keep = ["site_no", "station_nm", "dec_lat_va", "dec_long_va", "drain_area_va"]
    df = df[[c for c in keep if c in df.columns]].copy()

    for num_col in ("dec_lat_va", "dec_long_va", "drain_area_va"):
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    if "drain_area_va" in df.columns:
        df["drain_area_km2"] = df["drain_area_va"] * 2.58999    # mi² → km²

    return df.reset_index(drop=True)


def search_usgs_sites(bbox: tuple[float, float, float, float],
                      parameter_cd: str = _PARAMETER_DISCHARGE) -> pd.DataFrame:
    """
    Find USGS streamflow stations within a bounding box.

    Args:
        bbox: ``(west, south, east, north)`` in decimal degrees.
        parameter_cd: NWIS parameter code to filter by.

    Returns:
        pd.DataFrame with station metadata (same columns as get_usgs_site_info).
    """
    west, south, east, north = bbox
    params = {
        "format":       "rdb",
        "bBox":         f"{west},{south},{east},{north}",
        "parameterCd":  parameter_cd,
        "siteType":     "ST",
        "siteOutput":   "expanded",
        "hasDataTypeCd":"dv",
    }
    url = "https://waterservices.usgs.gov/nwis/site/"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    lines = [ln for ln in r.text.splitlines()
             if not ln.startswith("#") and ln.strip()]
    if len(lines) < 3:
        return pd.DataFrame()

    headers = lines[0].split("\t")
    data_lines = lines[2:]
    rows = [dict(zip(headers, ln.split("\t"))) for ln in data_lines]
    df = pd.DataFrame(rows)

    keep = ["site_no", "station_nm", "dec_lat_va", "dec_long_va", "drain_area_va"]
    df = df[[c for c in keep if c in df.columns]].copy()
    for num_col in ("dec_lat_va", "dec_long_va", "drain_area_va"):
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
    if "drain_area_va" in df.columns:
        df["drain_area_km2"] = df["drain_area_va"] * 2.58999
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_site(site: str, start: str, end: str,
                param: str, max_retries: int) -> Optional[pd.DataFrame]:
    """Download and parse NWIS daily values for one site."""
    params = {
        "format":      "json",
        "sites":       site,
        "startDT":     start,
        "endDT":       end,
        "parameterCd": param,
        "statCd":      _STAT_MEAN,
    }
    for attempt in range(max_retries):
        try:
            r = requests.get(_BASE_URL, params=params, timeout=60)
            r.raise_for_status()
            return _parse_nwis_json(r.json(), site)
        except Exception as exc:
            wait = 2 ** attempt
            print(f"Retry {attempt + 1}/{max_retries} for site {site}: {exc} (wait {wait}s)")
            time.sleep(wait)
    print(f"Failed to download site {site} after {max_retries} attempts.")
    return None


def _parse_nwis_json(data: dict, site: str) -> pd.DataFrame:
    """Parse NWIS JSON response into a DataFrame with date and discharge columns."""
    try:
        ts_list = data["value"]["timeSeries"]
    except (KeyError, TypeError):
        return pd.DataFrame(columns=["date", "discharge"])

    if not ts_list:
        return pd.DataFrame(columns=["date", "discharge"])

    values = ts_list[0]["values"][0]["value"]
    records = []
    for v in values:
        try:
            date = pd.to_datetime(v["dateTime"]).normalize()
            val  = float(v["value"])
            qual = v.get("qualifiers", [])
            if val < 0 or "Ice" in " ".join(qual):
                val = float("nan")
        except (ValueError, TypeError, KeyError):
            continue
        records.append({"date": date, "discharge": val})

    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values("date").reset_index(drop=True)

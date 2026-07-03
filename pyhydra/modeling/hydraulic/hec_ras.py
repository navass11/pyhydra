"""
HEC-RAS automation utilities.

Provides functions to modify HEC-RAS input files programmatically, run
simulations from Python, and read back DSS output time series, enabling
large-scale scenario automation.

Requires:
    - HEC-RAS installed (version 6.x recommended, Windows) for ``run_hec_ras``.
    - rascontrol: ``pip install rascontrol`` (execution; Windows COM only).
    - hecdss: ``pip install hecdss`` (DSS read, numpy-version-agnostic,
      cross-platform — replaces the older pydsstools dependency).
    - pandas, numpy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ── File modification helpers ─────────────────────────────────────────────────

_BOUNDARY_LOCATION_PREFIX = "Boundary Location="
_FLOW_HYDROGRAPH_PREFIX = "Flow Hydrograph="


def _boundary_location_name(line: str) -> str:
    """Extract the BC name (8th comma field) from a 'Boundary Location=' line."""
    fields = line[len(_BOUNDARY_LOCATION_PREFIX):].split(",")
    return fields[7].strip() if len(fields) > 7 else ""


def modify_unsteady_file(
    path_project: str,
    name_project: str,
    file_number: int,
    rainfall_plan_name: int,
    flow_series: pd.DataFrame,
    bc_pathnames: list[str],
) -> None:
    """Write a hydrograph into a HEC-RAS unsteady flow file (.u##).

    Reads the existing file, and for each boundary condition named in
    ``bc_pathnames`` replaces its ``Flow Hydrograph=`` block (count + fixed-
    width data rows) with the new series, leaving every other line intact.
    The boundary is located by matching the DSS-style pathname's B-part
    against the name field of the file's ``Boundary Location=`` line — e.g.
    ``'/BCLINE/CC_aguas_arriba/FLOW/.../'`` matches
    ``Boundary Location=...,CC_aguas_arriba,...``.

    Args:
        path_project: Project directory.
        name_project: Project name (without extension).
        file_number: Source unsteady flow file number (e.g. 1 → '.u01').
        rainfall_plan_name: New unsteady file number, 1-99 (e.g. 7 → '.u07').
        flow_series: DataFrame with a datetime index and one column per BC line,
                      in the same order as ``bc_pathnames``.
        bc_pathnames: List of DSS pathnames identifying each BC line, in the
                      same order as flow_series columns.

    Raises:
        ValueError: If a named boundary, or its ``Flow Hydrograph=`` block,
            cannot be found in the source file.
    """
    plan_id = int(rainfall_plan_name)
    src = Path(path_project, name_project + f".u{file_number:02d}")
    dst = Path(path_project, name_project + f".u{plan_id:02d}")

    lines = src.read_text().splitlines(keepends=True)
    bc_names = [pn.split("/")[2].strip() for pn in bc_pathnames]
    pending = dict(zip(bc_names, flow_series.columns))

    out_lines: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        out_lines.append(line)
        i += 1

        if not line.startswith(_BOUNDARY_LOCATION_PREFIX):
            continue
        bc_name = _boundary_location_name(line)
        if bc_name not in pending:
            continue
        col = pending.pop(bc_name)

        # Copy through lines (e.g. 'Interval=...') until the hydrograph block.
        block_end = i
        while block_end < n and not lines[block_end].startswith(_BOUNDARY_LOCATION_PREFIX):
            if lines[block_end].startswith(_FLOW_HYDROGRAPH_PREFIX):
                break
            block_end += 1
        if block_end >= n or not lines[block_end].startswith(_FLOW_HYDROGRAPH_PREFIX):
            raise ValueError(
                f"'{_FLOW_HYDROGRAPH_PREFIX}' block not found for boundary '{bc_name}'"
            )

        out_lines.extend(lines[i:block_end])  # e.g. 'Interval=1HOUR'
        old_count = int(lines[block_end].split("=", 1)[1].strip())
        old_data_rows = -(-old_count // 10)  # ceil division, 10 values/row

        series = flow_series[col]
        out_lines.append(f"{_FLOW_HYDROGRAPH_PREFIX} {len(series)} \n")
        out_lines.append(_series_to_ras_format(series))

        i = block_end + 1 + old_data_rows

    if pending:
        raise ValueError(
            f"Boundary condition(s) not found in {src.name}: {sorted(pending)}"
        )

    dst.write_text("".join(out_lines))
    print(f"✓ Unsteady file written → {dst.name}")


def _series_to_ras_format(series: pd.Series) -> str:
    """Convert a pandas Series to HEC-RAS fixed-width flow table format."""
    rows = []
    vals = series.values.tolist()
    for i in range(0, len(vals), 10):
        chunk = vals[i : i + 10]
        rows.append("".join(f"{v:8.2f}" for v in chunk) + "\n")
    return "".join(rows)


def modify_plan_file(
    path_project: str,
    name_project: str,
    plan_number: int,
    rainfall_plan_name: int,
) -> None:
    """Update the unsteady flow file reference in a HEC-RAS plan file (.p##).

    Args:
        path_project: Project directory.
        name_project: Project name.
        plan_number: Source plan number.
        rainfall_plan_name: New plan identifier to write, 1-99.
    """
    plan_id = int(rainfall_plan_name)
    plan_path = Path(path_project, name_project + f".p{plan_number:02d}")
    text = plan_path.read_text()
    new_ref = f"Unsteady File={name_project}.u{plan_id:02d}"
    text = re.sub(r"Unsteady File=.*", new_ref, text)
    plan_path.write_text(text)
    print(f"✓ Plan file {plan_path.name} updated.")


def modify_project_file(
    path_project: str,
    name_project: str,
    plan_number: int,
    rainfall_plan_name: int,
) -> None:
    """Set the current plan in the HEC-RAS project file (.prj).

    Args:
        path_project: Project directory.
        name_project: Project name.
        plan_number: Original plan number.
        rainfall_plan_name: New plan number to activate, 1-99.
    """
    plan_id = int(rainfall_plan_name)
    prj_path = Path(path_project, name_project + ".prj")
    text = prj_path.read_text()
    text = re.sub(
        r"Current Plan=.*",
        f"Current Plan={name_project}.p{plan_id:02d}",
        text,
    )
    prj_path.write_text(text)
    print(f"✓ Project file {prj_path.name} updated (plan {plan_id}).")


def create_flow_series(df: pd.DataFrame, col: str, window: int = 5) -> pd.Series:
    """Smooth a discharge series with a rolling maximum.

    Useful for preparing boundary condition inputs where abrupt spikes should
    be avoided.

    Args:
        df: DataFrame containing the raw discharge series.
        col: Column name to process.
        window: Rolling window size in time steps.

    Returns:
        Smoothed discharge Series.
    """
    return df[col].rolling(window, min_periods=1, center=True).max()


# ── Execution ─────────────────────────────────────────────────────────────────

def run_hec_ras(
    path_project: str,
    name_project: str,
    ras_version: int = 641,
) -> None:
    """Open and run HEC-RAS via the rascontrol COM interface (Windows only).

    Args:
        path_project: Project directory.
        name_project: Project name (without extension).
        ras_version: Integer version code (e.g. 641 for v6.4.1, 631 for v6.3.1).

    Raises:
        RuntimeError: If rascontrol cannot connect or the run fails.
    """
    try:
        import rascontrol
    except ImportError as exc:
        raise ImportError("rascontrol is required: pip install rascontrol") from exc

    prj_file = str(Path(path_project, name_project + ".prj"))
    rc = rascontrol.RasController(version=str(ras_version))
    rc.open_project(prj_file)
    rc.run_current_plan()
    rc.close_project()
    print(f"✓ HEC-RAS run completed for {name_project}.")


# ── Output reading ───────────────────────────────────────────────────────────

def read_ras_dss_timeseries(
    dss_path: str,
    pathname_prefix: str,
    plan_name: str | None = None,
) -> pd.DataFrame:
    """Read a HEC-RAS output time series (WSEL, FLOW, STAGE, ...) from DSS.

    HEC-RAS 5.x/6.x write plan results to a per-project ``.dss`` file. Uses
    hecdss (numpy-version-agnostic, cross-platform — see module docstring)
    to read the series, matching pathnames by B/C-part prefix and, if given,
    by the ``PLAN:<plan_name>`` F-part tag.

    Args:
        dss_path: Path to the ``.dss`` output file (usually
            ``<project>.dss`` next to the ``.prj``).
        pathname_prefix: DSS B/C prefix, e.g.
            ``'//White River - Muncie/WSEL'`` or ``'//Perimeter 1/STAGE'``.
        plan_name: Optional plan identifier to disambiguate the F-part when
            several plans' results share the same DSS file.

    Returns:
        DataFrame with ``datetime`` and ``value`` columns. Units are as
        stored by HEC-RAS (WSEL in the project vertical datum; FLOW in the
        project's unit system, typically m3/s or cfs).

    Raises:
        ImportError: If hecdss is not installed.
        ValueError: If no matching pathname is found in the DSS catalog.
    """
    try:
        from hecdss import HecDss
    except ImportError as exc:
        raise ImportError("hecdss is required: pip install hecdss") from exc

    prefix_key = (pathname_prefix.rstrip("/") + "/").upper()
    plan_key = f"PLAN:{plan_name}".upper() if plan_name else None

    with HecDss(str(dss_path)) as dss:
        catalog = dss.get_catalog()
        paths = []
        for path in catalog.uncondensed_paths:
            path_upper = path.upper()
            if not path_upper.startswith(prefix_key):
                continue
            if plan_key and plan_key not in path_upper:
                continue
            paths.append(path)

        if not paths:
            raise ValueError(
                f"No DSS paths found under '{pathname_prefix}'"
                f"{f' (plan {plan_name})' if plan_name else ''} in {dss_path}"
            )

        ts = dss.get(sorted(paths)[-1])
        return pd.DataFrame(
            {
                "datetime": pd.to_datetime(ts.get_dates()),
                "value": np.asarray(ts.get_values(), dtype=float),
            }
        )


def read_ras_max_wsel(
    dss_path: str,
    pathname_prefix: str,
    plan_name: str | None = None,
) -> float:
    """Peak value of a HEC-RAS DSS output series (e.g. max WSEL for a plan).

    Convenience wrapper around :func:`read_ras_dss_timeseries` for building a
    return-period WSEL curve from one plan per return period.

    Args:
        dss_path: Path to the ``.dss`` output file.
        pathname_prefix: DSS B/C prefix, e.g.
            ``'//White River - Muncie/WSEL'``.
        plan_name: Optional plan identifier (``PLAN:<plan_name>`` F-part tag).

    Returns:
        Maximum value in the matched series.
    """
    df = read_ras_dss_timeseries(dss_path, pathname_prefix, plan_name=plan_name)
    return float(df["value"].max())

"""
HEC-RAS automation utilities.

Provides functions to modify HEC-RAS input files programmatically and run
simulations from Python, enabling large-scale scenario automation.

Requires:
    - HEC-RAS installed (version 6.x recommended, Windows).
    - rascontrol: ``pip install rascontrol``
    - pydsstools: ``pip install pydsstools``
    - pandas, numpy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ── File modification helpers ─────────────────────────────────────────────────

def modify_unsteady_file(
    path_project: str,
    name_project: str,
    file_number: int,
    rainfall_plan_name: int,
    flow_series: pd.DataFrame,
    bc_pathnames: list[str],
) -> None:
    """Write a hyetograph / hydrograph into a HEC-RAS unsteady flow file (.u##).

    Reads the existing file, updates the flow data for every BC line, and
    writes the modified content back.

    Args:
        path_project: Project directory.
        name_project: Project name (without extension).
        file_number: Unsteady flow file number (e.g. 1 → '.u01').
        rainfall_plan_name: Integer used as the new plan identifier (e.g. 99).
        flow_series: DataFrame with a datetime index and one column per BC line.
        bc_pathnames: List of DSS pathnames identifying each BC line, in the
                      same order as flow_series columns.
    """
    src = Path(path_project, name_project + f".u{file_number:02d}").read_text()
    dst = Path(path_project, name_project + f".u{rainfall_plan_name:02d}")

    # Replace flow data section for each BC line
    lines = src.splitlines(keepends=True)
    out_lines: list[str] = []
    bc_iter = iter(zip(bc_pathnames, flow_series.columns))
    for line in lines:
        out_lines.append(line)
        for pn, col in bc_iter:
            if pn.split("/")[2] in line:
                series_str = _series_to_ras_format(flow_series[col])
                out_lines.append(series_str)
                break

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
        rainfall_plan_name: New plan identifier to write.
    """
    plan_path = Path(path_project, name_project + f".p{plan_number:02d}")
    text = plan_path.read_text()
    new_ref = f"Unsteady File={name_project}.u{rainfall_plan_name:02d}"
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
        rainfall_plan_name: New plan number to activate.
    """
    prj_path = Path(path_project, name_project + ".prj")
    text = prj_path.read_text()
    text = re.sub(
        r"Current Plan=.*",
        f"Current Plan={name_project}.p{rainfall_plan_name:02d}",
        text,
    )
    prj_path.write_text(text)
    print(f"✓ Project file {prj_path.name} updated (plan {rainfall_plan_name}).")


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

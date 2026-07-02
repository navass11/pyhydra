from __future__ import annotations

import numpy as np
import pandas as pd

from pyhydra.modeling.hydraulic import hec_ras
from pyhydra.modeling.hydrology import swat


def test_swatplus_precipitation_files_include_cli_and_missing_values(tmp_path):
    stations = pd.DataFrame(
        {
            "name": ["p1", "p2"],
            "lat": [43.0, 43.5],
            "lon": [-4.0, -4.5],
            "elev": [100.0, 200.0],
        }
    )
    dates = pd.date_range("2024-01-01", periods=2)
    series = pd.DataFrame(
        {"p1": [1.2, np.nan], "p2": [3.4, 5.6]},
        index=dates,
    )

    swat.write_swatplus_precipitation_files(stations, series, tmp_path)

    assert (tmp_path / "pcp.cli").read_text() == (
        "pcp.cli: Precipitation station files - written by pyhydra\n"
        "filename\n"
        "p1.pcp\n"
        "p2.pcp\n"
    )
    assert "-99.00000" in (tmp_path / "p1.pcp").read_text()
    assert "2024    1     1.20000" in (tmp_path / "p1.pcp").read_text()


def test_swatplus_temperature_files_write_station_files_and_cli(tmp_path):
    stations = pd.DataFrame(
        {
            "name": ["t1"],
            "lat": [42.0],
            "lon": [-3.0],
            "elev": [900.0],
        }
    )
    dates = pd.date_range("2024-02-01", periods=1)
    tmax = pd.DataFrame({"t1": [12.5]}, index=dates)
    tmin = pd.DataFrame({"t1": [2.5]}, index=dates)

    swat.write_swatplus_temperature_files(stations, tmax, tmin, tmp_path)

    assert (tmp_path / "tmp.cli").read_text().endswith("filename\nt1.tmp\n")
    assert "2024   32    12.50000     2.50000" in (tmp_path / "t1.tmp").read_text()


def test_edit_file_cio_updates_start_year_and_number_of_years(tmp_path):
    cio = tmp_path / "file.cio"
    cio.write_text("header\n   1990    IYR\n   1    NBYR\nfooter\n")

    swat.edit_file_cio(str(cio), start_year=2001, end_year=2005)

    text = cio.read_text()
    assert "   2001    IYR" in text
    assert "   5    NBYR" in text


def test_hec_ras_series_format_uses_ten_values_per_line():
    series = pd.Series(np.arange(12, dtype=float))

    lines = hec_ras._series_to_ras_format(series).splitlines()

    assert len(lines) == 2
    assert lines[0].count(".00") == 10
    assert lines[1].count(".00") == 2


def test_hec_ras_plan_and_project_modifiers_update_references(tmp_path):
    (tmp_path / "Demo.p01").write_text("Header\nUnsteady File=Demo.u01\nFooter\n")
    (tmp_path / "Demo.prj").write_text("Header\nCurrent Plan=Demo.p01\nFooter\n")

    hec_ras.modify_plan_file(str(tmp_path), "Demo", plan_number=1, rainfall_plan_name=7)
    hec_ras.modify_project_file(str(tmp_path), "Demo", plan_number=1, rainfall_plan_name=7)

    assert "Unsteady File=Demo.u07" in (tmp_path / "Demo.p01").read_text()
    assert "Current Plan=Demo.p07" in (tmp_path / "Demo.prj").read_text()


def test_create_flow_series_returns_centered_rolling_maximum():
    data = pd.DataFrame({"q": [1.0, 5.0, 2.0, 4.0, 1.0]})

    result = hec_ras.create_flow_series(data, "q", window=3)

    assert result.tolist() == [5.0, 5.0, 5.0, 4.0, 4.0]

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from pyhydra.modeling.hydraulic import hec_ras
from pyhydra.modeling.hydrology import swat

# SWAT+ channel_sd_*.txt header, trimmed to the columns exercised by the
# reader but keeping the real file's quirks: fixed 3-line preamble and a
# 'null' column name repeated for the (unlabeled, m3/s) flow-rate summary
# fields — see pyhydra/modeling/hydrology/swat.py:read_channel_sd.
_CHANNEL_SD_HEADER = (
    " Test watershed             SWAT+ 2024-07-08\n"
    "   jday   mon   day    yr     unit   name         flo_stor      null    flo_out      null\n"
    "                                                       m^3/s               m^3/s\n"
)


def _write_channel_sd(tmp_path, freq, rows):
    path = tmp_path / f"channel_sd_{freq}.txt"
    lines = [_CHANNEL_SD_HEADER]
    for row in rows:
        lines.append(
            f"{row['jday']:7d}{row['mon']:6d}{row['day']:6d}{row['yr']:6d}"
            f"{row['unit']:9d} {row['name']:<10} {row['flo_stor']:10.3f}"
            f"{row['null1']:10.3f}{row['flo_out']:10.3f}{row['null2']:10.3f}\n"
        )
    path.write_text("".join(lines))
    return path


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


def test_swatplus_precipitation_files_write_actual_year_count_as_nbyr(tmp_path):
    # Regression test: nbyr=0 was hardcoded regardless of the series length,
    # which SWAT+ uses to size its read buffers — a file with real multi-year
    # data but nbyr=0 crashes the SWAT+ runtime (SIGSEGV) instead of just
    # failing to parse.
    stations = pd.DataFrame({"name": ["p1"], "lat": [43.0], "lon": [-4.0], "elev": [100.0]})
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
    series = pd.DataFrame({"p1": np.linspace(0, 10, len(dates))}, index=dates)

    swat.write_swatplus_precipitation_files(stations, series, tmp_path)

    header = (tmp_path / "p1.pcp").read_text().splitlines()[2]
    assert header.split()[0] == "3"  # 2020-2022 inclusive


def test_swatplus_temperature_files_write_actual_year_count_as_nbyr(tmp_path):
    stations = pd.DataFrame({"name": ["t1"], "lat": [42.0], "lon": [-3.0], "elev": [900.0]})
    dates = pd.date_range("2020-01-01", "2021-12-31", freq="D")
    tmax = pd.DataFrame({"t1": np.linspace(10, 20, len(dates))}, index=dates)
    tmin = pd.DataFrame({"t1": np.linspace(0, 10, len(dates))}, index=dates)

    swat.write_swatplus_temperature_files(stations, tmax, tmin, tmp_path)

    header = (tmp_path / "t1.tmp").read_text().splitlines()[2]
    assert header.split()[0] == "2"  # 2020-2021 inclusive


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


def test_edit_file_cio_updates_time_sim_simulation_period(tmp_path):
    # SWAT+ rev 60+ (SWAT+ Editor projects) has no IYR/NBYR fields in
    # file.cio — the simulation period lives in time.sim instead. Regression
    # test: edit_file_cio was previously a silent no-op against real SWAT+
    # Editor projects, leaving stale simulation years in place.
    cio = tmp_path / "file.cio"
    cio.write_text("file.cio: written by SWAT+ editor\nsimulation  time.sim  print.prt\n")
    time_sim = tmp_path / "time.sim"
    time_sim.write_text(
        "time.sim: written by SWAT+ editor\n"
        "day_start  yrc_start   day_end   yrc_end      step  \n"
        "       0      2010       364      2015         0  \n"
    )

    swat.edit_file_cio(str(cio), start_year=2021, end_year=2040)

    lines = time_sim.read_text().splitlines()
    assert lines[2].split() == ["0", "2021", "364", "2040", "0"]
    # file.cio itself is untouched — it only lists filenames in this revision.
    assert cio.read_text() == "file.cio: written by SWAT+ editor\nsimulation  time.sim  print.prt\n"


def test_edit_file_cio_raises_when_time_sim_missing(tmp_path):
    cio = tmp_path / "file.cio"
    cio.write_text("file.cio: written by SWAT+ editor\n")

    with pytest.raises(FileNotFoundError, match="time.sim"):
        swat.edit_file_cio(str(cio), start_year=2021, end_year=2040)


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


# ── SWAT+ output reader ────────────────────────────────────────────────────

def test_read_channel_sd_day_builds_date_index_and_filters_unit(tmp_path):
    rows = [
        {"jday": 1, "mon": 1, "day": 1, "yr": 2020, "unit": 1, "name": "cha001",
         "flo_stor": 0.0, "null1": -5.7, "flo_out": 1.23, "null2": -5.7},
        {"jday": 1, "mon": 1, "day": 1, "yr": 2020, "unit": 2, "name": "cha002",
         "flo_stor": 0.0, "null1": -5.7, "flo_out": 2.34, "null2": -5.7},
        {"jday": 2, "mon": 1, "day": 2, "yr": 2020, "unit": 1, "name": "cha001",
         "flo_stor": 0.0, "null1": -5.7, "flo_out": 1.50, "null2": -5.7},
    ]
    _write_channel_sd(tmp_path, "day", rows)

    df_all = swat.read_channel_sd(str(tmp_path), freq="day")
    assert list(df_all.columns) == ["jday", "mon", "day", "yr", "unit", "name",
                                     "flo_stor", "null", "flo_out", "null_1"]
    assert len(df_all) == 3

    df_unit1 = swat.read_channel_sd(str(tmp_path), freq="day", unit=1)
    assert df_unit1["flo_out"].tolist() == [1.23, 1.50]
    assert list(df_unit1.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]


def test_read_channel_sd_mon_uses_month_end_index(tmp_path):
    rows = [
        {"jday": 31, "mon": 1, "day": 31, "yr": 2020, "unit": 1, "name": "cha001",
         "flo_stor": 0.0, "null1": -4.9, "flo_out": 10.0, "null2": 0.0},
    ]
    _write_channel_sd(tmp_path, "mon", rows)

    df = swat.read_channel_sd(str(tmp_path), freq="mon", unit=1)

    assert list(df.index) == [pd.Timestamp("2020-01-31")]


def test_read_channel_sd_rejects_unknown_freq(tmp_path):
    with pytest.raises(ValueError, match="Unknown freq"):
        swat.read_channel_sd(str(tmp_path), freq="week")


def test_read_swat_discharge_returns_flo_out_series(tmp_path):
    rows = [
        {"jday": 1, "mon": 1, "day": 1, "yr": 2020, "unit": 3, "name": "cha003",
         "flo_stor": 0.0, "null1": 0.0, "flo_out": 7.5, "null2": 0.0},
    ]
    _write_channel_sd(tmp_path, "day", rows)

    q = swat.read_swat_discharge(str(tmp_path), unit=3, freq="day")

    assert q.name == "flo_out"
    assert q.tolist() == [7.5]


# ── HEC-RAS unsteady-file hydrograph replacement ────────────────────────────

def _write_unsteady_fixture(tmp_path, filename="Demo.u01"):
    # Mirrors the real structure of a HEC-RAS 2D .u## file: a Boundary
    # Location line, an Interval line, a 'Flow Hydrograph= N' count line,
    # N fixed-width values (10/row), then unrelated trailer lines.
    text = (
        "Flow Title=Demo\n"
        "Program Version=6.20\n"
        "Boundary Location=                ,                ,        ,        ,"
        "                ,2D              ,                ,Upstream                         ,\n"
        "Interval=1HOUR\n"
        "Flow Hydrograph= 12 \n"
        + hec_ras._series_to_ras_format(pd.Series(np.arange(12, dtype=float)))
        + "Stage Hydrograph TW Check=0\n"
        "Flow Hydrograph Slope= 0.01 \n"
        "DSS Path=\n"
        "Boundary Location=                ,                ,        ,        ,"
        "                ,2D              ,                ,Downstream                       ,\n"
        "Friction Slope=0.01,0\n"
    )
    (tmp_path / filename).write_text(text)


def test_modify_unsteady_file_replaces_flow_hydrograph_block(tmp_path):
    _write_unsteady_fixture(tmp_path)
    new_series = pd.DataFrame({"flow": [100.0, 200.0, 300.0]})

    hec_ras.modify_unsteady_file(
        path_project=str(tmp_path),
        name_project="Demo",
        file_number=1,
        rainfall_plan_name=7,
        flow_series=new_series,
        bc_pathnames=["/BCLINE/Upstream/FLOW/Sep2008/10Minute/plan:base/"],
    )

    out = (tmp_path / "Demo.u07").read_text()
    assert "Flow Hydrograph= 3 \n" in out
    assert "  100.00  200.00  300.00\n" in out
    # Old 12-value block must be gone, trailer and other BC untouched.
    assert "10.00" not in out
    assert "Stage Hydrograph TW Check=0" in out
    assert "Downstream" in out
    assert out.count("Boundary Location=") == 2


def test_modify_unsteady_file_raises_for_unmatched_boundary(tmp_path):
    _write_unsteady_fixture(tmp_path)
    new_series = pd.DataFrame({"flow": [100.0]})

    with pytest.raises(ValueError, match="not found"):
        hec_ras.modify_unsteady_file(
            path_project=str(tmp_path),
            name_project="Demo",
            file_number=1,
            rainfall_plan_name=7,
            flow_series=new_series,
            bc_pathnames=["/BCLINE/NoSuchBoundary/FLOW/Sep2008/10Minute/plan:base/"],
        )


def test_modify_unsteady_file_rejects_non_numeric_plan_name(tmp_path):
    _write_unsteady_fixture(tmp_path)
    new_series = pd.DataFrame({"flow": [100.0]})

    with pytest.raises(ValueError):
        hec_ras.modify_unsteady_file(
            path_project=str(tmp_path),
            name_project="Demo",
            file_number=1,
            rainfall_plan_name="T2yr",
            flow_series=new_series,
            bc_pathnames=["/BCLINE/Upstream/FLOW/Sep2008/10Minute/plan:base/"],
        )


# ── HEC-RAS DSS output reader (hecdss mocked: the native library only ──────
# ── ships x86_64 binaries, see docker/Dockerfile.jupyter) ───────────────────

class _FakeTimeSeries:
    def __init__(self, dates, values):
        self._dates = dates
        self._values = values

    def get_dates(self):
        return self._dates

    def get_values(self):
        return self._values


class _FakeCatalog:
    def __init__(self, paths):
        self.uncondensed_paths = paths


class _FakeHecDss:
    _paths = [
        "//White River - Muncie/WSEL/01JAN2020/1HOUR/PLAN:T2yr/",
        "//White River - Muncie/WSEL/01JAN2020/1HOUR/PLAN:T100yr/",
        "//White River - Muncie/FLOW/01JAN2020/1HOUR/PLAN:T100yr/",
    ]
    _series = {
        "//White River - Muncie/WSEL/01JAN2020/1HOUR/PLAN:T2yr/":
            _FakeTimeSeries(["2020-01-01", "2020-01-02"], [10.0, 11.0]),
        "//White River - Muncie/WSEL/01JAN2020/1HOUR/PLAN:T100yr/":
            _FakeTimeSeries(["2020-01-01", "2020-01-02"], [20.0, 25.0]),
    }

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_catalog(self):
        return _FakeCatalog(self._paths)

    def get(self, pathname):
        return self._series[pathname]


@pytest.fixture
def fake_hecdss(monkeypatch):
    fake_module = types.SimpleNamespace(HecDss=_FakeHecDss)
    monkeypatch.setitem(sys.modules, "hecdss", fake_module)
    yield fake_module


def test_read_ras_dss_timeseries_filters_by_prefix_and_plan(fake_hecdss):
    df = hec_ras.read_ras_dss_timeseries(
        "dummy.dss", "//White River - Muncie/WSEL", plan_name="T100yr"
    )

    assert df["value"].tolist() == [20.0, 25.0]
    assert list(df["datetime"]) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]


def test_read_ras_dss_timeseries_raises_when_no_path_matches(fake_hecdss):
    with pytest.raises(ValueError, match="No DSS paths found"):
        hec_ras.read_ras_dss_timeseries("dummy.dss", "//Nonexistent/STAGE")


def test_read_ras_max_wsel_returns_peak_value(fake_hecdss):
    peak = hec_ras.read_ras_max_wsel(
        "dummy.dss", "//White River - Muncie/WSEL", plan_name="T100yr"
    )

    assert peak == 25.0

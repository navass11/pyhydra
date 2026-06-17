"""
Tests for pyhydra.data_sources.river_discharge (GRDC, USGS, GloFAS).

All tests are offline:
  - GRDC   : synthetic GRDC-format file written to a tmp directory
  - USGS   : requests.get mocked with minimal NWIS JSON
  - GloFAS : ImportError path tested (cdsapi not required)
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pyhydra.data_sources.river_discharge.grdc import (
    _parse_data_lines,
    _parse_header,
    _split_grdc_file,
    analyze_grdc_quality,
    read_grdc,
    read_grdc_folder,
    read_grdc_metadata,
)
from pyhydra.data_sources.river_discharge.usgs import (
    _parse_nwis_json,
    download_usgs,
    get_usgs_site_info,
    search_usgs_sites,
)


# ---------------------------------------------------------------------------
# GRDC fixtures
# ---------------------------------------------------------------------------

GRDC_CONTENT = textwrap.dedent("""\
    # GRDC-No.: 1234567
    # River: TEST RIVER
    # Station: MY STATION
    # Country: ES
    # Latitude: 40.123
    # Longitude: -3.456
    # Altitude: 850
    # Catchment area: 1200
    # YYYY-MM-DD;hh;Original;Calculated;Qualifier
    # format row (ignored)
    2000-01-01; 00; 45.30; 45.30; G
    2000-01-02; 00; 50.10; 50.10; G
    2000-01-03; 00; -999.00; -999.00; M
    2000-01-04; 00; 38.70; 38.70; G
""")


@pytest.fixture
def grdc_file(tmp_path) -> Path:
    p = tmp_path / "1234567.day"
    p.write_text(GRDC_CONTENT, encoding="latin-1")
    return p


# ---------------------------------------------------------------------------
# USGS fixtures
# ---------------------------------------------------------------------------

def _make_nwis_response(site: str, values: list[dict]) -> dict:
    """Build a minimal NWIS JSON structure."""
    return {
        "value": {
            "timeSeries": [
                {
                    "sourceInfo": {"siteCode": [{"value": site}]},
                    "values": [{"value": values}],
                }
            ]
        }
    }


NWIS_VALUES = [
    {"dateTime": "2010-01-01T00:00:00.000-05:00", "value": "100.0", "qualifiers": ["P"]},
    {"dateTime": "2010-01-02T00:00:00.000-05:00", "value": "120.5", "qualifiers": ["P"]},
    {"dateTime": "2010-01-03T00:00:00.000-05:00", "value": "-1.0",  "qualifiers": ["P"]},   # negative → NaN
    {"dateTime": "2010-01-04T00:00:00.000-05:00", "value": "95.0",  "qualifiers": ["Ice"]}, # Ice → NaN
    {"dateTime": "2010-01-05T00:00:00.000-05:00", "value": "bad",   "qualifiers": []},      # bad float → skipped
    {"dateTime": "2010-01-06T00:00:00.000-05:00", "value": "80.0",  "qualifiers": []},
]

NWIS_SITE_RDB = """\
# comment line
agency_cd\tsite_no\tstation_nm\tdec_lat_va\tdec_long_va\tdrain_area_va
5s\t15s\t50s\t16s\t16s\t8s
USGS\t08279500\tRIO GRANDE AT EMBUDO, NM\t36.2072\t-105.9589\t14124.8
"""


# ---------------------------------------------------------------------------
# GRDC: _parse_data_lines
# ---------------------------------------------------------------------------

class TestGRDCParseDataLines:
    def test_normal_rows(self):
        lines = [
            "2000-01-01; 00; 45.30; 45.30; G\n",
            "2000-01-02; 00; 50.10; 50.10; G\n",
        ]
        df = _parse_data_lines(lines)
        assert len(df) == 2
        assert list(df.columns) == ["date", "discharge"]
        assert df["discharge"].iloc[0] == pytest.approx(45.30)

    def test_missing_value_sentinel(self):
        lines = ["2000-01-01; 00; -999.00; -999.00; M\n"]
        df = _parse_data_lines(lines)
        assert df["discharge"].isna().all()

    def test_very_negative_sentinel(self):
        lines = ["2000-01-01; 00; -9999.0; -9999.0; M\n"]
        df = _parse_data_lines(lines)
        assert df["discharge"].isna().all()

    def test_skips_comment_lines(self):
        lines = [
            "# this is a comment\n",
            "2000-01-01; 00; 10.0; 10.0; G\n",
        ]
        df = _parse_data_lines(lines)
        assert len(df) == 1

    def test_skips_empty_lines(self):
        lines = ["   \n", "2000-01-01; 00; 10.0; 10.0; G\n"]
        df = _parse_data_lines(lines)
        assert len(df) == 1

    def test_bad_date_skipped(self):
        lines = ["not-a-date; 00; 10.0; 10.0; G\n"]
        df = _parse_data_lines(lines)
        assert df.empty

    def test_bad_value_becomes_nan(self):
        lines = ["2000-01-01; 00; text; text; G\n"]
        df = _parse_data_lines(lines)
        assert len(df) == 1
        assert pd.isna(df["discharge"].iloc[0])

    def test_sorted_output(self):
        lines = [
            "2000-01-03; 00; 30.0; 30.0; G\n",
            "2000-01-01; 00; 10.0; 10.0; G\n",
            "2000-01-02; 00; 20.0; 20.0; G\n",
        ]
        df = _parse_data_lines(lines)
        assert list(df["discharge"]) == [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# GRDC: _parse_header
# ---------------------------------------------------------------------------

class TestGRDCParseHeader:
    def test_station_and_river(self):
        lines = [
            "# River: TEST RIVER\n",
            "# Station: MY STATION\n",
        ]
        meta = _parse_header(lines)
        assert meta["river"] == "TEST RIVER"
        assert meta["station"] == "MY STATION"

    def test_numeric_fields(self):
        lines = [
            "# Latitude: 40.123\n",
            "# Longitude: -3.456\n",
            "# Altitude: 850\n",
        ]
        meta = _parse_header(lines)
        assert meta["latitude"] == pytest.approx(40.123)
        assert meta["longitude"] == pytest.approx(-3.456)
        assert meta["altitude_m"] == pytest.approx(850)

    def test_catchment_area(self):
        lines = ["# Catchment area: 1200\n"]
        meta = _parse_header(lines)
        assert meta["catchment_area_km2"] == pytest.approx(1200)

    def test_grdc_no(self):
        lines = ["# GRDC-No.: 1234567\n"]
        meta = _parse_header(lines)
        assert meta["grdc_no"] == pytest.approx(1234567)


# ---------------------------------------------------------------------------
# GRDC: read_grdc (full file)
# ---------------------------------------------------------------------------

class TestReadGRDC:
    def test_returns_dataframe(self, grdc_file):
        df = read_grdc(grdc_file)
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "discharge" in df.columns

    def test_row_count(self, grdc_file):
        df = read_grdc(grdc_file)
        assert len(df) == 4

    def test_missing_row_is_nan(self, grdc_file):
        df = read_grdc(grdc_file)
        assert pd.isna(df.loc[df["date"] == pd.Timestamp("2000-01-03"), "discharge"].iloc[0])

    def test_valid_rows_not_nan(self, grdc_file):
        df = read_grdc(grdc_file)
        assert not pd.isna(df.loc[df["date"] == pd.Timestamp("2000-01-01"), "discharge"].iloc[0])

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_grdc(tmp_path / "nonexistent.day")


# ---------------------------------------------------------------------------
# GRDC: read_grdc_metadata
# ---------------------------------------------------------------------------

class TestReadGRDCMetadata:
    def test_returns_dict(self, grdc_file):
        meta = read_grdc_metadata(grdc_file)
        assert isinstance(meta, dict)

    def test_station_name(self, grdc_file):
        meta = read_grdc_metadata(grdc_file)
        assert meta["station"] == "MY STATION"

    def test_latitude(self, grdc_file):
        meta = read_grdc_metadata(grdc_file)
        assert meta["latitude"] == pytest.approx(40.123)


# ---------------------------------------------------------------------------
# GRDC: read_grdc_folder
# ---------------------------------------------------------------------------

class TestReadGRDCFolder:
    def test_reads_all_files(self, tmp_path):
        for name in ("a.day", "b.day"):
            (tmp_path / name).write_text(GRDC_CONTENT, encoding="latin-1")
        result = read_grdc_folder(tmp_path, "*.day")
        assert set(result.keys()) == {"a", "b"}
        for df in result.values():
            assert isinstance(df, pd.DataFrame)

    def test_empty_folder(self, tmp_path):
        result = read_grdc_folder(tmp_path, "*.day")
        assert result == {}

    def test_skips_unreadable_file(self, tmp_path, capsys):
        bad = tmp_path / "bad.day"
        bad.write_text("no data at all", encoding="latin-1")
        # must not raise; warning printed to stdout
        result = read_grdc_folder(tmp_path, "*.day")
        captured = capsys.readouterr()
        assert "Warning" in captured.out


# ---------------------------------------------------------------------------
# GRDC: analyze_grdc_quality
# ---------------------------------------------------------------------------

class TestAnalyzeGRDCQuality:
    def test_returns_expected_keys(self, grdc_file):
        df = read_grdc(grdc_file)
        stats = analyze_grdc_quality(df)
        assert {"start", "end", "n_days", "missing_pct", "mean_m3s", "max_m3s"} == set(stats)

    def test_n_days(self, grdc_file):
        df = read_grdc(grdc_file)
        assert analyze_grdc_quality(df)["n_days"] == 4

    def test_missing_pct(self, grdc_file):
        df = read_grdc(grdc_file)
        # 1 NaN out of 4 rows = 25%
        assert analyze_grdc_quality(df)["missing_pct"] == pytest.approx(25.0)

    def test_empty_df_returns_empty_dict(self):
        assert analyze_grdc_quality(pd.DataFrame()) == {}

    def test_max_value(self, grdc_file):
        df = read_grdc(grdc_file)
        assert analyze_grdc_quality(df)["max_m3s"] == pytest.approx(50.10)


# ---------------------------------------------------------------------------
# USGS: _parse_nwis_json
# ---------------------------------------------------------------------------

class TestParseNWISJson:
    def test_basic_parse(self):
        data = _make_nwis_response("08279500", NWIS_VALUES)
        df = _parse_nwis_json(data, "08279500")
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns and "discharge" in df.columns

    def test_row_count_excludes_bad_float(self):
        # bad float is skipped entirely (not NaN); 5 remaining rows
        data = _make_nwis_response("08279500", NWIS_VALUES)
        df = _parse_nwis_json(data, "08279500")
        assert len(df) == 5

    def test_negative_value_becomes_nan(self):
        # NWIS_VALUES[2] = 2010-01-03, value=-1 → NaN; sorted index = 2
        data = _make_nwis_response("08279500", NWIS_VALUES)
        df = _parse_nwis_json(data, "08279500")
        assert pd.isna(df["discharge"].iloc[2])

    def test_ice_qualifier_becomes_nan(self):
        # NWIS_VALUES[3] = 2010-01-04, qualifier=Ice → NaN; sorted index = 3
        data = _make_nwis_response("08279500", NWIS_VALUES)
        df = _parse_nwis_json(data, "08279500")
        assert pd.isna(df["discharge"].iloc[3])

    def test_valid_value_preserved(self):
        # NWIS_VALUES[0] = 2010-01-01, value=100.0; sorted index = 0
        data = _make_nwis_response("08279500", NWIS_VALUES)
        df = _parse_nwis_json(data, "08279500")
        assert df["discharge"].iloc[0] == pytest.approx(100.0)

    def test_empty_timeseries_returns_empty_df(self):
        data = {"value": {"timeSeries": []}}
        df = _parse_nwis_json(data, "08279500")
        assert df.empty

    def test_malformed_json_returns_empty_df(self):
        df = _parse_nwis_json({}, "08279500")
        assert df.empty

    def test_output_is_sorted_by_date(self):
        shuffled = list(reversed(NWIS_VALUES))
        data = _make_nwis_response("08279500", shuffled)
        df = _parse_nwis_json(data, "08279500")
        assert df["date"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# USGS: download_usgs (mocked)
# ---------------------------------------------------------------------------

class TestDownloadUSGS:
    def _mock_response(self):
        data = _make_nwis_response("08279500", [
            {"dateTime": "2010-01-01T00:00:00.000-05:00", "value": "100.0", "qualifiers": []},
            {"dateTime": "2010-01-02T00:00:00.000-05:00", "value": "200.0", "qualifiers": []},
        ])
        mock_r = MagicMock()
        mock_r.json.return_value = data
        mock_r.raise_for_status.return_value = None
        return mock_r

    def test_returns_dataframe(self):
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=self._mock_response()):
            df = download_usgs("08279500", "2010-01-01", "2010-01-02")
        assert isinstance(df, pd.DataFrame)

    def test_column_named_q_site(self):
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=self._mock_response()):
            df = download_usgs("08279500", "2010-01-01", "2010-01-02")
        assert "Q_08279500" in df.columns

    def test_metric_conversion(self):
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=self._mock_response()):
            df = download_usgs("08279500", "2010-01-01", "2010-01-02", units="metric")
        # 100 ft³/s * 0.028316847 ≈ 2.832 m³/s
        assert df["Q_08279500"].iloc[0] == pytest.approx(100.0 * 0.028316847, rel=1e-5)

    def test_imperial_no_conversion(self):
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=self._mock_response()):
            df = download_usgs("08279500", "2010-01-01", "2010-01-02", units="imperial")
        assert df["Q_08279500"].iloc[0] == pytest.approx(100.0)

    def test_multiple_sites(self):
        mock_r = self._mock_response()
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=mock_r):
            df = download_usgs(["08279500", "01646500"], "2010-01-01", "2010-01-02")
        assert "Q_08279500" in df.columns
        assert "Q_01646500" in df.columns

    def test_empty_response_returns_empty_df(self):
        mock_r = MagicMock()
        mock_r.json.return_value = {"value": {"timeSeries": []}}
        mock_r.raise_for_status.return_value = None
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=mock_r):
            df = download_usgs("00000000", "2010-01-01", "2010-01-02")
        assert df.empty

    def test_index_name_is_date(self):
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=self._mock_response()):
            df = download_usgs("08279500", "2010-01-01", "2010-01-02")
        assert df.index.name == "date"

    def test_retry_on_failure_then_success(self):
        mock_ok = self._mock_response()
        mock_fail = MagicMock()
        mock_fail.raise_for_status.side_effect = Exception("timeout")

        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   side_effect=[mock_fail, mock_ok]), \
             patch("pyhydra.data_sources.river_discharge.usgs.time.sleep"):
            df = download_usgs("08279500", "2010-01-01", "2010-01-02", max_retries=3)
        assert not df.empty


# ---------------------------------------------------------------------------
# USGS: get_usgs_site_info / search_usgs_sites (mocked)
# ---------------------------------------------------------------------------

class TestGetUSGSSiteInfo:
    def _mock_rdb(self):
        mock_r = MagicMock()
        mock_r.text = NWIS_SITE_RDB
        mock_r.raise_for_status.return_value = None
        return mock_r

    def test_returns_dataframe(self):
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=self._mock_rdb()):
            df = get_usgs_site_info("08279500")
        assert isinstance(df, pd.DataFrame)

    def test_site_no_column(self):
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=self._mock_rdb()):
            df = get_usgs_site_info("08279500")
        assert "site_no" in df.columns

    def test_drain_area_km2_computed(self):
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=self._mock_rdb()):
            df = get_usgs_site_info("08279500")
        assert "drain_area_km2" in df.columns
        assert df["drain_area_km2"].iloc[0] == pytest.approx(14124.8 * 2.58999, rel=1e-4)

    def test_lat_lon_numeric(self):
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=self._mock_rdb()):
            df = get_usgs_site_info("08279500")
        assert df["dec_lat_va"].iloc[0] == pytest.approx(36.2072)
        assert df["dec_long_va"].iloc[0] == pytest.approx(-105.9589)

    def test_accepts_list_of_sites(self):
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=self._mock_rdb()):
            df = get_usgs_site_info(["08279500", "01646500"])
        assert isinstance(df, pd.DataFrame)


class TestSearchUSGSSites:
    def test_returns_dataframe(self):
        mock_r = MagicMock()
        mock_r.text = NWIS_SITE_RDB
        mock_r.raise_for_status.return_value = None
        with patch("pyhydra.data_sources.river_discharge.usgs.requests.get",
                   return_value=mock_r):
            df = search_usgs_sites((-106.5, 35.0, -104.0, 37.5))
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# GloFAS: mock cdsapi — no real network calls
# ---------------------------------------------------------------------------

import importlib
import sys

from pyhydra.data_sources.river_discharge import glofas as _glofas_mod


def _glofas_with_mock_cdsapi(mock_cdsapi):
    """Return the glofas module reloaded with a mock cdsapi injected."""
    with patch.dict(sys.modules, {"cdsapi": mock_cdsapi}):
        importlib.reload(_glofas_mod)
        yield _glofas_mod
    # restore real state after yield
    importlib.reload(_glofas_mod)


class TestGloFASDownload:
    """Tests for download_glofas and download_glofas_by_year using mocked cdsapi."""

    def _make_cdsapi_mock(self):
        mock_cdsapi = MagicMock()
        mock_client  = MagicMock()
        mock_cdsapi.Client.return_value = mock_client
        return mock_cdsapi, mock_client

    # --- import guard ---

    def test_raises_import_error_without_cdsapi(self, tmp_path):
        with patch.dict(sys.modules, {"cdsapi": None}):
            importlib.reload(_glofas_mod)
            with pytest.raises(ImportError, match="cdsapi"):
                _glofas_mod.download_glofas(
                    area=[44, -10, 35, 5], years=[2000],
                    output_dir=str(tmp_path),
                )
        importlib.reload(_glofas_mod)

    def test_raises_import_error_by_year_without_cdsapi(self, tmp_path):
        with patch.dict(sys.modules, {"cdsapi": None}):
            importlib.reload(_glofas_mod)
            with pytest.raises(ImportError, match="cdsapi"):
                _glofas_mod.download_glofas_by_year(
                    area=[44, -10, 35, 5], years=[2000],
                    output_dir=str(tmp_path),
                )
        importlib.reload(_glofas_mod)

    # --- EWDS URL ---

    def test_default_url_is_ewds(self, tmp_path):
        mock_cdsapi, _ = self._make_cdsapi_mock()
        with patch.dict(sys.modules, {"cdsapi": mock_cdsapi}):
            importlib.reload(_glofas_mod)
            _glofas_mod.download_glofas(
                area=[44, -10, 35, 5], years=[2000],
                output_dir=str(tmp_path),
            )
        call_kwargs = mock_cdsapi.Client.call_args[1]
        assert call_kwargs["url"] == "https://ewds.climate.copernicus.eu/api"
        importlib.reload(_glofas_mod)

    def test_custom_api_key_passed_to_client(self, tmp_path):
        mock_cdsapi, _ = self._make_cdsapi_mock()
        with patch.dict(sys.modules, {"cdsapi": mock_cdsapi}):
            importlib.reload(_glofas_mod)
            _glofas_mod.download_glofas(
                area=[44, -10, 35, 5], years=[2000],
                output_dir=str(tmp_path), api_key="mykey123",
            )
        call_kwargs = mock_cdsapi.Client.call_args[1]
        assert call_kwargs["key"] == "mykey123"
        importlib.reload(_glofas_mod)

    # --- request structure ---

    def test_retrieve_called_with_correct_dataset(self, tmp_path):
        mock_cdsapi, mock_client = self._make_cdsapi_mock()
        with patch.dict(sys.modules, {"cdsapi": mock_cdsapi}):
            importlib.reload(_glofas_mod)
            _glofas_mod.download_glofas(
                area=[44, -10, 35, 5], years=[2000],
                output_dir=str(tmp_path),
            )
        dataset_arg = mock_client.retrieve.call_args[0][0]
        assert dataset_arg == "cems-glofas-historical"
        importlib.reload(_glofas_mod)

    def test_request_contains_area(self, tmp_path):
        mock_cdsapi, mock_client = self._make_cdsapi_mock()
        with patch.dict(sys.modules, {"cdsapi": mock_cdsapi}):
            importlib.reload(_glofas_mod)
            _glofas_mod.download_glofas(
                area=[44, -10, 35, 5], years=[2000],
                output_dir=str(tmp_path),
            )
        request = mock_client.retrieve.call_args[0][1]
        assert request["area"] == [44, -10, 35, 5]
        importlib.reload(_glofas_mod)

    def test_request_contains_all_months_by_default(self, tmp_path):
        mock_cdsapi, mock_client = self._make_cdsapi_mock()
        with patch.dict(sys.modules, {"cdsapi": mock_cdsapi}):
            importlib.reload(_glofas_mod)
            _glofas_mod.download_glofas(
                area=[44, -10, 35, 5], years=[2000],
                output_dir=str(tmp_path),
            )
        request = mock_client.retrieve.call_args[0][1]
        assert request["month"] == [f"{m:02d}" for m in range(1, 13)]
        importlib.reload(_glofas_mod)

    def test_request_system_version(self, tmp_path):
        mock_cdsapi, mock_client = self._make_cdsapi_mock()
        with patch.dict(sys.modules, {"cdsapi": mock_cdsapi}):
            importlib.reload(_glofas_mod)
            _glofas_mod.download_glofas(
                area=[44, -10, 35, 5], years=[2000],
                output_dir=str(tmp_path),
            )
        request = mock_client.retrieve.call_args[0][1]
        assert request["system_version"] == "version_4_0"
        importlib.reload(_glofas_mod)

    def test_returns_list_of_paths(self, tmp_path):
        mock_cdsapi, _ = self._make_cdsapi_mock()
        with patch.dict(sys.modules, {"cdsapi": mock_cdsapi}):
            importlib.reload(_glofas_mod)
            result = _glofas_mod.download_glofas(
                area=[44, -10, 35, 5], years=[2000],
                output_dir=str(tmp_path),
            )
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].endswith(".nc")
        importlib.reload(_glofas_mod)

    # --- download_glofas_by_year ---

    def test_by_year_one_retrieve_per_year(self, tmp_path):
        mock_cdsapi, mock_client = self._make_cdsapi_mock()
        with patch.dict(sys.modules, {"cdsapi": mock_cdsapi}):
            importlib.reload(_glofas_mod)
            result = _glofas_mod.download_glofas_by_year(
                area=[44, -10, 35, 5], years=[2000, 2001, 2002],
                output_dir=str(tmp_path),
            )
        assert mock_client.retrieve.call_count == 3
        assert len(result) == 3
        importlib.reload(_glofas_mod)

    def test_by_year_ewds_url(self, tmp_path):
        mock_cdsapi, _ = self._make_cdsapi_mock()
        with patch.dict(sys.modules, {"cdsapi": mock_cdsapi}):
            importlib.reload(_glofas_mod)
            _glofas_mod.download_glofas_by_year(
                area=[44, -10, 35, 5], years=[2000],
                output_dir=str(tmp_path),
            )
        call_kwargs = mock_cdsapi.Client.call_args[1]
        assert call_kwargs["url"] == "https://ewds.climate.copernicus.eu/api"
        importlib.reload(_glofas_mod)

    # --- read_glofas_nc import guard ---

    def test_read_glofas_nc_raises_without_xarray(self):
        with patch.dict(sys.modules, {"xarray": None}):
            importlib.reload(_glofas_mod)
            with pytest.raises(ImportError, match="xarray"):
                _glofas_mod.read_glofas_nc("/tmp/fake.nc", lat=40.0, lon=-3.5)
        importlib.reload(_glofas_mod)

"""Tests for AEMET offline helpers."""

import numpy as np
import pandas as pd
import pytest

from pyhydra.data_sources.rainfall.aemet import AemetCSVLoader, _parse_dms_coord, _to_float


class TestAemetParsing:
    def test_to_float_accepts_decimal_comma(self):
        assert _to_float("12,5") == pytest.approx(12.5)

    def test_to_float_invalid_returns_nan(self):
        assert np.isnan(_to_float("not-a-number"))

    def test_parse_dms_latitude(self):
        assert _parse_dms_coord("415135N") == pytest.approx(41 + 51 / 60 + 35 / 3600)

    def test_parse_dms_west_longitude_uses_negative_sign(self):
        assert _parse_dms_coord("0011224O") == pytest.approx(-(1 + 12 / 60 + 24 / 3600))

    def test_parse_decimal_string(self):
        assert _parse_dms_coord("-3,7038") == pytest.approx(-3.7038)


class TestAemetCSVLoader:
    def test_load_station_data_missing_file_returns_empty_dataframe(self, tmp_path):
        loader = AemetCSVLoader(tmp_path)

        stations = loader.load_station_data()

        assert stations.empty

    def test_load_series_data_merges_station_columns_and_drops_duplicate_times(self, tmp_path):
        times = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])
        pd.DataFrame({"time": times, "001_prec": [1.0, 99.0, 2.0]}).to_csv(
            tmp_path / "AEMET_001_series.csv",
            index=False,
        )
        pd.DataFrame({"time": times[:2], "002_prec": [3.0, 4.0]}).to_csv(
            tmp_path / "AEMET_002_series.csv",
            index=False,
        )

        loader = AemetCSVLoader(tmp_path)
        series = loader.load_series_data("prec")

        assert list(series.columns) == ["001", "002"]
        assert list(series.index) == pd.to_datetime(["2020-01-01", "2020-01-02"]).tolist()
        assert series.loc[pd.Timestamp("2020-01-01"), "001"] == pytest.approx(1.0)

    def test_analyze_series_quality_reports_missing_percentage(self, tmp_path):
        pd.DataFrame(
            {
                "time": pd.date_range("2020-01-01", periods=4, freq="D"),
                "001_prec": [1.0, np.nan, 3.0, 4.0],
            }
        ).to_csv(tmp_path / "AEMET_001_series.csv", index=False)
        loader = AemetCSVLoader(tmp_path)
        loader.load_series_data("prec")

        quality = loader.analyze_series_quality()

        assert quality.loc[0, "station_id"] == "001"
        assert quality.loc[0, "missing_percent"] == pytest.approx(25.0)

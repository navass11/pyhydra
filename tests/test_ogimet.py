"""Tests for pyhydra.data_sources.rainfall.ogimet — offline functions only."""

import inspect
import numpy as np
import pandas as pd
import pytest

from pyhydra.data_sources.rainfall.ogimet import (
    OGIMETDownloader,
    OgimetCSVLoader,
    get_default_ogimet_stations_csv,
    normalize_filename,
    process_all_meteorological_variables,
)


class TestNormalizeFilename:
    def test_plain_ascii(self):
        assert normalize_filename("Madrid") == "Madrid"

    def test_spaces_become_underscore(self):
        assert normalize_filename("San Sebastian") == "San_Sebastian"

    def test_unicode_normalized(self):
        result = normalize_filename("Málaga")
        assert result == "Malaga"

    def test_accented_characters_stripped(self):
        result = normalize_filename("Cádiz")
        assert result == "Cadiz"

    def test_special_chars_replaced(self):
        result = normalize_filename("A/B-C.D")
        assert "_" in result
        assert "/" not in result
        assert "-" not in result
        assert "." not in result

    def test_leading_trailing_underscores_stripped(self):
        result = normalize_filename("_test_")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_consecutive_underscores_collapsed(self):
        result = normalize_filename("A   B")
        assert "__" not in result

    def test_numeric_input(self):
        result = normalize_filename(12345)
        assert result == "12345"

    def test_empty_string(self):
        result = normalize_filename("")
        assert isinstance(result, str)

    def test_only_special_chars(self):
        result = normalize_filename("---")
        # All hyphens → underscores → stripped
        assert result == "" or result.replace("_", "") == ""


class TestProcessMeteorologicalVariables:
    """Tests for process_all_meteorological_variables using synthetic MultiIndex DataFrames."""

    def _make_raw_df(self, n_days=5, include_precip=True, include_wind=True):
        """Build a minimal MultiIndex DataFrame that mimics OGIMET output."""
        dates = pd.date_range("2020-01-01", periods=n_days)

        # Build MultiIndex columns
        col_tuples = [("Fecha / hora UTC", "Fecha / hora UTC")]
        if include_precip:
            col_tuples.append(("Prec mm", "Prec mm"))
        if include_wind:
            col_tuples.append(("Viento kmh", "Dir"))
            col_tuples.append(("Viento kmh", "Vel"))
        col_tuples.append(("Temperatura C", "Max"))
        col_tuples.append(("Temperatura C", "Min"))

        columns = pd.MultiIndex.from_tuples(col_tuples)

        rows = []
        for d in dates:
            row = [d]
            if include_precip:
                row.append(5.0)
            if include_wind:
                row.append("N")
                row.append(10.0)
            row.extend([20.0, 10.0])
            rows.append(row)

        df = pd.DataFrame(rows, columns=columns)
        return df

    def test_returns_dataframe(self):
        df_raw = self._make_raw_df()
        result = process_all_meteorological_variables(df_raw)
        assert isinstance(result, pd.DataFrame)

    def test_date_column_present(self):
        df_raw = self._make_raw_df()
        result = process_all_meteorological_variables(df_raw)
        assert "date" in result.columns

    def test_one_row_per_day(self):
        df_raw = self._make_raw_df(n_days=5)
        result = process_all_meteorological_variables(df_raw)
        assert len(result) == 5

    def test_wind_direction_converted_to_degrees(self):
        df_raw = self._make_raw_df(include_wind=True)
        result = process_all_meteorological_variables(df_raw)
        wind_cols = [c for c in result.columns if "dir" in c.lower()]
        if wind_cols:
            # Values should be in 0-360 range (N=0°); coerce to numeric for check
            vals = pd.to_numeric(result[wind_cols[0]], errors="coerce")
            assert vals.between(0, 360).all()

    def test_trace_precip_becomes_01(self):
        df_raw = self._make_raw_df(include_precip=True)
        # Replace one precip value with 'ip' (trace)
        prec_col = [c for c in df_raw.columns if "prec" in str(c).lower()][0]
        df_raw.loc[df_raw.index[0], prec_col] = "ip"
        result = process_all_meteorological_variables(df_raw)
        prec_cols_out = [c for c in result.columns if "prec" in c.lower()]
        if prec_cols_out:
            assert result[prec_cols_out[0]].iloc[0] == pytest.approx(0.1)

    def test_raises_on_flat_multiindex(self):
        """Must raise ValueError if columns are not MultiIndex."""
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=3), "val": [1, 2, 3]})
        with pytest.raises((ValueError, AttributeError)):
            process_all_meteorological_variables(df)


class TestOgimetDefaults:
    def test_default_station_catalogue_is_discoverable(self):
        path = get_default_ogimet_stations_csv()
        assert path.endswith("estaciones_ogimet_all.csv")

    def test_loader_creates_default_folder(self, tmp_path):
        folder = tmp_path / "ogimet"
        loader = OgimetCSVLoader(folder)
        assert folder.exists()
        assert loader.load_station_data().empty

    def test_downloader_station_csv_is_optional(self):
        assert inspect.signature(OGIMETDownloader).parameters["stations_csv"].default is None

"""Tests for extreme-value helpers."""

import numpy as np
import pandas as pd
import pytest

from pyhydra.climate.time_series.extremes import (
    extract_block_maxima,
    extract_pot,
    return_level_gev,
    return_level_gpd,
    return_levels,
)


class TestBlockMaxima:
    def test_extract_block_maxima_uses_calendar_years(self):
        series = pd.Series(
            [1.0, 7.0, 3.0, 2.0, 9.0, 4.0],
            index=pd.to_datetime([
                "2000-01-01",
                "2000-06-01",
                "2000-12-31",
                "2001-01-01",
                "2001-02-01",
                "2001-12-31",
            ]),
        )

        maxima = extract_block_maxima(series, freq="YE")

        assert list(maxima.values) == [7.0, 9.0]


class TestPotExtraction:
    def test_extract_pot_separates_events_by_min_distance(self):
        series = pd.Series(
            [0.0, 5.0, 7.0, 0.0, 6.0, 0.0, 8.0, 0.0],
            index=pd.date_range("2000-01-01", periods=8, freq="D"),
        )

        peaks = extract_pot(series, threshold=4.0, min_gap=2)

        assert list(peaks.values) == [7.0, 6.0, 8.0]


class TestReturnLevels:
    def test_gev_gumbel_limit_formula(self):
        params = {"mu": 10.0, "sigma": 2.0, "xi": 0.0}
        expected = 10.0 - 2.0 * np.log(-np.log(0.98))

        assert return_level_gev(params, 50) == pytest.approx(expected)

    def test_gpd_gumbel_limit_formula(self):
        params = {"threshold": 3.0, "scale": 2.0, "shape": 0.0, "lambda_rate": 1.5}
        expected = 3.0 + 2.0 * np.log(1.5 * 10.0)

        assert return_level_gpd(params, 10) == pytest.approx(expected)

    def test_return_levels_preserves_requested_periods(self):
        params = {"mu": 10.0, "sigma": 2.0, "xi": 0.1}

        levels = return_levels(params, [2, 10, 100], dist="gev")

        assert isinstance(levels, pd.Series)
        assert list(levels.index) == [2, 10, 100]
        assert levels.is_monotonic_increasing

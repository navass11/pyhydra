"""Tests for pyhydra.climate.time_series.events."""

import numpy as np
import pandas as pd
import pytest

from pyhydra.climate.time_series.events import (
    extract_discharge_events,
    extract_events,
    extract_precipitation_events,
)


class TestExtractDischargeEvents:
    def test_detects_two_peaks(self, discharge_series):
        stats, bounds = extract_discharge_events(discharge_series, threshold=50)
        assert len(stats) == 2

    def test_peak_values_correct(self, discharge_series):
        stats, _ = extract_discharge_events(discharge_series, threshold=50)
        assert stats["peak"].iloc[0] == pytest.approx(120, abs=5)
        assert stats["peak"].iloc[1] == pytest.approx(200, abs=5)

    def test_bounds_order(self, discharge_series):
        _, bounds = extract_discharge_events(discharge_series, threshold=50)
        assert (bounds["start"] < bounds["end"]).all()

    def test_threshold2_filters_small_events(self, discharge_series):
        stats_all, _ = extract_discharge_events(discharge_series, threshold=50)
        stats_filtered, _ = extract_discharge_events(
            discharge_series, threshold=50, threshold2=150
        )
        assert len(stats_filtered) < len(stats_all)
        assert (stats_filtered["peak"] >= 150).all()

    def test_no_events_below_threshold(self, discharge_series):
        stats, bounds = extract_discharge_events(discharge_series, threshold=300)
        assert len(stats) == 0
        assert len(bounds) == 0

    def test_stats_columns(self, discharge_series):
        stats, bounds = extract_discharge_events(discharge_series, threshold=50)
        assert set(stats.columns) == {"peak", "mean", "duration", "volume", "date_peak"}
        assert set(bounds.columns) == {"start", "end"}

    def test_duration_positive(self, discharge_series):
        stats, _ = extract_discharge_events(discharge_series, threshold=50)
        assert (stats["duration"] > 0).all()

    def test_volume_consistent_with_mean_duration(self, discharge_series):
        stats, _ = extract_discharge_events(discharge_series, threshold=50)
        # volume is in m³ (Q [m³/s] × duration [days] × 86400 s/day); ±1 step rounding
        expected = stats["mean"] * stats["duration"] * 86400
        assert np.allclose(stats["volume"], expected, rtol=0.05)

    def test_date_peak_in_index(self, discharge_series):
        stats, _ = extract_discharge_events(discharge_series, threshold=50)
        for dp in stats["date_peak"]:
            assert dp in discharge_series.index

    def test_empty_series_returns_empty(self):
        s = pd.Series(np.zeros(50), index=pd.date_range("2000-01-01", periods=50))
        stats, bounds = extract_discharge_events(s, threshold=10)
        assert len(stats) == 0


class TestExtractPrecipitationEvents:
    def test_detects_two_events_with_min_duration_2(self, precipitation_series):
        stats, _ = extract_precipitation_events(
            precipitation_series, threshold=0, min_duration=2
        )
        assert len(stats) == 2

    def test_detects_three_events_no_filter(self, precipitation_series):
        stats, _ = extract_precipitation_events(
            precipitation_series, threshold=0, min_duration=1
        )
        assert len(stats) == 3

    def test_peak_column(self, precipitation_series):
        stats, _ = extract_precipitation_events(precipitation_series, threshold=0)
        assert stats["peak"].max() == pytest.approx(12.0)

    def test_total_accumulation(self, precipitation_series):
        stats, _ = extract_precipitation_events(
            precipitation_series, threshold=0, min_duration=1
        )
        # sum of all event totals == total wet accumulation
        assert stats["total"].sum() == pytest.approx(
            precipitation_series[precipitation_series > 0].sum(), rel=1e-3
        )

    def test_gap_bridging(self):
        """Two wet spells separated by 1 dry day should merge when min_gap=1."""
        dates = pd.date_range("2000-01-01", periods=10)
        pr = pd.Series([0, 5, 8, 0, 6, 4, 0, 0, 0, 0], index=dates)
        stats_merged, _ = extract_precipitation_events(pr, threshold=0, min_gap=1)
        stats_separate, _ = extract_precipitation_events(pr, threshold=0, min_gap=0)
        assert len(stats_merged) == 1
        assert len(stats_separate) == 2

    def test_threshold_filters_light_rain(self):
        dates = pd.date_range("2000-01-01", periods=20)
        pr = pd.Series([0.1] * 5 + [0] * 5 + [5, 10, 8] + [0] * 7, index=dates)
        stats_strict, _ = extract_precipitation_events(pr, threshold=1.0, min_duration=1)
        assert len(stats_strict) == 1
        assert stats_strict["peak"].iloc[0] == pytest.approx(10.0)

    def test_stats_columns(self, precipitation_series):
        stats, bounds = extract_precipitation_events(precipitation_series)
        assert set(stats.columns) == {"peak", "total", "duration", "mean_intensity", "date_start"}
        assert set(bounds.columns) == {"start", "end"}

    def test_empty_returns_empty(self):
        s = pd.Series(np.zeros(30), index=pd.date_range("2000-01-01", periods=30))
        stats, bounds = extract_precipitation_events(s)
        assert len(stats) == 0

    def test_bounds_order(self, precipitation_series):
        _, bounds = extract_precipitation_events(precipitation_series, min_duration=1)
        assert (bounds["start"] <= bounds["end"]).all()


class TestExtractEventsUnified:
    def test_dispatch_discharge(self, discharge_series):
        stats, _ = extract_events(discharge_series, threshold=50, variable="discharge")
        assert "peak" in stats.columns

    def test_dispatch_precipitation(self, precipitation_series):
        stats, _ = extract_events(
            precipitation_series, threshold=0, variable="precipitation"
        )
        assert "total" in stats.columns

    def test_invalid_variable_raises(self, discharge_series):
        with pytest.raises(ValueError, match="Unknown variable"):
            extract_events(discharge_series, threshold=10, variable="wind")

    def test_threshold2_forwarded_to_discharge(self, discharge_series):
        stats_all, _ = extract_events(
            discharge_series, threshold=50, variable="discharge"
        )
        stats_filt, _ = extract_events(
            discharge_series, threshold=50, variable="discharge", threshold2=150
        )
        assert len(stats_filt) <= len(stats_all)

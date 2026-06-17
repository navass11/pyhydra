"""Tests for pure helper functions in pyhydra.data_sources.climate_change.esgf."""

import pytest

from pyhydra.data_sources.climate_change.esgf import (
    _extract_years,
    _valid_year_range,
)


class TestExtractYears:
    def test_long_format(self):
        name = "tas_day_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_19500101-20141231.nc"
        y1, y2 = _extract_years(name)
        assert y1 == 1950
        assert y2 == 2014

    def test_short_format(self):
        name = "pr_day_ACCESS-CM2_ssp245_r1i1p1f1_gn_2015-2100.nc"
        y1, y2 = _extract_years(name)
        assert y1 == 2015
        assert y2 == 2100

    def test_no_match_returns_none(self):
        y1, y2 = _extract_years("README.txt")
        assert y1 is None
        assert y2 is None

    def test_long_format_takes_priority(self):
        # Long format (8-digit groups) should match before short format
        name = "var_19500101-20141231.nc"
        y1, y2 = _extract_years(name)
        assert y1 == 1950
        assert y2 == 2014

    def test_ssp585_filename(self):
        name = "pr_day_GFDL-ESM4_ssp585_r1i1p1f1_gr1_20150101-21001231.nc"
        y1, y2 = _extract_years(name)
        assert y1 == 2015
        assert y2 == 2100


class TestValidYearRange:
    def test_historical_valid(self):
        assert _valid_year_range("historical", 1950, 2014) is True

    def test_historical_overlap(self):
        # File covering 1970-2014 overlaps historical range 1950-2014
        assert _valid_year_range("historical", 1970, 2014) is True

    def test_historical_after_range(self):
        # 2015-2020 is outside historical (1950-2014)
        assert _valid_year_range("historical", 2015, 2020) is False

    def test_historical_before_range(self):
        # 1900-1949 is before historical start
        assert _valid_year_range("historical", 1900, 1949) is False

    def test_ssp245_valid(self):
        assert _valid_year_range("ssp245", 2015, 2100) is True

    def test_ssp245_overlap_start(self):
        assert _valid_year_range("ssp245", 2010, 2050) is True

    def test_ssp245_before_range(self):
        # 1950-2010 ends before ssp245 starts (2015), so no overlap → False
        assert _valid_year_range("ssp245", 1950, 2010) is False

    def test_ssp585_valid(self):
        assert _valid_year_range("ssp585", 2050, 2100) is True

    def test_unknown_scenario_always_valid(self):
        assert _valid_year_range("rcp45", 1900, 2200) is True
        assert _valid_year_range("piControl", 850, 1850) is True

    def test_all_ssps_defined(self):
        for scenario in ("ssp126", "ssp245", "ssp370", "ssp585"):
            assert _valid_year_range(scenario, 2050, 2075) is True

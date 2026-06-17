"""Tests for pyhydra.climate.bias_correction."""

import numpy as np
import pandas as pd
import pytest

from pyhydra.climate.bias_correction import BiasCorrection, delta_method


class TestBiasCorrection:
    def test_qm_no_negatives(self, bc_arrays):
        obs, mod, sce = bc_arrays
        bc = BiasCorrection(obs, mod, sce)
        result = bc.quantile_mapping()
        assert (result >= 0).all()

    def test_qm_reduces_bias(self, bc_arrays):
        obs, mod, sce = bc_arrays
        bc = BiasCorrection(obs, mod, sce)
        result = bc.quantile_mapping()
        # After QM the mean should be closer to obs than raw sce was
        raw_error = abs(sce.mean() - obs.mean())
        corrected_error = abs(result.mean() - obs.mean())
        assert corrected_error < raw_error

    def test_qdm_no_negatives(self, bc_arrays):
        obs, mod, sce = bc_arrays
        bc = BiasCorrection(obs, mod, sce)
        result = bc.quantile_deltamapping()
        assert (result >= 0).all()

    def test_qdm_reduces_bias(self, bc_arrays):
        obs, mod, sce = bc_arrays
        bc = BiasCorrection(obs, mod, sce)
        result = bc.quantile_deltamapping()
        raw_error = abs(sce.mean() - obs.mean())
        corrected_error = abs(result.mean() - obs.mean())
        assert corrected_error < raw_error

    def test_qdm_returns_same_length(self, bc_arrays):
        obs, mod, sce = bc_arrays
        bc = BiasCorrection(obs, mod, sce)
        assert len(bc.quantile_deltamapping()) == len(sce)

    def test_qm_returns_same_length(self, bc_arrays):
        obs, mod, sce = bc_arrays
        bc = BiasCorrection(obs, mod, sce)
        assert len(bc.quantile_mapping()) == len(sce)

    def test_sdm_precipitation_no_negatives(self, bc_arrays):
        obs, mod, sce = bc_arrays
        bc = BiasCorrection(obs, mod, sce)
        result = bc.scaled_distribution_mapping("precipitation")
        assert (result >= 0).all()

    def test_sdm_precipitation_reduces_bias(self, bc_arrays):
        obs, mod, sce = bc_arrays
        bc = BiasCorrection(obs, mod, sce)
        result = bc.scaled_distribution_mapping("precipitation")
        raw_error = abs(sce.mean() - obs.mean())
        corrected_error = abs(result.mean() - obs.mean())
        assert corrected_error < raw_error

    def test_sdm_temperature(self):
        rng = np.random.default_rng(99)
        obs = rng.normal(15, 3, 500)
        mod = rng.normal(17, 3, 400)   # +2°C bias
        sce = rng.normal(20, 3, 300)
        bc = BiasCorrection(obs, mod, sce)
        result = bc.scaled_distribution_mapping("temperature")
        assert result.shape == sce.shape

    def test_sdm_invalid_variable_raises(self, bc_arrays):
        obs, mod, sce = bc_arrays
        bc = BiasCorrection(obs, mod, sce)
        with pytest.raises(NotImplementedError):
            bc.scaled_distribution_mapping("wind")

    def test_accepts_list_input(self):
        obs = list(range(1, 101))
        mod = list(range(3, 103))
        sce = list(range(5, 55))
        bc = BiasCorrection(obs, mod, sce)
        result = bc.quantile_mapping()
        assert len(result) == 50

    def test_original_arrays_not_mutated(self, bc_arrays):
        obs, mod, sce = bc_arrays
        sce_copy = sce.copy()
        bc = BiasCorrection(obs, mod, sce)
        bc.quantile_mapping()
        np.testing.assert_array_equal(sce, sce_copy)


class TestDeltaMethod:
    def test_temperature_output_length(self, temperature_series):
        obs, hist, fut = temperature_series
        result = delta_method(obs, hist, fut, var="tas", stat="mean")
        assert len(result) == len(obs)

    def test_temperature_index_shifted(self, temperature_series):
        obs, hist, fut = temperature_series
        result = delta_method(obs, hist, fut, var="tas", stat="mean")
        expected_shift = fut.index.year[0] - obs.index.year[0]
        actual_shift = result.index.year[0] - obs.index.year[0]
        assert actual_shift == expected_shift

    def test_temperature_warming_applied(self, temperature_series):
        obs, hist, fut = temperature_series
        result = delta_method(obs, hist, fut, var="tas", stat="mean")
        # fut is 3°C warmer → result should be warmer than obs
        assert result.mean() > obs.mean() + 1.0

    def test_precipitation_multiplicative(self, precipitation_series_bc):
        obs, hist, fut = precipitation_series_bc
        result = delta_method(obs, hist, fut, var="pr", stat="mean")
        assert len(result) == len(obs)
        assert (result >= 0).all()

    def test_precipitation_no_negatives(self, precipitation_series_bc):
        obs, hist, fut = precipitation_series_bc
        result = delta_method(obs, hist, fut, var="pr", stat="mean")
        assert (result >= 0).all()

    def test_median_stat(self, temperature_series):
        obs, hist, fut = temperature_series
        result_mean   = delta_method(obs, hist, fut, var="tas", stat="mean")
        result_median = delta_method(obs, hist, fut, var="tas", stat="median")
        # Both should return series of same length; values may differ
        assert len(result_mean) == len(result_median)

    def test_returns_pandas_series(self, temperature_series):
        obs, hist, fut = temperature_series
        result = delta_method(obs, hist, fut, var="tas", stat="mean")
        assert isinstance(result, pd.Series)

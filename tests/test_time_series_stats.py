"""Tests for hydrological performance statistics."""

import numpy as np
import pytest

from pyhydra.climate.time_series.stats import compute_kge, compute_nse, compute_pbias


class TestPerformanceStats:
    def test_perfect_simulation_scores_are_ideal(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        sim = obs.copy()

        assert compute_nse(obs, sim) == pytest.approx(1.0)
        assert compute_kge(obs, sim) == pytest.approx(1.0)
        assert compute_pbias(obs, sim) == pytest.approx(0.0)

    def test_nan_pairs_are_ignored(self):
        obs = np.array([1.0, np.nan, 3.0, 4.0])
        sim = np.array([1.0, 99.0, 3.0, 5.0])

        expected_nse = 1.0 - ((0.0**2 + 0.0**2 + 1.0**2) / np.sum((np.array([1.0, 3.0, 4.0]) - 8.0 / 3.0) ** 2))
        assert compute_nse(obs, sim) == pytest.approx(expected_nse)
        assert compute_pbias(obs, sim) == pytest.approx(100.0 * 1.0 / 8.0)

    def test_degenerate_observations_return_nan(self):
        obs = np.array([2.0, 2.0, 2.0])
        sim = np.array([2.0, 3.0, 4.0])

        assert np.isnan(compute_nse(obs, sim))
        assert np.isnan(compute_kge(obs, sim))

    def test_zero_observed_sum_pbias_returns_nan(self):
        assert np.isnan(compute_pbias(np.array([1.0, -1.0]), np.array([2.0, 0.0])))

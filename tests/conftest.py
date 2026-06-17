"""
Shared pytest fixtures for pyhydra tests.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Time index helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_index():
    return pd.date_range("2000-01-01", periods=365 * 5, freq="D")


# ---------------------------------------------------------------------------
# Event extraction fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def discharge_series():
    """Two clean flood peaks above threshold=50."""
    np.random.seed(0)
    n = 300
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    q = np.full(n, 5.0)
    # Peak 1: days 30-70
    q[30:50] = np.linspace(5, 120, 20)
    q[50:71] = np.linspace(120, 5, 21)
    # Peak 2: days 150-200
    q[150:175] = np.linspace(5, 200, 25)
    q[175:201] = np.linspace(200, 5, 26)
    return pd.Series(q, index=dates)


@pytest.fixture
def precipitation_series():
    """Three wet spells of varying length."""
    n = 200
    dates = pd.date_range("2010-01-01", periods=n, freq="D")
    pr = np.zeros(n)
    pr[5:10]   = [1, 5, 12, 8, 2]   # 5-day event
    pr[40:44]  = [0.5, 3, 9, 1]     # 4-day event
    pr[100]    = 0.3                 # single-day event (filtered if min_duration=2)
    return pd.Series(pr, index=dates)


# ---------------------------------------------------------------------------
# Bias correction fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bc_arrays():
    """obs/mod/sce arrays with a known +5 additive bias in mod/sce."""
    rng = np.random.default_rng(42)
    obs = rng.gamma(2, 2, 500)          # reference
    mod = obs[:400] + 5.0               # biased historical
    sce = rng.gamma(2, 2, 300) + 5.0   # biased future
    return obs, mod, sce


@pytest.fixture
def temperature_series():
    """Monthly pandas Series for delta method (obs / hist / future)."""
    idx_obs  = pd.date_range("1980-01-01", "2009-12-31", freq="D")
    idx_hist = pd.date_range("1980-01-01", "2009-12-31", freq="D")
    idx_mod  = pd.date_range("2070-01-01", "2099-12-31", freq="D")

    rng = np.random.default_rng(7)
    # Use a fixed-length base so obs/hist/fut can differ in calendar length
    n_obs = len(idx_obs)
    n_mod = len(idx_mod)
    base_obs = 15 + 10 * np.sin(np.linspace(0, 6 * np.pi, n_obs))
    base_mod = 15 + 10 * np.sin(np.linspace(0, 6 * np.pi, n_mod))
    obs  = pd.Series(base_obs + rng.normal(0, 1, n_obs), index=idx_obs)
    hist = pd.Series(base_obs + rng.normal(0, 1, n_obs), index=idx_hist)
    fut  = pd.Series(base_mod + 3.0 + rng.normal(0, 1, n_mod), index=idx_mod)
    return obs, hist, fut


@pytest.fixture
def precipitation_series_bc():
    """Daily precipitation Series for delta method (multiplicative)."""
    idx_obs  = pd.date_range("1980-01-01", "2009-12-31", freq="D")
    idx_hist = pd.date_range("1980-01-01", "2009-12-31", freq="D")
    idx_mod  = pd.date_range("2070-01-01", "2099-12-31", freq="D")

    rng = np.random.default_rng(13)
    obs  = pd.Series(np.abs(rng.exponential(3, len(idx_obs))),  index=idx_obs)
    hist = pd.Series(np.abs(rng.exponential(3, len(idx_hist))), index=idx_hist)
    fut  = pd.Series(np.abs(rng.exponential(5, len(idx_mod))),  index=idx_mod)
    return obs, hist, fut


# ---------------------------------------------------------------------------
# Soils fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def texture_arrays():
    """3×3 pixel grids with known USDA classes in each cell."""
    shape = (3, 3)
    sand = np.array([[90, 70, 50],  # Sand, Loamy Sand, Loam
                     [50, 30, 10],  # Loam, Clay Loam, Silt
                     [50, 60, 30]], dtype=np.uint8)
    silt = np.array([[5,  20, 40],
                     [40, 30, 85],
                     [20,  5, 35]], dtype=np.uint8)
    clay = np.array([[5,  10, 10],
                     [10, 40, 5 ],
                     [30, 35, 35]], dtype=np.uint8)
    return sand, silt, clay

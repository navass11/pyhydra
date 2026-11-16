"""Pruebas de los hallazgos de los experimentos sintéticos (τ < 0, nivel de Kendall, API de pyvinecopulib)."""
import warnings
import numpy as np
import pytest
from scipy import stats, optimize

from pyhydra.climate.spatial_analysis.copulas import BivariateCopula, _tau_to_theta, _kendall_level, _default_kendall_grid


def _copula(family, theta):
    c = BivariateCopula(family)
    c._theta = theta
    c._mx = c._my = (stats.uniform, ())
    return c


@pytest.mark.parametrize("family", ["gumbel", "clayton"])
def test_negative_tau_warns(family):
    with pytest.warns(UserWarning, match="only represents positive dependence"):
        _tau_to_theta(-0.4, family)


def test_no_warning_for_frank_or_positive_or_noise():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _tau_to_theta(-0.4, "frank")
        _tau_to_theta(0.3, "gumbel")
        _tau_to_theta(-0.01, "clayton")      # ruido muestral, no dependencia negativa


def test_kendall_grid_refined_towards_one():
    g = _default_kendall_grid()
    assert 190 <= len(g) <= 200 and g[0] == pytest.approx(0.01) and g[-1] == pytest.approx(1 - 1e-4) and np.all(np.diff(g) > 0)


def _exact_tstar(theta, K):
    """Gumbel: K(t) = t - t ln t / theta."""
    return optimize.brentq(lambda t: t - t * np.log(t) / theta - K, 1e-6, 1 - 1e-12)


@pytest.mark.parametrize("T, mu", [(500, 0.5), (100, 0.5), (1000, 0.5)])
def test_kendall_level_not_truncated(T, mu):
    c = _copula("gumbel", 1.5)                       # tau = 1/3
    res = c.kendall_return_period(T, mu=mu, n_sim=400_000, random_state=1)
    assert res["t_star"] == pytest.approx(_exact_tstar(1.5, 1 - mu / T), abs=2e-3)
    if T >= 500:
        assert res["t_star"] > 0.99                  # antes quedaba en 0.9900


def test_kendall_level_warns_when_pool_is_too_small():
    pool = np.random.default_rng(0).uniform(size=2000)
    with pytest.warns(UserWarning, match="needs K"):
        _kendall_level(pool, T=500, mu=0.5)


def test_vine_sampling_without_deprecation():
    pytest.importorskip("pyvinecopulib")
    import pandas as pd
    from pyhydra.climate.spatial_analysis.copulas import VineCopula
    z = np.random.default_rng(0).multivariate_normal([0, 0, 0], [[1, .5, .3], [.5, 1, .4], [.3, .4, 1]], 300)
    v = VineCopula().fit(pd.DataFrame(np.exp(z), columns=["a", "b", "c"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        u = v._simulate(500, np.random.default_rng(1))
    assert u.shape == (500, 3) and np.all((u > 0) & (u < 1))

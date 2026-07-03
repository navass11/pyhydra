from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyhydra.climate.spatial_analysis import rfa


def test_regional_index_flood_normalises_each_station_by_mean():
    data = {
        "A": np.array([10.0, 20.0, 30.0]),
        "B": np.array([5.0, 10.0, 15.0]),
    }

    normalised, index_flood = rfa.regional_index_flood(data)

    assert index_flood.to_dict() == {"A": 20.0, "B": 10.0}
    assert np.allclose(normalised["A"], [0.5, 1.0, 1.5])
    assert np.allclose(normalised["B"], [0.5, 1.0, 1.5])


def test_fit_regional_gev_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        rfa.fit_regional_gev({"A": np.array([1.0, 2.0, 3.0])}, method="invalid")


def test_regional_return_levels_scale_regional_quantiles_by_station_index(monkeypatch):
    def fake_fit_regional_gev(data_dict, method="lmom"):
        return {"shape": 0.0}, pd.Series({"A": 10.0, "B": 20.0}, name="index_flood")

    def fake_return_level(params, return_periods):
        return np.asarray(return_periods, dtype=float) * 0.1

    monkeypatch.setattr(rfa, "fit_regional_gev", fake_fit_regional_gev)
    monkeypatch.setattr(rfa, "return_level", fake_return_level)

    result = rfa.regional_return_levels(
        {"A": np.array([1.0, 2.0]), "B": np.array([2.0, 4.0])},
        T_values=[10, 50],
    )

    assert list(result.index) == ["A", "B"]
    assert list(result.columns) == ["T10", "T50"]
    assert result.loc["A"].to_dict() == {"T10": 10.0, "T50": 50.0}
    assert result.loc["B"].to_dict() == {"T10": 20.0, "T50": 100.0}


def test_regional_return_levels_bayes_scales_posterior_ci_by_index_flood(monkeypatch):
    fake_posterior = pd.DataFrame({"mu": [1.0], "sigma": [0.2], "xi": [0.1]})

    def fake_fit_regional_gev(data_dict, method="lmom", n_chains=4, n_samples=1000):
        assert method == "bayes"
        return fake_posterior, pd.Series({"A": 10.0, "B": 20.0}, name="index_flood")

    def fake_return_level_bayes(posterior, T, credible=0.90):
        assert posterior is fake_posterior
        return {"median": T * 0.1, "lower": T * 0.08, "upper": T * 0.12}

    monkeypatch.setattr(rfa, "fit_regional_gev", fake_fit_regional_gev)
    monkeypatch.setattr(rfa, "return_level_bayes", fake_return_level_bayes)

    median, lower, upper = rfa.regional_return_levels(
        {"A": np.array([1.0, 2.0]), "B": np.array([2.0, 4.0])},
        T_values=[10, 50],
        method="bayes",
    )

    assert median.loc["A"].to_dict() == {"T10": 10.0, "T50": 50.0}
    assert median.loc["B"].to_dict() == {"T10": 20.0, "T50": 100.0}
    assert lower.loc["A"].to_dict() == {"T10": 8.0, "T50": 40.0}
    assert upper.loc["B"].to_dict() == {"T10": 24.0, "T50": 120.0}


def test_fit_regional_gev_bayes_returns_posterior_dataframe():
    rng = np.random.default_rng(0)
    from scipy.stats import genextreme

    data = {
        f"S{i}": genextreme.rvs(-0.1, loc=100, scale=20, size=25, random_state=rng)
        for i in range(3)
    }

    posterior, index_floods = rfa.fit_regional_gev(
        data, method="bayes", n_chains=2, n_samples=50
    )

    assert set(posterior.columns) >= {"mu", "sigma", "xi"}
    assert len(posterior) > 0
    assert set(index_floods.index) == set(data)


# ── Hosking & Wallis homogeneity diagnostics ────────────────────────────────
# Formulas cross-checked against `lmomRFA::regtst.s()` (Hosking & Wallis's
# own reference R implementation): Di uses N/(3*(N-1)) * mahalanobis(u,
# colMeans(u), cov(u)); H1 uses a kappa-distribution Monte Carlo null built
# from record-length-weighted regional-average L-moments.

def _gev_region(n_sites, xi, scale, n_per_site=35, seed=0):
    from scipy.stats import genextreme

    rng = np.random.default_rng(seed)
    return {
        f"S{i}": genextreme.rvs(-xi, loc=1000, scale=scale, size=n_per_site, random_state=rng)
        for i in range(n_sites)
    }


def test_discordancy_critical_value_matches_hosking_wallis_table():
    assert rfa.discordancy_critical_value(5) == pytest.approx(1.3330)
    assert rfa.discordancy_critical_value(9) == pytest.approx(2.3287)
    assert rfa.discordancy_critical_value(14) == pytest.approx(2.9709)
    assert rfa.discordancy_critical_value(3) == 3.0
    assert rfa.discordancy_critical_value(20) == 3.0


def test_regional_discordancy_requires_at_least_four_sites():
    with pytest.raises(ValueError, match="at least 4 sites"):
        rfa.regional_discordancy(_gev_region(3, xi=0.15, scale=300))


def test_regional_discordancy_flags_clear_outlier_station():
    data = _gev_region(8, xi=0.15, scale=300, seed=1)
    from scipy.stats import genextreme

    rng = np.random.default_rng(1)
    data["Outlier"] = genextreme.rvs(0.9, loc=1000, scale=900, size=35, random_state=rng)

    result = rfa.regional_discordancy(data)

    assert list(result.columns) == ["Di", "critical", "discordant"]
    assert result.loc["Outlier", "Di"] == result["Di"].max()
    assert result.loc["Outlier", "discordant"]
    assert result["Di"].sum() == pytest.approx(len(data), rel=1e-6)


def test_regional_heterogeneity_is_low_for_a_homogeneous_region():
    data = _gev_region(8, xi=0.15, scale=300, seed=2)

    result = rfa.regional_heterogeneity(data, n_sim=100, seed=42)

    assert result["H"] < 1.0


def test_regional_heterogeneity_is_high_for_a_heterogeneous_region():
    from scipy.stats import genextreme

    rng = np.random.default_rng(3)
    xis = [-0.35, 0.4, -0.1, 0.5, 0.0, 0.3, -0.4, 0.2]
    scales = [100, 500, 200, 700, 50, 600, 150, 450]
    data = {
        f"S{i}": genextreme.rvs(-xis[i], loc=1000, scale=scales[i], size=35, random_state=rng)
        for i in range(8)
    }

    result = rfa.regional_heterogeneity(data, n_sim=100, seed=42)

    assert result["H"] > 2.0


def test_regional_heterogeneity_requires_at_least_two_sites():
    with pytest.raises(ValueError, match="at least 2 sites"):
        rfa.regional_heterogeneity(_gev_region(1, xi=0.15, scale=300))

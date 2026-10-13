from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("openturns")

from pyhydra.climate.spatial_analysis.copulas import (  # noqa: E402
    FloodEventCopula,
    fit_marginal,
)

POSITIVE_FAMILIES = ["Exponential", "Gamma", "LogNormal", "WeibullMin"]


def _zero_inflated_events(n=1500, p0=0.4, seed=0):
    """Two correlated rain-like variables sharing dry events, plus a duration."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], size=n)
    wet = rng.uniform(size=n) > p0
    pmax = np.where(wet, np.exp(0.5 + 0.6 * z[:, 0]), 0.0)
    pmed = np.where(wet, 0.2 * np.exp(0.6 * z[:, 1]), 0.0)
    duration = rng.gamma(2.0, 8.0, size=n)
    shape_type = rng.integers(1, 6, size=n).astype(float)
    return pd.DataFrame(
        {"Pmax": pmax, "Pmed": pmed, "Duracion": duration, "Tipo": shape_type}
    )


def test_fit_marginal_restricts_candidate_families():
    data = np.random.default_rng(1).gamma(2.0, 3.0, size=400)
    dist, _ = fit_marginal(data, families=POSITIVE_FAMILIES, verbose=False)
    assert dist.getName() in POSITIVE_FAMILIES


def test_fit_marginal_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unknown openturns distribution family"):
        fit_marginal([1.0, 2.0, 3.0], families=["NotADistribution"], verbose=False)


def test_zero_inflated_vars_must_be_continuous():
    with pytest.raises(ValueError, match="zero_inflated_vars"):
        FloodEventCopula(["Pmax"], ["Tipo"], zero_inflated_vars=["Tipo"])


def test_zero_inflated_marginals_reproduce_dry_share_and_positive_tail():
    events = _zero_inflated_events()
    copula = FloodEventCopula(
        continuous_vars=["Pmax", "Pmed", "Duracion"],
        discrete_vars=["Tipo"],
        families=POSITIVE_FAMILIES,
        zero_inflated_vars=["Pmax", "Pmed"],
        random_state=7,
        verbose=False,
    ).fit(events)

    synthetic = copula.sample(6000)

    for var in ("Pmax", "Pmed"):
        observed_dry = (events[var] == 0).mean()
        assert copula._zero_mass[var] == pytest.approx(observed_dry)
        assert (synthetic[var] == 0).mean() == pytest.approx(observed_dry, abs=0.03)
        assert (synthetic[var] >= 0).all()
        assert np.isfinite(synthetic[var]).all()
        observed_wet_median = events.loc[events[var] > 0, var].median()
        synthetic_wet_median = synthetic.loc[synthetic[var] > 0, var].median()
        assert synthetic_wet_median == pytest.approx(observed_wet_median, rel=0.15)

    # The shared dry events and the latent correlation must survive the
    # randomised probability-integral transform.
    observed_rho = events[["Pmax", "Pmed"]].corr("spearman").iloc[0, 1]
    synthetic_rho = synthetic[["Pmax", "Pmed"]].corr("spearman").iloc[0, 1]
    assert synthetic_rho == pytest.approx(observed_rho, abs=0.1)
    assert set(np.unique(synthetic["Tipo"])) <= set(np.unique(events["Tipo"]))


def test_sample_is_reproducible_with_random_state():
    events = _zero_inflated_events(n=600)
    kwargs = dict(
        continuous_vars=["Pmax", "Pmed", "Duracion"],
        discrete_vars=["Tipo"],
        families=POSITIVE_FAMILIES,
        zero_inflated_vars=["Pmax", "Pmed"],
        random_state=11,
        verbose=False,
    )
    first = FloodEventCopula(**kwargs).fit(events).sample(200)
    second = FloodEventCopula(**kwargs).fit(events).sample(200)
    pd.testing.assert_frame_equal(first, second)


def test_default_behaviour_without_new_options_still_samples():
    rng = np.random.default_rng(3)
    events = pd.DataFrame(
        {
            "Qmax": rng.gamma(3.0, 20.0, size=300),
            "Qmed": rng.gamma(3.0, 5.0, size=300),
            "Duracion": rng.gamma(2.0, 10.0, size=300),
            "shape_type": rng.integers(1, 4, size=300).astype(float),
        }
    )
    synthetic = FloodEventCopula(verbose=False).fit(events).sample(50)
    assert list(synthetic.columns) == ["Qmax", "Qmed", "Duracion", "shape_type"]
    assert len(synthetic) == 50

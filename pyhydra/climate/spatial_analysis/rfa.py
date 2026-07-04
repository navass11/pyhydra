"""
Regional Frequency Analysis (RFA) of extreme events.

Three fitting approaches for GEV (Generalized Extreme Value) distributions,
applicable to annual maxima of precipitation, discharge, or any hydro-
meteorological variable:

- **MLE**   — Maximum Likelihood Estimation via scipy.
- **L-moments** — Method of L-moments via lmoments3.
- **Bayesian** — MCMC via PyMC + NUTS (weakly informative priors).

Regional analysis normalises each station's series by its index flood
(mean annual maximum) before fitting a single regional GEV, then
re-scales the regional quantiles back to each station.

Dependencies
------------
- scipy (always available)
- lmoments3: ``pip install lmoments3``
- pymc: ``pip install pymc``  (Bayesian method only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import genextreme


# ---------------------------------------------------------------------------
# Dependency guards
# ---------------------------------------------------------------------------

def _require_lmoments():
    try:
        import lmoments3 as lm
        import lmoments3.distr as lmd
        return lm, lmd
    except ImportError as exc:
        raise ImportError(
            "lmoments3 is required for L-moment fitting.\n"
            "Install it with: pip install lmoments3"
        ) from exc




# ---------------------------------------------------------------------------
# Point frequency analysis
# ---------------------------------------------------------------------------

def fit_gev_mle(data, xi_bounds=(-0.5, 0.8)):
    """
    Fit a GEV distribution by Maximum Likelihood Estimation.

    Uses multi-start optimisation with bounded shape parameter to avoid
    degenerate solutions (xi → ±∞) that unconstrained MLE can find on
    small samples (n < 50).

    Args:
        data: 1-D array of annual maxima.
        xi_bounds: (lower, upper) bounds on xi — keeps estimates physical.

    Returns:
        dict with keys ``mu`` (location), ``sigma`` (scale), ``xi`` (shape).
        Note: xi > 0 → Fréchet (heavy tail); xi = 0 → Gumbel; xi < 0 → Weibull.
    """
    import warnings
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    mu0, sig0 = float(np.mean(arr)), float(np.std(arr))

    # Build starting points: L-moments first, then a grid
    starts = []
    try:
        p0 = fit_gev_lmom(arr)
        starts.append((-p0["xi"], p0["mu"], p0["sigma"]))
    except Exception:
        pass
    for xi0 in [0.0, 0.1, -0.1, 0.2, -0.2]:
        starts.append((xi0, mu0, sig0 * 0.5))

    best_nll, best = np.inf, None
    for c0, loc0, scale0 in starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c_fit, loc_fit, scale_fit = genextreme.fit(arr, c0, loc=loc0, scale=scale0)
            xi_fit = -c_fit
            if not (xi_bounds[0] <= xi_fit <= xi_bounds[1]) or scale_fit <= 0:
                continue
            nll = -np.sum(genextreme.logpdf(arr, c_fit, loc=loc_fit, scale=scale_fit))
            if np.isfinite(nll) and nll < best_nll:
                best_nll, best = nll, {"mu": float(loc_fit), "sigma": float(scale_fit), "xi": xi_fit}
        except Exception:
            continue

    if best is None:
        # fall back to L-moments
        return fit_gev_lmom(arr)
    return best


def fit_gev_lmom(data):
    """
    Fit a GEV distribution by the method of L-moments.

    Args:
        data: 1-D array of annual maxima.

    Returns:
        dict with keys ``mu`` (location), ``sigma`` (scale), ``xi`` (shape).
    """
    lm, lmd = _require_lmoments()
    ratios = lm.lmom_ratios(list(data), nmom=4)
    params = lmd.gev.lmom_fit(list(data), lmom_ratios=ratios)
    # lmoments3 uses the same sign convention as scipy genextreme: c = -xi.
    # Negate to convert to pyhydra convention (xi > 0 = Fréchet = heavy tail).
    return {
        "mu":    float(params.get("loc", 0.0)),
        "sigma": float(params.get("scale", 1.0)),
        "xi":   -float(params.get("c", params.get("shape", 0.0))),
    }


def return_level(params, T):
    """
    Compute the T-year return level from GEV parameters.

    Args:
        params: dict with ``mu``, ``sigma``, ``xi`` (from fit_gev_mle
                or fit_gev_lmom).
        T:      Return period in years (scalar or array).

    Returns:
        Return level(s) — same shape as T.
    """
    # scipy genextreme uses c = -xi sign convention
    return genextreme.ppf(1 - 1 / np.asarray(T),
                          -params["xi"],
                          loc=params["mu"],
                          scale=params["sigma"])


def fit_gev_bayes(
    data,
    n_chains=4,
    n_samples=1000,
    warmup=1000,
    adapt_delta=0.95,
    progressbar=True,
    random_seed=None,
    prior="stan",
):
    """
    Fit a GEV distribution by Bayesian MCMC (PyMC + NUTS).

    Args:
        data:      1-D array of annual maxima.
        n_chains:  Number of MCMC chains.
        n_samples: Samples per chain.
        warmup:    Tuning samples per chain.
        adapt_delta: NUTS target acceptance rate.
        progressbar: Show PyMC sampler progress bar.
        random_seed: Random seed passed to PyMC.
        prior: Prior family passed to ``fit_gev_mcmc``.

    Returns:
        pd.DataFrame of posterior samples with columns ``mu``, ``sigma``, ``xi``.
    """
    from pyhydra.climate.time_series.extremes import fit_gev_mcmc
    return fit_gev_mcmc(
        data,
        n_samples=n_samples,
        n_chains=n_chains,
        adapt_delta=adapt_delta,
        warmup=warmup,
        progressbar=progressbar,
        random_seed=random_seed,
        prior=prior,
    )


def return_level_bayes(posterior, T, credible=0.95):
    """
    Compute return-level posterior distribution from MCMC samples.

    Args:
        posterior: pd.DataFrame from :func:`fit_gev_bayes`.
        T:         Return period (scalar).
        credible:  Credible interval width (default 0.95).

    Returns:
        dict with keys ``median``, ``lower``, ``upper``.
    """
    alpha = (1 - credible) / 2
    levels = genextreme.ppf(
        1 - 1 / T,
        -posterior["xi"].values,       # scipy c = -xi (pyhydra convention: xi>0 = Fréchet)
        loc=posterior["mu"].values,
        scale=posterior["sigma"].values,
    )
    levels = levels[np.isfinite(levels)]
    if len(levels) == 0:
        return {"median": np.nan, "lower": np.nan, "upper": np.nan}
    return {
        "median": float(np.median(levels)),
        "lower":  float(np.quantile(levels, alpha)),
        "upper":  float(np.quantile(levels, 1 - alpha)),
    }


# ---------------------------------------------------------------------------
# Regional frequency analysis
# ---------------------------------------------------------------------------

def regional_index_flood(data_dict):
    """
    Normalise each station's series by its index flood (mean annual maximum).

    Args:
        data_dict: dict mapping station name → 1-D array of annual maxima.

    Returns:
        normalised: dict mapping station name → normalised series.
        index_floods: pd.Series mapping station name → mean annual maximum.
    """
    index_floods = {k: np.mean(v) for k, v in data_dict.items()}
    normalised = {k: np.asarray(v) / index_floods[k] for k, v in data_dict.items()}
    return normalised, pd.Series(index_floods, name="index_flood")


def fit_regional_gev(
    data_dict,
    method="lmom",
    n_chains=4,
    n_samples=1000,
    warmup=1000,
    adapt_delta=0.95,
    progressbar=True,
    random_seed=None,
    prior="stan",
):
    """
    Fit a regional GEV to normalised pooled data from multiple stations.

    Args:
        data_dict: dict mapping station name → 1-D array of annual maxima.
        method:    ``'mle'``, ``'lmom'`` (default) or ``'bayes'``.
        n_chains:  MCMC chains, used only when ``method='bayes'``.
        n_samples: Samples per chain, used only when ``method='bayes'``.
        warmup:    Tuning samples per chain, used only when ``method='bayes'``.
        adapt_delta: NUTS target acceptance rate, used only when ``method='bayes'``.
        progressbar: Show PyMC sampler progress bar, used only when ``method='bayes'``.
        random_seed: Random seed passed to PyMC, used only when ``method='bayes'``.
        prior: Prior family passed to ``fit_gev_bayes``, used only when
            ``method='bayes'``.

    Returns:
        regional_params: dict with ``mu``, ``sigma``, ``xi`` (for ``'mle'``/
            ``'lmom'``), or a pd.DataFrame of posterior samples with columns
            ``mu``, ``sigma``, ``xi`` (for ``'bayes'``) — fitted to the
            pooled, index-flood-normalised data (index ≈ 1).
        index_floods:    pd.Series with each station's index flood.
    """
    normalised, index_floods = regional_index_flood(data_dict)
    pooled = np.concatenate(list(normalised.values()))

    if method == "lmom":
        regional_params = fit_gev_lmom(pooled)
    elif method == "mle":
        regional_params = fit_gev_mle(pooled)
    elif method == "bayes":
        regional_params = fit_gev_bayes(
            pooled,
            n_chains=n_chains,
            n_samples=n_samples,
            warmup=warmup,
            adapt_delta=adapt_delta,
            progressbar=progressbar,
            random_seed=random_seed,
            prior=prior,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'mle', 'lmom' or 'bayes'.")

    return regional_params, index_floods


def regional_return_levels(data_dict, T_values=(2, 5, 10, 20, 50, 100),
                           method="lmom", credible=0.90, n_chains=4, n_samples=1000,
                           warmup=1000, adapt_delta=0.95, progressbar=True,
                           random_seed=None, prior="stan"):
    """
    Compute T-year return levels for each station via regional GEV.

    Args:
        data_dict: dict mapping station name → 1-D array of annual maxima.
        T_values:  Iterable of return periods in years.
        method:    ``'mle'``, ``'lmom'`` or ``'bayes'``.
        credible:  Credible interval width, used only when ``method='bayes'``.
        n_chains:  MCMC chains, used only when ``method='bayes'``.
        n_samples: Samples per chain, used only when ``method='bayes'``.
        warmup:    Tuning samples per chain, used only when ``method='bayes'``.
        adapt_delta: NUTS target acceptance rate, used only when ``method='bayes'``.
        progressbar: Show PyMC sampler progress bar, used only when ``method='bayes'``.
        random_seed: Random seed passed to PyMC, used only when ``method='bayes'``.
        prior: Prior family passed to ``fit_regional_gev``, used only when
            ``method='bayes'``.

    Returns:
        If ``method`` is ``'mle'`` or ``'lmom'``: a single pd.DataFrame
        (stations × T values) with return levels.

        If ``method='bayes'``: a tuple ``(median, lower, upper)`` of three
        pd.DataFrame (stations × T values) — the posterior median and the
        ``credible``-width interval bounds, each station's regional
        dimensionless posterior scaled by its own index flood.
    """
    T_arr = np.asarray(T_values)
    col_names = [f"T{int(t)}" for t in T_arr]

    if method == "bayes":
        try:
            posterior, index_floods = fit_regional_gev(
                data_dict,
                method="bayes",
                n_chains=n_chains,
                n_samples=n_samples,
                warmup=warmup,
                adapt_delta=adapt_delta,
                progressbar=progressbar,
                random_seed=random_seed,
                prior=prior,
            )
        except TypeError as exc:
            # Backward-compatible path for tests or user wrappers monkeypatching
            # fit_regional_gev with the pre-warmup signature.
            if "unexpected keyword argument" not in str(exc):
                raise
            posterior, index_floods = fit_regional_gev(
                data_dict,
                method="bayes",
                n_chains=n_chains,
                n_samples=n_samples,
            )
        medians, lowers, uppers = {}, {}, {}
        for station, mu in index_floods.items():
            m, lo, hi = [], [], []
            for T in T_arr:
                ci = return_level_bayes(posterior, T, credible=credible)
                m.append(ci["median"] * mu)
                lo.append(ci["lower"] * mu)
                hi.append(ci["upper"] * mu)
            medians[station], lowers[station], uppers[station] = m, lo, hi

        return (
            pd.DataFrame(medians, index=col_names).T,
            pd.DataFrame(lowers, index=col_names).T,
            pd.DataFrame(uppers, index=col_names).T,
        )

    regional_params, index_floods = fit_regional_gev(data_dict, method=method)

    rows = {}
    for station, mu in index_floods.items():
        regional_q = return_level(regional_params, T_arr)
        rows[station] = regional_q * mu

    return pd.DataFrame(rows, index=col_names).T


# ---------------------------------------------------------------------------
# Hosking & Wallis (1997) homogeneity diagnostics
# ---------------------------------------------------------------------------
#
# Formulas verified against the reference implementation `regtst.s()` in the
# `lmomRFA` R package (J. R. M. Hosking's own package; source:
# https://github.com/cran/lmomRFA/blob/master/R/lmomRFA.r), which accompanies
# Hosking, J.R.M. and Wallis, J.R. (1997), "Regional Frequency Analysis: An
# Approach Based on L-Moments", Cambridge University Press.

# Discordancy critical values, Hosking & Wallis (1997) Table 3.1, indexed by
# number of sites N (as tabulated in lmomRFA's `dc1`). N <= 4 or N >= 15: 3.0.
_DISCORDANCY_CRITICAL = {
    5: 1.3330, 6: 1.6481, 7: 1.9166, 8: 2.1401, 9: 2.3287, 10: 2.4906,
    11: 2.6321, 12: 2.7573, 13: 2.8694, 14: 2.9709,
}


def _site_lmoment_ratios(data_dict):
    """At-site (n, t, t3, t4) — record length, L-CV, L-skew, L-kurtosis."""
    lm, _lmd = _require_lmoments()
    names = list(data_dict)
    n = np.array([len(np.asarray(data_dict[k])) for k in names], dtype=float)
    t, t3, t4 = [], [], []
    for name in names:
        r = lm.lmom_ratios(list(data_dict[name]), nmom=4)
        t.append(r[1] / r[0])
        t3.append(r[2])
        t4.append(r[3])
    return names, n, np.asarray(t), np.asarray(t3), np.asarray(t4)


def discordancy_critical_value(n_sites):
    """
    Critical value for the discordancy statistic (Hosking & Wallis 1997,
    Table 3.1). Sites with ``Di`` above this value are discordant.

    Args:
        n_sites: Number of sites in the region.

    Returns:
        Critical Di value (3.0 for n_sites <= 4 or >= 15).
    """
    if n_sites <= 4 or n_sites >= 15:
        return 3.0
    return _DISCORDANCY_CRITICAL[n_sites]


def regional_discordancy(data_dict):
    """
    Hosking & Wallis (1997) discordancy statistic ``Di`` for each site.

    Flags sites whose L-moment ratios (L-CV, L-skew, L-kurtosis) are unusual
    relative to the (unweighted) regional average — a likely data error or a
    site that should be excluded before regional pooling.

    Args:
        data_dict: dict mapping station name → 1-D array of annual maxima.
            At least 4 sites are required (need an invertible 3x3 covariance
            of the L-moment-ratio vectors); Hosking & Wallis tabulate
            critical values from 5 sites up.

    Returns:
        pd.DataFrame indexed by station name with columns ``Di``,
        ``critical`` (region-size critical value) and ``discordant``
        (``Di > critical``).
    """
    names, _n, t, t3, t4 = _site_lmoment_ratios(data_dict)
    n_sites = len(names)
    if n_sites <= 3:
        raise ValueError("regional_discordancy requires at least 4 sites.")

    u = np.column_stack([t, t3, t4])
    ubar = u.mean(axis=0)
    d = u - ubar
    s = (d.T @ d) / (n_sites - 1)
    s_inv = np.linalg.inv(s)

    scale = n_sites / (3.0 * (n_sites - 1))
    di = np.array([scale * (row @ s_inv @ row) for row in d])
    critical = discordancy_critical_value(n_sites)

    return pd.DataFrame(
        {"Di": di, "critical": critical, "discordant": di > critical},
        index=names,
    )


def _weighted_l_cv_dispersion(t, weights):
    """Weighted RMS deviation of L-CV about its own weighted mean."""
    t_bar = np.average(t, weights=weights)
    return float(np.sqrt(np.average((t - t_bar) ** 2, weights=weights)))


def regional_heterogeneity(data_dict, n_sim=1000, seed=None):
    """
    Hosking & Wallis (1993/1997) heterogeneity statistic ``H`` (``H1``).

    Compares the observed record-length-weighted dispersion of at-site L-CV
    to the dispersion expected from ``n_sim`` simulated homogeneous regions,
    drawn from a 4-parameter kappa distribution fitted to the
    record-length-weighted regional-average L-moment ratios. Interpretation
    (Hosking & Wallis 1997): ``H < 1`` acceptably homogeneous, ``1 <= H < 2``
    possibly heterogeneous, ``H >= 2`` definitely heterogeneous.

    Args:
        data_dict: dict mapping station name → 1-D array of annual maxima.
            At least 2 sites are required.
        n_sim: Number of Monte Carlo regions to simulate (1000 in Hosking &
            Wallis; reduce for quick exploratory checks).
        seed: Random seed for the simulation.

    Returns:
        dict with keys ``H`` (the H1 statistic), ``V_obs`` (observed
        weighted L-CV dispersion), ``V_sim_mean`` and ``V_sim_std``
        (moments of the simulated null distribution).

    Raises:
        ImportError: If lmoments3 is not installed.
        ValueError: If fewer than 2 sites are given, or no valid kappa
            distribution can be fitted to the regional-average L-moments
            (rare; indicates a very unusual region).
    """
    _lm, lmd = _require_lmoments()
    names, n, t, t3, t4 = _site_lmoment_ratios(data_dict)
    if len(names) < 2:
        raise ValueError("regional_heterogeneity requires at least 2 sites.")

    v_obs = _weighted_l_cv_dispersion(t, n)

    t_r = np.average(t, weights=n)
    t3_r = np.average(t3, weights=n)
    t4_r = np.average(t4, weights=n)

    try:
        kappa_params = lmd.kap.lmom_fit(lmom_ratios=[1.0, t_r, t3_r, t4_r])
        frozen = lmd.kap(**kappa_params)
    except Exception as exc:
        raise ValueError(
            "Could not fit a kappa distribution to the regional-average "
            "L-moments; the region may be too unusual for this diagnostic."
        ) from exc

    rng = np.random.default_rng(seed)
    n_int = n.astype(int)
    v_sim = np.empty(n_sim)
    for s in range(n_sim):
        t_sim = np.empty(len(names))
        for j, nj in enumerate(n_int):
            sample = frozen.rvs(size=nj, random_state=rng)
            r = _lm.lmom_ratios(list(sample), nmom=2)
            t_sim[j] = r[1] / r[0]
        v_sim[s] = _weighted_l_cv_dispersion(t_sim, n)

    mu_v, sigma_v = float(v_sim.mean()), float(v_sim.std(ddof=1))
    return {
        "H": (v_obs - mu_v) / sigma_v,
        "V_obs": v_obs,
        "V_sim_mean": mu_v,
        "V_sim_std": sigma_v,
    }

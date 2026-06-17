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


def fit_gev_bayes(data, n_chains=4, n_samples=1000):
    """
    Fit a GEV distribution by Bayesian MCMC (PyMC + NUTS).

    Args:
        data:      1-D array of annual maxima.
        n_chains:  Number of MCMC chains.
        n_samples: Samples per chain.

    Returns:
        pd.DataFrame of posterior samples with columns ``mu``, ``sigma``, ``xi``.
    """
    from pyhydra.climate.time_series.extremes import fit_gev_mcmc
    return fit_gev_mcmc(data, n_samples=n_samples, n_chains=n_chains)


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


def fit_regional_gev(data_dict, method="lmom"):
    """
    Fit a regional GEV to normalised pooled data from multiple stations.

    Args:
        data_dict: dict mapping station name → 1-D array of annual maxima.
        method:    ``'mle'`` or ``'lmom'`` (default ``'lmom'``).

    Returns:
        regional_params: dict with ``shape``, ``loc``, ``scale`` (for index=1).
        index_floods:    pd.Series with each station's index flood.
    """
    normalised, index_floods = regional_index_flood(data_dict)
    pooled = np.concatenate(list(normalised.values()))

    if method == "lmom":
        regional_params = fit_gev_lmom(pooled)
    elif method == "mle":
        regional_params = fit_gev_mle(pooled)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'mle' or 'lmom'.")

    return regional_params, index_floods


def regional_return_levels(data_dict, T_values=(2, 5, 10, 20, 50, 100),
                           method="lmom"):
    """
    Compute T-year return levels for each station via regional GEV.

    Args:
        data_dict: dict mapping station name → 1-D array of annual maxima.
        T_values:  Iterable of return periods in years.
        method:    ``'mle'`` or ``'lmom'``.

    Returns:
        pd.DataFrame (stations × T values) with return levels.
    """
    regional_params, index_floods = fit_regional_gev(data_dict, method=method)
    T_arr = np.asarray(T_values)

    col_names = [f"T{int(t)}" for t in T_arr]

    rows = {}
    for station, mu in index_floods.items():
        regional_q = return_level(regional_params, T_arr)
        rows[station] = regional_q * mu

    return pd.DataFrame(rows, index=col_names).T

"""
Extreme value analysis for hydro-meteorological time series.

Two standard approaches:

Block Maxima (GEV)
    Extract annual (or seasonal) maxima and fit a Generalised Extreme Value
    distribution.  Suitable for long records with clear seasonal structure.

Peaks Over Threshold (GPD)
    Extract independent peaks above a threshold and fit a Generalised Pareto
    Distribution.  More data-efficient for short records; threshold choice
    is critical.

Fitting methods
---------------
- MLE     : Maximum Likelihood with robust multi-start optimisation and shape
            parameter bounds to avoid non-physical estimates.
- L-mom   : L-moments (lmoments3) — resistant to outliers, good starting point
            for MLE.
- Fisher  : MLE + asymptotic covariance from the numerical Hessian of the
            negative log-likelihood → samples from a multivariate normal.
            Fast uncertainty estimation without MCMC.
- MAP     : Maximum A Posteriori — fast Bayesian point estimate with weakly
            informative priors via scipy optimisation.
- MCMC    : Full posterior via PyMC + NUTS (non-centred parameterisation for
            better convergence; no C++ compilation at runtime).

All fitting functions fall back gracefully:
  1. MLE with L-moments warm-start.
  2. If MLE fails: L-moments estimate.
  3. Bayesian MAP via scipy; full MCMC only when pymc is installed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _require_lmoments():
    try:
        import lmoments3
        return lmoments3
    except ImportError as exc:
        raise ImportError(
            "lmoments3 is required for L-moment fitting.\n"
            "Install with: pip install lmoments3"
        ) from exc


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_block_maxima(series: pd.Series, freq: str = "YE") -> pd.Series:
    """
    Extract block maxima from a time series.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex.
    freq : str
        Resampling frequency for blocks (default 'YE' = annual).
        Use 'QE' for seasonal, 'ME' for monthly.

    Returns
    -------
    pd.Series of block maxima indexed by block end date.
    Only blocks with at least one non-NaN value are kept.

    Examples
    --------
    >>> am = extract_block_maxima(discharge_series, freq='YE')
    """
    return series.resample(freq).max().dropna()


def extract_pot(series: pd.Series, threshold: float,
                min_gap: int = 7) -> pd.Series:
    """
    Extract independent Peaks Over Threshold (POT).

    Identifies local maxima above `threshold`, then enforces independence by
    keeping only the highest peak within any `min_gap`-step window.

    Parameters
    ----------
    series : pd.Series
        Time series with DatetimeIndex.
    threshold : float
        Minimum value for a peak to be retained.
    min_gap : int
        Minimum number of time steps between independent peaks (default 7).

    Returns
    -------
    pd.Series of peak values indexed by their timestamp.

    Examples
    --------
    >>> peaks = extract_pot(discharge_series, threshold=500, min_gap=7)
    """
    values = series.values.astype(float)
    idx = series.index
    n = len(values)

    # find all local maxima above threshold
    candidates = []
    for i in range(1, n - 1):
        if values[i] >= threshold and values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            candidates.append(i)

    if not candidates:
        return pd.Series(dtype=float, name=series.name)

    # enforce independence via min_gap sliding window
    selected = [candidates[0]]
    for ci in candidates[1:]:
        if ci - selected[-1] >= min_gap:
            selected.append(ci)
        elif values[ci] > values[selected[-1]]:
            selected[-1] = ci

    return pd.Series(values[selected], index=idx[selected], name=series.name)


def threshold_stability_plot(series: pd.Series,
                             thresholds: np.ndarray | None = None,
                             min_peaks: int = 10):
    """
    GPD threshold stability plot (mean excess and shape parameter vs threshold).

    A good threshold is the lowest value where both plots are approximately
    linear/flat — indicating GPD behaviour.

    Parameters
    ----------
    series : pd.Series
        Time series of values.
    thresholds : ndarray or None
        Thresholds to evaluate. Defaults to the 50th–95th percentile range
        at 1 % quantile steps.
    min_peaks : int
        Skip thresholds with fewer than this many exceedances (default 10).

    Returns
    -------
    pd.DataFrame with columns: threshold, n_exceed, mean_excess,
                                gpd_shape, gpd_scale.
    """
    x = series.dropna().values.astype(float)
    if thresholds is None:
        thresholds = np.percentile(x, np.arange(50, 96, 1))

    rows = []
    for u in thresholds:
        exc = x[x > u] - u
        if len(exc) < min_peaks:
            continue
        params = _fit_gpd_mle(exc)
        rows.append({
            "threshold":  float(u),
            "n_exceed":   len(exc),
            "mean_excess": float(exc.mean()),
            "gpd_shape":  float(params["shape"]),
            "gpd_scale":  float(params["scale"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Robust GEV fitting
# ---------------------------------------------------------------------------

def _lmom_gev(data: np.ndarray) -> dict:
    """L-moment GEV estimates (fast, outlier-resistant)."""
    _require_lmoments()
    from lmoments3 import distr
    para = distr.gev.lmom_fit(data.tolist())
    # lmoments3 uses the same c convention as scipy.stats.genextreme:
    # c > 0 = Weibull (bounded upper tail), c < 0 = Fréchet (heavy tail).
    # pyhydra uses xi = -c (xi > 0 = Fréchet), so negate here.
    return {"mu": float(para["loc"]), "sigma": float(para["scale"]),
            "xi": -float(para["c"])}


def _fit_gev_mle_robust(data: np.ndarray,
                        xi_bounds: tuple = (-0.5, 0.8)) -> dict:
    """
    MLE GEV fitting with multiple starts and bounded shape parameter.

    Starts from L-moments estimate, then tries a grid of starting points.
    Returns the parameter set with the highest log-likelihood.
    """
    from scipy.stats import genextreme

    # Try L-moments starting point first
    try:
        lm = _lmom_gev(data)
        starts = [(-lm["xi"], lm["mu"], lm["sigma"])]
    except Exception:
        starts = []

    # Add grid of generic starting points
    mu0 = float(np.mean(data))
    sig0 = float(np.std(data))
    for xi0 in [0.0, 0.1, -0.1, 0.2, -0.2]:
        starts.append((xi0, mu0, sig0 * 0.5))

    best_nll = np.inf
    best_params = None

    for c0, loc0, scale0 in starts:
        # scipy uses -xi convention: GEV(c) with c = -xi
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c_fit, loc_fit, scale_fit = genextreme.fit(
                    data, c0, loc=loc0, scale=scale0
                )
            xi_fit = -c_fit
            if not (xi_bounds[0] <= xi_fit <= xi_bounds[1]):
                continue
            if scale_fit <= 0:
                continue
            nll = -np.sum(genextreme.logpdf(data, c_fit, loc=loc_fit, scale=scale_fit))
            if np.isfinite(nll) and nll < best_nll:
                best_nll = nll
                best_params = {"mu": float(loc_fit), "sigma": float(scale_fit),
                               "xi": float(xi_fit)}
        except Exception:
            continue

    if best_params is None:
        raise RuntimeError(
            "GEV MLE failed for all starting points. "
            "Try xi_bounds=(-1, 1) or method='lmom'."
        )
    return best_params


def fit_gev(data: np.ndarray | pd.Series,
            method: str = "mle",
            xi_bounds: tuple = (-0.5, 0.8)) -> dict:
    """
    Fit a GEV distribution to block maxima.

    Parameters
    ----------
    data : array-like
        Annual (or block) maxima.
    method : str
        'mle'  — robust MLE with L-moments warm-start (default).
        'lmom' — L-moment estimation (fast, no iteration).
        'both' — run both and return a dict with both results.
    xi_bounds : tuple
        (lower, upper) bounds on the shape parameter for MLE.
        Default (-0.5, 0.5) rejects non-physical estimates.

    Returns
    -------
    dict with keys: mu (location), sigma (scale), xi (shape).
    When method='both': dict with sub-dicts 'mle' and 'lmom'.

    Examples
    --------
    >>> params = fit_gev(annual_maxima, method='mle')
    >>> print(params)
    {'mu': 450.2, 'sigma': 120.3, 'xi': 0.08}
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        raise ValueError("At least 5 data points required for GEV fitting.")

    if method == "mle":
        return _fit_gev_mle_robust(arr, xi_bounds=xi_bounds)
    elif method == "lmom":
        return _lmom_gev(arr)
    elif method == "both":
        result = {}
        try:
            result["mle"] = _fit_gev_mle_robust(arr, xi_bounds=xi_bounds)
        except RuntimeError as e:
            result["mle"] = {"error": str(e)}
        try:
            result["lmom"] = _lmom_gev(arr)
        except ImportError:
            result["lmom"] = {"error": "lmoments3 not installed"}
        return result
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'mle', 'lmom', or 'both'.")


# ---------------------------------------------------------------------------
# Robust GPD fitting
# ---------------------------------------------------------------------------

def _fit_gpd_mle(exceedances: np.ndarray,
                 xi_bounds: tuple = (-0.5, 1.0)) -> dict:
    """MLE GPD fitting with bounded shape and multiple starts."""
    from scipy.stats import genpareto

    starts = []
    sig0 = float(np.mean(exceedances))
    for xi0 in [0.0, 0.1, 0.2, -0.1, 0.3]:
        starts.append((xi0, sig0))

    best_nll = np.inf
    best_params = None

    for xi0, sig0 in starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c_fit, loc_fit, scale_fit = genpareto.fit(
                    exceedances, xi0, loc=0, scale=sig0
                )
            xi_fit = float(c_fit)
            if not (xi_bounds[0] <= xi_fit <= xi_bounds[1]):
                continue
            if scale_fit <= 0:
                continue
            nll = -np.sum(genpareto.logpdf(exceedances, c_fit, loc=0, scale=scale_fit))
            if np.isfinite(nll) and nll < best_nll:
                best_nll = nll
                best_params = {"scale": float(scale_fit), "shape": xi_fit}
        except Exception:
            continue

    if best_params is None:
        raise RuntimeError(
            "GPD MLE failed. Try xi_bounds=(-0.5, 2.0) or a lower threshold."
        )
    return best_params


def _lmom_gpd(exceedances: np.ndarray) -> dict:
    lm = _require_lmoments()
    from lmoments3 import distr
    para = distr.gpa.lmom_fit(exceedances.tolist())
    return {"scale": float(para["scale"]), "shape": float(para["c"])}


def fit_gpd(series: pd.Series | np.ndarray,
            threshold: float,
            method: str = "mle",
            min_gap: int = 7,
            xi_bounds: tuple = (-0.5, 1.0)) -> dict:
    """
    Fit a GPD to exceedances above `threshold`.

    Parameters
    ----------
    series : pd.Series or ndarray
        Full time series (not pre-extracted exceedances).
    threshold : float
        Exceedance threshold.
    method : str
        'mle' (default) or 'lmom'.
    min_gap : int
        Minimum steps between independent peaks (passed to extract_pot).
    xi_bounds : tuple
        Bounds on the shape parameter for MLE.

    Returns
    -------
    dict with keys:
        scale   — GPD scale parameter.
        shape   — GPD shape parameter (xi).
        threshold — the threshold used.
        n_exceed  — number of exceedances used.
        lambda_rate — mean annual rate of exceedances (peaks/year).

    Examples
    --------
    >>> params = fit_gpd(discharge, threshold=500, method='mle')
    """
    if isinstance(series, np.ndarray):
        series = pd.Series(series)

    peaks = extract_pot(series, threshold=threshold, min_gap=min_gap)
    exc = peaks.values - threshold

    if len(exc) < 5:
        raise ValueError(
            f"Only {len(exc)} peaks above threshold {threshold}. "
            "Lower the threshold or reduce min_gap."
        )

    if method == "mle":
        p = _fit_gpd_mle(exc, xi_bounds=xi_bounds)
    elif method == "lmom":
        p = _lmom_gpd(exc)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'mle' or 'lmom'.")

    n_years = (series.index[-1] - series.index[0]).days / 365.25
    lambda_rate = len(peaks) / n_years if n_years > 0 else np.nan

    return {
        "scale":       p["scale"],
        "shape":       p["shape"],
        "threshold":   threshold,
        "n_exceed":    len(exc),
        "lambda_rate": float(lambda_rate),
    }


# ---------------------------------------------------------------------------
# Return levels
# ---------------------------------------------------------------------------

def return_level_gev(params: dict, T: float) -> float:
    """
    GEV return level for return period T (years).

    Parameters
    ----------
    params : dict
        Output of fit_gev() — keys: mu, sigma, xi.
    T : float
        Return period in years.

    Returns
    -------
    float — T-year return level.
    """
    mu, sigma, xi = params["mu"], params["sigma"], params["xi"]
    p = 1.0 - 1.0 / T
    if abs(xi) < 1e-6:
        return mu - sigma * np.log(-np.log(p))
    return mu + sigma / xi * ((-np.log(p)) ** (-xi) - 1.0)


def return_level_gpd(params: dict, T: float) -> float:
    """
    GPD return level for return period T (years).

    Uses the Poisson process model:
        x_T = threshold + (sigma/xi) * ((lambda * T)^xi - 1)

    Parameters
    ----------
    params : dict
        Output of fit_gpd() — keys: scale, shape, threshold, lambda_rate.
    T : float
        Return period in years.

    Returns
    -------
    float — T-year return level.
    """
    sigma = params["scale"]
    xi    = params["shape"]
    u     = params["threshold"]
    lam   = params["lambda_rate"]

    if abs(xi) < 1e-6:
        return u + sigma * np.log(lam * T)
    return u + sigma / xi * ((lam * T) ** xi - 1.0)


def return_levels(params: dict, T_values, dist: str = "gev") -> pd.Series:
    """
    Compute return levels for multiple return periods.

    Parameters
    ----------
    params : dict
        Output of fit_gev() or fit_gpd().
    T_values : list of float
        Return periods.
    dist : str
        'gev' (default) or 'gpd'.

    Returns
    -------
    pd.Series indexed by return period.
    """
    fn = return_level_gev if dist == "gev" else return_level_gpd
    return pd.Series(
        {T: fn(params, T) for T in T_values},
        name="return_level",
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def return_level_ci(data: np.ndarray | pd.Series,
                    T: float, dist: str = "gev",
                    method: str = "mle",
                    n_bootstrap: int = 500,
                    ci: float = 0.95,
                    threshold: float | None = None,
                    **kwargs) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for a return level.

    Parameters
    ----------
    data : array-like
        Block maxima (GEV) or full series (GPD).
    T : float
        Return period.
    dist : str
        'gev' or 'gpd'.
    method : str
        Fitting method — 'mle' or 'lmom'.
    n_bootstrap : int
        Bootstrap replicates (default 500).
    ci : float
        Coverage (default 0.95).
    threshold : float
        Required when dist='gpd'.
    **kwargs
        Passed to fit_gev or fit_gpd.

    Returns
    -------
    (point_estimate, lower, upper)
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    rng = np.random.default_rng(0)

    if dist == "gev":
        point = return_level_gev(fit_gev(arr, method=method, **kwargs), T)
        samples = []
        for _ in range(n_bootstrap):
            boot = rng.choice(arr, size=len(arr), replace=True)
            try:
                p = fit_gev(boot, method=method, **kwargs)
                samples.append(return_level_gev(p, T))
            except Exception:
                continue
    elif dist == "gpd":
        if threshold is None:
            raise ValueError("threshold is required for GPD.")
        point = return_level_gpd(fit_gpd(arr, threshold=threshold,
                                         method=method, **kwargs), T)
        samples = []
        series = pd.Series(arr)
        for _ in range(n_bootstrap):
            boot_idx = rng.choice(len(arr), size=len(arr), replace=True)
            boot = pd.Series(arr[boot_idx])
            try:
                p = fit_gpd(boot, threshold=threshold, method=method, **kwargs)
                samples.append(return_level_gpd(p, T))
            except Exception:
                continue
    else:
        raise ValueError("dist must be 'gev' or 'gpd'.")

    alpha = (1 - ci) / 2
    samples = np.array(samples)
    lower = float(np.quantile(samples, alpha)) if len(samples) else np.nan
    upper = float(np.quantile(samples, 1 - alpha)) if len(samples) else np.nan
    return float(point), lower, upper


# ---------------------------------------------------------------------------
# Bayesian GEV — MAP + optional MCMC
# ---------------------------------------------------------------------------

def fit_gev_map(data: np.ndarray | pd.Series,
                mu_prior_mean: float | None = None,
                sigma_prior_scale: float = 0.5,
                xi_prior_std: float = 0.3) -> dict:
    """
    Maximum A Posteriori (MAP) GEV estimate.

    Uses log-Normal prior on sigma and Normal priors on mu and xi.
    No MCMC required — fast and stable with weakly informative priors.

    Parameters
    ----------
    data : array-like
        Block maxima.
    mu_prior_mean : float or None
        Prior mean for mu. Defaults to sample mean.
    sigma_prior_scale : float
        Log-scale prior std for sigma (default 0.5).
    xi_prior_std : float
        Prior std for xi (default 0.3, centred at 0).

    Returns
    -------
    dict with keys: mu, sigma, xi, map_logpost.
    """
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    mu0 = mu_prior_mean if mu_prior_mean is not None else float(arr.mean())
    logsig0 = float(np.log(arr.std()))

    from scipy.stats import genextreme as _gev, norm as _norm

    def neg_logpost(params):
        mu, logsig, xi = params
        sigma = np.exp(logsig)
        # Log-likelihood
        ll = np.sum(_gev.logpdf(arr, -xi, loc=mu, scale=sigma))
        if not np.isfinite(ll):
            return 1e12
        # Log-priors
        lp_mu    = _norm.logpdf(mu, mu0, arr.std() * 2)
        lp_logsig = _norm.logpdf(logsig, logsig0, sigma_prior_scale)
        lp_xi    = _norm.logpdf(xi, 0.0, xi_prior_std)
        return -(ll + lp_mu + lp_logsig + lp_xi)

    # MLE warm-start
    try:
        start_mle = _fit_gev_mle_robust(arr)
        x0 = [start_mle["mu"], np.log(start_mle["sigma"]), start_mle["xi"]]
    except Exception:
        x0 = [mu0, logsig0, 0.0]

    res = minimize(neg_logpost, x0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6})
    mu_map, logsig_map, xi_map = res.x
    return {
        "mu":           float(mu_map),
        "sigma":        float(np.exp(logsig_map)),
        "xi":           float(xi_map),
        "map_logpost":  float(-res.fun),
    }


def fit_gev_fisher(data: np.ndarray | pd.Series,
                   n_samples: int = 1000) -> pd.DataFrame:
    """
    GEV uncertainty via Fisher information matrix (asymptotic covariance).

    Fits GEV by MLE, then computes the asymptotic covariance matrix from the
    numerical Hessian of the negative log-likelihood and samples from the
    resulting multivariate normal approximation.

    Much faster than MCMC — valid when the record is long enough for the
    asymptotic normal approximation to hold (rule of thumb: n ≥ 20).

    Parameters
    ----------
    data : array-like
        Block maxima.
    n_samples : int
        Number of parameter samples to draw (default 1000).

    Returns
    -------
    pd.DataFrame with columns mu, sigma, xi — one row per sample.

    Examples
    --------
    >>> samples = fit_gev_fisher(annual_maxima, n_samples=2000)
    >>> print(samples.describe().round(2))
    """
    from scipy.stats import genextreme, multivariate_normal

    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]

    # MLE point estimate
    params_mle = _fit_gev_mle_robust(arr)
    # scipy convention: shape c = -xi
    theta = (-params_mle["xi"], params_mle["mu"], params_mle["sigma"])

    # Numerical Hessian of the negative log-likelihood
    pm = 1e-5 * np.abs(np.array(theta))
    pm[pm < 1e-8] = 1e-5
    n = 3
    FI = np.full((n, n), np.nan)

    def nll(t):
        return genextreme.nnlf(t, arr)

    for i in range(n):
        p1, p2 = np.array(theta), np.array(theta)
        p1[i] += pm[i]; p2[i] -= pm[i]
        FI[i, i] = (nll(p1) - 2 * nll(theta) + nll(p2)) / pm[i] ** 2
        for j in range(i + 1, n):
            p3 = np.array(theta); p3[i] -= pm[i]
            p4 = np.array(theta); p4[j] -= pm[j]
            p5 = np.array(theta); p5[i] -= pm[i]; p5[j] -= pm[j]
            cov = (nll(theta) - nll(p3) - nll(p4) + nll(p5)) / (pm[i] * pm[j])
            FI[i, j] = FI[j, i] = cov

    try:
        acov = np.linalg.inv(FI)
        # Ensure positive-definite
        eigvals = np.linalg.eigvalsh(acov)
        if np.any(eigvals <= 0):
            acov += np.eye(n) * (-eigvals.min() + 1e-8)
    except np.linalg.LinAlgError:
        raise RuntimeError(
            "Fisher matrix inversion failed. The MLE may be at the boundary — "
            "try a longer record or use fit_gev_map() instead."
        )

    raw = multivariate_normal.rvs(mean=theta, cov=acov, size=n_samples)
    # raw columns: c (=-xi), mu, sigma; keep only samples with sigma > 0
    df = pd.DataFrame(raw, columns=["_c", "mu", "sigma"])
    df["xi"] = -df["_c"]
    df = df[df["sigma"] > 0][["mu", "sigma", "xi"]].reset_index(drop=True)
    return df


# Stan code for single-site GEV (non-centred parameterisation for mu/xi)
_GEV_STAN_CODE = """
data {
    int<lower=1> N;
    vector[N] y;
    real y_mean;
    real<lower=0> y_sd;
}
parameters {
    real mu_raw;
    real<lower=0> sigma;
    real<lower=-1, upper=1> xi;
}
transformed parameters {
    real mu = y_mean + y_sd * mu_raw;
}
model {
    mu_raw ~ normal(0, 1);
    sigma  ~ lognormal(log(y_sd), 1);
    xi     ~ normal(0, 0.5);
    for (n in 1:N) {
        real z = (y[n] - mu) / sigma;
        if (abs(xi) > 1e-6) {
            real t = 1.0 + xi * z;
            if (t > 0)
                target += -log(sigma) - (1.0 + 1.0/xi) * log(t)
                          - pow(t, -1.0/xi);
            else
                target += negative_infinity();
        } else {
            target += -log(sigma) - z - exp(-z);
        }
    }
}
"""


def fit_gev_mcmc(data: np.ndarray | pd.Series,
                 n_samples: int = 2000,
                 n_chains: int = 4,
                 adapt_delta: float = 0.95) -> pd.DataFrame:
    """
    Full Bayesian GEV via MCMC (PyMC + NUTS sampler).

    Non-centred parameterisation: mu = y_mean + y_sd * mu_raw.
    Priors:
      mu_raw ~ Normal(0, 1)
      sigma  ~ LogNormal(log(y_sd), 1)
      xi     ~ Normal(0, 0.5), bounded to (-1, 1)

    Parameters
    ----------
    data : array-like
        Block maxima (NaN-free).
    n_samples : int
        Posterior draws per chain (default 2000).
    n_chains : int
        Number of independent chains (default 4).
    adapt_delta : float
        NUTS target acceptance rate (default 0.95).

    Returns
    -------
    pd.DataFrame with columns mu, sigma, xi — all chains concatenated.

    Notes
    -----
    Requires: pip install pymc
    """
    try:
        import pymc as pm
        import pytensor.tensor as pt
    except ImportError as exc:
        raise ImportError(
            "pymc is required for full MCMC Bayesian GEV fitting.\n"
            "Install with: pip install pymc"
        ) from exc

    arr    = np.asarray(data, dtype=float)
    arr    = arr[np.isfinite(arr)]
    y_mean = float(arr.mean())
    y_sd   = float(max(arr.std(), 1e-6))

    with pm.Model() as model:
        # Non-centred parameterisation for mu
        mu_raw = pm.Normal("mu_raw", mu=0.0, sigma=1.0)
        sigma  = pm.LogNormal("sigma", mu=np.log(y_sd), sigma=1.0)
        xi     = pm.TruncatedNormal("xi", mu=0.0, sigma=0.5, lower=-1.0, upper=1.0)
        mu     = pm.Deterministic("mu", y_mean + y_sd * mu_raw)

        # GEV log-likelihood via CustomDist
        def gev_logp(y, mu, sigma, xi):
            z = (y - mu) / sigma
            # Safe xi: substitute 1e-6 when |xi|<=1e-6; Gumbel branch handles that.
            xi_safe = pt.where(pt.abs(xi) > 1e-6, xi, pt.ones_like(xi) * 1e-6)
            # Clip t away from zero so log/power are always finite.
            t = pt.clip(1.0 + xi_safe * z, 1e-10, 1e30)
            gev_lp = pt.sum(
                -pt.log(sigma)
                - (1.0 + 1.0 / xi_safe) * pt.log(t)
                - t ** (-1.0 / xi_safe)
            )
            gumbel_lp = pt.sum(-pt.log(sigma) - z - pt.exp(-z))
            return pt.switch(pt.abs(xi) > 1e-6, gev_lp, gumbel_lp)

        pm.CustomDist("obs", mu, sigma, xi,
                      logp=gev_logp, observed=arr)

        # Warm-start from MLE
        try:
            mle    = _fit_gev_mle_robust(arr)
            start  = {
                "mu_raw": (mle["mu"] - y_mean) / y_sd,
                "sigma":  float(np.clip(mle["sigma"], y_sd * 0.05, y_sd * 10)),
                "xi":     float(np.clip(mle["xi"], -0.8, 0.8)),
            }
        except Exception:
            start = None

        idata = pm.sample(
            draws=n_samples,
            chains=n_chains,
            target_accept=adapt_delta,
            initvals=start,
            progressbar=True,
            return_inferencedata=True,
        )

    posterior = idata.posterior
    return pd.DataFrame({
        "mu":    posterior["mu"].values.flatten(),
        "sigma": posterior["sigma"].values.flatten(),
        "xi":    posterior["xi"].values.flatten(),
    })


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_return_levels(data: np.ndarray | pd.Series,
                       params: dict,
                       dist: str = "gev",
                       T_range: tuple = (1.5, 1000),
                       n_bootstrap: int = 200,
                       ci: float = 0.95,
                       ax=None):
    """
    Return level plot with bootstrap confidence band.

    Parameters
    ----------
    data : array-like
        Block maxima (GEV) or full series (GPD).
    params : dict
        Output of fit_gev() or fit_gpd().
    dist : str
        'gev' or 'gpd'.
    T_range : tuple
        (min_T, max_T) for the fitted curve.
    n_bootstrap : int
        Bootstrap replicates for CI band (default 200).
    ci : float
        Confidence level (default 0.95).
    ax : matplotlib Axes or None
        Existing axes to plot on.

    Returns
    -------
    matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]

    T_grid = np.logspace(np.log10(T_range[0]), np.log10(T_range[1]), 200)
    rl_fn = return_level_gev if dist == "gev" else return_level_gpd
    rl_fit = np.array([rl_fn(params, T) for T in T_grid])

    rng_boot = np.random.default_rng(0)
    boot_curves = []

    if dist == "gev":
        # Block maxima bootstrap: resample annual maxima directly.
        for _ in range(n_bootstrap):
            try:
                boot = rng_boot.choice(arr, size=len(arr), replace=True)
                p = fit_gev(boot)
                boot_curves.append([return_level_gev(p, T) for T in T_grid])
            except Exception:
                continue
        # Empirical Gringorten positions on block maxima
        emp_sorted = np.sort(arr)
        n_emp = len(emp_sorted)
        prob_emp = (np.arange(1, n_emp + 1) - 0.44) / (n_emp + 0.12)
        T_emp = 1.0 / (1.0 - prob_emp)
        emp_label = "Empirical (block maxima)"
    else:
        # POT bootstrap: extract peaks once, then resample exceedances.
        u   = params["threshold"]
        lam = params["lambda_rate"]
        peaks = extract_pot(pd.Series(arr), threshold=u)
        exc   = peaks.values - u
        for _ in range(n_bootstrap):
            try:
                boot_exc = rng_boot.choice(exc, size=len(exc), replace=True)
                p_boot   = _fit_gpd_mle(boot_exc)
                p_full   = {"scale": p_boot["scale"], "shape": p_boot["shape"],
                            "threshold": u, "lambda_rate": lam}
                boot_curves.append([return_level_gpd(p_full, T) for T in T_grid])
            except Exception:
                continue
        # Empirical return periods for POT peaks:
        # T = 1 / (lambda × (1 − F_emp)) where F_emp uses Gringorten on peaks.
        emp_sorted = np.sort(peaks.values)
        n_emp      = len(emp_sorted)
        prob_emp   = (np.arange(1, n_emp + 1) - 0.44) / (n_emp + 0.12)
        T_emp      = 1.0 / (lam * (1.0 - prob_emp))
        emp_label  = "Empirical (POT peaks)"

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    if boot_curves:
        boot_arr = np.array(boot_curves)
        alpha = (1 - ci) / 2
        lower = np.quantile(boot_arr, alpha, axis=0)
        upper = np.quantile(boot_arr, 1 - alpha, axis=0)
        ax.fill_between(T_grid, lower, upper, alpha=0.2,
                        color="steelblue", label=f"{int(ci*100)}% CI")

    ax.semilogx(T_grid, rl_fit, "b-", lw=2, label="Fitted")
    ax.semilogx(T_emp, emp_sorted, "ko", ms=4, zorder=5, label=emp_label)

    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Return level")
    ax.set_title(f"Return level plot — {dist.upper()}", fontsize=11)
    ax.legend()
    ax.grid(which="both", alpha=0.3)
    return ax


def plot_diagnostic(data: np.ndarray | pd.Series, params: dict,
                    dist: str = "gev"):
    """
    2×2 diagnostic panel: return levels, density, Q-Q, P-P.

    Parameters
    ----------
    data : array-like
        Block maxima.
    params : dict
        Output of fit_gev() or fit_gpd().
    dist : str
        'gev' or 'gpd'.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import genextreme, genpareto

    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]

    if dist == "gev":
        rv = genextreme(-params["xi"], loc=params["mu"], scale=params["sigma"])
        label = "GEV"
    else:
        rv = genpareto(params["shape"], loc=params["threshold"], scale=params["scale"])
        label = "GPD"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Return level plot
    plot_return_levels(arr, params, dist=dist, n_bootstrap=100, ax=axes[0, 0])

    # Density
    x_pdf = np.linspace(arr.min(), arr.max() * 1.1, 200)
    axes[0, 1].hist(arr, bins="auto", density=True, alpha=0.5,
                    color="steelblue", label="Observed")
    axes[0, 1].plot(x_pdf, rv.pdf(x_pdf), "r-", lw=2, label=f"Fitted {label}")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Density fit", fontsize=11)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Q-Q plot
    emp_q = np.sort(arr)
    probs = (np.arange(1, len(arr) + 1) - 0.5) / len(arr)
    theo_q = rv.ppf(probs)
    axes[1, 0].scatter(theo_q, emp_q, s=20, alpha=0.7, c="steelblue")
    lims = [min(theo_q.min(), emp_q.min()), max(theo_q.max(), emp_q.max())]
    axes[1, 0].plot(lims, lims, "k--", lw=1)
    axes[1, 0].set_xlabel(f"Theoretical quantiles ({label})")
    axes[1, 0].set_ylabel("Empirical quantiles")
    axes[1, 0].set_title("Q-Q plot", fontsize=11)
    axes[1, 0].grid(alpha=0.3)

    # P-P plot
    emp_cdf = np.arange(1, len(arr) + 1) / (len(arr) + 1)
    theo_cdf = rv.cdf(emp_q)
    axes[1, 1].scatter(theo_cdf, emp_cdf, s=20, alpha=0.7, c="steelblue")
    axes[1, 1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1, 1].set_xlabel(f"Theoretical CDF ({label})")
    axes[1, 1].set_ylabel("Empirical CDF")
    axes[1, 1].set_title("P-P plot", fontsize=11)
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"Extreme value diagnostics — {label}", fontsize=13)
    plt.tight_layout()
    return fig

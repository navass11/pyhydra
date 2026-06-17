"""
Hierarchical Bayesian GEV model for regional frequency analysis (PyMC).

Shares GEV parameters across stations through a population distribution,
so stations with short records borrow strength from the regional signal.
The model is estimated by MCMC via PyMC + NUTS.

Model structure (non-centered parameterization)
------------------------------------------------
Population hyperpriors:
    mu_pop    ~ Normal(y_mean, y_sd)
    sigma_pop ~ LogNormal(log(y_sd), 1)
    xi_pop    ~ TruncatedNormal(0, 0.5)  in (-1, 1)

Scale hyperpriors:
    tau_mu    ~ Exponential(1 / y_sd)   # mean = y_sd
    tau_sigma ~ Exponential(2 / y_sd)   # mean = y_sd / 2
    tau_xi    ~ Exponential(4)           # dimensionless

Station-level parameters (non-centered):
    mu_station[s]    = mu_pop + tau_mu * mu_raw[s]
    sigma_station[s] = sigma_pop * exp(tau_sigma * sigma_raw[s])
    xi_station[s]    = xi_pop + tau_xi * xi_raw[s]

    y[n, s] ~ GEV(mu_station[s], sigma_station[s], xi_station[s])
              (NaN entries masked; explicit GEV log-likelihood for stability)

Return levels are computed analytically from the posterior samples.

Dependency: ``pip install pymc``
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class HierarchicalGEV:
    """
    Hierarchical Bayesian GEV model estimated by MCMC (PyMC + NUTS).

    Uses a non-centered parameterization for better MCMC mixing when station
    records are short or heterogeneous. Missing values (NaN) in the annual-maxima
    matrix are masked in the likelihood.

    Parameters
    ----------
    T_values : list of float
        Return periods (years) for return-level estimation (default: 2, 10, 50, 100).
    n_chains : int
        Number of MCMC chains (default 4).
    n_samples : int
        Posterior samples per chain after tuning (default 1000).
    warmup : int
        Tuning samples per chain (default 1000).
    adapt_delta : float
        Target acceptance rate for NUTS (default 0.99).

    Examples
    --------
    >>> model = HierarchicalGEV()
    >>> model.fit({'StationA': am_a, 'StationB': am_b})
    >>> summary = model.posterior_summary()
    >>> rl = model.return_levels()
    """

    def __init__(
        self,
        T_values=(2, 10, 50, 100),
        n_chains=4,
        n_samples=1000,
        warmup=1000,
        adapt_delta=0.99,
    ):
        self.T_values    = list(T_values)
        self.n_chains    = n_chains
        self.n_samples   = n_samples
        self.warmup      = warmup
        self.adapt_delta = adapt_delta
        self._idata      = None
        self._stations   = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_matrix(data_dict, stations):
        """Pad ragged station arrays into a (N_years, N_stations) matrix with NaN."""
        max_len = max(len(v) for v in data_dict.values())
        mat = np.full((max_len, len(stations)), np.nan)
        for s, name in enumerate(stations):
            vals = np.asarray(data_dict[name], dtype=float)
            mat[: len(vals), s] = vals
        return mat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Stan model code (used by pystan fallback)
    # ------------------------------------------------------------------
    _STAN_MODEL = """
    data {
        int<lower=1> S;
        int<lower=1> N_max;
        array[S] int<lower=1> n_s;
        array[N_max, S] real y;
        real y_mean;
        real<lower=0> y_sd;
    }
    parameters {
        real mu_pop;
        real<lower=0> sigma_pop;
        real<lower=-1, upper=1> xi_pop;
        real<lower=0> tau_mu;
        real<lower=0> tau_sigma;
        real<lower=0> tau_xi;
        vector[S] mu_raw;
        vector[S] sigma_raw;
        vector[S] xi_raw;
    }
    transformed parameters {
        vector[S] mu_station;
        vector<lower=0>[S] sigma_station;
        vector[S] xi_station;
        mu_station = mu_pop + tau_mu * mu_raw;
        sigma_station = sigma_pop * exp(tau_sigma * sigma_raw);
        xi_station = xi_pop + tau_xi * xi_raw;
    }
    model {
        mu_pop    ~ normal(y_mean, y_sd);
        sigma_pop ~ lognormal(log(y_sd), 1.0);
        xi_pop    ~ normal(0, 0.5);
        tau_mu    ~ exponential(1.0 / y_sd);
        tau_sigma ~ exponential(2.0 / y_sd);
        tau_xi    ~ exponential(4.0);
        mu_raw    ~ std_normal();
        sigma_raw ~ std_normal();
        xi_raw    ~ std_normal();
        for (s in 1:S) {
            for (n in 1:n_s[s]) {
                real z = (y[n, s] - mu_station[s]) / sigma_station[s];
                if (fabs(xi_station[s]) < 1e-6) {
                    target += -log(sigma_station[s]) - z - exp(-z);
                } else {
                    real t = fmax(1.0 + xi_station[s] * z, 1e-10);
                    target += -log(sigma_station[s])
                              - (1.0 + 1.0/xi_station[s]) * log(t)
                              - pow(t, -1.0/xi_station[s]);
                }
            }
        }
    }
    """

    # ------------------------------------------------------------------

    def fit(self, data_dict):
        """
        Fit the hierarchical model to annual maxima from multiple stations.

        Uses PyMC (preferred) if installed; falls back to PyStan 3 automatically.

        Parameters
        ----------
        data_dict : dict
            Maps station name → 1-D array-like of annual maxima.
            Arrays may have different lengths; missing years should be NaN.

        Returns
        -------
        self (for chaining).
        """
        _has_pymc = False
        try:
            import pymc as pm          # noqa: F401
            import pytensor.tensor as pt  # noqa: F401
            _has_pymc = True
        except ImportError:
            pass

        if _has_pymc:
            return self._fit_pymc(data_dict)

        try:
            import stan  # noqa: F401
            return self._fit_pystan(data_dict)
        except ImportError:
            pass

        raise ImportError(
            "Neither pymc nor stan (pystan 3) is installed.\n"
            "Install one of them:\n"
            "  pip install pymc          # recommended\n"
            "  pip install pystan        # alternative"
        )

    def _fit_pymc(self, data_dict):
        """PyMC backend."""
        import pymc as pm
        import pytensor.tensor as pt

        self._stations = list(data_dict.keys())
        S = len(self._stations)

        y_mat  = self._build_matrix(data_dict, self._stations)
        valid  = np.isfinite(y_mat).astype(float)   # (N_years, S) — 1=observed, 0=missing
        y_obs  = np.where(np.isfinite(y_mat), y_mat, 0.0)  # NaN replaced with 0

        all_vals = y_mat[np.isfinite(y_mat)]
        y_mean   = float(np.mean(all_vals))
        y_sd     = float(np.std(all_vals))

        with pm.Model():
            # ── Population hyperpriors ────────────────────────────────────────
            mu_pop    = pm.Normal("mu_pop",    mu=y_mean,        sigma=y_sd)
            sigma_pop = pm.LogNormal("sigma_pop", mu=np.log(y_sd), sigma=1.0)
            xi_pop    = pm.TruncatedNormal("xi_pop", mu=0.0, sigma=0.5,
                                            lower=-1.0, upper=1.0)

            # ── Scale hyperpriors ─────────────────────────────────────────────
            tau_mu    = pm.Exponential("tau_mu",    lam=1.0 / y_sd)
            tau_sigma = pm.Exponential("tau_sigma", lam=2.0 / y_sd)
            tau_xi    = pm.Exponential("tau_xi",    lam=4.0)

            # ── Station-level parameters (non-centered) ───────────────────────
            mu_raw    = pm.Normal("mu_raw",    mu=0.0, sigma=1.0, shape=S)
            sigma_raw = pm.Normal("sigma_raw", mu=0.0, sigma=1.0, shape=S)
            xi_raw    = pm.Normal("xi_raw",    mu=0.0, sigma=1.0, shape=S)

            mu_s    = pm.Deterministic("mu_station",
                          mu_pop + tau_mu * mu_raw)
            sigma_s = pm.Deterministic("sigma_station",
                          sigma_pop * pt.exp(tau_sigma * sigma_raw))
            xi_s    = pm.Deterministic("xi_station",
                          xi_pop + tau_xi * xi_raw)

            # ── GEV log-likelihood (vectorized, NaN-masked) ───────────────────
            # Standardised anomaly z: shape (N_years, S)
            z = (y_obs - mu_s[np.newaxis, :]) / sigma_s[np.newaxis, :]

            # Avoid xi=0 singularity in GEV branch: safe substitute used only
            # when |xi| <= 1e-6; the Gumbel formula handles that case instead.
            xi_safe = pt.where(pt.abs(xi_s) > 1e-6, xi_s,
                               pt.ones_like(xi_s) * 1e-6)

            # GEV transformation t = 1 + xi * z; clipped for numerical safety
            t_raw = 1.0 + xi_safe[np.newaxis, :] * z
            t     = pt.clip(t_raw, 1e-10, 1e30)

            gev_lp = (
                -pt.log(sigma_s[np.newaxis, :])
                - (1.0 + 1.0 / xi_safe[np.newaxis, :]) * pt.log(t)
                - t ** (-1.0 / xi_safe[np.newaxis, :])
            )
            gumbel_lp = (
                -pt.log(sigma_s[np.newaxis, :])
                - z
                - pt.exp(-z)
            )

            lp_per_obs = pt.where(
                pt.abs(xi_s[np.newaxis, :]) > 1e-6,
                gev_lp,
                gumbel_lp,
            )

            # Sum only over observed (non-NaN) entries
            pm.Potential("gev_lik", pt.sum(lp_per_obs * valid))

            # ── Warm-start from pooled observed statistics ────────────────────
            start = {
                "mu_pop":    y_mean,
                "sigma_pop": max(y_sd, 1.0),
                "xi_pop":    0.1,
                "mu_raw":    np.zeros(S),
                "sigma_raw": np.zeros(S),
                "xi_raw":    np.zeros(S),
                "tau_mu":    y_sd,
                "tau_sigma": y_sd / 2.0,
                "tau_xi":    0.25,
            }

            self._idata = pm.sample(
                draws=self.n_samples,
                chains=self.n_chains,
                tune=self.warmup,
                target_accept=self.adapt_delta,
                initvals=start,
                progressbar=True,
                return_inferencedata=True,
            )

        return self

    def _fit_pystan(self, data_dict):
        """PyStan 3 fallback backend."""
        import io
        import logging
        import os
        import sys
        import stan

        self._stations = list(data_dict.keys())
        S = len(self._stations)

        y_mat   = self._build_matrix(data_dict, self._stations)
        n_s     = [int(np.sum(np.isfinite(y_mat[:, s]))) for s in range(S)]
        N_max   = int(y_mat.shape[0])
        y_obs   = np.where(np.isfinite(y_mat), y_mat, 0.0)

        all_vals = y_mat[np.isfinite(y_mat)]
        y_mean   = float(np.mean(all_vals))
        y_sd     = float(np.std(all_vals))

        stan_data = {
            "S": S, "N_max": N_max, "n_s": n_s,
            "y": y_obs.tolist(),
            "y_mean": y_mean, "y_sd": max(y_sd, 1.0),
        }

        # pystan 3 uses asyncio internally — apply nest_asyncio so it works
        # inside a Jupyter kernel (which already has a running event loop).
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass  # outside Jupyter, no conflict

        # Suppress Stan/httpstan C++ compilation warnings (cosmetic only)
        _old_stderr = sys.stderr
        _old_level  = logging.root.level
        logging.root.setLevel(logging.CRITICAL)
        sys.stderr = io.StringIO()
        try:
            print("Building Stan model (compiles once, cached afterwards)...")
            posterior = stan.build(self._STAN_MODEL, data=stan_data,
                                   random_seed=42)
        finally:
            sys.stderr = _old_stderr
            logging.root.setLevel(_old_level)

        print(f"Sampling {self.n_chains} chains × {self.n_samples} draws...")
        fit = posterior.sample(
            num_chains=self.n_chains,
            num_samples=self.n_samples,
            num_warmup=self.warmup,
        )

        # Convert to arviz InferenceData so posterior_summary / return_levels work
        import arviz as az

        scalars = ["mu_pop", "sigma_pop", "xi_pop",
                   "tau_mu", "tau_sigma", "tau_xi"]
        vectors = ["mu_station", "sigma_station", "xi_station"]

        posterior_dict = {}
        for p in scalars:
            arr = np.array(fit[p])         # (chains * draws,)
            posterior_dict[p] = arr.reshape(self.n_chains, self.n_samples)
        for p in vectors:
            arr = np.array(fit[p])         # (chains * draws, S)
            posterior_dict[p] = arr.reshape(self.n_chains, self.n_samples, S)

        self._idata = az.from_dict(posterior=posterior_dict)
        return self

    def posterior_summary(self):
        """
        Summarise the posterior of population, tau, and station-level parameters.

        Returns
        -------
        pd.DataFrame with mean, std, and 95 % credible intervals.
        """
        if self._idata is None:
            raise RuntimeError("Call fit() first.")

        posterior = self._idata.posterior
        rows = {}

        for param in ["mu_pop", "sigma_pop", "xi_pop",
                      "tau_mu", "tau_sigma", "tau_xi"]:
            samples = posterior[param].values.flatten()
            rows[param] = {
                "mean":  float(np.mean(samples)),
                "std":   float(np.std(samples)),
                "q2.5":  float(np.quantile(samples, 0.025)),
                "q97.5": float(np.quantile(samples, 0.975)),
            }

        for s, name in enumerate(self._stations):
            for base in ["mu_station", "sigma_station", "xi_station"]:
                samples = posterior[base].values[..., s].flatten()
                rows[f"{base}[{name}]"] = {
                    "mean":  float(np.mean(samples)),
                    "std":   float(np.std(samples)),
                    "q2.5":  float(np.quantile(samples, 0.025)),
                    "q97.5": float(np.quantile(samples, 0.975)),
                }

        return pd.DataFrame(rows).T

    def return_levels(self, credible=0.95):
        """
        Return-level posterior summaries for each station and return period.

        Returns
        -------
        pd.DataFrame indexed by station with columns T{T}_median, T{T}_lower,
        T{T}_upper for each configured return period T.
        """
        if self._idata is None:
            raise RuntimeError("Call fit() first.")

        alpha    = (1 - credible) / 2
        posterior = self._idata.posterior

        mu_post    = posterior["mu_station"].values    # (chains, draws, S)
        sigma_post = posterior["sigma_station"].values
        xi_post    = posterior["xi_station"].values

        records = {}
        for s, name in enumerate(self._stations):
            mu_s    = mu_post[..., s].flatten()
            sigma_s = sigma_post[..., s].flatten()
            xi_s    = xi_post[..., s].flatten()
            row = {}
            for T in self.T_values:
                p  = 1.0 - 1.0 / T
                nl = -np.log(p)                       # reduced variate
                # GEV quantile: mu + sigma/xi * (y^(-xi) - 1),  y = -log(p)
                gev_rl    = mu_s + (sigma_s / xi_s) * (nl ** (-xi_s) - 1.0)
                # Gumbel quantile: mu - sigma * log(y)
                gumbel_rl = mu_s - sigma_s * np.log(nl)
                rl = np.where(np.abs(xi_s) > 1e-6, gev_rl, gumbel_rl)
                rl = rl[np.isfinite(rl)]
                row[f"T{int(T)}_median"] = float(np.median(rl))
                row[f"T{int(T)}_lower"]  = float(np.quantile(rl, alpha))
                row[f"T{int(T)}_upper"]  = float(np.quantile(rl, 1 - alpha))
            records[name] = row

        return pd.DataFrame(records).T

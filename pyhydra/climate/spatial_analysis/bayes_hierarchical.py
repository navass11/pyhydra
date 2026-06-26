"""
Hierarchical Bayesian GEV model for regional frequency analysis (PyMC).

Shares GEV parameters across stations through a population distribution,
so stations with short records borrow strength from the regional signal.
The model is estimated by MCMC via PyMC + NUTS.

Model structure (centered parameterization in log-space)
---------------------------------------------------------
Population hyperpriors:
    mu_pop          ~ Normal(y_mean, y_sd)
    log_sigma_pop   ~ Normal(log(y_sd), 1)     → sigma_pop = exp(log_sigma_pop)
    xi_pop          ~ TruncatedNormal(0, 0.5)  in (-1, 1)

Scale hyperpriors (log-normal, sigma=0.75 to avoid Neal's funnel near 0):
    log_tau_mu    ~ Normal(log(y_sd*0.1), 0.75) → tau_mu    = exp(log_tau_mu)
    log_tau_sigma ~ Normal(log(0.1), 0.75)       → tau_sigma = exp(log_tau_sigma)
    log_tau_xi    ~ Normal(log(0.05), 0.75)      → tau_xi    = exp(log_tau_xi)

Station-level parameters (centered):
    mu_station[s]        ~ Normal(mu_pop, tau_mu)
    log_sigma_station[s] ~ Normal(log_sigma_pop, tau_sigma)
    xi_station[s]        ~ Normal(xi_pop, tau_xi)
    sigma_station[s]      = exp(log_sigma_station[s])

    y[n, s] ~ GEV(mu_station[s], sigma_station[s], xi_station[s])
              (NaN entries masked; explicit GEV log-likelihood for stability)

The centered parameterization avoids Neal's funnel geometry that the non-centered
form suffers when tau → 0 (strong pooling), which is typical in regional analysis
where stations share similar parameter values.

Return levels are computed analytically from the posterior samples.

Dependency: ``pip install pymc``
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class HierarchicalGEV:
    """
    Hierarchical Bayesian GEV model estimated by MCMC (PyMC + NUTS).

    Uses a centered parameterization in log-space for better MCMC mixing in the
    typical regional-analysis regime where stations share similar parameter values
    (strong pooling, small tau). Missing values (NaN) in the annual-maxima matrix
    are masked in the likelihood.

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
        tau_mu    ~ lognormal(log(y_sd * 0.1), 0.75);
        tau_sigma ~ lognormal(log(0.1), 0.75);
        tau_xi    ~ lognormal(log(0.05), 0.75);
        mu_raw    ~ std_normal();
        sigma_raw ~ std_normal();
        xi_raw    ~ std_normal();
        for (s in 1:S) {
            for (n in 1:n_s[s]) {
                real z = (y[n, s] - mu_station[s]) / sigma_station[s];
                if (abs(xi_station[s]) < 1e-6) {
                    target += -log(sigma_station[s]) - z - exp(-z);
                } else {
                    real t = fmax(1.0 + xi_station[s] * z, 1e-10);  // fmax ok in Stan
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
        """PyMC backend — centered parameterization in log-space."""
        import pymc as pm
        import pytensor.tensor as pt

        self._stations = list(data_dict.keys())
        S = len(self._stations)

        y_mat  = self._build_matrix(data_dict, self._stations)
        valid  = np.isfinite(y_mat).astype(float)   # (N_years, S) — 1=observed, 0=missing
        y_obs  = np.where(np.isfinite(y_mat), y_mat, 0.0)  # NaN replaced with 0

        all_vals  = y_mat[np.isfinite(y_mat)]
        y_mean    = float(np.mean(all_vals))
        y_sd      = float(np.std(all_vals))
        log_y_sd  = float(np.log(max(y_sd, 1.0)))

        # Per-station statistics for warm start
        st_means  = np.array([float(np.nanmean(data_dict[n])) for n in self._stations])
        st_sds    = np.array([float(np.nanstd(data_dict[n]))  for n in self._stations])

        with pm.Model():
            # ── Population hyperpriors ────────────────────────────────────────
            mu_pop        = pm.Normal("mu_pop", mu=y_mean, sigma=y_sd)
            log_sigma_pop = pm.Normal("log_sigma_pop", mu=log_y_sd, sigma=1.0)
            sigma_pop     = pm.Deterministic("sigma_pop", pt.exp(log_sigma_pop))
            xi_pop        = pm.TruncatedNormal("xi_pop", mu=0.0, sigma=0.5,
                                               lower=-1.0, upper=1.0)

            # ── Scale hyperpriors (log-normal, tight sigma=0.75 to avoid funnel) ─
            # Prior medians are physically motivated: tau_mu ≈ 10 % of y_sd;
            # tau_sigma ≈ 0.1 log-scale; tau_xi ≈ 0.05 (xi varies little in floods)
            log_tau_mu    = pm.Normal("log_tau_mu",
                                      mu=float(np.log(max(y_sd * 0.1, 1.0))), sigma=0.75)
            log_tau_sigma = pm.Normal("log_tau_sigma", mu=float(np.log(0.1)), sigma=0.75)
            log_tau_xi    = pm.Normal("log_tau_xi",    mu=float(np.log(0.05)), sigma=0.75)
            tau_mu    = pm.Deterministic("tau_mu",    pt.exp(log_tau_mu))
            tau_sigma = pm.Deterministic("tau_sigma", pt.exp(log_tau_sigma))
            tau_xi    = pm.Deterministic("tau_xi",    pt.exp(log_tau_xi))

            # ── Station-level parameters (non-centered) ───────────────────────
            mu_raw        = pm.Normal("mu_raw", mu=0.0, sigma=1.0, shape=S)
            log_sigma_raw = pm.Normal("log_sigma_raw", mu=0.0, sigma=1.0, shape=S)
            xi_raw        = pm.Normal("xi_raw", mu=0.0, sigma=1.0, shape=S)

            mu_s        = pm.Deterministic("mu_station", mu_pop + tau_mu * mu_raw)
            log_sigma_s = pm.Deterministic("log_sigma_station", log_sigma_pop + tau_sigma * log_sigma_raw)
            sigma_s     = pm.Deterministic("sigma_station", pt.exp(log_sigma_s))
            xi_s        = pm.Deterministic("xi_station", xi_pop + tau_xi * xi_raw)

            # ── GEV log-likelihood (vectorized, NaN-masked) ───────────────────
            z = (y_obs - mu_s[np.newaxis, :]) / sigma_s[np.newaxis, :]

            xi_safe = pt.where(pt.abs(xi_s) > 1e-6, xi_s,
                               pt.ones_like(xi_s) * 1e-6)
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
            pm.Potential("gev_lik", pt.sum(lp_per_obs * valid))

            # ── Warm-start from per-station observed statistics ───────────────
            tau_mu_init = max(y_sd * 0.1, 1.0)
            start = {
                "mu_pop":              y_mean,
                "log_sigma_pop":       log_y_sd,
                "xi_pop":              0.1,
                "log_tau_mu":          float(np.log(tau_mu_init)),
                "log_tau_sigma":       float(np.log(0.1)),
                "log_tau_xi":          float(np.log(0.05)),
                "mu_raw":              (st_means - y_mean) / tau_mu_init,
                "log_sigma_raw":       (np.log(np.maximum(st_sds, 1.0)) - log_y_sd) / 0.1,
                "xi_raw":              np.zeros(S),
            }

            self._idata = pm.sample(
                draws=self.n_samples,
                chains=self.n_chains,
                tune=self.warmup,
                target_accept=self.adapt_delta,
                initvals=start,
                init="adapt_diag",  # skip random jitter — GEV log-lik rejects bad starts
                progressbar=True,
                return_inferencedata=True,
            )

        return self

    def _fit_pystan(self, data_dict):
        """PyStan 3 fallback backend."""
        import logging
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

        # Suppress only the logger noise; leave sys.stderr open so httpstan
        # error messages (e.g. model compilation failures) remain readable.
        _old_level = logging.root.level
        logging.root.setLevel(logging.CRITICAL)
        try:
            print("Building Stan model (compiles once, cached afterwards)...")
            posterior = stan.build(self._STAN_MODEL, data=stan_data,
                                   random_seed=42)
        finally:
            logging.root.setLevel(_old_level)

        # Warm start: same sensible initial point for all chains.
        # Without this, Stan's default uniform init in unconstrained space can put
        # mu_pop near 0 and sigma_pop near exp(±2) ≈ 0.1–7, far from the data scale.
        init_point = {
            "mu_pop":    y_mean,
            "sigma_pop": max(y_sd, 1.0),
            "xi_pop":    0.1,
            "mu_raw":    [0.0] * S,
            "sigma_raw": [0.0] * S,
            "xi_raw":    [0.0] * S,
            "tau_mu":    y_sd,
            "tau_sigma": 0.5,
            "tau_xi":    0.25,
        }
        print(f"Sampling {self.n_chains} chains × {self.n_samples} draws...")
        fit = posterior.sample(
            num_chains=self.n_chains,
            num_samples=self.n_samples,
            num_warmup=self.warmup,
            init=[init_point] * self.n_chains,
        )

        # Convert to arviz InferenceData so posterior_summary / return_levels work
        import arviz as az

        scalars = ["mu_pop", "sigma_pop", "xi_pop",
                   "tau_mu", "tau_sigma", "tau_xi"]
        vectors = ["mu_station", "sigma_station", "xi_station"]

        # PyStan 3 returns:
        #   scalars → (1,  n_chains * n_samples)
        #   vectors → (S,  n_chains * n_samples)
        # We need (n_chains, n_samples) and (n_chains, n_samples, S) respectively.
        posterior_dict = {}
        for p in scalars:
            arr = np.array(fit[p]).reshape(-1)          # flatten to (n_chains * n_samples,)
            posterior_dict[p] = arr.reshape(self.n_chains, self.n_samples)
        for p in vectors:
            arr = np.array(fit[p])                      # (S, n_chains * n_samples)
            posterior_dict[p] = arr.T.reshape(self.n_chains, self.n_samples, S)

        self._idata = az.from_dict({"posterior": posterior_dict})
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

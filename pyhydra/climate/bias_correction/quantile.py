import numpy as np


class BiasCorrection:
    """
    Apply bias correction to climate scenario series.

    Three methods available:
    - quantile_mapping: additive empirical correction
    - quantile_deltamapping: multiplicative empirical correction
    - scaled_distribution_mapping: parametric (gamma for precipitation, normal for temperature)

    Args:
        obs: Observational data (array-like)
        mod: Model data for the historical/reference period (array-like)
        sce: Scenario data to be corrected (array-like)
    """

    def __init__(self, obs, mod, sce):
        self.obs = np.asarray(obs, dtype=float)
        self.mod = np.asarray(mod, dtype=float)
        self.sce = np.asarray(sce, dtype=float)

    def quantile_mapping(self):
        """Additive empirical quantile mapping correction."""
        from statsmodels.distributions.empirical_distribution import ECDF

        obs_data = self.obs.copy()
        mod_data = self.mod.copy()
        sce_data = self.sce.copy()

        mod_ecdf = ECDF(mod_data)
        p = mod_ecdf(sce_data).flatten() * 100
        corr = np.percentile(obs_data, p) - np.percentile(mod_data, p)
        sce_data += corr
        sce_data[sce_data < 0] = 0
        return sce_data

    def quantile_deltamapping(self):
        """Multiplicative empirical quantile mapping correction (suited for precipitation)."""
        from statsmodels.distributions.empirical_distribution import ECDF

        obs_data = self.obs.copy()
        mod_data = self.mod.copy()
        sce_data = self.sce.copy()

        mod_ecdf = ECDF(mod_data)
        p = mod_ecdf(sce_data).flatten() * 100
        corr = np.percentile(obs_data, p) / np.percentile(mod_data, p)
        sce_data *= corr
        sce_data[sce_data < 0] = 0
        return sce_data

    def scaled_distribution_mapping(self, variable, **kwargs):
        """
        Parametric Scaled Distribution Mapping (SDM).

        Args:
            variable: 'precipitation' (gamma) or 'temperature' (normal)
            **kwargs: lower_limit (default 0.1), cdf_threshold
        """
        if variable == "precipitation":
            return self._relative_sdm(**kwargs)
        elif variable == "temperature":
            return self._absolute_sdm(**kwargs)
        else:
            raise NotImplementedError(f"SDM not implemented for variable '{variable}'.")

    def _relative_sdm(self, lower_limit=0.1, **kwargs):
        """
        Relative (multiplicative) SDM for precipitation.
        Maps scenario wet-day amounts through gamma distributions fitted on
        obs and mod historical wet days, then adjusts wet-day frequency.
        """
        from scipy.stats import gamma

        obs_wet = self.obs[self.obs >= lower_limit]
        mod_wet = self.mod[self.mod >= lower_limit]
        sce_wet_mask = self.sce >= lower_limit
        sce_wet = self.sce[sce_wet_mask]

        if len(obs_wet) < 5 or len(mod_wet) < 5 or len(sce_wet) < 5:
            raise ValueError("Insufficient wet days for gamma SDM fit (need ≥5 each).")

        obs_ga = gamma.fit(obs_wet, floc=0)
        mod_ga = gamma.fit(mod_wet, floc=0)
        sce_ga = gamma.fit(sce_wet, floc=0)

        # Wet-day frequency adjustment: scale sce freq by obs/mod ratio
        obs_wf = float((self.obs >= lower_limit).mean())
        mod_wf = float((self.mod >= lower_limit).mean())
        sce_wf = float(sce_wet_mask.mean())
        adj_wf = float(np.clip(obs_wf * sce_wf / max(mod_wf, 1e-9), 0.0, 1.0))

        # Parametric multiplicative transfer for each wet scenario day:
        #   p = F_sce(x),  x_corr = F_obs^{-1}(p) * x / F_mod^{-1}(p)
        p = np.clip(gamma.cdf(sce_wet, *sce_ga), 1e-6, 1 - 1e-6)
        obs_q = gamma.ppf(p, *obs_ga)
        mod_q = np.maximum(gamma.ppf(p, *mod_ga), 1e-9)
        corrected_wet = np.clip(obs_q * sce_wet / mod_q, 0.0, None)

        # Assign corrected wet values to the n_wet_out largest sce positions
        n_wet_out = max(1, int(np.round(len(self.sce) * adj_wf)))
        result = np.zeros(len(self.sce), dtype=float)
        wet_pos = np.argsort(self.sce)[-n_wet_out:]

        # Sort both arrays to maintain rank order
        n_assign = min(len(corrected_wet), n_wet_out)
        result[np.sort(wet_pos)[-n_assign:]] = np.sort(corrected_wet)[-n_assign:]
        return result

    def _absolute_sdm(self, **kwargs):
        """
        Absolute (additive) SDM for temperature.
        Detrends, fits normal distributions, applies variance-scaled delta
        correction, then adds back the scenario trend.
        """
        from scipy.signal import detrend
        from scipy.stats import norm

        obs_det = detrend(self.obs.copy())
        mod_det = detrend(self.mod.copy())
        sce_det = detrend(self.sce.copy())
        sce_trend = self.sce - sce_det

        obs_params = norm.fit(obs_det)
        mod_params = norm.fit(mod_det)
        sce_params = norm.fit(sce_det)

        # CDF of sorted detrended arrays (clipped to avoid ±inf in ppf)
        eps = 1e-6
        obs_cdf = np.clip(norm.cdf(np.sort(obs_det), *obs_params), eps, 1 - eps)
        mod_cdf = np.clip(norm.cdf(np.sort(mod_det), *mod_params), eps, 1 - eps)
        sce_cdf = np.clip(norm.cdf(np.sort(sce_det), *sce_params), eps, 1 - eps)

        n = len(sce_det)
        obs_cdf_r = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(obs_cdf)), obs_cdf)
        mod_cdf_r = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(mod_cdf)), mod_cdf)

        # Delta correction: corrected = obs_ppf(obs_cdf_r) + σ_ratio*(sce_ppf − mod_ppf)
        sigma_ratio = obs_params[1] / max(mod_params[1], 1e-9)
        corrected_sorted = (
            norm.ppf(obs_cdf_r, *obs_params)
            + sigma_ratio * (
                norm.ppf(sce_cdf, *sce_params)
                - norm.ppf(mod_cdf_r, *mod_params)
            )
        )

        # Map back to original time order and add scenario trend
        result = np.empty(n, dtype=float)
        result[np.argsort(sce_det)] = corrected_sorted
        return result + sce_trend


# Backward-compatible alias
bias_correction = BiasCorrection

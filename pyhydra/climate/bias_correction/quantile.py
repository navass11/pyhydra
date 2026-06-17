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

    def _relative_sdm(self, lower_limit=0.1, cdf_threshold=0.99999999, **kwargs):
        from scipy.stats import gamma

        obs_c = self.obs.copy()
        mod_c = self.mod.copy()
        sce_c = self.sce.copy()

        obs_raindays = obs_c[obs_c >= lower_limit]
        mod_raindays = mod_c[mod_c >= lower_limit]

        if len(obs_raindays) < 10 or len(mod_raindays) < 10:
            raise ValueError("Insufficient rainy days for gamma fit in obs or mod data.")

        obs_frequency = len(obs_raindays) / len(obs_c)
        mod_frequency = len(mod_raindays) / len(mod_c)

        obs_gamma = gamma.fit(obs_raindays, floc=0)
        mod_gamma = gamma.fit(mod_raindays, floc=0)

        obs_cdf = gamma.cdf(np.sort(obs_raindays), *obs_gamma)
        mod_cdf = gamma.cdf(np.sort(mod_raindays), *mod_gamma)
        obs_cdf[obs_cdf > cdf_threshold] = cdf_threshold
        mod_cdf[mod_cdf > cdf_threshold] = cdf_threshold

        sce_raindays = sce_c[sce_c >= lower_limit]
        if len(sce_raindays) < 10:
            raise ValueError("Insufficient rainy days for gamma fit in scenario data.")

        sce_argsort = np.argsort(sce_c)
        sce_gamma = gamma.fit(sce_raindays, floc=0)

        expected_sce_raindays = int(min(
            np.round(len(sce_c) * obs_frequency * len(sce_raindays) / mod_frequency),
            len(sce_c)
        ))

        sce_cdf = gamma.cdf(np.sort(sce_raindays), *sce_gamma)
        sce_cdf[sce_cdf > cdf_threshold] = cdf_threshold

        obs_cdf_resampled = np.interp(
            np.linspace(0, 1, len(sce_raindays)),
            np.linspace(0, 1, len(obs_raindays)),
            obs_cdf,
        )
        mod_cdf_resampled = np.interp(
            np.linspace(0, 1, len(sce_raindays)),
            np.linspace(0, 1, len(mod_raindays)),
            mod_cdf,
        )

        obs_inverse = 1.0 / (1 - obs_cdf_resampled)
        mod_inverse = 1.0 / (1 - mod_cdf_resampled)
        sce_inverse = 1.0 / (1 - sce_cdf)

        adapted_cdf = 1 - 1.0 / (obs_inverse * sce_inverse / mod_inverse)
        adapted_cdf[adapted_cdf < 0.0] = 0.0

        xvals = (
            gamma.ppf(np.sort(adapted_cdf), *obs_gamma)
            * gamma.ppf(np.sort(sce_cdf), *sce_gamma)
            / gamma.ppf(np.sort(sce_cdf), *mod_gamma)
        )

        correction = np.zeros(len(sce_c))
        if len(sce_raindays) > expected_sce_raindays:
            xvals = np.interp(
                np.linspace(1, len(sce_raindays), expected_sce_raindays),
                np.linspace(1, len(sce_raindays), len(sce_raindays)),
                xvals,
            )
        else:
            xvals = np.hstack((np.zeros(expected_sce_raindays - len(sce_raindays)), xvals))

        correction[sce_argsort[-expected_sce_raindays:].flatten()] = xvals
        return correction

    def _absolute_sdm(self, cdf_threshold=0.99999, **kwargs):
        from scipy.signal import detrend
        from scipy.stats import norm

        obs_c = self.obs.copy()
        mod_c = self.mod.copy()
        sce_c = self.sce.copy()

        obs_detrended = detrend(obs_c)
        mod_detrended = detrend(mod_c)
        sce_detrended = detrend(sce_c)

        obs_norm = norm.fit(obs_detrended)
        mod_norm = norm.fit(mod_detrended)

        obs_cdf = norm.cdf(np.sort(obs_detrended), *obs_norm)
        mod_cdf = norm.cdf(np.sort(mod_detrended), *mod_norm)

        obs_cdf_resampled = np.interp(
            np.linspace(0, 1, len(sce_detrended)),
            np.linspace(0, 1, len(obs_detrended)),
            obs_cdf,
        )
        mod_cdf_resampled = np.interp(
            np.linspace(0, 1, len(sce_detrended)),
            np.linspace(0, 1, len(mod_detrended)),
            mod_cdf,
        )

        correction = np.zeros(len(sce_c))
        correction[np.argsort(sce_detrended)] = norm.ppf(
            np.sort(obs_cdf_resampled), *obs_norm
        ) + obs_norm[-1] / mod_norm[-1] * (
            norm.ppf(np.sort(sce_detrended), *norm.fit(sce_detrended))
            - norm.ppf(np.sort(mod_cdf_resampled), *mod_norm)
        )

        return correction


# Backward-compatible alias
bias_correction = BiasCorrection

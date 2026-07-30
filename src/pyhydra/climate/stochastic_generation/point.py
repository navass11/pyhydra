"""
Point stochastic generation — single-site models.

Two backends depending on variable type:

General hydro-meteorological variables (discharge, temperature, wind…)
    → CoSMoS_py: analyze_ts / simulate_ts / generate_ts
      https://github.com/navass11/CoSMoS_py

Precipitation (Neyman-Scott Rectangular Pulses Model)
    → NEOPRENE: NSRPModel
      https://github.com/IHCantabria/NEOPRENE
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _require_cosmos():
    try:
        import cosmos_py
        return cosmos_py
    except ImportError as exc:
        raise ImportError(
            "cosmos_py is required for general time-series stochastic generation.\n"
            "Install it with: pip install -e /path/to/CoSMoS_py\n"
            "Source: https://github.com/navass11/CoSMoS_py"
        ) from exc


def _require_neoprene():
    try:
        import NEOPRENE
        return NEOPRENE
    except ImportError as exc:
        raise ImportError(
            "NEOPRENE is required for stochastic precipitation generation.\n"
            "Install it with: pip install NEOPRENE\n"
            "Source: https://github.com/IHCantabria/NEOPRENE"
        ) from exc


# ---------------------------------------------------------------------------
# General series — CoSMoS_py wrappers
# ---------------------------------------------------------------------------

def analyze_ts(series, season="month", dist="gengamma", acs_id="weibull",
               norm_type="N1", n_points=30, lag_max=30, opts=None):
    """
    Fit a seasonal stochastic model to a daily time series.

    Wraps cosmos_py.timeseries.analyze.analyzeTS. Fits a marginal distribution
    and an autocorrelation structure for each calendar month (or ISO week).
    Suitable for any variable: discharge, temperature, wind speed, etc.
    For precipitation use NSRPModel instead.

    Args:
        series: pd.Series with DatetimeIndex and numeric values. Zero values
                are treated as dry/zero-flow steps (p0 estimated automatically).
        season: Seasonal grouping — 'month' (default) or 'week'.
        dist: Marginal distribution for non-zero values.
              Options: 'gengamma', 'paretoII', 'burrXII', 'burrIII', 'gev', 'lnorm'.
        acs_id: Autocorrelation structure model ('weibull', 'paretoII', …).
        norm_type: Normalisation type passed to fit_dist ('N1' default).
        n_points: Number of points used in ECDF matching.
        lag_max: Maximum lag for empirical ACF estimation.
        opts: Extra options dict forwarded to fit_dist.

    Returns:
        Dict with fitted model (input to simulate_ts / report_ts).
    """
    _require_cosmos()
    from cosmos_py.timeseries.analyze import analyzeTS

    if isinstance(series, pd.Series):
        df = series.reset_index()
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
    else:
        df = series.copy()

    return analyzeTS(df, season=season, dist=dist, acs_id=acs_id,
                     norm_type=norm_type, n_points=n_points, lag_max=lag_max,
                     opts=opts or {})


def report_ts(analyzed, method="stat"):
    """
    Summarise a fitted stochastic model.

    Args:
        analyzed: Output of analyze_ts().
        method: 'stat' (DataFrame with statistics — default),
                'dist' (plots marginal fits), 'acs' (plots ACS fits).

    Returns:
        DataFrame of seasonal statistics when method='stat', else None.
    """
    _require_cosmos()
    from cosmos_py.timeseries.analyze import reportTS
    return reportTS(analyzed, method=method)


def simulate_ts(analyzed, from_date=None, to_date=None):
    """
    Generate a synthetic daily time series from a fitted stochastic model.

    Args:
        analyzed: Output of analyze_ts().
        from_date: Start date of simulation (defaults to calibration period).
        to_date: End date of simulation (defaults to calibration period).

    Returns:
        pd.DataFrame with columns ['date', 'value'].
    """
    _require_cosmos()
    from cosmos_py.timeseries.analyze import simulateTS
    return simulateTS(analyzed, from_date=from_date, to_date=to_date)


def generate_ts(n, dist, dist_params, acs_values, p0=0.0, n_series=1):
    """
    Generate synthetic series directly from distribution and ACS parameters.

    Lower-level entry point that bypasses seasonal fitting — useful when
    parameters are already known (e.g. from a bias-corrected model).

    Args:
        n: Length of each series.
        dist: Marginal distribution name (e.g. 'gengamma').
        dist_params: Dict of distribution parameters.
        acs_values: Array of target autocorrelations starting at lag 0.
        p0: Probability of zero values (default 0).
        n_series: Number of independent series to generate (default 1).

    Returns:
        List of numpy arrays, one per series.
    """
    _require_cosmos()
    from cosmos_py.timeseries.generateTS import generate_ts as _gen
    return _gen(n=n, margdist=dist, margarg=dist_params, p0=p0,
                TSn=n_series, acsvalue=np.asarray(acs_values))


def fit_distribution(values, dist="gengamma", norm_type="N1", n_points=30, opts=None):
    """
    Fit a single marginal distribution to non-zero values.

    Args:
        values: 1-D array or pd.Series of positive values.
        dist: Distribution name.
        norm_type: Normalisation type for ECDF matching.
        n_points: Number of ECDF matching points.
        opts: Extra options forwarded to fit_dist.

    Returns:
        Dict with keys: 'dist', 'params_dict', 'edf', 'objective'.
    """
    _require_cosmos()
    from cosmos_py.utils.fitDist import fit_dist
    arr = pd.Series(values).dropna()
    arr = arr[arr > 0]
    return fit_dist(arr, dist, norm_type=norm_type, n_points=n_points, opts=opts)


def fit_acs(series, acs_id="weibull", season="month", lag_max=30):
    """
    Fit an autocorrelation structure to a daily series.

    Args:
        series: pd.Series with DatetimeIndex.
        acs_id: ACS model identifier ('weibull', 'paretoII', …).
        season: Seasonal grouping ('month' or 'week').
        lag_max: Maximum lag for empirical ACF.

    Returns:
        List of ACS fit dicts, one per season.
    """
    _require_cosmos()
    from cosmos_py.timeseries.seasonal_acf import seasonal_acf
    from cosmos_py.correlation.fitACS import fit_acs as _fit

    if isinstance(series, pd.Series):
        df = series.reset_index()
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
    else:
        df = series.copy()

    ea = seasonal_acf(df, season=season, lag_max=lag_max)
    return [_fit(x, acs_id=acs_id) for x in ea.values()]


# ---------------------------------------------------------------------------
# NEOPRENE YAML helpers (shared with spatial.py)
# ---------------------------------------------------------------------------

def _write_yaml(config: dict, path: str) -> None:
    import yaml
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _default_statistics():
    # Format: mean | {var,fih,fiWW,fiDD,M3}_{h} | autocorr_{l}_{h}
    # h = aggregation level (days for daily, hours for hourly), l = lag — must be integers.
    # fiDD and fiWW are omitted from defaults: NEOPRENE divides by pdry/pwet which can be
    # zero for some PSO candidates, causing ZeroDivisionError during optimisation.
    return ["mean", "var_1", "autocorr_1_1", "fih_1"]


def _default_weights(statistics_name):
    return {stat: 1.0 for stat in statistics_name}


def _default_model_bounds():
    # Bounds tuned for daily resolution with low wet-day fraction (~0.25–0.45).
    # All values in hours; NEOPRENE converts internally with t=24.
    #
    # The official NEOPRENE example YAMLs use wider bounds (number_storm_cells up to 100,
    # cell_duration down to 0.15 h). For daily data those wider bounds let PSO converge to a
    # degenerate solution: 100 very short, very intense cells per storm with high storm
    # frequency → near-100% wet fraction regardless of fih_1 target. These tighter bounds
    # prevent that degeneracy.
    #
    # Key constraint: storm_cell_displacement_min (10 h) > cell_duration_max (8 h).
    # This guarantees storm_cell_displacement > cell_duration for every PSO candidate,
    # so log(η/(η−β)) never produces NaN in the Cowpertwait (1991) NSRP_pdry formula.
    #
    # storm_cell_displacement upper bound (48 h): cells stay within two calendar days of
    # their storm origin. Larger values cause adjacent-storm cell interleaving that breaks
    # the analytical P_dry approximation (PSO sees fih_1 ≈ 0.80 but simulation gives 0.20).
    #
    # number_storm_cells minimum = 2: the Cowpertwait (1991) P_dry approximation contains
    # the term 1/(mu_c − 1). When mu_c → 1 this diverges (0/0 indeterminate form), making
    # the approximation inaccurate and causing large formula/simulation divergence in wet
    # fraction. Forcing mu_c ≥ 2 keeps the approximation in a valid regime.
    return {
        "time_between_storms":     [12.0, 500.0],  # h
        "number_storm_cells":      [2.0,   20.0],  #   min=2 avoids Cowpertwait 1/(mu_c-1) breakdown
        "cell_duration":           [0.5,    8.0],  # h  — max < storm_cell_displacement_min
        "cell_intensity":          [0.1,  500.0],  #    — needs ≥150 for ~3 mm/day mean
        "storm_cell_displacement": [10.0,  48.0],  # h  — min > cell_duration_max; max ≤ 48 h
    }


def _build_calibration_yaml(temporal_resolution, seasonality, process,
                             statistics_name, weights, n_iterations, n_bees,
                             n_initializations, model_bounds, seasonality_user=None):
    cfg = {
        "temporal_resolution": temporal_resolution,
        "Seasonality_type": seasonality,
        "Seasonality_user": seasonality_user,  # always required by NEOPRENE HiperParams; null when not user_defined
        "process": process,
        "statistics_name": statistics_name,
        # NEOPRENE indexes weights by position; convert dict → ordered list
        "weights": [weights[s] for s in statistics_name] if isinstance(weights, dict) else weights,
        "number_iterations": n_iterations,
        "number_bees": n_bees,
        "number_initializations": n_initializations,
    }
    cfg.update(model_bounds)
    return cfg


def _build_simulation_yaml(temporal_resolution, seasonality, process,
                            statistics_name, year_ini, year_fin,
                            seasonality_user=None):
    return {
        "temporal_resolution": temporal_resolution,
        "Seasonality_type": seasonality,
        "Seasonality_user": seasonality_user,  # always required by NEOPRENE HiperParams
        "process": process,
        "statistics_name": statistics_name,
        "year_ini": int(year_ini),
        "year_fin": int(year_fin),
    }


def _series_to_neoprene_df(series):
    # NEOPRENE Statistics.statistics_from_serie expects index.month to work,
    # so we must keep a DatetimeIndex — not reset_index().
    return pd.DataFrame({"Rain": series.values}, index=pd.to_datetime(series.index))


# ---------------------------------------------------------------------------
# Precipitation — NEOPRENE NSRP single-site model
# ---------------------------------------------------------------------------

class NSRPModel:
    """
    Single-site stochastic rainfall generator (Neyman-Scott Rectangular Pulses).

    Typical usage::

        model = NSRPModel(temporal_resolution='d', seasonality='monthly')
        model.fit(daily_rainfall_series)
        synthetic = model.simulate(year_ini=2000, year_fin=2050)

    Args:
        temporal_resolution: Time step — 'd' (daily) or 'h' (hourly).
        seasonality: Seasonal grouping — 'annual', 'seasonal', 'monthly'
                     (default), or 'user_defined'.
        process: NSRP variant — 'normal' (default) or 'storms'.
        statistics: List of statistics to match during calibration.
        weights: Dict mapping statistic → optimisation weight (default 1.0).
        n_iterations: PSO iterations (default 100).
        n_bees: PSO swarm size (default 100). Official NEOPRENE examples use 1000
                for production-quality calibration; 100 is a faster starting point.
        n_initializations: Independent PSO runs (default 1).
        model_bounds: NSRP parameter search bounds dict.
        seasonality_user: Season labels when seasonality='user_defined'.
        calibration_yaml: Path to existing NEOPRENE calibration YAML
                          (bypasses all parameter arguments).
    """

    def __init__(self, temporal_resolution="d", seasonality="monthly",
                 process="normal", statistics=None, weights=None,
                 n_iterations=100, n_bees=100, n_initializations=1,
                 model_bounds=None, seasonality_user=None, calibration_yaml=None):
        self.temporal_resolution = temporal_resolution
        self.seasonality = seasonality
        self.process = process
        self.statistics_name = statistics or _default_statistics()
        self.weights = weights or _default_weights(self.statistics_name)
        self.n_iterations = n_iterations
        self.n_bees = n_bees
        self.n_initializations = n_initializations
        self.model_bounds = model_bounds or _default_model_bounds()
        self.seasonality_user = seasonality_user
        self._calibration_yaml = calibration_yaml

        self._cal_hiperparams = None
        self._statistics = None
        self._calibration_result = None

    def _get_cal_hiperparams(self):
        _require_neoprene()
        from NEOPRENE.NSRP.HiperParams import Calibration as HpCal
        if self._calibration_yaml:
            return HpCal(self._calibration_yaml)
        cfg = _build_calibration_yaml(
            self.temporal_resolution, self.seasonality, self.process,
            self.statistics_name, self.weights, self.n_iterations,
            self.n_bees, self.n_initializations, self.model_bounds,
            self.seasonality_user,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            tmp = f.name
        _write_yaml(cfg, tmp)
        try:
            return HpCal(tmp)
        finally:
            os.unlink(tmp)

    def compute_statistics(self, series):
        """Compute observed statistics from a rainfall series (pd.Series)."""
        _require_neoprene()
        from NEOPRENE.NSRP.Statistics import Statistics
        hp = self._get_cal_hiperparams()
        self._cal_hiperparams = hp
        self._statistics = Statistics(hp, time_series=_series_to_neoprene_df(series))
        return self._statistics

    def calibrate(self, series_or_stats=None, verbose=False):
        """
        Calibrate NSRPM via PSO.

        Args:
            series_or_stats: pd.Series (statistics computed automatically) or
                             NEOPRENE Statistics object from compute_statistics().
            verbose: Print PSO progress.

        Returns:
            NEOPRENE Calibration result (``Fitted_parameters``,
            ``statististics_Fit``, ``statististics_Real``).
        """
        if series_or_stats is not None and isinstance(series_or_stats, pd.Series):
            self.compute_statistics(series_or_stats)
        elif series_or_stats is not None:
            self._statistics = series_or_stats

        if self._statistics is None:
            raise RuntimeError("Call compute_statistics() or pass a series before calibrate().")

        _require_neoprene()
        from NEOPRENE.NSRP.Calibration import Calibration
        if self._cal_hiperparams is None:
            self._cal_hiperparams = self._get_cal_hiperparams()
        cal = Calibration(self._cal_hiperparams)
        self._calibration_result = cal(self._statistics, verbose=verbose)
        return self._calibration_result

    def fit(self, series, verbose=False):
        """Compute statistics and calibrate in one step."""
        self.compute_statistics(series)
        return self.calibrate(verbose=verbose)

    def simulate(self, year_ini, year_fin, simulation_yaml=None):
        """
        Generate synthetic rainfall from calibrated parameters.

        Args:
            year_ini: First year of the simulation.
            year_fin: Last year of the simulation.
            simulation_yaml: Optional path to NEOPRENE simulation YAML.

        Returns:
            NEOPRENE Simulation result (``Daily_Simulation``,
            ``Hourly_Simulation``, ``statististics_Simulated``).
        """
        if self._calibration_result is None:
            raise RuntimeError("Call fit() or calibrate() before simulate().")

        _require_neoprene()
        from NEOPRENE.NSRP.HiperParams import Simulation as HpSim
        from NEOPRENE.NSRP.Simulation import Simulation

        if simulation_yaml:
            hp_sim = HpSim(simulation_yaml)
        else:
            cfg = _build_simulation_yaml(
                self.temporal_resolution, self.seasonality, self.process,
                self.statistics_name, year_ini, year_fin, self.seasonality_user,
            )
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
                tmp = f.name
            _write_yaml(cfg, tmp)
            try:
                hp_sim = HpSim(tmp)
            finally:
                os.unlink(tmp)

        return Simulation(hp_sim)(self._calibration_result)

    def summary(self):
        """Return DataFrame comparing observed vs fitted statistics, or None."""
        if self._calibration_result is None:
            return None
        obs = getattr(self._calibration_result, "statististics_Real", None)
        fit = getattr(self._calibration_result, "statististics_Fit", None)
        if obs is None or fit is None:
            return None
        # Both are DataFrames with one column per season; take the first (only) column.
        return pd.concat([obs.iloc[:, 0].rename("observed"),
                          fit.iloc[:, 0].rename("fitted")], axis=1)

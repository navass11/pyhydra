"""
Spatial interpolation of hydro-meteorological fields.

Four methods — choose depending on data density, variogram structure, and
available covariates:

- **IDWInterpolator**            — Inverse Distance Weighting (no extra deps).
- **KrigingInterpolator**        — Universal Kriging via pykrige.
- **GaussianProcessInterpolator**— GP regression with covariates (scikit-learn).
- **NeopreneGPReconstructor**    — Daily precipitation reconstruction: NEOPRENE
                                   calibration then GP spatial interpolation per
                                   time step.
- **CopulaCDFEmulator**          — GP surrogate for a multivariate copula CDF.

Optional dependencies
---------------------
pykrige   : pip install pykrige           (KrigingInterpolator)
sklearn   : pip install scikit-learn      (GP-based classes)
NEOPRENE  : pip install NEOPRENE          (NeopreneGPReconstructor)
openturns : conda install -c conda-forge openturns  (CopulaCDFEmulator)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

def _require_pykrige():
    try:
        from pykrige.uk import UniversalKriging
        return UniversalKriging
    except ImportError as exc:
        raise ImportError(
            "pykrige is required for Kriging interpolation.\n"
            "Install it with: pip install pykrige"
        ) from exc


def _require_sklearn():
    try:
        import sklearn
        return sklearn
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for GP interpolation.\n"
            "Install it with: pip install scikit-learn"
        ) from exc


# ---------------------------------------------------------------------------
# IDW Interpolator
# ---------------------------------------------------------------------------

class IDWInterpolator:
    """
    Inverse Distance Weighting spatial interpolation.

    No external dependencies — pure numpy implementation.

    Parameters
    ----------
    power : float
        Distance decay exponent (default 2). Higher values increase the
        influence of nearby points relative to distant ones.

    Examples
    --------
    >>> idw = IDWInterpolator(power=2)
    >>> idw.fit(station_df, x_col='lon', y_col='lat', value_col='precip')
    >>> predictions = idw.predict(grid_df, x_col='lon', y_col='lat')
    """

    def __init__(self, power: float = 2.0):
        self.power = power
        self._x = None
        self._y = None
        self._values = None

    def fit(self, data: pd.DataFrame, x_col: str = "lon",
            y_col: str = "lat", value_col: str = "value"):
        """
        Store station coordinates and observed values.

        Args:
            data:       DataFrame with station observations.
            x_col:      Column name for x-coordinate (longitude).
            y_col:      Column name for y-coordinate (latitude).
            value_col:  Column name for the field to interpolate.

        Returns:
            self
        """
        self._x = data[x_col].values.astype(float)
        self._y = data[y_col].values.astype(float)
        self._values = data[value_col].values.astype(float)
        return self

    def predict(self, grid_data: pd.DataFrame, x_col: str = "lon",
                y_col: str = "lat") -> np.ndarray:
        """
        Interpolate onto new locations.

        Args:
            grid_data:  DataFrame with target locations.
            x_col:      Column name for x-coordinate.
            y_col:      Column name for y-coordinate.

        Returns:
            ndarray of shape (n_grid,) with interpolated values.
        """
        if self._x is None:
            raise RuntimeError("Call fit() before predict().")

        gx = grid_data[x_col].values.astype(float)
        gy = grid_data[y_col].values.astype(float)

        results = np.empty(len(gx))
        for i, (xi, yi) in enumerate(zip(gx, gy)):
            dist = np.sqrt((self._x - xi) ** 2 + (self._y - yi) ** 2)
            if np.any(dist == 0):
                results[i] = self._values[dist == 0][0]
            else:
                w = 1.0 / dist ** self.power
                results[i] = np.sum(w * self._values) / np.sum(w)

        return results

    def predict_timeseries(self, grid_data: pd.DataFrame,
                           timeseries_df: pd.DataFrame,
                           x_col: str = "lon",
                           y_col: str = "lat") -> pd.DataFrame:
        """
        Reconstruct a daily field for every time step via IDW.

        Args:
            grid_data:     DataFrame with target grid locations.
            timeseries_df: DataFrame (dates × stations) of observed values.
                           Columns must match the station order used in fit().
            x_col, y_col:  Coordinate column names in grid_data.

        Returns:
            pd.DataFrame (dates × grid cells).
        """
        if self._x is None:
            raise RuntimeError("Call fit() before predict_timeseries().")

        gx = grid_data[x_col].values.astype(float)
        gy = grid_data[y_col].values.astype(float)

        n_grid = len(gx)
        n_time = len(timeseries_df)
        out = np.empty((n_time, n_grid))

        for i, (xi, yi) in enumerate(zip(gx, gy)):
            dist = np.sqrt((self._x - xi) ** 2 + (self._y - yi) ** 2)
            if np.any(dist == 0):
                col_idx = np.where(dist == 0)[0][0]
                out[:, i] = timeseries_df.iloc[:, col_idx].values
            else:
                w = 1.0 / dist ** self.power
                w /= w.sum()
                out[:, i] = timeseries_df.values @ w

        return pd.DataFrame(
            out,
            index=timeseries_df.index,
            columns=[f"cell_{i}" for i in range(n_grid)],
        )


# ---------------------------------------------------------------------------
# Kriging Interpolator
# ---------------------------------------------------------------------------

class KrigingInterpolator:
    """
    Universal Kriging spatial interpolation (pykrige).

    Supports external drift functions (covariates) passed as nlags-friendly
    arrays, consistent with the UniversalKriging API.

    Parameters
    ----------
    variogram_model : str
        pykrige variogram model: 'linear', 'power', 'gaussian', 'spherical',
        'exponential', 'hole-effect' (default 'spherical').
    drift_terms : list of str or None
        External drift terms; pykrige built-ins: 'regional_linear',
        'specified', 'functional' (default None → ordinary kriging).
    nlags : int
        Number of variogram lags (default 6).
    weight : bool
        Weight variogram by number of point pairs (default False).

    Examples
    --------
    >>> krig = KrigingInterpolator(variogram_model='spherical')
    >>> krig.fit(station_df, x_col='lon', y_col='lat', value_col='precip')
    >>> mean, var = krig.predict(grid_df, x_col='lon', y_col='lat')
    """

    def __init__(self, variogram_model: str = "spherical",
                 drift_terms=None, nlags: int = 6, weight: bool = False):
        self.variogram_model = variogram_model
        self.drift_terms = drift_terms
        self.nlags = nlags
        self.weight = weight
        self._model = None

    def fit(self, data: pd.DataFrame, x_col: str = "lon",
            y_col: str = "lat", value_col: str = "value",
            external_drift=None):
        """
        Fit variogram and build the kriging model.

        Args:
            data:           DataFrame with station observations.
            x_col:          Longitude column.
            y_col:          Latitude column.
            value_col:      Target field column.
            external_drift: ndarray passed as ``external_drift`` to
                            UniversalKriging when drift_terms include
                            'specified'.

        Returns:
            self
        """
        UniversalKriging = _require_pykrige()

        kwargs = dict(
            variogram_model=self.variogram_model,
            nlags=self.nlags,
            weight=self.weight,
            enable_plotting=False,
        )
        if self.drift_terms:
            kwargs["drift_terms"] = self.drift_terms
        if external_drift is not None:
            kwargs["specified_drift"] = [external_drift]

        self._model = UniversalKriging(
            data[x_col].values,
            data[y_col].values,
            data[value_col].values,
            **kwargs,
        )
        return self

    def predict(self, grid_data: pd.DataFrame, x_col: str = "lon",
                y_col: str = "lat"):
        """
        Krige onto new locations.

        Args:
            grid_data:  DataFrame with target locations.
            x_col:      Longitude column.
            y_col:      Latitude column.

        Returns:
            mean: ndarray of kriged predictions.
            var:  ndarray of kriging variances.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        mean, var = self._model.execute(
            "points",
            grid_data[x_col].values,
            grid_data[y_col].values,
        )
        return np.array(mean), np.array(var)


# ---------------------------------------------------------------------------
# Gaussian Process Interpolator
# ---------------------------------------------------------------------------

class GaussianProcessInterpolator:
    """
    Gaussian Process regression for spatial interpolation with covariates.

    Applies MinMaxScaler to features before GP fitting, matching the pattern
    used in the ``Analisis Climatico-Precipitation.ipynb`` notebook. Supports
    both single-field prediction and full temporal reconstruction (one GP fit
    per time step).

    Parameters
    ----------
    covariates : list of str or None
        Feature columns. Defaults to ['lon', 'lat'].
    kernel : sklearn kernel or None
        GP kernel. Defaults to RationalQuadratic (good general-purpose choice).
        For precipitation a composite kernel is recommended::

            from sklearn.gaussian_process.kernels import (
                ConstantKernel, DotProduct, RationalQuadratic, Exponentiation
            )
            kernel = ConstantKernel() * Exponentiation(
                        DotProduct() ** 2 * RationalQuadratic(), exponent=2)

    n_restarts_optimizer : int
        Kernel hyper-parameter restarts (default 5).
    test_size : float
        Fraction held out for cross-validation (default 0.2).
    random_state : int
        Reproducibility seed (default 42).

    Examples
    --------
    >>> gp = GaussianProcessInterpolator(covariates=['lon', 'lat', 'elevation', 'trmm'])
    >>> metrics = gp.fit(station_df, target_col='precip_mean')
    >>> mean, std = gp.predict(grid_df)
    >>> daily = gp.predict_timeseries(grid_df, timeseries_df)
    """

    def __init__(self, covariates=None, kernel=None,
                 n_restarts_optimizer: int = 5,
                 test_size: float = 0.2, random_state: int = 42):
        _require_sklearn()
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RationalQuadratic
        from sklearn.preprocessing import MinMaxScaler

        self.covariates = covariates or ["lon", "lat"]
        self.test_size = test_size
        self.random_state = random_state

        _kernel = kernel or RationalQuadratic()
        self._gp = GaussianProcessRegressor(
            kernel=_kernel,
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer,
        )
        self._scaler = MinMaxScaler()
        self._target_col = None

    def fit(self, data: pd.DataFrame, target_col: str) -> dict:
        """
        Fit the GP to station observations.

        Args:
            data:       DataFrame with covariate and target columns.
            target_col: Column to model.

        Returns:
            dict with cross-validation metrics: 'r2', 'rmse'.
        """
        from sklearn.model_selection import train_test_split

        self._target_col = target_col
        X = self._scaler.fit_transform(data[self.covariates].values)
        self._X_train = X  # stored for predict_timeseries
        y = data[target_col].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self._gp.fit(X_tr, y_tr)

        y_pred = self._gp.predict(X_te)
        ss_res = np.sum((y_te - y_pred) ** 2)
        ss_tot = np.sum((y_te - y_te.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
        print(f"Cross-validation — R²: {r2:.3f}  RMSE: {rmse:.3f}")
        return {"r2": r2, "rmse": rmse}

    def predict(self, grid_data: pd.DataFrame):
        """
        Predict the spatial field at new locations.

        Args:
            grid_data: DataFrame with covariate columns.

        Returns:
            mean: ndarray of predictions.
            std:  ndarray of posterior standard deviations.
        """
        X = self._scaler.transform(grid_data[self.covariates].values)
        mean, std = self._gp.predict(X, return_std=True)
        return mean, std

    def predict_timeseries(self, grid_data: pd.DataFrame,
                           timeseries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct a daily field by fitting a new GP for each time step.

        This reproduces the temporal reconstruction in
        ``Analisis Climatico-Precipitation.ipynb``: for each day a GP is fit
        to the station observations of that day and then predicted on the
        full grid.

        Args:
            grid_data:     DataFrame with grid locations (covariate columns).
            timeseries_df: DataFrame (dates × stations) of observed daily values.
                           Columns must correspond to the stations whose
                           coordinates and covariates are in *grid_data* (only
                           the station rows).

        Returns:
            pd.DataFrame (dates × grid cells).
        """
        _require_sklearn()
        from sklearn.gaussian_process import GaussianProcessRegressor

        X_grid = self._scaler.transform(grid_data[self.covariates].values)
        n_time = len(timeseries_df)
        n_grid = len(grid_data)
        out = np.empty((n_time, n_grid))

        for i, (_, row) in enumerate(timeseries_df.iterrows()):
            y_day = row.values.astype(float)
            gp_day = GaussianProcessRegressor(
                kernel=self._gp.kernel_,
                normalize_y=True,
                n_restarts_optimizer=0,
            )
            # Use the training station features stored during fit()
            gp_day.fit(self._X_train, y_day)
            out[i] = gp_day.predict(X_grid)

        return pd.DataFrame(
            out,
            index=timeseries_df.index,
            columns=[f"cell_{i}" for i in range(n_grid)],
        )


# ---------------------------------------------------------------------------
# NEOPRENE + GP daily precipitation reconstructor
# ---------------------------------------------------------------------------

class NeopreneGPReconstructor:
    """
    Daily precipitation field reconstruction combining NEOPRENE and GP.

    Pipeline (matching the ``gaup`` branch of NEOPRENE):

    1. Calibrate a single-site NSRP model per station with NEOPRENE.
    2. Simulate multi-year synthetic daily series at each station.
    3. For each day, fit a GP to the station values and predict on a grid.

    This gives a spatially coherent daily precipitation field whose marginal
    statistics at each station are controlled by the calibrated NSRP model.

    Parameters
    ----------
    gp_covariates : list of str or None
        Feature columns for the GP spatial interpolation (default ['lon', 'lat']).
    temporal_resolution : str
        NEOPRENE time step — 'd' (daily) or 'h' (hourly).
    seasonality : str
        NEOPRENE seasonal grouping — 'annual', 'seasonal', 'monthly'.
    n_restarts_optimizer : int
        GP kernel optimiser restarts (default 3).

    Examples
    --------
    >>> rec = NeopreneGPReconstructor(gp_covariates=['lon', 'lat', 'elevation'])
    >>> rec.fit(station_obs, station_meta)   # station_obs: dict name→pd.Series
    >>> grid_precip = rec.simulate_spatial(station_meta, grid_meta,
    ...                                    year_ini=2000, year_fin=2050)
    """

    def __init__(self, gp_covariates=None, temporal_resolution: str = "d",
                 seasonality: str = "monthly", n_restarts_optimizer: int = 3,
                 statistics=None, n_iterations: int = 100, n_bees: int = 20):
        self.gp_covariates = gp_covariates or ["lon", "lat"]
        self.temporal_resolution = temporal_resolution
        self.seasonality = seasonality
        self.n_restarts_optimizer = n_restarts_optimizer
        self.statistics = statistics
        self.n_iterations = n_iterations
        self.n_bees = n_bees

        self._nsrp_models: dict = {}
        self._simulated: dict = {}

    def fit(self, station_obs: dict, verbose: bool = False):
        """
        Calibrate one NSRP model per station.

        Args:
            station_obs: dict mapping station name → pd.Series (DatetimeIndex,
                         daily precipitation in mm).
            verbose:     Print PSO progress per station.

        Returns:
            self
        """
        from pyhydra.climate.stochastic_generation.point import NSRPModel

        for name, series in station_obs.items():
            model = NSRPModel(
                temporal_resolution=self.temporal_resolution,
                seasonality=self.seasonality,
                statistics=self.statistics,
                n_iterations=self.n_iterations,
                n_bees=self.n_bees,
            )
            model.fit(series, verbose=verbose)
            self._nsrp_models[name] = model

        return self

    def simulate_stations(self, year_ini: int, year_fin: int) -> dict:
        """
        Simulate daily series at each calibrated station.

        Args:
            year_ini: First year of simulation.
            year_fin: Last year of simulation.

        Returns:
            dict mapping station name → pd.Series of simulated daily precip.
        """
        if not self._nsrp_models:
            raise RuntimeError("Call fit() first.")

        if not hasattr(np, 'in1d'):
            np.in1d = np.isin  # NumPy 2.0 removed np.in1d; NEOPRENE still uses it

        for name, model in self._nsrp_models.items():
            result = model.simulate(year_ini=year_ini, year_fin=year_fin)
            daily = getattr(result, "Daily_Simulation", None)
            if daily is None:
                raise AttributeError(
                    "NEOPRENE simulation result has no 'Daily_Simulation' attribute."
                )
            # NEOPRENE returns a DataFrame with a DatetimeIndex and a "Rain" column.
            if isinstance(daily, pd.DataFrame):
                self._simulated[name] = daily.iloc[:, 0]
            else:
                self._simulated[name] = pd.Series(np.asarray(daily))

        return self._simulated

    def simulate_spatial(self, station_meta: pd.DataFrame,
                         grid_meta: pd.DataFrame,
                         year_ini: int, year_fin: int) -> pd.DataFrame:
        """
        Simulate spatially coherent daily precipitation on a grid.

        Runs ``simulate_stations`` then fits a GP per day and predicts on
        *grid_meta*.

        Args:
            station_meta: DataFrame with station rows; must contain
                          ``gp_covariates`` columns and an index or column
                          that matches keys in ``station_obs``.
            grid_meta:    DataFrame with grid-cell rows; must contain
                          ``gp_covariates`` columns.
            year_ini:     First simulation year.
            year_fin:     Last simulation year.

        Returns:
            pd.DataFrame (dates × grid cells) of simulated daily precipitation.
        """
        _require_sklearn()
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RationalQuadratic
        from sklearn.preprocessing import MinMaxScaler

        if not self._simulated:
            self.simulate_stations(year_ini, year_fin)

        stations = list(self._simulated.keys())
        sim_df = pd.DataFrame(self._simulated)

        X_stations = station_meta.loc[stations, self.gp_covariates].values.astype(float)
        X_grid = grid_meta[self.gp_covariates].values.astype(float)

        scaler = MinMaxScaler()
        X_stations_s = scaler.fit_transform(X_stations)
        X_grid_s = scaler.transform(X_grid)

        n_time = len(sim_df)
        n_grid = len(X_grid)
        out = np.empty((n_time, n_grid))

        base_kernel = RationalQuadratic()
        for i, (_, row) in enumerate(sim_df.iterrows()):
            y = row.values.astype(float)
            y = np.where(np.isnan(y), 0.0, y)  # NEOPRENE can produce NaN → treat as dry
            gp = GaussianProcessRegressor(
                kernel=base_kernel,
                normalize_y=True,
                n_restarts_optimizer=self.n_restarts_optimizer,
            )
            gp.fit(X_stations_s, y)
            pred = gp.predict(X_grid_s)
            out[i] = np.clip(pred, 0.0, None)

        return pd.DataFrame(
            out,
            index=sim_df.index,
            columns=[f"cell_{i}" for i in range(n_grid)],
        )


# ---------------------------------------------------------------------------
# GP emulator for copula CDF
# ---------------------------------------------------------------------------

class CopulaCDFEmulator:
    """
    Gaussian Process surrogate for a multivariate copula CDF.

    Approximates an expensive copula CDF via a GP emulator trained on
    Monte Carlo samples. Useful when the copula is a vine or other
    high-dimensional model where direct evaluation is slow.

    Parameters
    ----------
    n_mc_samples : int
        Number of Monte Carlo samples used to train the emulator (default 1_000_000).
    batch_size : int
        Prediction batch size (default 50_000).

    Examples
    --------
    >>> emulator = CopulaCDFEmulator(n_mc_samples=500_000)
    >>> emulator.fit(vine_copula_model, n_dims=5)
    >>> cdf_values = emulator.predict(query_points)
    """

    def __init__(self, n_mc_samples: int = 1_000_000, batch_size: int = 50_000):
        _require_sklearn()
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RationalQuadratic

        self.n_mc_samples = n_mc_samples
        self.batch_size = batch_size
        self._gp = GaussianProcessRegressor(
            kernel=RationalQuadratic(),
            normalize_y=True,
            n_restarts_optimizer=3,
        )
        self._fitted = False

    def fit(self, copula_model, n_dims: int):
        """
        Train the GP emulator on Monte Carlo samples from *copula_model*.

        Args:
            copula_model: Object with a ``.computeCDF(sample)`` method
                          (e.g. an openturns distribution or copula).
            n_dims:       Dimensionality of the copula.

        Returns:
            self
        """
        X_train = np.random.uniform(0, 1, size=(self.n_mc_samples, n_dims))

        try:
            import openturns as ot
        except ImportError as exc:
            raise ImportError(
                "openturns is required for copula CDF evaluation.\n"
                "Install with: conda install -c conda-forge openturns"
            ) from exc

        ot_sample = ot.Sample(X_train.tolist())
        y_train = np.array(copula_model.computeCDF(ot_sample)).flatten()

        self._gp.fit(X_train, y_train)
        self._fitted = True
        return self

    def predict(self, query_points) -> np.ndarray:
        """
        Evaluate the emulated CDF at *query_points*.

        Args:
            query_points: ndarray of shape (n_query, n_dims), values in [0, 1].

        Returns:
            ndarray of shape (n_query,) — emulated CDF values.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(query_points)
        n = len(X)
        results = np.empty(n)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            results[start:end] = self._gp.predict(X[start:end])
        return np.clip(results, 0.0, 1.0)

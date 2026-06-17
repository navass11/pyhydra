"""
Spatial random field generation — CoSMoS_py wrappers.

Generates spatially (and spatio-temporally) correlated random fields on a grid
or at irregular locations, preserving a target marginal distribution and a
user-specified spatio-temporal correlation structure (STCS).

Methodology: Biller-Nelson (2003) VAR approach + AutoCorrelation Transformation
Function (ACTF) — essentially a copula-based mapping from Gaussian to target.

STCS options
------------
'clayton'    : Clayton copula combining marginal space and time ACFs.
'gneiting14' : Gneiting (2002) Eq.14 — differentiable ST covariance.
'gneiting16' : Gneiting (2002) Eq.16 — Matérn-type ST covariance.

Dependence structure (dsid) for VAR spatial model
---------------------------------------------------
'gauss'    : Gaussian copula (default).
'student'  : Student-t copula (heavier joint tails).
'bardossy' : Bardossy (2006) copula.

Reference: Papalexiou & Serinaldi (2020), WRR 56(2), e2019WR026331
Source: https://github.com/navass11/CoSMoS_py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------

def _require_cosmos():
    try:
        import cosmos_py
        return cosmos_py
    except ImportError as exc:
        raise ImportError(
            "cosmos_py is required for spatial random field generation.\n"
            "Install it with: pip install -e /path/to/CoSMoS_py\n"
            "Source: https://github.com/navass11/CoSMoS_py"
        ) from exc


# ---------------------------------------------------------------------------
# Low-level wrappers (functional API)
# ---------------------------------------------------------------------------

def fit_spatial_model(spacepoints, p, dist, dist_params, p0=0.0,
                      stcs_id="clayton", stcs_params=None,
                      scale_factor=1.0, dep_structure="gauss",
                      dep_arg=None, anisotropy_id="affine",
                      anisotropy_params=None, advection_id="uniform",
                      advection_params=None):
    """
    Fit a VAR(p) spatio-temporal model to a set of spatial locations.

    Wraps ``cosmos_py.fields.fit_var``. Returns a model dict that can be
    passed directly to ``generate_random_field`` or ``generate_multisite_ts``.

    Parameters
    ----------
    spacepoints : int or ndarray (n_sites × 2)
        Integer m → simulate on an m×m regular grid.
        Array → simulate at n_sites irregular locations (column order: x, y).
    p : int
        VAR order (temporal lag depth). p=1 is usually sufficient.
    dist : str
        Marginal distribution name: 'gengamma', 'burrXII', 'paretoII',
        'burrIII', 'gev'.
    dist_params : dict
        Distribution parameters matching CoSMoS_py conventions.
    p0 : float
        Probability of zero values (intermittency, default 0).
    stcs_id : str
        Spatio-temporal correlation structure:
        'clayton' (default), 'gneiting14', 'gneiting16'.
    stcs_params : dict or None
        Parameters for the chosen STCS function.
        Clayton default:  ``{'scfid': 'weibull', 'tcfid': 'weibull',
                             'copulaarg': 1.0,
                             'scfarg': {'scale': 0.5, 'shape': 0.5},
                             'tcfarg': {'scale': 0.5, 'shape': 0.8}}``
        Gneiting default: ``{'a': 1, 'c': 1, 'alpha': 0.5, 'beta': 0.5,
                              'gamma': 0.5, 'tau': 1}``
    scale_factor : float
        Distance between pixel centres for regular grids (default 1).
    dep_structure : str
        Copula for spatial dependence: 'gauss' (default), 'student',
        'bardossy', 'bardossyF'.
    dep_arg : float or None
        Degrees of freedom for Student-t; m parameter for Bardossy.
    anisotropy_id : str
        Anisotropy transformation: 'affine' (default), 'swirl', 'wave'.
    anisotropy_params : dict or None
        Parameters for the anisotropy transformation.
    advection_id : str
        Advection field: 'uniform' (default), 'rotation', 'spiral'.
    advection_params : dict or None
        Parameters for the advection field (e.g. ``{'u': 1.0, 'v': 0.0}``).

    Returns
    -------
    dict
        Fitted model ready for ``generate_random_field`` /
        ``generate_multisite_ts``.

    Examples
    --------
    >>> model = fit_spatial_model(
    ...     spacepoints=10,   # 10×10 grid
    ...     p=1,
    ...     dist='gengamma',
    ...     dist_params={'scale': 5.0, 'shape1': 0.8, 'shape2': 0.5},
    ...     p0=0.7,
    ...     stcs_id='clayton',
    ...     stcs_params={'scfid': 'weibull', 'tcfid': 'weibull',
    ...                  'copulaarg': 2.0,
    ...                  'scfarg': {'scale': 0.3, 'shape': 0.5},
    ...                  'tcfarg': {'scale': 0.5, 'shape': 0.8}},
    ... )
    """
    _require_cosmos()
    from cosmos_py.fields.fitVAR import fit_var

    _stcs = stcs_params or _default_stcs_params(stcs_id)
    _aniso = anisotropy_params or {"phi1": 1, "phi2": 1, "phi12": 0, "theta": 0}
    _adv = advection_params or {"u": 0, "v": 0}

    return fit_var(
        spacepoints=spacepoints,
        p=p,
        margdist=dist,
        margarg=dist_params,
        p0=p0,
        stcsid=stcs_id,
        stcsarg=_stcs,
        scalefactor=scale_factor,
        anisotropyid=anisotropy_id,
        anisotropyarg=_aniso,
        advectionid=advection_id,
        advectionarg=_adv,
        dsid=dep_structure,
        dsarg=dep_arg,
    )


def generate_random_field(n_steps, model):
    """
    Simulate n time steps of a spatially-correlated random field.

    Wraps ``cosmos_py.fields.generate_mts_fast`` for speed or
    ``generate_mts`` for reliability.

    Parameters
    ----------
    n_steps : int
        Number of time steps to simulate.
    model : dict
        Output of ``fit_spatial_model()``.

    Returns
    -------
    ndarray, shape (n_steps, n_sites)
        Simulated field values. For a regular grid of size m×m the
        n_sites = m*m columns are in row-major order.

    Examples
    --------
    >>> field = generate_random_field(365, model)
    >>> field.shape   # (365, 100) for a 10×10 grid
    """
    _require_cosmos()
    from cosmos_py.fields.generateMTS import generate_mts
    return generate_mts(n_steps, model)


def generate_random_field_fast(n_steps, model):
    """
    Fast variant of ``generate_random_field`` using pre-computed Cholesky.

    Slightly less numerically robust but significantly faster for large grids.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    model : dict
        Output of ``fit_spatial_model()``.

    Returns
    -------
    ndarray, shape (n_steps, n_sites)
    """
    _require_cosmos()
    from cosmos_py.fields.generateMTSFast import generate_mts_fast
    return generate_mts_fast(
        n=n_steps,
        spacepoints=model["spacepoints"],
        margdist=model["margdist"],
        margarg=model["margarg"],
        p0=model["p0"],
        distbounds=model.get("distbounds", (-np.inf, np.inf)),
        stcsarg=model.get("stcs"),
        dsid=model.get("dsid", "gauss"),
        dsarg=model.get("dsarg"),
    )


def check_random_field(field, model):
    """
    Diagnostic check: compare simulated vs target statistics.

    Wraps ``cosmos_py.fields.check_rf``.

    Parameters
    ----------
    field : ndarray, shape (n_steps, n_sites)
        Output of ``generate_random_field``.
    model : dict
        Output of ``fit_spatial_model``.

    Returns
    -------
    dict with keys: 'marginal' (DataFrame), 'spatial_acf' (ndarray),
                    'temporal_acf' (ndarray).
    """
    _require_cosmos()
    from cosmos_py.fields.checkRF import check_rf
    return check_rf(field, model)


# ---------------------------------------------------------------------------
# Default STCS parameter helpers
# ---------------------------------------------------------------------------

def _default_stcs_params(stcs_id: str) -> dict:
    if stcs_id == "clayton":
        return {
            "scfid":     "weibull",
            "tcfid":     "weibull",
            "copulaarg": 1.0,
            "scfarg":    {"scale": 0.5, "shape": 0.5},
            "tcfarg":    {"scale": 0.5, "shape": 0.8},
        }
    if stcs_id == "gneiting14":
        return {"a": 1.0, "c": 1.0, "alpha": 0.5, "beta": 0.5, "gamma": 0.5, "tau": 1.0}
    if stcs_id == "gneiting16":
        return {"a": 1.0, "c": 1.0, "alpha": 0.5, "beta": 0.5, "nu": 1.5, "tau": 1.0}
    return {}


# ---------------------------------------------------------------------------
# High-level class (object-oriented API)
# ---------------------------------------------------------------------------

class SpatialFieldModel:
    """
    Spatio-temporal random field generator (CoSMoS_py VAR approach).

    Generates gridded or multi-site fields with user-specified marginal
    distribution and spatio-temporal correlation structure. The dependence
    between sites is modelled via a copula (Gaussian by default).

    Parameters
    ----------
    dist : str
        Marginal distribution: 'gengamma', 'burrXII', 'paretoII', 'burrIII', 'gev'.
    dist_params : dict
        Distribution parameters.
    p0 : float
        Probability of zero (intermittency). Use 0 for temperature or wind.
    p : int
        VAR model order (default 1).
    stcs_id : str
        Spatio-temporal correlation structure: 'clayton' (default),
        'gneiting14', 'gneiting16'.
    stcs_params : dict or None
        STCS parameters. Uses sensible defaults when None.
    dep_structure : str
        Copula for spatial dependence: 'gauss' (default), 'student',
        'bardossy', 'bardossyF'.
    dep_arg : float or None
        Degrees of freedom (Student-t) or m (Bardossy).
    scale_factor : float
        Distance between regular grid pixels (default 1).
    anisotropy_id : str
        Anisotropy transformation: 'affine' (default), 'swirl', 'wave'.
    anisotropy_params : dict or None
        Anisotropy parameters.
    advection_id : str
        Advection type: 'uniform' (default), 'rotation', 'spiral'.
    advection_params : dict or None
        Advection parameters (e.g. ``{'u': 2.0, 'v': 0.5}`` for uniform).

    Examples
    --------
    **Regular 10×10 grid, daily precipitation, Clayton STCS:**

    >>> model = SpatialFieldModel(
    ...     dist='gengamma',
    ...     dist_params={'scale': 5.0, 'shape1': 0.8, 'shape2': 0.5},
    ...     p0=0.65,
    ...     stcs_id='clayton',
    ... )
    >>> model.fit(spacepoints=10)
    >>> field = model.simulate(n_steps=365)   # shape (365, 100)

    **Irregular station locations:**

    >>> coords = np.array([[0, 0], [1, 0], [0.5, 1], [2, 0.5]])
    >>> model.fit(spacepoints=coords)
    >>> field = model.simulate(n_steps=365)   # shape (365, 4)

    **Student-t spatial copula (heavier joint extremes):**

    >>> model = SpatialFieldModel(
    ...     dist='gengamma',
    ...     dist_params={'scale': 8.0, 'shape1': 0.7, 'shape2': 0.4},
    ...     dep_structure='student',
    ...     dep_arg=5.0,   # degrees of freedom
    ... )
    """

    def __init__(self, dist: str, dist_params: dict, p0: float = 0.0,
                 p: int = 1, stcs_id: str = "clayton", stcs_params=None,
                 dep_structure: str = "gauss", dep_arg=None,
                 scale_factor: float = 1.0,
                 anisotropy_id: str = "affine", anisotropy_params=None,
                 advection_id: str = "uniform", advection_params=None):
        self.dist = dist
        self.dist_params = dist_params
        self.p0 = p0
        self.p = p
        self.stcs_id = stcs_id
        self.stcs_params = stcs_params
        self.dep_structure = dep_structure
        self.dep_arg = dep_arg
        self.scale_factor = scale_factor
        self.anisotropy_id = anisotropy_id
        self.anisotropy_params = anisotropy_params
        self.advection_id = advection_id
        self.advection_params = advection_params
        self._model = None

    def fit(self, spacepoints):
        """
        Build the VAR(p) model for the given spatial layout.

        Args:
            spacepoints: int → regular m×m grid; ndarray (n×2) → irregular sites.

        Returns:
            self
        """
        self._model = fit_spatial_model(
            spacepoints=spacepoints,
            p=self.p,
            dist=self.dist,
            dist_params=self.dist_params,
            p0=self.p0,
            stcs_id=self.stcs_id,
            stcs_params=self.stcs_params,
            scale_factor=self.scale_factor,
            dep_structure=self.dep_structure,
            dep_arg=self.dep_arg,
            anisotropy_id=self.anisotropy_id,
            anisotropy_params=self.anisotropy_params,
            advection_id=self.advection_id,
            advection_params=self.advection_params,
        )
        return self

    def simulate(self, n_steps: int, fast: bool = False) -> np.ndarray:
        """
        Simulate n_steps of the fitted spatio-temporal field.

        Args:
            n_steps: Number of time steps.
            fast:    Use the fast Cholesky variant (default False).

        Returns:
            ndarray, shape (n_steps, n_sites).
        """
        if self._model is None:
            raise RuntimeError("Call fit() before simulate().")
        if fast:
            return generate_random_field_fast(n_steps, self._model)
        return generate_random_field(n_steps, self._model)

    def simulate_dataframe(self, n_steps: int, start_date=None,
                           freq: str = "D", fast: bool = False) -> pd.DataFrame:
        """
        Simulate and return a DataFrame with a DatetimeIndex.

        Args:
            n_steps:    Number of time steps.
            start_date: Start of the index (default '2000-01-01').
            freq:       Pandas frequency string (default 'D' = daily).
            fast:       Use fast Cholesky variant.

        Returns:
            pd.DataFrame (n_steps × n_sites).
        """
        field = self.simulate(n_steps, fast=fast)
        idx = pd.date_range(
            start=start_date or "2000-01-01",
            periods=n_steps,
            freq=freq,
        )
        return pd.DataFrame(
            field,
            index=idx,
            columns=[f"site_{i}" for i in range(field.shape[1])],
        )

    def diagnostics(self, n_steps: int = 1000) -> dict:
        """
        Run a quick simulation and compare statistics vs targets.

        Args:
            n_steps: Length of the diagnostic simulation (default 1000).

        Returns:
            dict from ``check_random_field``.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before diagnostics().")
        field = self.simulate(n_steps)
        return check_random_field(field, self._model)

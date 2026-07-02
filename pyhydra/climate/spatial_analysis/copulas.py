"""
Multivariate copula fitting — synthetic event sampling and return period analysis.

Three complementary tools
-------------------------
1. **FloodEventCopula** (openturns)
   Gaussian copula for synthetic flood event generation.  Fits BIC-selected
   marginals and a Normal copula, then samples arbitrary synthetic catalogues.

2. **BivariateCopula / TrivariateCopula** (scipy only)
   Archimedean copulas (Gumbel, Clayton, Frank) for compound-flood return
   period analysis.  Computes AND / OR iso-return-period contours in physical
   variable space — the standard output for bivariate frequency analysis.

   Typical application:
     River discharge Q + storm surge SL + catchment rainfall P
     → joint return periods for coastal compound flooding events.

   Reference:
     Salvadori et al. (2016) Multivariate return periods in hydrology.
     Serinaldi (2015) Dismissing return periods.

3. **GaussianCopulaSampler** (scipy only, no openturns required)
   Parameterised Gaussian copula for uncertainty propagation.  The user
   specifies marginal distributions (fitted externally or via BIC on data)
   and a correlation structure (equi-correlation scalar or full matrix).
   Designed for sensitivity analyses where the correlation between variables
   is prescribed rather than estimated from observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _require_ot():
    try:
        import openturns as ot
        return ot
    except ImportError as exc:
        raise ImportError(
            "openturns is required for copula fitting.\n"
            "Install it with: conda install -c conda-forge openturns"
        ) from exc


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def fit_marginal(data, variable_name=""):
    """
    Select the best-fitting univariate distribution for *data* by BIC.

    Args:
        data:          1-D array-like of observed values.
        variable_name: Label used in printed output (optional).

    Returns:
        dist:   Fitted openturns distribution object.
        sample: openturns Sample of the input data.
    """
    ot = _require_ot()
    sample = ot.Sample([[float(v)] for v in data])
    factories = ot.DistributionFactory.GetContinuousUniVariateFactories()
    dist, _ = ot.FittingTest.BestModelBIC(sample, factories)
    label = f" [{variable_name}]" if variable_name else ""
    print(f"Best marginal{label}: {dist}")
    return dist, sample


def fit_discrete_marginal(data):
    """
    Fit a discrete (empirical) distribution to integer-valued *data*.

    Used for variables such as hydrograph shape type.

    Returns:
        dist:   openturns UserDefined distribution.
        sample: openturns Sample.
    """
    ot = _require_ot()
    sample = ot.Sample([[float(v)] for v in data])
    dist = ot.UserDefinedFactory().build(sample)
    return dist, sample


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class FloodEventCopula:
    """
    Multivariate Normal copula for synthetic flood event generation.

    Fits marginal distributions to each variable of the observed flood event
    table and a Normal copula to their joint dependence. Once fitted, an
    arbitrary number of synthetic events can be sampled.

    Parameters
    ----------
    continuous_vars : list of str
        Names of continuous variables (e.g. ``['Qmax', 'Qmed', 'Duracion']``).
        Each gets an automatic BIC-selected marginal distribution.
    discrete_vars : list of str
        Names of discrete / categorical variables (e.g. ``['shape_type']``).
        Each gets an empirical discrete marginal.

    Examples
    --------
    >>> copula = FloodEventCopula(
    ...     continuous_vars=['Qmax', 'Qmed', 'Duracion'],
    ...     discrete_vars=['shape_type'],
    ... )
    >>> copula.fit(classified_events_df)
    >>> synthetic = copula.sample(5000)
    """

    def __init__(
        self,
        continuous_vars=("Qmax", "Qmed", "Duracion"),
        discrete_vars=("shape_type",),
    ):
        self.continuous_vars = list(continuous_vars)
        self.discrete_vars = list(discrete_vars)
        self._marginals = {}   # {var: (dist, sample)}
        self._copula = None
        self._all_vars = self.continuous_vars + self.discrete_vars

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data):
        """
        Fit marginal distributions and the Normal copula to *data*.

        Args:
            data: pd.DataFrame with one column per variable in
                  ``continuous_vars`` + ``discrete_vars``.

        Returns:
            self (for chaining).
        """
        ot = _require_ot()

        # 1. Fit marginals and transform to uniform space
        u_cols = []
        for var in self.continuous_vars:
            dist, sample = fit_marginal(data[var].values, variable_name=var)
            self._marginals[var] = (dist, sample)
            u_cols.append(np.array(dist.computeCDF(sample)).flatten())

        for var in self.discrete_vars:
            dist, sample = fit_discrete_marginal(data[var].values)
            self._marginals[var] = (dist, sample)
            u_cols.append(np.array(dist.computeCDF(sample)).flatten())

        # 2. Fit Normal copula to joint uniform margins
        U = np.vstack(u_cols).T          # (n_obs, n_vars)
        U = np.clip(U, 1e-6, 1 - 1e-6)  # numerical stability
        ot_U = ot.Sample(U.tolist())
        self._copula = ot.NormalCopulaFactory().build(ot_U)
        print(f"Fitted copula: {self._copula}")
        return self

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples):
        """
        Draw *n_samples* synthetic events from the fitted copula model.

        Args:
            n_samples: Number of synthetic events to generate.

        Returns:
            pd.DataFrame with the same column names as the fitted data.
        """
        if self._copula is None:
            raise RuntimeError("Call fit() before sample().")

        ot = _require_ot()
        u_synth = np.array(self._copula.getSample(n_samples))  # (n, n_vars)

        result = {}
        for k, var in enumerate(self._all_vars):
            dist, _ = self._marginals[var]
            result[var] = np.array([
                float(dist.computeQuantile(float(v))[0])
                for v in u_synth[:, k]
            ])

        return pd.DataFrame(result)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def plot_marginals(self):
        """Plot ECDF vs fitted CDF for every variable."""
        try:
            import openturns.viewer as otv
            import openturns as ot
        except ImportError:
            raise ImportError("openturns is required for plotting.")

        for var, (dist, sample) in self._marginals.items():
            g = ot.UserDefined(sample).drawCDF()
            cdf = dist.drawCDF()
            cdf.setColors(["steelblue"])
            g.add(cdf)
            g.setTitle(f"Marginal fit — {var}")
            g.setLegends(["ECDF", dist.getName()])
            g.setXTitle(var)
            otv.View(g)

    def plot_dependence(self, observed, synthetic):
        """
        Scatter plots comparing observed vs synthetic event pairs.

        Args:
            observed:  pd.DataFrame of observed events.
            synthetic: pd.DataFrame returned by :meth:`sample`.
        """
        import matplotlib.pyplot as plt

        pairs = [
            (self.continuous_vars[i], self.continuous_vars[j])
            for i in range(len(self.continuous_vars))
            for j in range(i + 1, len(self.continuous_vars))
        ]
        if not pairs:
            return

        fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 5))
        if len(pairs) == 1:
            axes = [axes]

        for ax, (xv, yv) in zip(axes, pairs):
            ax.scatter(synthetic[xv], synthetic[yv], s=8, alpha=0.3,
                       color="steelblue", label="Synthetic")
            ax.scatter(observed[xv], observed[yv], s=30, alpha=0.8,
                       color="orange", label="Observed")
            ax.set_xlabel(xv)
            ax.set_ylabel(yv)
            ax.grid(alpha=0.3)

        axes[0].legend()
        fig.suptitle("Observed vs Synthetic — dependence structure", fontsize=13)
        fig.tight_layout()
        plt.show()


# ============================================================================
# Bivariate & Trivariate return period analysis (Archimedean copulas)
# ============================================================================

# ── Copula CDFs ──────────────────────────────────────────────────────────────

def _eps(u):
    return np.clip(u, 1e-10, 1 - 1e-10)


def _gumbel_cdf(u, v, theta):
    """Gumbel–Hougaard copula (upper tail dependence, θ ≥ 1)."""
    u, v = _eps(u), _eps(v)
    return np.exp(-((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1.0 / theta))


def _gumbel_cdf_3d(u, v, w, theta):
    u, v, w = _eps(u), _eps(v), _eps(w)
    return np.exp(-((-np.log(u)) ** theta + (-np.log(v)) ** theta + (-np.log(w)) ** theta) ** (1.0 / theta))


def _clayton_cdf(u, v, theta):
    """Clayton copula (lower tail dependence, θ > 0)."""
    u, v = _eps(u), _eps(v)
    return np.maximum(u ** (-theta) + v ** (-theta) - 1.0, 0.0) ** (-1.0 / theta)


def _clayton_cdf_3d(u, v, w, theta):
    u, v, w = _eps(u), _eps(v), _eps(w)
    return np.maximum(u ** (-theta) + v ** (-theta) + w ** (-theta) - 2.0, 0.0) ** (-1.0 / theta)


def _frank_cdf(u, v, theta):
    """Frank copula (symmetric, no tail dependence, θ ≠ 0)."""
    u, v = _eps(u), _eps(v)
    if abs(theta) < 1e-8:
        return u * v
    num = (np.exp(-theta * u) - 1.0) * (np.exp(-theta * v) - 1.0)
    den = np.exp(-theta) - 1.0
    return -1.0 / theta * np.log(1.0 + num / den)


def _frank_cdf_3d(u, v, w, theta):
    u, v, w = _eps(u), _eps(v), _eps(w)
    if abs(theta) < 1e-8:
        return u * v * w
    e0 = np.exp(-theta) - 1.0
    num = (np.exp(-theta * u) - 1.0) * (np.exp(-theta * v) - 1.0) * (np.exp(-theta * w) - 1.0)
    return -1.0 / theta * np.log(1.0 + num / e0 ** 2)


_COPULA_CDF = {
    "gumbel":  _gumbel_cdf,
    "clayton": _clayton_cdf,
    "frank":   _frank_cdf,
}
_COPULA_CDF_3D = {
    "gumbel":  _gumbel_cdf_3d,
    "clayton": _clayton_cdf_3d,
    "frank":   _frank_cdf_3d,
}


# ── Parameter estimation via Kendall's τ ─────────────────────────────────────

def _kendall_tau(x, y):
    """Kendall's rank correlation τ."""
    from scipy.stats import kendalltau
    return kendalltau(x, y).statistic


def _tau_to_theta(tau, family):
    """Convert Kendall τ to Archimedean copula parameter θ."""
    from scipy.optimize import brentq

    if family == "gumbel":
        return max(1.0 / (1.0 - tau), 1.001)  # θ = 1/(1-τ), θ ≥ 1

    if family == "clayton":
        return max(2.0 * tau / (1.0 - tau), 1e-4)  # θ = 2τ/(1-τ)

    if family == "frank":
        # τ = 1 - 4/θ * (1 - 1/θ * ∫₀^θ t/(e^t-1) dt)  (Debye function relation)
        def _frank_tau(theta):
            if abs(theta) < 1e-8:
                return 0.0
            from scipy.integrate import quad
            D1 = quad(lambda t: t / (np.exp(t) - 1.0), 0, theta)[0] / theta
            return 1.0 - 4.0 / theta * (1.0 - D1)
        if abs(tau) < 1e-6:
            return 1e-4
        lo, hi = (1e-4, 50.0) if tau > 0 else (-50.0, -1e-4)
        try:
            return brentq(lambda t: _frank_tau(t) - tau, lo, hi)
        except ValueError:
            return np.sign(tau) * 5.0

    raise ValueError(f"Unknown copula family: {family!r}")


# ── Marginal fitting (scipy, no extra deps) ──────────────────────────────────

def _fit_marginal_scipy(data, families=("gev", "lognorm", "gamma")):
    """Fit univariate distributions to *data* and return the best by AIC."""
    from scipy import stats

    _SCIPY_DISTS = {
        "gev":     stats.genextreme,
        "gumbel":  stats.gumbel_r,
        "lognorm": stats.lognorm,
        "gamma":   stats.gamma,
        "expon":   stats.expon,
        "norm":    stats.norm,
    }
    # Fix loc=0 for distributions with natural support [0, ∞) to prevent
    # unphysical negative values in hydrological variables (Q, SL, P).
    # AIC is computed with n_free = len(params) - 1 (loc is not estimated).
    _POSITIVE_SUPPORT = {"lognorm", "gamma", "expon"}

    best, best_aic = None, np.inf
    for name in families:
        dist = _SCIPY_DISTS[name]
        try:
            if name in _POSITIVE_SUPPORT:
                params = dist.fit(data, floc=0)
                n_free = len(params) - 1
            else:
                params = dist.fit(data)
                n_free = len(params)
            logL   = dist.logpdf(data, *params).sum()
            aic    = 2 * n_free - 2 * logL
            if aic < best_aic:
                best_aic = aic
                best = (dist, params, name)
        except Exception:
            continue
    if best is None:
        d, p = stats.lognorm, stats.lognorm.fit(data, floc=0)
        best = (d, p, "lognorm")
    print(f"  Best marginal: {best[2]}  (AIC={best_aic:.1f})")
    return best[0], best[1], best[2], best_aic   # (dist, params, name, aic)


# ── BivariateCopula ───────────────────────────────────────────────────────────

class BivariateCopula:
    """
    Bivariate Archimedean copula for compound-flood return period analysis.

    Fits univariate marginals (GEV / lognormal / gamma via AIC) and a
    parametric copula (Gumbel, Clayton or Frank) calibrated by Kendall's τ.

    Computes AND and OR iso-return-period contours in the original variable
    space — the standard plot in bivariate flood frequency analysis.

    Args:
        family: ``'gumbel'`` | ``'clayton'`` | ``'frank'``.
        marginal_families: Tuple of candidate distribution names passed to
            each marginal fit (``'gev'``, ``'gumbel'``, ``'lognorm'``,
            ``'gamma'``, ``'expon'``).

    Example::

        cop = BivariateCopula(family='gumbel')
        cop.fit(Q_annual_max, SL_annual_max,
                labels=('Peak discharge Q (m³/s)', 'Storm surge SL (m)'))
        cop.plot_contours([2, 5, 10, 25, 50, 100], scenario='both')
    """

    def __init__(self, family="gumbel",
                 marginal_families=("gev", "lognorm", "gamma")):
        self.family           = family
        self.marginal_families = marginal_families
        self._cdf_fn          = _COPULA_CDF[family]
        self._theta           = None
        self._mx = self._my  = None   # (scipy_dist, params)
        self._labels          = ("X", "Y")
        self._x = self._y    = None   # raw data

    # ------------------------------------------------------------------
    def fit(self, x, y, labels=("X", "Y")):
        """
        Fit marginals and copula parameter to observed paired data.

        Args:
            x, y:   1-D array-like of paired observations (same length).
            labels: Variable labels for plotting.

        Returns:
            self
        """
        x, y           = np.asarray(x, float), np.asarray(y, float)
        self._x, self._y = x, y
        self._labels   = labels

        print(f"Fitting BivariateCopula [{self.family}]  n={len(x)}")
        print(f"  Marginal X ({labels[0]}):")
        self._mx = _fit_marginal_scipy(x, self.marginal_families)
        print(f"  Marginal Y ({labels[1]}):")
        self._my = _fit_marginal_scipy(y, self.marginal_families)

        tau          = _kendall_tau(x, y)
        self._tau    = tau
        self._theta  = _tau_to_theta(tau, self.family)
        self._family_x, self._aic_x = self._mx[2], self._mx[3]
        self._family_y, self._aic_y = self._my[2], self._my[3]
        print(f"  Kendall τ = {tau:.3f}  →  θ = {self._theta:.3f}")
        return self

    # ------------------------------------------------------------------
    def _u(self, x):
        d, p = self._mx[0], self._mx[1]
        return np.clip(d.cdf(x, *p), 1e-10, 1 - 1e-10)

    def _v(self, y):
        d, p = self._my[0], self._my[1]
        return np.clip(d.cdf(y, *p), 1e-10, 1 - 1e-10)

    def _x_ppf(self, u):
        d, p = self._mx[0], self._mx[1]
        return d.ppf(u, *p)

    def _y_ppf(self, v):
        d, p = self._my[0], self._my[1]
        return d.ppf(v, *p)

    # ── Public marginal helpers ───────────────────────────────────────────────
    def marginal_cdf_x(self, x):
        """CDF of the fitted X marginal evaluated at x."""
        return self._u(x)

    def marginal_cdf_y(self, y):
        """CDF of the fitted Y marginal evaluated at y."""
        return self._v(y)

    def marginal_ppf_x(self, p):
        """Quantile (inverse CDF) of the fitted X marginal at probability p."""
        return self._x_ppf(p)

    def marginal_ppf_y(self, p):
        """Quantile (inverse CDF) of the fitted Y marginal at probability p."""
        return self._y_ppf(p)

    def cdf(self, u, v):
        """Copula CDF at uniform margins u, v."""
        return self._cdf_fn(u, v, self._theta)

    def joint_exceedance(self, x0, y0, scenario="OR"):
        """P(X > x0 AND/OR Y > y0)."""
        u, v = self._u(x0), self._v(y0)
        if scenario == "OR":
            return 1.0 - self.cdf(u, v)
        if scenario == "AND":
            return 1.0 - u - v + self.cdf(u, v)
        raise ValueError("scenario must be 'OR' or 'AND'")

    def return_period(self, x0, y0, scenario="OR"):
        """T = 1 / P(exceedance)."""
        p = self.joint_exceedance(x0, y0, scenario)
        return 1.0 / np.maximum(p, 1e-12)

    # ------------------------------------------------------------------
    def return_period_contour(self, T, scenario="OR", n_pts=300):
        """
        Iso-return-period curve in physical (x, y) space.

        For OR:  C(u,v) = 1 − 1/T   → solutions exist for u ∈ (1−1/T, 1)
        For AND: 1 − u − v + C(u,v) = 1/T  → solutions exist for u ∈ (0, 1−1/T)

        Returns:
            x_curve, y_curve: arrays of physical coordinates.
        """
        from scipy.optimize import brentq

        target = 1.0 / T

        # Adaptive u-sweep: concentrate points in the region where solutions exist.
        if scenario == "OR":
            level = 1.0 - target
            u_arr = np.linspace(level + 1e-4, 1.0 - 1e-4, n_pts)
        else:  # AND
            u_arr = np.linspace(1e-4, max(1.0 - target - 1e-4, 2e-4), n_pts)

        x_out, y_out = [], []
        for u in u_arr:
            try:
                if scenario == "OR":
                    if self._cdf_fn(u, 1 - 1e-10, self._theta) < level:
                        continue
                    v = brentq(
                        lambda vv: self._cdf_fn(u, vv, self._theta) - level,
                        1e-10, 1 - 1e-10, xtol=1e-8,
                    )
                else:  # AND
                    lo_val = 1.0 - u - 1e-10 + self._cdf_fn(u, 1e-10, self._theta)
                    hi_val = 1.0 - u - (1 - 1e-10) + self._cdf_fn(u, 1 - 1e-10, self._theta)
                    if lo_val < target or hi_val > target:
                        continue
                    v = brentq(
                        lambda vv: 1.0 - u - vv + self._cdf_fn(u, vv, self._theta) - target,
                        1e-10, 1 - 1e-10, xtol=1e-8,
                    )
                x_out.append(self._x_ppf(u))
                y_out.append(self._y_ppf(v))
            except Exception:
                continue

        return np.array(x_out), np.array(y_out)

    # ------------------------------------------------------------------
    def most_probable_event(self, T, scenario="OR", n_pts=500):
        """
        Most Probable Design Event (MPDE) on the T-year iso-return-period contour.

        Among all (x, y) combinations on the iso-contour, returns the one that
        maximises the joint density f(x,y) = c(F_X(x), F_Y(y))·f_X(x)·f_Y(y).

        Args:
            T:        Return period (years).
            scenario: ``'OR'`` or ``'AND'``.
            n_pts:    Contour resolution (higher → more precise MPDE location).

        Returns:
            (x_mpde, y_mpde) as floats, or (None, None) if the contour is empty.
        """
        xc, yc = self.return_period_contour(T, scenario=scenario, n_pts=n_pts)
        if len(xc) == 0:
            return None, None

        u = self._u(xc)
        v = self._v(yc)

        # Copula density via central finite differences
        eps = 1e-5
        u_lo, u_hi = np.maximum(u - eps, eps), np.minimum(u + eps, 1 - eps)
        v_lo, v_hi = np.maximum(v - eps, eps), np.minimum(v + eps, 1 - eps)
        c_uv = (self._cdf_fn(u_hi, v_hi, self._theta)
                - self._cdf_fn(u_hi, v_lo, self._theta)
                - self._cdf_fn(u_lo, v_hi, self._theta)
                + self._cdf_fn(u_lo, v_lo, self._theta)) / (4 * eps ** 2)
        c_uv = np.maximum(c_uv, 0.0)

        # Marginal PDFs
        fx = self._mx[0].pdf(xc, *self._mx[1])
        fy = self._my[0].pdf(yc, *self._my[1])

        idx = int(np.argmax(c_uv * fx * fy))
        return float(xc[idx]), float(yc[idx])

    # ------------------------------------------------------------------
    def joint_return_period(self, t, kind="or"):
        """
        Alias for :meth:`return_period_contour` with a lowercase ``kind``
        argument (``'or'``/``'and'``), matching the AND/OR terminology used
        elsewhere in the compound-flooding literature.

        Args:
            t:    Return period (years).
            kind: ``'or'`` or ``'and'``.

        Returns:
            (x_vals, y_vals): iso-return-period contour in physical space.
        """
        return self.return_period_contour(t, scenario=kind.upper())

    def mpde(self, t, scenario="OR"):
        """
        Alias for :meth:`most_probable_event` (Most Probable Design Event).

        Args:
            t:        Return period (years).
            scenario: ``'OR'`` or ``'AND'``.

        Returns:
            (x_mpde, y_mpde): coordinates of the most probable design event.
        """
        return self.most_probable_event(t, scenario=scenario)

    # ------------------------------------------------------------------
    def sample(self, n, random_state=None):
        """
        Draw *n* synthetic paired samples from the fitted bivariate copula.

        Uses family-specific exact algorithms (Marshall–Olkin for Gumbel,
        Gamma-frailty for Clayton, conditional inversion for Frank).

        Returns:
            (x_synth, y_synth): two 1-D arrays of length *n* in physical space.
        """
        rng   = np.random.default_rng(random_state)
        theta = self._theta

        if self.family == "gumbel":
            alpha = 1.0 / theta
            U  = rng.uniform(0.0, np.pi, n)
            E  = rng.exponential(1.0, n)
            V  = ((np.sin(alpha * U) / np.sin(U))
                  * (np.sin((1 - alpha) * U) / (E * np.sin(U))) ** ((1 - alpha) / alpha))
            e1 = rng.exponential(1.0, n)
            e2 = rng.exponential(1.0, n)
            u1 = np.exp(-(e1 / V) ** (1.0 / theta))
            u2 = np.exp(-(e2 / V) ** (1.0 / theta))

        elif self.family == "clayton":
            V  = rng.gamma(1.0 / theta, 1.0, n)
            e1 = rng.exponential(1.0, n)
            e2 = rng.exponential(1.0, n)
            u1 = (1.0 + e1 / V) ** (-1.0 / theta)
            u2 = (1.0 + e2 / V) ** (-1.0 / theta)

        elif self.family == "frank":
            # Analytical conditional inversion: set h(v|u) = t and solve for v.
            # Result: ev = [eu*(1-t) + t*e0] / [eu*(1-t) + t]  where
            # e0 = exp(-θ), eu = exp(-θu), ev = exp(-θv); then v = -log(ev)/θ.
            u1 = rng.uniform(0.0, 1.0, n)
            t  = np.clip(rng.uniform(0.0, 1.0, n), 1e-9, 1 - 1e-9)
            e0 = np.exp(-theta)
            eu = np.exp(-theta * u1)
            ev = (eu * (1.0 - t) + t * e0) / np.maximum(eu * (1.0 - t) + t, 1e-15)
            u2 = np.clip(-np.log(np.maximum(ev, 1e-15)) / theta, 1e-10, 1 - 1e-10)

        else:
            raise ValueError(f"Unknown family: {self.family!r}")

        u1 = np.clip(u1, 1e-10, 1 - 1e-10)
        u2 = np.clip(u2, 1e-10, 1 - 1e-10)
        return self._x_ppf(u1), self._y_ppf(u2)

    # ------------------------------------------------------------------
    def plot_contours(self, T_list=(2, 5, 10, 25, 50, 100),
                      scenario="both", data=True, ax=None, figsize=(13, 5)):
        """
        Standard bivariate return period plot.

        Args:
            T_list:   Return periods to draw.
            scenario: ``'OR'``, ``'AND'``, or ``'both'`` (side-by-side).
            data:     If True, scatter the fitted observations.
            ax:       Existing Axes (only used when scenario != 'both').
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        scenarios = ["OR", "AND"] if scenario == "both" else [scenario]
        if scenario == "both":
            fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        else:
            if ax is None:
                fig, axes = plt.subplots(figsize=(6, 5))
            else:
                fig = ax.figure
                axes = ax
            axes = [axes]

        colors = cm.plasma_r(np.linspace(0.15, 0.9, len(T_list)))

        for ax_, scen in zip(axes, scenarios):
            for col, T in zip(colors, T_list):
                xc, yc = self.return_period_contour(T, scenario=scen)
                if len(xc):
                    ax_.plot(xc, yc, color=col, lw=1.8, label=f"T={T} yr")

            if data and self._x is not None:
                ax_.scatter(self._x, self._y, s=18, alpha=0.5,
                            color="steelblue", zorder=5, label="Observed")

            lbl = "OR  [P(X>x ∪ Y>y) = 1/T]" if scen == "OR" \
                else "AND  [P(X>x ∩ Y>y) = 1/T]"
            ax_.set_title(lbl, fontsize=11)
            ax_.set_xlabel(self._labels[0])
            ax_.set_ylabel(self._labels[1])
            ax_.legend(fontsize=8, loc="upper left")
            ax_.grid(alpha=0.3)

        title = (f"Bivariate return periods — {self.family.capitalize()} copula"
                 f"  (τ={_kendall_tau(self._x, self._y):.2f}, θ={self._theta:.2f})")
        if scenario == "both":
            fig.suptitle(title, fontsize=12, fontweight="bold")
            fig.tight_layout()
            return fig, axes
        else:
            ax_.set_title(f"{lbl}\n{title}", fontsize=10)
            return fig, axes[0]

    # ------------------------------------------------------------------
    def plot_joint(self, T_list=(2, 10, 50, 100, 500),
                   n_synthetic=2000, figsize=(6, 6),
                   colors=("indianred", "b"),
                   n_grid=200):
        """
        Composite joint plot — GridSpec(4, 4) layout matching the reference style.

        * **Main panel** (top-right 3×3): grey synthetic scatter, black observed
          scatter, AND (*colors[0]*) and OR (*colors[1]*) iso-return-period
          contours on the same axes, points coloured by joint PDF density,
          MaxProb 'x' markers per contour level.
        * **Bottom panel** (xMarg): marginal PDF of X (olive fill, y-axis
          inverted so it grows upward toward main), T-year quantile annotations.
        * **Left panel** (yMarg): marginal PDF of Y (horizontal, x-axis
          inverted so it grows rightward), T-year quantile annotations.
        * **Legend**: placed in a separate axes to the right of the figure.

        Args:
            T_list:      Return periods to draw.
            n_synthetic: Synthetic copula samples for background scatter (0 to skip).
            figsize:     Figure size.
            colors:      ``(and_color, or_color)`` for base contour lines.
            n_grid:      Contour grid resolution.

        Returns:
            ``(fig, (main_ax, xMarg_ax, yMarg_ax))``
        """
        import matplotlib.pyplot as plt

        Ts = np.asarray(sorted(T_list), dtype=float)

        # --- Physical-space extent from data ---
        if self._x is not None:
            dx = (self._x.max() - self._x.min()) * 0.15
            dy = (self._y.max() - self._y.min()) * 0.15
            extent = [self._x.min() - dx, self._x.max() + dx,
                      self._y.min() - dy, self._y.max() + dy]
        else:
            u_e = np.linspace(1e-3, 1 - 1e-3, n_grid)
            extent = [float(self._x_ppf(u_e).min()), float(self._x_ppf(u_e).max()),
                      float(self._y_ppf(u_e).min()), float(self._y_ppf(u_e).max())]

        # --- Return-period grids ---
        u_lin = np.linspace(1e-3, 1 - 1e-3, n_grid)
        UU, VV = np.meshgrid(u_lin, u_lin)
        C     = self._cdf_fn(UU, VV, self._theta)
        T_or  = 1.0 / np.maximum(1.0 - C, 1e-12)
        T_and = 1.0 / np.maximum(1.0 - UU - VV + C, 1e-12)
        XX = self._x_ppf(UU)
        YY = self._y_ppf(VV)

        # --- Layout: GridSpec(4, 4) identical to reference ---
        fig  = plt.figure(figsize=figsize)
        grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2, figure=fig)
        main  = fig.add_subplot(grid[:-1, 1:])
        yMarg = fig.add_subplot(grid[:-1, 0])
        xMarg = fig.add_subplot(grid[-1,  1:])
        main.set_xlim(extent[:2])
        main.set_ylim(extent[2:])

        # --- Synthetic scatter (grey, small) ---
        if n_synthetic > 0:
            xs, ys = self.sample(n_synthetic)
            main.scatter(xs, ys, marker='.', c='grey', s=1, alpha=0.5,
                         rasterized=True, zorder=2, label='aleatorios')

        # --- Observed scatter (black) ---
        if self._x is not None:
            main.scatter(self._x, self._y, marker='.', c='k', s=12,
                         zorder=3, label='observados')

        # --- Base contour lines ---
        cs_and = main.contour(XX, YY, T_and, levels=Ts.tolist(),
                              colors=colors[0], linewidths=0.8, alpha=0.3, zorder=4)
        cs_or  = main.contour(XX, YY, T_or,  levels=Ts.tolist(),
                              colors=colors[1], linewidths=0.8, alpha=0.3, zorder=4)
        main.clabel(cs_and, fmt='%g', fontsize=8, inline=True)
        main.clabel(cs_or,  fmt='%g', fontsize=8, inline=True)

        # --- Colour contour points by joint PDF + MaxProb 'x' markers ---
        eps = 1e-5
        c_and, c_or = colors[0], colors[1]
        for scn, cs, cmap, c_mpde in zip(
                ['AND', 'OR'], [cs_and, cs_or],
                ['Reds', 'Blues'], [c_and, c_or]):
            try:
                level_paths = [col.get_paths() for col in cs.collections]
            except AttributeError:
                level_paths = [[p] for p in cs.get_paths()]

            for i, T in enumerate(Ts):
                paths = level_paths[i] if i < len(level_paths) else []
                if not paths:
                    continue
                pts = max(paths, key=lambda p: len(p.vertices)).vertices
                mask = ((pts[:, 0] >= extent[0]) & (pts[:, 0] <= extent[1]) &
                        (pts[:, 1] >= extent[2]) & (pts[:, 1] <= extent[3]))
                pts = pts[mask]
                if len(pts) < 2:
                    continue
                u   = self._u(pts[:, 0])
                v   = self._v(pts[:, 1])
                u_lo = np.clip(u - eps, 1e-10, 1 - 1e-10)
                u_hi = np.clip(u + eps, 1e-10, 1 - 1e-10)
                v_lo = np.clip(v - eps, 1e-10, 1 - 1e-10)
                v_hi = np.clip(v + eps, 1e-10, 1 - 1e-10)
                c_uv = (self._cdf_fn(u_hi, v_hi, self._theta)
                        - self._cdf_fn(u_hi, v_lo, self._theta)
                        - self._cdf_fn(u_lo, v_hi, self._theta)
                        + self._cdf_fn(u_lo, v_lo, self._theta)) / (4 * eps ** 2)
                pdf  = np.clip(c_uv
                               * self._mx[0].pdf(pts[:, 0], *self._mx[1])
                               * self._my[0].pdf(pts[:, 1], *self._my[1]), 0, None)
                main.scatter(pts[:, 0], pts[:, 1], c=pdf, cmap=cmap,
                             s=0.25, zorder=5)
                mp = pts[int(np.argmax(pdf))]
                main.scatter(mp[0], mp[1], marker='x', color=c_mpde,
                             s=80, linewidths=1.5, zorder=7)

            # Off-screen proxy artists for legend
            main.plot([-1e9], [-1e9], c=c_mpde, label=scn)
            main.scatter([-1e9], [-1e9], marker='x', c=c_mpde,
                         s=80, linewidths=1.5, label=f'MaxProb {scn}')

        main.set(xlim=extent[:2], ylim=extent[2:])
        main.set_xticklabels([])
        main.set_yticklabels([])

        # --- X marginal (bottom): PDF, y-axis inverted ---
        v0     = np.linspace(extent[0], extent[1], 400)
        pdf_x  = self._mx[0].pdf(v0, *self._mx[1])
        ymax_x = float(np.nanmax(pdf_x))
        if ymax_x > 0:
            xMarg.fill_between(v0, pdf_x, color='olive', alpha=0.25)
            xMarg.set(xlim=(extent[0], extent[1]), ylim=(ymax_x, 0))
            ppfs_x = np.clip(self._mx[0].ppf(1 - 1 / Ts, *self._mx[1]),
                             extent[0], extent[1])
            for T, ppf in zip(Ts, ppfs_x):
                xMarg.annotate(f'T{int(T)}',
                               xy=(ppf, 0), xytext=(ppf, ymax_x / 3),
                               color='olive', fontsize=11,
                               ha='center', va='top', rotation=90,
                               arrowprops=dict(arrowstyle='->', color='olive', lw=1))
            xMarg.set_xlabel(self._labels[0], fontsize=13)
            yticks = np.round(np.linspace(0, ymax_x, 3), 2)
            xMarg.set_yticks(yticks)
            xMarg.set_yticklabels(yticks)

        # --- Y marginal (left): PDF horizontal, x-axis inverted ---
        v1     = np.linspace(extent[2], extent[3], 400)
        pdf_y  = self._my[0].pdf(v1, *self._my[1])
        xmax_y = float(np.nanmax(pdf_y))
        if xmax_y > 0:
            yMarg.fill_between(pdf_y, v1, color='olive', alpha=0.25,
                               label='Distribución (PDF)')
            yMarg.set(xlim=(xmax_y, 0), ylim=(extent[2], extent[3]))
            ppfs_y = np.clip(self._my[0].ppf(1 - 1 / Ts, *self._my[1]),
                             extent[2], extent[3])
            for T, ppf in zip(Ts, ppfs_y):
                yMarg.annotate(f'T{int(T)}',
                               xy=(0, ppf), xytext=(xmax_y / 3, ppf),
                               color='olive', fontsize=11,
                               ha='right', va='center',
                               arrowprops=dict(arrowstyle='->', color='olive', lw=1))
            yMarg.set_ylabel(self._labels[1], fontsize=13)
            xticks = np.round(np.linspace(0, xmax_y, 3), 2)
            yMarg.set_xticks(xticks)
            yMarg.set_xticklabels(xticks)

        # --- Legend: outside to the right (reference style) ---
        lgnd_ax = fig.add_axes([0.97, 0.3, 0.08, 0.4])
        lgnd_ax.axis('off')
        # Handles from main: [0]=aleatorios [1]=observados [2]=AND [3]=MaxProbAND
        #                    [4]=OR [5]=MaxProbOR  → reorder [2,3,0,4,1,5]
        hndls, lbls = main.get_legend_handles_labels()
        if len(hndls) >= 6:
            order = [2, 3, 0, 4, 1, 5]
            hndls = [hndls[i] for i in order]
            lbls  = [lbls[i]  for i in order]
        hndls1, lbls1 = yMarg.get_legend_handles_labels()
        lgnd_ax.legend(hndls + hndls1, lbls + lbls1,
                       ncol=1, loc='center left', fontsize=9)

        return fig, (main, xMarg, yMarg)

    # ------------------------------------------------------------------
    def kendall_function(self, t_values=None, n_sim=50_000, random_state=None):
        """
        Monte Carlo estimate of the Kendall distribution K(t) = P[C(U,V) ≤ t].

        The Kendall function is the basis of the Kendall return period
        (Salvadori et al. 2011, 2016) — dimension-invariant and comparable
        across different copula families.

        Args:
            t_values:     Grid of t ∈ (0, 1); default 200 equally-spaced points.
            n_sim:        Monte Carlo pool size.
            random_state: RNG seed.

        Returns:
            ``(t_arr, K_arr)`` — evaluation points and K(t) values.
        """
        rng = np.random.default_rng(random_state)
        xs, ys = self.sample(n_sim, random_state=rng)
        c_vals = self._cdf_fn(self._u(xs), self._v(ys), self._theta)
        if t_values is None:
            t_values = np.linspace(0.01, 0.99, 200)
        t_arr = np.asarray(t_values, float)
        return t_arr, np.array([np.mean(c_vals <= t) for t in t_arr])

    # ------------------------------------------------------------------
    def kendall_return_period(self, T, mu=1.0, n_sim=50_000, random_state=None):
        """
        Kendall return period analysis (Salvadori et al. 2011).

        Inverts K(t*) = 1 − μ/T to find the critical Kendall level t*, then
        builds the iso-probability curve C(u, v) = t* in physical (x, y) space.

        Args:
            T:            Target return period (same units as μ).
            mu:           Mean inter-occurrence time; 1 for annual maxima.
            n_sim:        Monte Carlo sample size for K estimation.
            random_state: RNG seed.

        Returns:
            dict with keys:

            * ``t_star``  — Kendall probability level.
            * ``K_t``     — K(t*) ≈ 1 − μ/T.
            * ``T_K``     — Kendall return period (= T by construction).
            * ``contour`` — ``(x_arr, y_arr)`` on the C(u,v) = t* isoline.
        """
        from scipy.interpolate import interp1d
        from scipy.optimize import brentq

        target_K = np.clip(1.0 - mu / T, 0.001, 0.999)
        t_arr, K_arr = self.kendall_function(n_sim=n_sim, random_state=random_state)
        f_inv = interp1d(K_arr, t_arr, kind='linear',
                         bounds_error=False, fill_value=(t_arr[0], t_arr[-1]))
        t_star = float(f_inv(target_K))

        # C(u,v) ≤ min(u,v), so the isoline C=t* only exists for u > t_star.
        # Focus the grid there to avoid wasting 300 points below the threshold.
        u_lo = float(np.clip(t_star + 5e-4, 1e-4, 1 - 2e-4))
        xc, yc = [], []
        for u in np.linspace(u_lo, 1 - 1e-4, 300):
            try:
                hi = self._cdf_fn(u, 1 - 1e-8, self._theta)
                if hi < t_star:
                    continue
                vv = brentq(lambda v_: self._cdf_fn(u, v_, self._theta) - t_star,
                            1e-8, 1 - 1e-8, xtol=1e-8)
                xc.append(self._x_ppf(u))
                yc.append(self._y_ppf(vv))
            except Exception:
                continue

        return {'t_star': t_star, 'K_t': float(target_K), 'T_K': T,
                'contour': (np.array(xc), np.array(yc))}

    # ------------------------------------------------------------------
    def design_storm_ensemble(self, T, n_events=50, n_sim=100_000,
                              mu=1.0, random_state=None, eps_rel=0.05):
        """
        Ensemble of design events on the Kendall critical layer T_K = T.

        Draws *n_sim* samples from the copula, selects those within an ε-band
        of the critical Kendall level t*, then ranks by joint density.  The
        top *n_events* rows represent a physically diverse set of events that
        all share (approximately) the same Kendall return period T.

        Args:
            T:            Kendall return period.
            n_events:     Number of design events to return.
            n_sim:        Monte Carlo pool size.
            mu:           Mean inter-occurrence time (1 for annual maxima).
            random_state: RNG seed.
            eps_rel:      Relative band half-width around t* (0.05 = ±5 %).

        Returns:
            pd.DataFrame columns: variable labels, ``'C_uv'``, ``'joint_pdf'``,
            ``'t_star'``; sorted by descending joint density.
        """
        rng = np.random.default_rng(random_state)
        kr = self.kendall_return_period(T, mu=mu,
                                        n_sim=min(n_sim // 2, 50_000),
                                        random_state=rng)
        t_star = kr['t_star']
        eps = eps_rel * t_star

        xs, ys = self.sample(n_sim, random_state=rng)
        u = self._u(xs)
        v = self._v(ys)
        c_vals = self._cdf_fn(u, v, self._theta)

        mask = np.abs(c_vals - t_star) < eps
        if mask.sum() < max(n_events, 10):
            eps = np.sort(np.abs(c_vals - t_star))[
                min(n_events * 5, len(c_vals) - 1)]
            mask = np.abs(c_vals - t_star) < eps

        xf, yf, cf = xs[mask], ys[mask], c_vals[mask]
        uf, vf = u[mask], v[mask]

        eps_d = 1e-5
        cu = np.clip(uf + eps_d, 1e-10, 1 - 1e-10)
        cl = np.clip(uf - eps_d, 1e-10, 1 - 1e-10)
        vu = np.clip(vf + eps_d, 1e-10, 1 - 1e-10)
        vl = np.clip(vf - eps_d, 1e-10, 1 - 1e-10)
        c_uv = (self._cdf_fn(cu, vu, self._theta) - self._cdf_fn(cu, vl, self._theta)
              - self._cdf_fn(cl, vu, self._theta) + self._cdf_fn(cl, vl, self._theta)
               ) / (4 * eps_d ** 2)
        pdf = np.clip(c_uv * self._mx[0].pdf(xf, *self._mx[1])
                           * self._my[0].pdf(yf, *self._my[1]), 0, None)

        idx = np.argsort(-pdf)[:n_events]
        return pd.DataFrame({
            self._labels[0]: xf[idx],
            self._labels[1]: yf[idx],
            'C_uv':      cf[idx],
            'joint_pdf': pdf[idx],
            't_star':    t_star,
        })


# ── TrivariateCopula ──────────────────────────────────────────────────────────

class TrivariateCopula:
    """
    Trivariate Archimedean copula for compound-flood return period analysis.

    Fits three marginals and a single copula parameter (exchangeable
    Archimedean family: Gumbel, Clayton or Frank) via the average of the
    three pairwise Kendall τ values.

    Typical application:
        X = peak river discharge Q  (m³/s)
        Y = storm surge / sea level SL  (m)
        Z = catchment rainfall P  (mm)

    The key outputs are:
    * ``joint_exceedance(x0, y0, z0, scenario)`` — scalar probability.
    * ``conditional_contours(z_quantile, T_list)`` — 2-D iso-return-period
      curves in the (X, Y) plane conditional on Z exceeding its α-quantile.
    * ``plot(T_list, z_quantiles)`` — standard trivariate plot grid.

    Args:
        family: ``'gumbel'`` | ``'clayton'`` | ``'frank'``.
    """

    def __init__(self, family="gumbel",
                 marginal_families=("gev", "lognorm", "gamma")):
        self.family            = family
        self.marginal_families = marginal_families
        self._cdf_fn           = _COPULA_CDF[family]
        self._cdf_3d           = _COPULA_CDF_3D[family]
        self._theta            = None
        self._mx = self._my = self._mz = None
        self._labels = ("X", "Y", "Z")
        self._x = self._y = self._z = None

    # ------------------------------------------------------------------
    def fit(self, x, y, z, labels=("X", "Y", "Z")):
        """
        Fit marginals and trivariate copula parameter.

        The copula parameter θ is estimated from the mean pairwise Kendall τ
        (exchangeable copula assumption).

        Returns:
            self
        """
        x, y, z = (np.asarray(a, float) for a in (x, y, z))
        self._x, self._y, self._z = x, y, z
        self._labels = labels

        print(f"Fitting TrivariateCopula [{self.family}]  n={len(x)}")
        for lbl, arr, attr in zip(labels, (x, y, z), ("_mx", "_my", "_mz")):
            print(f"  Marginal {lbl}:")
            setattr(self, attr, _fit_marginal_scipy(arr, self.marginal_families))

        tau_xy = _kendall_tau(x, y)
        tau_xz = _kendall_tau(x, z)
        tau_yz = _kendall_tau(y, z)
        tau_mean = (tau_xy + tau_xz + tau_yz) / 3.0
        self._tau = tau_mean
        self._tau_pairs = {
            f"{labels[0]}–{labels[1]}": tau_xy,
            f"{labels[0]}–{labels[2]}": tau_xz,
            f"{labels[1]}–{labels[2]}": tau_yz,
        }
        self._theta = _tau_to_theta(tau_mean, self.family)
        print(f"  Pairwise τ: XY={tau_xy:.3f}  XZ={tau_xz:.3f}  YZ={tau_yz:.3f}")
        print(f"  Mean τ = {tau_mean:.3f}  →  θ = {self._theta:.3f}")
        return self

    # ------------------------------------------------------------------
    def _u(self, x): d, p = self._mx[0], self._mx[1]; return np.clip(d.cdf(x, *p), 1e-10, 1-1e-10)
    def _v(self, y): d, p = self._my[0], self._my[1]; return np.clip(d.cdf(y, *p), 1e-10, 1-1e-10)
    def _w(self, z): d, p = self._mz[0], self._mz[1]; return np.clip(d.cdf(z, *p), 1e-10, 1-1e-10)
    def _x_ppf(self, u): d, p = self._mx[0], self._mx[1]; return d.ppf(u, *p)
    def _y_ppf(self, v): d, p = self._my[0], self._my[1]; return d.ppf(v, *p)
    def _z_ppf(self, w): d, p = self._mz[0], self._mz[1]; return d.ppf(w, *p)

    # ── Public marginal helpers ───────────────────────────────────────────────
    def marginal_cdf_x(self, x):
        """CDF of the fitted X marginal evaluated at x."""
        return self._u(x)

    def marginal_cdf_y(self, y):
        """CDF of the fitted Y marginal evaluated at y."""
        return self._v(y)

    def marginal_cdf_z(self, z):
        """CDF of the fitted Z marginal evaluated at z."""
        return self._w(z)

    def marginal_ppf_x(self, p):
        """Quantile (inverse CDF) of the fitted X marginal at probability p."""
        return self._x_ppf(p)

    def marginal_ppf_y(self, p):
        """Quantile (inverse CDF) of the fitted Y marginal at probability p."""
        return self._y_ppf(p)

    def marginal_ppf_z(self, p):
        """Quantile (inverse CDF) of the fitted Z marginal at probability p."""
        return self._z_ppf(p)

    def joint_exceedance(self, x0, y0, z0, scenario="OR"):
        """
        P(X>x0, Y>y0, Z>z0) for OR or AND scenarios.

        OR:  P = 1 − C₃(u,v,w)
        AND: P = 1 − u − v − w + C(u,v) + C(u,w) + C(v,w) − C₃(u,v,w)
             (Fréchet–Hoeffding inclusion–exclusion)
        """
        u, v, w = self._u(x0), self._v(y0), self._w(z0)
        c3 = self._cdf_3d(u, v, w, self._theta)
        if scenario == "OR":
            return 1.0 - c3
        if scenario == "AND":
            cxy = self._cdf_fn(u, v, self._theta)
            cxz = self._cdf_fn(u, w, self._theta)
            cyz = self._cdf_fn(v, w, self._theta)
            return np.maximum(1.0 - u - v - w + cxy + cxz + cyz - c3, 0.0)
        raise ValueError("scenario must be 'OR' or 'AND'")

    def return_period(self, x0, y0, z0, scenario="OR"):
        """T = 1 / P(exceedance)."""
        return 1.0 / np.maximum(self.joint_exceedance(x0, y0, z0, scenario), 1e-12)

    # ------------------------------------------------------------------
    def conditional_contours(self, z_quantile, T_list, scenario="AND",
                              n_pts=250):
        """
        2-D **conditional** iso-return-period curves in (X, Y) space, with Z
        conditioned on Z > z₀ = F_Z^{-1}(z_quantile).

        **AND**: T = P(Z>z₀) / P(X>x ∧ Y>y ∧ Z>z₀)
        **OR**:  T = 1 / P(X>x ∨ Y>y | Z>z₀)
                   = 1 / [1 − (C₂(u,v) − C₃(u,v,w₀)) / P(Z>z₀)]

        Both formulae give conditional return periods that are comparable
        across different z_quantile levels.

        Returns:
            dict {T: (x_curve, y_curve)}
        """
        from scipy.optimize import brentq

        z0  = self._z_ppf(z_quantile)
        w0  = self._w(z0)
        pZ  = max(1.0 - w0, 1e-10)   # P(Z > z0)

        contours = {}
        for T in T_list:
            target = 1.0 / T
            u_arr  = np.linspace(1e-4, 1.0 - 1e-4, n_pts)

            xc, yc = [], []
            for u in u_arr:
                def _residual(vv, _u=u):
                    c3 = self._cdf_3d(_u, vv, w0, self._theta)
                    if scenario == "AND":
                        cxy = self._cdf_fn(_u, vv, self._theta)
                        cxz = self._cdf_fn(_u, w0, self._theta)
                        cyz = self._cdf_fn(vv, w0, self._theta)
                        p_and = max(1.0 - _u - vv - w0 + cxy + cxz + cyz - c3, 0.0)
                        return p_and / pZ - target
                    else:
                        # P(X>x OR Y>y | Z>z0) = 1-(C2(u,v)-C3(u,v,w0))/pZ
                        c2 = self._cdf_fn(_u, vv, self._theta)
                        return 1.0 - (c2 - c3) / pZ - target

                try:
                    lo_val = _residual(1e-10)
                    hi_val = _residual(1 - 1e-10)
                    if lo_val * hi_val > 0:
                        continue
                    vv = brentq(_residual, 1e-10, 1 - 1e-10, xtol=1e-8)
                    xc.append(self._x_ppf(u))
                    yc.append(self._y_ppf(vv))
                except Exception:
                    continue
            contours[T] = (np.array(xc), np.array(yc))
        return contours

    # ------------------------------------------------------------------
    def plot(self, T_list=(2, 10, 50, 100),
             z_quantiles=(0.5, 0.8, 0.95),
             scenario="AND",
             figsize=None):
        """
        Grid of 2-D conditional return period contours.

        Rows: different z₀ levels (Z fixed at its α-quantile).
        Columns: AND and OR scenarios side-by-side.

        Each panel shows iso-return-period curves in the (X, Y) plane
        conditional on Z ≥ z₀, plus the observed scatter.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        scenarios = ["AND", "OR"]
        n_q       = len(z_quantiles)
        if figsize is None:
            figsize = (7 * len(scenarios), 5 * n_q)

        fig, axes = plt.subplots(n_q, len(scenarios), figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
        colors = cm.plasma_r(np.linspace(0.15, 0.9, len(T_list)))

        for row, alpha in enumerate(z_quantiles):
            z0_val = self._z_ppf(alpha)
            z_lbl  = f"{self._labels[2]} ≥ {z0_val:.1f}  (α={alpha:.0%})"

            for col, scen in enumerate(scenarios):
                ax   = axes[row][col]
                ctrs = self.conditional_contours(alpha, T_list, scenario=scen)

                for col_c, T in zip(colors, T_list):
                    xc, yc = ctrs[T]
                    if len(xc):
                        ax.plot(xc, yc, color=col_c, lw=1.8, label=f"T={T} yr")

                # Observed scatter — filter to Z ≥ z₀
                if self._x is not None:
                    mask = self._z >= z0_val
                    ax.scatter(self._x[mask], self._y[mask],
                               s=18, alpha=0.6, color="steelblue", zorder=5)
                    ax.scatter(self._x[~mask], self._y[~mask],
                               s=10, alpha=0.15, color="grey", zorder=4)

                ax.set_xlabel(self._labels[0])
                ax.set_ylabel(self._labels[1])
                ax.set_title(f"{scen} | {z_lbl}", fontsize=10)
                ax.grid(alpha=0.3)
                if row == 0 and col == 0:
                    ax.legend(fontsize=8)

        fig.suptitle(
            f"Trivariate compound flood return periods\n"
            f"{self.family.capitalize()} copula  (θ={self._theta:.2f})  —  "
            f"{self._labels[0]} × {self._labels[1]} | {self._labels[2]}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------
    def sample(self, n, random_state=None):
        """
        Draw *n* samples from the fitted trivariate copula.

        Returns:
            ``(x_samples, y_samples, z_samples)`` — arrays of shape (n,).
        """
        rng   = np.random.default_rng(random_state)
        theta = self._theta

        if self.family == "gumbel":
            if theta < 1.01:
                # θ→1 is independence; the stable-distribution sampler degenerates
                # and produces effectively Uniform(0,1) marginals anyway
                u1 = rng.uniform(0.0, 1.0, n)
                u2 = rng.uniform(0.0, 1.0, n)
                u3 = rng.uniform(0.0, 1.0, n)
            else:
                alpha = 1.0 / theta
                U_    = rng.uniform(0.0, np.pi, n)
                W_    = rng.exponential(1.0, n)
                log_V = ((1.0 / alpha) * np.log(np.sin(alpha * U_))
                         + ((1.0 - alpha) / alpha) * np.log(np.sin((1.0 - alpha) * U_))
                         - np.log(np.sin(U_))
                         - ((1.0 - alpha) / alpha) * np.log(W_))
                V  = np.exp(log_V)
                e1 = rng.exponential(1.0, n)
                e2 = rng.exponential(1.0, n)
                e3 = rng.exponential(1.0, n)
                u1 = np.exp(-(e1 / V) ** (1.0 / theta))
                u2 = np.exp(-(e2 / V) ** (1.0 / theta))
                u3 = np.exp(-(e3 / V) ** (1.0 / theta))

        elif self.family == "clayton":
            if theta < 0.05:
                # θ → 0 is independence; gamma(1/θ) degenerates and causes
                # extreme quantile values, so fall back to independent uniforms
                u1 = rng.uniform(0.0, 1.0, n)
                u2 = rng.uniform(0.0, 1.0, n)
                u3 = rng.uniform(0.0, 1.0, n)
            else:
                V  = rng.gamma(1.0 / theta, 1.0, n)
                e1 = rng.exponential(1.0, n)
                e2 = rng.exponential(1.0, n)
                e3 = rng.exponential(1.0, n)
                u1 = (1.0 + e1 / V) ** (-1.0 / theta)
                u2 = (1.0 + e2 / V) ** (-1.0 / theta)
                u3 = (1.0 + e3 / V) ** (-1.0 / theta)

        elif self.family == "frank":
            # Sequential conditional inversion.
            # u1 ~ Unif; u2|u1 via bivariate Frank inversion (analytic);
            # u3|u1,u2 via trivariate Frank inversion (analytic quadratic).
            u1_  = rng.uniform(0.0, 1.0, n)
            t2   = np.clip(rng.uniform(0.0, 1.0, n), 1e-9, 1 - 1e-9)
            t3   = np.clip(rng.uniform(0.0, 1.0, n), 1e-9, 1 - 1e-9)

            e0   = np.exp(-theta)           # scalar
            eu1  = np.exp(-theta * u1_)
            ev2  = ((eu1 * (1.0 - t2) + t2 * e0)
                    / np.maximum(eu1 * (1.0 - t2) + t2, 1e-15))
            u2_  = np.clip(-np.log(np.maximum(ev2, 1e-15)) / theta, 1e-10, 1 - 1e-10)

            # Solve t3·K²·c² + (2t3·B·K − A·S2²)·c + t3·B² = 0
            # c = e^{−θ·u3}−1; K = a·b; S2 = A+K; B = A²
            A      = e0 - 1.0              # < 0 for θ > 0
            a      = eu1 - 1.0
            b      = np.exp(-theta * u2_) - 1.0
            K      = a * b                 # > 0
            S2     = A + K
            B      = A ** 2

            two_aq = 2.0 * t3 * K ** 2
            b_q    = 2.0 * t3 * B * K - A * S2 ** 2
            inner  = np.maximum(S2 ** 2 - 4.0 * t3 * A * K, 0.0)
            sqrt_d = (-A) * np.abs(S2) * np.sqrt(inner)

            degen  = np.abs(K) < 1e-8
            safe   = np.where(two_aq > 1e-20, two_aq, 1e-20)
            c3     = np.where(degen, A * t3, (-b_q + sqrt_d) / safe)
            # c = exp(-θ·u3) - 1 must lie in (A, 0) for θ>0 (A<0)
            # or in (0, A) for θ<0 (A>0); bounds are swapped depending on sign of θ
            if theta > 0:
                c3 = np.clip(c3, A + 1e-10, -1e-10)
            else:
                c3 = np.clip(c3, 1e-10, A - 1e-10)
            u3_    = np.clip(-np.log(np.maximum(c3 + 1.0, 1e-15)) / theta,
                             1e-10, 1 - 1e-10)
            u1, u2, u3 = u1_, u2_, u3_

        else:
            raise ValueError(f"Unknown family: {self.family!r}")

        u1 = np.clip(u1, 1e-10, 1 - 1e-10)
        u2 = np.clip(u2, 1e-10, 1 - 1e-10)
        u3 = np.clip(u3, 1e-10, 1 - 1e-10)
        return self._x_ppf(u1), self._y_ppf(u2), self._z_ppf(u3)

    # ------------------------------------------------------------------
    def plot_joint(self, z_quantiles=(0.50, 0.80, 0.95),
                   T_list=(2, 10, 50, 100, 500),
                   n_synthetic=2000, figsize=(6, 6),
                   colors=("indianred", "b"), n_grid=200):
        """
        One figure per z-quantile: conditional joint return period plot.

        Each figure mirrors the ``BivariateCopula.plot_joint`` GridSpec(4,4)
        layout, conditioning all contours on Z ≥ z₀ = F_Z^{-1}(α).

        * **Main panel**: grey synthetic scatter filtered by Z ≥ z₀, black
          observed scatter filtered, AND/OR conditional iso-return-period
          contours, points coloured by conditional joint PDF, MaxProb 'x'.
        * **Bottom panel** (xMarg): marginal PDF of X with T-year annotations.
        * **Left panel** (yMarg): marginal PDF of Y with T-year annotations.
        * **Legend**: separate axes to the right.

        Returns:
            list of ``(fig, (main_ax, xMarg_ax, yMarg_ax))`` — one per α.
        """
        import matplotlib.pyplot as plt

        Ts = np.asarray(sorted(T_list), dtype=float)

        if n_synthetic > 0:
            xs_all, ys_all, zs_all = self.sample(n_synthetic)
        else:
            xs_all = ys_all = zs_all = None

        results = []
        for alpha in z_quantiles:
            z0 = self._z_ppf(alpha)
            w0 = float(self._w(z0))

            if self._x is not None:
                dx = (self._x.max() - self._x.min()) * 0.15
                dy = (self._y.max() - self._y.min()) * 0.15
                extent = [self._x.min() - dx, self._x.max() + dx,
                          self._y.min() - dy, self._y.max() + dy]
            else:
                u_e = np.linspace(1e-3, 1 - 1e-3, n_grid)
                extent = [float(self._x_ppf(u_e).min()),
                          float(self._x_ppf(u_e).max()),
                          float(self._y_ppf(u_e).min()),
                          float(self._y_ppf(u_e).max())]

            fig  = plt.figure(figsize=figsize)
            grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2, figure=fig)
            main  = fig.add_subplot(grid[:-1, 1:])
            yMarg = fig.add_subplot(grid[:-1, 0])
            xMarg = fig.add_subplot(grid[-1,  1:])
            main.set_xlim(extent[:2])
            main.set_ylim(extent[2:])

            if xs_all is not None:
                mask_s = zs_all >= z0
                main.scatter(xs_all[~mask_s], ys_all[~mask_s],
                             marker='.', c='lightgrey', s=1, alpha=0.3,
                             rasterized=True, zorder=2)
                main.scatter(xs_all[mask_s], ys_all[mask_s],
                             marker='.', c='grey', s=1, alpha=0.5,
                             rasterized=True, zorder=2, label='aleatorios')

            if self._x is not None:
                mask_o = self._z >= z0
                main.scatter(self._x[~mask_o], self._y[~mask_o],
                             marker='.', c='lightgrey', s=12, alpha=0.3, zorder=3)
                main.scatter(self._x[mask_o], self._y[mask_o],
                             marker='.', c='k', s=12, zorder=3, label='observados')

            # eps for 3-D copula density ∂³C/∂u∂v∂w: use 1e-4 so the
            # denominator 8·eps³ ≈ 8e-12 stays well above float64 noise.
            eps3 = 1e-4
            w0p  = min(w0 + eps3, 1 - 1e-10)
            w0m  = max(w0 - eps3, 1e-10)
            for scn, cmap, c_line in zip(['AND', 'OR'], ['Reds', 'Blues'], colors):
                ctrs = self.conditional_contours(alpha, Ts.tolist(), scenario=scn)
                for T in Ts:
                    xc, yc = ctrs[T]
                    if len(xc) < 2:
                        continue
                    main.plot(xc, yc, color=c_line, lw=0.8, alpha=0.3, zorder=4)
                    u_c  = np.clip(self._u(xc), 1e-10, 1 - 1e-10)
                    v_c  = np.clip(self._v(yc), 1e-10, 1 - 1e-10)
                    u_hi = np.clip(u_c + eps3, 1e-10, 1 - 1e-10)
                    u_lo = np.clip(u_c - eps3, 1e-10, 1 - 1e-10)
                    v_hi = np.clip(v_c + eps3, 1e-10, 1 - 1e-10)
                    v_lo = np.clip(v_c - eps3, 1e-10, 1 - 1e-10)
                    # Trivariate copula density ∂³C/∂u∂v∂w via 8-term finite diff
                    c3_uvw = (
                        self._cdf_3d(u_hi, v_hi, w0p, self._theta)
                      - self._cdf_3d(u_hi, v_hi, w0m, self._theta)
                      - self._cdf_3d(u_hi, v_lo, w0p, self._theta)
                      + self._cdf_3d(u_hi, v_lo, w0m, self._theta)
                      - self._cdf_3d(u_lo, v_hi, w0p, self._theta)
                      + self._cdf_3d(u_lo, v_hi, w0m, self._theta)
                      + self._cdf_3d(u_lo, v_lo, w0p, self._theta)
                      - self._cdf_3d(u_lo, v_lo, w0m, self._theta)
                    ) / (8 * eps3 ** 3)
                    pdf   = np.clip(
                        c3_uvw
                        * self._mx[0].pdf(xc, *self._mx[1])
                        * self._my[0].pdf(yc, *self._my[1]),
                        0, None)
                    main.scatter(xc, yc, c=pdf, cmap=cmap, s=0.25, zorder=5)
                    mp_i = int(np.argmax(pdf))
                    main.scatter(xc[mp_i], yc[mp_i], marker='x', color=c_line,
                                 s=80, linewidths=1.5, zorder=7)
                main.plot([-1e9], [-1e9], c=c_line, label=scn)
                main.scatter([-1e9], [-1e9], marker='x', c=c_line,
                             s=80, linewidths=1.5, label=f'MaxProb {scn}')

            main.set(xlim=extent[:2], ylim=extent[2:])
            main.set_xticklabels([])
            main.set_yticklabels([])
            z_lbl = self._labels[2] if len(self._labels) > 2 else 'Z'
            main.set_title(f'α={alpha:.0%}  |  {z_lbl} ≥ {z0:.1f}', fontsize=10)

            v0x    = np.linspace(extent[0], extent[1], 400)
            pdf_x  = self._mx[0].pdf(v0x, *self._mx[1])
            ymax_x = float(np.nanmax(pdf_x))
            if ymax_x > 0:
                xMarg.fill_between(v0x, pdf_x, color='olive', alpha=0.25)
                xMarg.set(xlim=(extent[0], extent[1]), ylim=(ymax_x, 0))
                ppfs_x = np.clip(self._mx[0].ppf(1 - 1 / Ts, *self._mx[1]),
                                 extent[0], extent[1])
                for T, ppf in zip(Ts, ppfs_x):
                    xMarg.annotate(f'T{int(T)}',
                                   xy=(ppf, 0), xytext=(ppf, ymax_x / 3),
                                   color='olive', fontsize=11,
                                   ha='center', va='top', rotation=90,
                                   arrowprops=dict(arrowstyle='->', color='olive', lw=1))
                xMarg.set_xlabel(self._labels[0], fontsize=13)
                yticks = np.round(np.linspace(0, ymax_x, 3), 2)
                xMarg.set_yticks(yticks)
                xMarg.set_yticklabels(yticks)

            v1y    = np.linspace(extent[2], extent[3], 400)
            pdf_y  = self._my[0].pdf(v1y, *self._my[1])
            xmax_y = float(np.nanmax(pdf_y))
            if xmax_y > 0:
                yMarg.fill_between(pdf_y, v1y, color='olive', alpha=0.25,
                                   label='Distribución (PDF)')
                yMarg.set(xlim=(xmax_y, 0), ylim=(extent[2], extent[3]))
                ppfs_y = np.clip(self._my[0].ppf(1 - 1 / Ts, *self._my[1]),
                                 extent[2], extent[3])
                for T, ppf in zip(Ts, ppfs_y):
                    yMarg.annotate(f'T{int(T)}',
                                   xy=(0, ppf), xytext=(xmax_y / 3, ppf),
                                   color='olive', fontsize=11,
                                   ha='right', va='center',
                                   arrowprops=dict(arrowstyle='->', color='olive', lw=1))
                yMarg.set_ylabel(self._labels[1], fontsize=13)
                xticks = np.round(np.linspace(0, xmax_y, 3), 2)
                yMarg.set_xticks(xticks)
                yMarg.set_xticklabels(xticks)

            lgnd_ax = fig.add_axes([0.97, 0.3, 0.08, 0.4])
            lgnd_ax.axis('off')
            hndls, lbls = main.get_legend_handles_labels()
            if len(hndls) >= 6:
                order = [2, 3, 0, 4, 1, 5]
                hndls = [hndls[i] for i in order]
                lbls  = [lbls[i]  for i in order]
            hndls1, lbls1 = yMarg.get_legend_handles_labels()
            lgnd_ax.legend(hndls + hndls1, lbls + lbls1,
                           ncol=1, loc='center left', fontsize=9)

            results.append((fig, (main, xMarg, yMarg)))

        return results

    # ------------------------------------------------------------------
    def kendall_function(self, t_values=None, n_sim=50_000, random_state=None):
        """
        Monte Carlo estimate of the trivariate Kendall distribution
        K₃(t) = P[C₃(U, V, W) ≤ t].

        Args:
            t_values:     Grid of t ∈ (0, 1); default 200 points.
            n_sim:        Monte Carlo pool size.
            random_state: RNG seed.

        Returns:
            ``(t_arr, K_arr)`` — evaluation points and K₃(t) values.
        """
        rng = np.random.default_rng(random_state)
        xs, ys, zs = self.sample(n_sim, random_state=rng)
        c_vals = self._cdf_3d(self._u(xs), self._v(ys), self._w(zs), self._theta)
        if t_values is None:
            t_values = np.linspace(0.01, 0.99, 200)
        t_arr = np.asarray(t_values, float)
        return t_arr, np.array([np.mean(c_vals <= t) for t in t_arr])

    # ------------------------------------------------------------------
    def kendall_return_period(self, T, mu=1.0, n_sim=50_000, random_state=None):
        """
        Kendall return period for the trivariate copula.

        Inverts K₃(t*) = 1 − μ/T.  The Kendall level t* defines a critical
        layer C₃(u,v,w) = t* in the unit cube, which can be used to sample
        a physically diverse design event ensemble via ``design_storm_ensemble``.

        Args:
            T:            Target return period.
            mu:           Mean inter-occurrence time (1 for annual maxima).
            n_sim:        Monte Carlo sample size.
            random_state: RNG seed.

        Returns:
            dict with keys ``t_star``, ``K_t``, ``T_K``.
        """
        from scipy.interpolate import interp1d

        target_K = np.clip(1.0 - mu / T, 0.001, 0.999)
        t_arr, K_arr = self.kendall_function(n_sim=n_sim, random_state=random_state)
        f_inv = interp1d(K_arr, t_arr, kind='linear',
                         bounds_error=False, fill_value=(t_arr[0], t_arr[-1]))
        t_star = float(f_inv(target_K))
        return {'t_star': t_star, 'K_t': float(target_K), 'T_K': T}

    # ------------------------------------------------------------------
    def design_storm_ensemble(self, T, n_events=50, n_sim=100_000,
                              mu=1.0, random_state=None, eps_rel=0.05):
        """
        Ensemble of trivariate design events on the Kendall critical layer T_K = T.

        Draws *n_sim* samples from the trivariate copula, selects those within
        an ε-band of the critical Kendall level t*, ranks them by trivariate
        joint density, and returns the top *n_events*.

        Args:
            T:            Kendall return period.
            n_events:     Number of design events to return.
            n_sim:        Monte Carlo pool size.
            mu:           Mean inter-occurrence time (1 for annual maxima).
            random_state: RNG seed.
            eps_rel:      Relative band half-width around t* (0.05 = ±5 %).

        Returns:
            pd.DataFrame columns: variable labels, ``'C3_uvw'``, ``'joint_pdf'``,
            ``'t_star'``; sorted by descending joint density.
        """
        rng = np.random.default_rng(random_state)
        kr = self.kendall_return_period(T, mu=mu,
                                        n_sim=min(n_sim // 2, 50_000),
                                        random_state=rng)
        t_star = kr['t_star']
        eps = eps_rel * t_star

        xs, ys, zs = self.sample(n_sim, random_state=rng)
        u = self._u(xs)
        v = self._v(ys)
        w = self._w(zs)
        c_vals = self._cdf_3d(u, v, w, self._theta)

        mask = np.abs(c_vals - t_star) < eps
        if mask.sum() < max(n_events, 10):
            eps = np.sort(np.abs(c_vals - t_star))[
                min(n_events * 5, len(c_vals) - 1)]
            mask = np.abs(c_vals - t_star) < eps

        xf, yf, zf, cf = xs[mask], ys[mask], zs[mask], c_vals[mask]
        uf, vf, wf = u[mask], v[mask], w[mask]

        # 3D copula density ∂³C/∂u∂v∂w via 8-term finite differences
        eps3 = 1e-4
        uh = np.clip(uf + eps3, 1e-10, 1 - 1e-10)
        ul = np.clip(uf - eps3, 1e-10, 1 - 1e-10)
        vh = np.clip(vf + eps3, 1e-10, 1 - 1e-10)
        vl = np.clip(vf - eps3, 1e-10, 1 - 1e-10)
        wh = np.clip(wf + eps3, 1e-10, 1 - 1e-10)
        wl = np.clip(wf - eps3, 1e-10, 1 - 1e-10)
        c3_uvw = (
            self._cdf_3d(uh, vh, wh, self._theta) - self._cdf_3d(uh, vh, wl, self._theta)
          - self._cdf_3d(uh, vl, wh, self._theta) + self._cdf_3d(uh, vl, wl, self._theta)
          - self._cdf_3d(ul, vh, wh, self._theta) + self._cdf_3d(ul, vh, wl, self._theta)
          + self._cdf_3d(ul, vl, wh, self._theta) - self._cdf_3d(ul, vl, wl, self._theta)
        ) / (8 * eps3 ** 3)
        pdf = np.clip(
            c3_uvw
            * self._mx[0].pdf(xf, *self._mx[1])
            * self._my[0].pdf(yf, *self._my[1])
            * self._mz[0].pdf(zf, *self._mz[1]),
            0, None)

        idx = np.argsort(-pdf)[:n_events]
        return pd.DataFrame({
            self._labels[0]: xf[idx],
            self._labels[1]: yf[idx],
            self._labels[2]: zf[idx],
            'C3_uvw':    cf[idx],
            'joint_pdf': pdf[idx],
            't_star':    t_star,
        })


# ============================================================================
# VineCopula — d-dimensional flexible vine copula (pyvinecopulib)
# ============================================================================

def _require_pvc():
    try:
        import pyvinecopulib as pv
        return pv
    except ImportError as exc:
        raise ImportError(
            "pyvinecopulib is required for VineCopula.\n"
            "Install it with: pip install pyvinecopulib"
        ) from exc


class VineCopula:
    """
    Flexible d-dimensional vine copula using pyvinecopulib.

    Implements R-vine pair-copula constructions (PCC) fitted by maximum
    pseudo-likelihood with automatic structure and family selection.

    Supports:

    * ``fit`` — marginals (scipy AIC) + R-vine structure auto-selection.
    * ``sample`` — synthetic sampling in physical space.
    * ``jcdf`` — joint CDF evaluation.
    * ``kendall_function`` / ``kendall_return_period`` — Kendall return period
      analysis following Salvadori et al. (2011, 2016).
    * ``most_probable_event`` — MPDE on the Kendall critical layer.
    * ``design_storm_ensemble`` — ensemble of design events ranked by joint density
      (Urrea Méndez et al. 2026, J. Hydrology).

    Parameters
    ----------
    family_set : list of str or None
        Pair-copula families considered during selection.
        Supported names: ``'gaussian'``, ``'student'``, ``'gumbel'``,
        ``'clayton'``, ``'frank'``, ``'joe'``, ``'bb1'``, ``'bb8'``.
        Defaults to ``['gaussian','student','gumbel','clayton','frank','joe','bb1']``.
    marginal_families : tuple of str
        Candidates for each univariate marginal (AIC selection).

    References
    ----------
    Nagler & Vatter (2023). pyvinecopulib.
    Urrea Méndez et al. (2026). Multivariate Design Storms Using Vine
        Copulas and GPR. J. Hydrology.
    Salvadori et al. (2011). On the return period and design in a
        multivariate framework. Hydrology & Earth System Sciences.
    """

    _FAMILY_MAP = None   # populated lazily to avoid import at module load

    def __init__(self, family_set=None,
                 marginal_families=("gev", "lognorm", "gamma")):
        self.family_set       = family_set
        self.marginal_families = marginal_families
        self._vine            = None
        self._marginals       = []   # list of (dist, params, name, aic)
        self._labels          = []
        self._data            = None
        self._d               = None

    # ------------------------------------------------------------------
    @staticmethod
    def _pv_family_set(pv, names):
        mapping = {
            'gaussian': pv.BicopFamily.gaussian,
            'student':  pv.BicopFamily.student,
            'gumbel':   pv.BicopFamily.gumbel,
            'clayton':  pv.BicopFamily.clayton,
            'frank':    pv.BicopFamily.frank,
            'joe':      pv.BicopFamily.joe,
            'bb1':      pv.BicopFamily.bb1,
            'bb8':      pv.BicopFamily.bb8,
        }
        return [mapping[n] for n in names if n in mapping]

    # ------------------------------------------------------------------
    def fit(self, data, labels=None):
        """
        Fit marginals and R-vine structure to observed data.

        Args:
            data:   array-like (n_obs, d) or pd.DataFrame.
            labels: list of variable names; defaults to column names or X0…Xd-1.

        Returns:
            self
        """
        pv = _require_pvc()

        if isinstance(data, pd.DataFrame):
            if labels is None:
                labels = list(data.columns)
            data = data.values
        data = np.asarray(data, float)
        n, d = data.shape
        self._d    = d
        self._data = data
        self._labels = labels if labels else [f"X{k}" for k in range(d)]

        print(f"Fitting VineCopula  d={d}  n={n}")

        # 1. Marginals
        pseudo = np.empty((n, d))
        self._marginals = []
        for k in range(d):
            print(f"  Marginal {self._labels[k]}:")
            mx = _fit_marginal_scipy(data[:, k], self.marginal_families)
            self._marginals.append(mx)
            pseudo[:, k] = np.clip(mx[0].cdf(data[:, k], *mx[1]), 1e-6, 1 - 1e-6)

        # 2. Vine structure + pair-copula families
        if self.family_set is None:
            fset = self._pv_family_set(
                pv, ['gaussian', 'student', 'gumbel', 'clayton', 'frank', 'joe', 'bb1'])
        else:
            fset = self._pv_family_set(pv, self.family_set)

        controls = pv.FitControlsVinecop(family_set=fset)
        self._vine = pv.Vinecop.from_data(np.asfortranarray(pseudo), controls=controls)
        try:
            summary = repr(self._vine)[:120]
        except Exception:
            summary = f"d={d} vine"
        print(f"  Fitted R-vine: {summary}")
        return self

    # ------------------------------------------------------------------
    def _to_uniform(self, data):
        data = np.asarray(data, float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        u = np.empty_like(data)
        for k, mx in enumerate(self._marginals):
            u[:, k] = np.clip(mx[0].cdf(data[:, k], *mx[1]), 1e-6, 1 - 1e-6)
        return u

    def _from_uniform(self, u):
        out = np.empty_like(u)
        for k, mx in enumerate(self._marginals):
            out[:, k] = mx[0].ppf(u[:, k], *mx[1])
        return out

    def _simulate(self, n, rng=None):
        """Draw n uniform samples from the vine with optional RNG seeding."""
        if rng is not None:
            seeds = [int(s) for s in rng.integers(0, 2 ** 31, size=4)]
            try:
                return self._vine.simulate(n, seeds=seeds)
            except TypeError:
                pass
        return self._vine.simulate(n)

    # ------------------------------------------------------------------
    def sample(self, n, random_state=None):
        """
        Draw *n* synthetic samples in physical space.

        Returns:
            np.ndarray of shape ``(n, d)``.
        """
        rng = np.random.default_rng(random_state)
        u = self._simulate(n, rng)
        return self._from_uniform(u)

    # ------------------------------------------------------------------
    def jcdf(self, data):
        """
        Joint CDF at rows of *data* (physical space).

        Args:
            data: array (n, d).

        Returns:
            1-D array of length n.
        """
        u = self._to_uniform(np.asarray(data, float))
        return self._vine.cdf(u)

    # ------------------------------------------------------------------
    def kendall_function(self, t_values=None, n_sim=20_000, random_state=None):
        """
        Monte Carlo estimate of the d-variate Kendall function
        K_d(t) = P[C(U) ≤ t].

        Note: for large *d*, vine CDF evaluation can be slow; prefer
        n_sim ≤ 20 000 for d ≥ 5.

        Args:
            t_values:     Grid of t ∈ (0, 1); default 200 points.
            n_sim:        Monte Carlo pool size.
            random_state: RNG seed.

        Returns:
            ``(t_arr, K_arr)`` — evaluation points and K_d(t) values.
        """
        rng = np.random.default_rng(random_state)
        u = self._simulate(n_sim, rng)
        c_vals = self._vine.cdf(u)
        if t_values is None:
            t_values = np.linspace(0.01, 0.99, 200)
        t_arr = np.asarray(t_values, float)
        return t_arr, np.array([np.mean(c_vals <= t) for t in t_arr])

    # ------------------------------------------------------------------
    def kendall_return_period(self, T, mu=1.0, n_sim=20_000, random_state=None):
        """
        Critical Kendall level for return period T.

        K_d(t*) = 1 − μ/T  ⟹  T_K = μ / (1 − K_d(t*))

        Args:
            T:            Target return period.
            mu:           Mean inter-occurrence time (1 for annual maxima).
            n_sim:        Monte Carlo size for K estimation.
            random_state: RNG seed.

        Returns:
            dict with keys ``t_star``, ``K_t``, ``T_K``.
        """
        from scipy.interpolate import interp1d

        target_K = np.clip(1.0 - mu / T, 0.001, 0.999)
        t_arr, K_arr = self.kendall_function(n_sim=n_sim, random_state=random_state)
        f_inv = interp1d(K_arr, t_arr, kind='linear',
                         bounds_error=False, fill_value=(t_arr[0], t_arr[-1]))
        t_star = float(f_inv(target_K))
        return {'t_star': t_star, 'K_t': float(target_K), 'T_K': T}

    # ------------------------------------------------------------------
    def most_probable_event(self, T, mu=1.0, n_sim=50_000,
                            random_state=None, eps_rel=0.05):
        """
        Most Probable Design Event (MPDE) on the Kendall critical layer T_K = T.

        Returns:
            1-D array of shape (d,) in physical variable space.
        """
        ens = self.design_storm_ensemble(T, n_events=1, n_sim=n_sim,
                                         mu=mu, random_state=random_state,
                                         eps_rel=eps_rel)
        return ens[self._labels].values[0]

    # ------------------------------------------------------------------
    def design_storm_ensemble(self, T, n_events=50, n_sim=50_000,
                              mu=1.0, random_state=None, eps_rel=0.05):
        """
        Ensemble of d-dimensional design events on the Kendall critical layer.

        Draws *n_sim* events from the vine copula, selects those within an
        ε-band of the critical Kendall level t*, and ranks them by joint
        density f(x) = c(F(x)) × Π fk(xk).  The top *n_events* rows cover
        the critical layer with physical diversity (Urrea Méndez et al. 2026).

        Args:
            T:            Kendall return period.
            n_events:     Number of design events to return.
            n_sim:        Monte Carlo pool size.
            mu:           Mean inter-occurrence time (1 for annual maxima).
            random_state: RNG seed.
            eps_rel:      Relative band half-width around t* (0.05 = ±5 %).

        Returns:
            pd.DataFrame: one column per variable, plus ``'C_u'``,
            ``'log_joint_pdf'``, ``'t_star'``; sorted by descending joint density.
        """
        rng = np.random.default_rng(random_state)

        kr = self.kendall_return_period(T, mu=mu,
                                        n_sim=min(n_sim // 2, 20_000),
                                        random_state=rng)
        t_star = kr['t_star']
        eps = eps_rel * t_star

        u_all = self._simulate(n_sim, rng)
        c_vals = self._vine.cdf(u_all)

        mask = np.abs(c_vals - t_star) < eps
        if mask.sum() < max(n_events, 10):
            eps = np.sort(np.abs(c_vals - t_star))[
                min(n_events * 5, len(c_vals) - 1)]
            mask = np.abs(c_vals - t_star) < eps

        u_f  = u_all[mask]
        c_f  = c_vals[mask]
        x_f  = self._from_uniform(u_f)

        # Joint log-density = log copula density + Σ log marginal densities
        cop_dens = self._vine.pdf(u_f)
        log_joint = np.log(np.maximum(cop_dens, 1e-300))
        for k, mx in enumerate(self._marginals):
            log_joint += np.log(np.maximum(mx[0].pdf(x_f[:, k], *mx[1]), 1e-300))

        idx = np.argsort(-log_joint)[:n_events]
        df = pd.DataFrame(x_f[idx], columns=self._labels)
        df['C_u']           = c_f[idx]
        df['log_joint_pdf'] = log_joint[idx]
        df['t_star']        = t_star
        return df

    # ------------------------------------------------------------------
    def plot_matrix(self, n_synthetic=2000, figsize=None, random_state=None):
        """
        Scatter matrix comparing observed (black) vs synthetic (grey) in
        physical space.

        Upper triangle: scatter plots.
        Diagonal: marginal histograms.

        Returns:
            ``(fig, axes)``
        """
        import matplotlib.pyplot as plt

        d = self._d
        if figsize is None:
            figsize = (3 * d, 3 * d)

        synth = self.sample(n_synthetic, random_state=random_state) if n_synthetic > 0 else None

        fig, axes = plt.subplots(d, d, figsize=figsize)
        for i in range(d):
            for j in range(d):
                ax = axes[i][j]
                if i == j:
                    if self._data is not None:
                        ax.hist(self._data[:, i], bins=20, color='steelblue',
                                alpha=0.6, density=True)
                    ax.set_xlabel(self._labels[i], fontsize=8)
                elif i < j:
                    if synth is not None:
                        ax.scatter(synth[:, j], synth[:, i], s=1, alpha=0.3,
                                   c='grey', rasterized=True)
                    if self._data is not None:
                        ax.scatter(self._data[:, j], self._data[:, i],
                                   s=15, c='k', zorder=5)
                else:
                    ax.axis('off')
                ax.tick_params(labelsize=7)

        fig.suptitle(f"VineCopula scatter matrix  (d={d})", fontsize=11)
        fig.tight_layout()
        return fig, axes


# ---------------------------------------------------------------------------
# GaussianCopulaSampler — scipy-only, parameterised correlation
# ---------------------------------------------------------------------------

class GaussianCopulaSampler:
    """Gaussian copula sampler with scipy marginals and a specified correlation.

    Unlike :class:`FloodEventCopula`, which fits the copula from observed data
    using openturns, this class accepts pre-specified marginal distributions
    and a correlation matrix, making it suitable for uncertainty-propagation
    studies where the correlation structure is prescribed (e.g., sensitivity
    analyses with equi-correlation between roughness classes).

    The sampling algorithm follows the standard Gaussian copula procedure:

    1. Draw :math:`\\mathbf{Z} \\sim \\mathcal{N}(\\mathbf{0}, \\boldsymbol{\\Sigma})`.
    2. Convert to uniform margins: :math:`U_i = \\Phi(Z_i)`.
    3. Apply marginal quantile functions: :math:`X_i = F_i^{-1}(U_i)`.

    Parameters
    ----------
    marginals : list of scipy.stats frozen distributions
        One fitted marginal distribution per variable, in order.
        Must expose ``.ppf(u)`` and ``.cdf(x)``.
    names : list of str
        Variable names — used as column labels in the returned DataFrame.

    Examples
    --------
    >>> from scipy.stats import lognorm, gamma
    >>> from pyhydra.climate.spatial_analysis.copulas import GaussianCopulaSampler
    >>> marginals = [lognorm(s=0.3, scale=0.05), gamma(a=3, scale=0.015)]
    >>> sampler = GaussianCopulaSampler(marginals, names=['class_A', 'class_B'])
    >>> df = sampler.sample(1000, rho=0.7, seed=42)
    >>> df.corr()          # empirical Pearson correlation ≈ 0.7
    """

    def __init__(self, marginals, names):
        if len(marginals) != len(names):
            raise ValueError("marginals and names must have the same length.")
        self._marginals = list(marginals)
        self._names     = list(names)
        self._k         = len(marginals)

    @classmethod
    def from_data(cls, data: "pd.DataFrame",
                  families: tuple = ("norm", "lognorm", "gamma")):
        """Fit marginals from data columns by KS test and return a sampler.

        Args:
            data: DataFrame; each column is one variable.
            families: Candidate scipy distribution names for KS selection.

        Returns:
            Fitted :class:`GaussianCopulaSampler`.
        """
        from scipy import stats as _stats

        marginals, names = [], list(data.columns)
        for col in names:
            x = data[col].dropna().values
            best_p, best_dist = -1.0, None
            for fname in families:
                dist_cls = getattr(_stats, fname)
                params   = dist_cls.fit(x)
                _, p     = _stats.kstest(x, fname, args=params)
                if p > best_p:
                    best_p = p
                    best_dist = dist_cls(*params[:-2],
                                        loc=params[-2], scale=params[-1])
            marginals.append(best_dist)
        return cls(marginals, names)

    def _build_sigma(self, rho_or_matrix):
        if np.isscalar(rho_or_matrix):
            rho = float(rho_or_matrix)
            if not (0.0 <= rho <= 1.0):
                raise ValueError("Scalar rho must be in [0, 1].")
            return rho * np.ones((self._k, self._k)) + (1.0 - rho) * np.eye(self._k)
        sigma = np.asarray(rho_or_matrix, dtype=float)
        if sigma.shape != (self._k, self._k):
            raise ValueError(f"Correlation matrix must be ({self._k}, {self._k}).")
        return sigma

    def sample(self, n: int, rho: "float | np.ndarray" = 0.0,
               seed: "int | None" = None) -> "pd.DataFrame":
        """Draw *n* samples from the Gaussian copula.

        Args:
            n:    Number of samples.
            rho:  Equi-correlation scalar ∈ [0, 1] **or** full (k×k)
                  correlation matrix.  Default 0 (independent).
            seed: Random seed for reproducibility.

        Returns:
            DataFrame of shape (n, k) with column names from *names*.
        """
        from scipy.stats import norm as _norm

        rng   = np.random.default_rng(seed)
        Sigma = self._build_sigma(rho)
        Z     = rng.multivariate_normal(np.zeros(self._k), Sigma, size=n)
        U     = _norm.cdf(Z)

        samples = {}
        for j, (name, dist) in enumerate(zip(self._names, self._marginals)):
            x = dist.ppf(U[:, j])
            x = np.clip(x, 1e-9, None)
            samples[name] = x

        return pd.DataFrame(samples)

    def cv_nbar(self, weights: "dict | np.ndarray", rho: float,
                n: int = 20_000, seed: int = 0) -> float:
        """Estimate CV of the area-weighted mean across variables.

        Args:
            weights: Dict {name: area_fraction} or 1-D array aligned to *names*.
            rho:     Equi-correlation coefficient.
            n:       Number of Monte Carlo draws for estimation.
            seed:    Random seed.

        Returns:
            CV in percent.
        """
        if isinstance(weights, dict):
            w = np.array([weights[name] for name in self._names])
        else:
            w = np.asarray(weights)
        w = w / w.sum()
        df   = self.sample(n, rho=rho, seed=seed)
        nbar = df.values @ w
        return 100.0 * nbar.std() / nbar.mean()

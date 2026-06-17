"""
Flood event selection pipeline for hybrid statistical-hydraulic downscaling.

Implements the full characterisation → classification → synthetic generation
workflow described in Navas et al. (2025):

1. Extract flood events from a continuous discharge series using a dual-threshold
   approach with inflection-point refinement.
2. Characterise each event by (Qmax, Qmed, Duration, shape_type) via PCA + K-means
   (``HydrographClassifier``).
3. Fit marginal distributions (best-BIC from a catalogue of scipy distributions)
   to Qmax, Qmed and Duration.
4. Fit a Gaussian copula to the probability-integral-transformed observations.
5. Sample a large synthetic ensemble from the copula + inverse marginals.
6. Visualise the event cloud and marginal fits.

The synthetic ensemble is then passed to ``HydrographReconstructor`` and
``FloodMapInterpolator`` for hydraulic reconstruction and return-period maps.

References
----------
Solari S. et al. (2017). WRR — unified POT statistical model.
Navas Fernández S. et al. (2025). Hybrid downscaling for fluvial flood hazard.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from pyhydra.climate.hybrid_downscaling.classification import HydrographClassifier

# ---------------------------------------------------------------------------
# Marginal distribution catalogue (scipy.stats continuous)
# ---------------------------------------------------------------------------

_MARGINAL_CANDIDATES = [
    stats.genextreme,   # GEV (scipy alias: genextreme, not gev)
    stats.genpareto,
    stats.lognorm,
    stats.gamma,
    stats.expon,
    stats.weibull_min,
    stats.pearson3,
    stats.norm,
    stats.gumbel_r,
]


def _best_marginal_bic(data: np.ndarray):
    """Fit all candidate distributions and return the one with lowest BIC.

    Returns
    -------
    best_dist : frozen scipy.stats distribution
    best_name : str
    best_bic  : float
    """
    best_bic  = np.inf
    best_dist = None
    best_name = ""

    for dist_cls in _MARGINAL_CANDIDATES:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist_cls.fit(data)
            ll = np.sum(dist_cls.logpdf(data, *params))
            k  = len(params)
            bic = k * np.log(len(data)) - 2 * ll
            if bic < best_bic:
                best_bic  = bic
                best_dist = dist_cls(*params)
                best_name = dist_cls.name
        except Exception:
            continue

    return best_dist, best_name, best_bic


# ---------------------------------------------------------------------------
# Gaussian copula helpers
# ---------------------------------------------------------------------------

def _to_uniform(data: np.ndarray, dist) -> np.ndarray:
    """Map *data* to [0,1] using the fitted marginal CDF."""
    u = dist.cdf(data)
    # Clip away exact 0/1 to avoid infinite normal quantiles
    return np.clip(u, 1e-6, 1 - 1e-6)


def _fit_gaussian_copula(U: np.ndarray) -> np.ndarray:
    """Fit a Gaussian copula correlation matrix from uniform samples U.

    Args:
        U : (n_samples × d) array of uniform marginals.

    Returns:
        corr : (d × d) Pearson correlation matrix in the normal space.
    """
    Z = stats.norm.ppf(U)   # transform to normal scores
    return np.corrcoef(Z, rowvar=False)


def _sample_gaussian_copula(corr: np.ndarray, n_samples: int,
                             marginals: list, rng=None) -> pd.DataFrame:
    """Sample from Gaussian copula and back-transform through marginals.

    Args:
        corr      : (d × d) correlation matrix (from _fit_gaussian_copula).
        n_samples : Number of synthetic observations to generate.
        marginals : List of frozen scipy.stats distributions (one per column).
        rng       : Optional numpy random generator for reproducibility.

    Returns:
        DataFrame with columns matching len(marginals).
    """
    if rng is None:
        rng = np.random.default_rng()

    d = len(marginals)
    Z = rng.multivariate_normal(np.zeros(d), corr, size=n_samples)
    U = stats.norm.cdf(Z)   # back to uniform
    X = np.column_stack([m.ppf(U[:, j]) for j, m in enumerate(marginals)])
    return X


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------

class FloodEventSelector:
    """
    Full event-selection pipeline for hybrid downscaling.

    Parameters
    ----------
    discharge : pd.Series
        Continuous daily discharge series (DatetimeIndex, m³/s).
    threshold : float
        Primary threshold Q1 (m³/s). Events must exceed this level.
    threshold2 : float, optional
        Secondary (higher) threshold Q2. Events must reach Q2 to be retained.
        If None, Q2 = threshold.
    n_types : int
        Number of hydrograph shape clusters (perfect square, e.g. 9, 16, 25).
    n_synthetic : int
        Size of the synthetic event ensemble to generate from the copula
        (default 5 000).
    plot : bool
        Whether to produce diagnostic plots at each step.
    output_dir : path-like, optional
        Directory for saving CSV outputs (selected events, synthetic matrix).
    """

    def __init__(
        self,
        discharge: pd.Series,
        threshold: float,
        threshold2: float | None = None,
        n_types: int = 25,
        n_synthetic: int = 5000,
        plot: bool = False,
        output_dir: str | Path | None = None,
    ):
        self.discharge    = discharge.copy()
        self.threshold    = threshold
        self.threshold2   = threshold2 if threshold2 is not None else threshold
        self.n_types      = n_types
        self.n_synthetic  = n_synthetic
        self.plot         = plot
        self.output_dir   = Path(output_dir) if output_dir else None

        # Results populated by each step
        self.events_bounds: pd.DataFrame | None = None
        self.classified:    pd.DataFrame | None = None
        self.marginals:     dict | None = None
        self.copula_corr:   np.ndarray | None = None
        self.synthetic:     pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Step 1 — Event extraction with dual threshold
    # ------------------------------------------------------------------

    def extract_events(self) -> pd.DataFrame:
        """
        Extract flood events by threshold crossing with inflection-point
        refinement (equivalent to *generacion_eventos_sinteticos.eventos_caudal*).

        Returns
        -------
        events_bounds : DataFrame with columns:
            ``Inicio_evento`` (int index), ``Fin_evento`` (int index).
        Also sets ``self.events_bounds``.
        """
        Q  = self.discharge.values.astype(float)
        th = self.threshold
        th2 = self.threshold2

        # ---- Locate threshold crossings ----------------------------------------
        x_ini, x_fin = [], []

        for i in range(len(Q) - 1):
            if Q[i] <= th < Q[i + 1]:
                x_ini.append(i)
            if Q[i] >= th > Q[i + 1]:
                x_fin.append(i + 1)

        # Match each start to the last end before it
        x_fin_matched = []
        for xi in x_ini:
            candidates = [xf for xf in x_fin if xf < xi]
            x_fin_matched.append(candidates[-1] if candidates else 0)

        # ---- Detect inflection points (slope sign changes) ----------------------
        slope  = np.sign(np.diff(Q))
        d_slope = np.diff(slope)

        inflect_up   = np.where(d_slope == 2)[0] + 1   # valley → rising
        inflect_down = np.where(d_slope == -2)[0] + 1  # peak → recession

        # ---- Refine start/end to nearest inflection point -----------------------
        p1, p2 = [], []

        for xi, xf_prev in zip(x_ini, x_fin_matched):
            # Inflection before threshold crossing → start of event
            cands_up = inflect_up[inflect_up >= xf_prev]
            cands_up = cands_up[cands_up <= xi]
            p1.append(int(cands_up[-1]) if len(cands_up) else xi)

        for i, (xi, xf) in enumerate(zip(x_ini, x_fin)):
            # Inflection after peak → end of event
            cands_down = inflect_down[inflect_down >= xi]
            if len(cands_down) == 0:
                p2.append(xf)
                continue
            cands_down = cands_down[cands_down <= xf + 1]
            p2.append(int(cands_down[-1]) if len(cands_down) else xf)

        # Pad p2 if lengths differ
        while len(p2) < len(p1):
            p2.append(p2[-1] if p2 else len(Q) - 1)
        p1 = p1[: len(p2)]

        # ---- Filter: event must reach secondary threshold -----------------------
        ini_filt, fin_filt = [], []
        for a, b in zip(p1, p2):
            seg = Q[a: b + 1]
            if len(seg) == 0:
                continue
            if seg.max() >= th2:
                ini_filt.append(a)
                fin_filt.append(b)

        self.events_bounds = pd.DataFrame({
            "Inicio_evento": ini_filt,
            "Fin_evento":    fin_filt,
        }).reset_index(drop=True)

        # ---- Optional plot -------------------------------------------------------
        if self.plot:
            self._plot_events()

        return self.events_bounds

    # ------------------------------------------------------------------
    # Step 2 — Characterise & classify event shapes (PCA + K-means)
    # ------------------------------------------------------------------

    def classify_events(self) -> pd.DataFrame:
        """
        Characterise events by (Qmax, Qmed, Duration, shape_type) via PCA + K-means.

        Calls ``extract_events`` automatically if not already done.

        Returns
        -------
        classified : DataFrame with columns Inicio_evento, Fin_evento,
            Qmax, Qmed, Duracion, shape_type.
        Also sets ``self.classified``.
        """
        if self.events_bounds is None:
            self.extract_events()

        clf = HydrographClassifier(
            discharge=self.discharge,
            events_bounds=self.events_bounds,
            n_types=self.n_types,
        )
        self.classified = clf.fit()
        self.classified = self.classified.reset_index(drop=True)

        if self.plot:
            self._plot_characterization()

        return self.classified

    # ------------------------------------------------------------------
    # Step 3 — Fit marginal distributions
    # ------------------------------------------------------------------

    def fit_marginals(self) -> dict:
        """
        Fit the best marginal distribution (lowest BIC) to Qmax, Qmed and Duration.

        Returns
        -------
        marginals : dict with keys 'Qmax', 'Qmed', 'Duracion' each holding
            a frozen scipy.stats distribution.
        Also sets ``self.marginals``.
        """
        if self.classified is None:
            self.classify_events()

        self.marginals = {}
        for col in ("Qmax", "Qmed", "Duracion"):
            data = self.classified[col].values.astype(float)
            data = data[np.isfinite(data) & (data > 0)]
            dist, name, bic = _best_marginal_bic(data)
            self.marginals[col] = dist
            print(f"  {col:10s}: best fit = {name:15s}  BIC = {bic:.1f}")

        if self.plot:
            self._plot_marginals()

        return self.marginals

    # ------------------------------------------------------------------
    # Step 4 — Fit Gaussian copula
    # ------------------------------------------------------------------

    def fit_copula(self) -> np.ndarray:
        """
        Fit a Gaussian copula on (Qmax, Qmed, Duracion) via Pearson correlation
        in the normal-score space.

        Returns
        -------
        corr : (3 × 3) correlation matrix.
        Also sets ``self.copula_corr``.
        """
        if self.marginals is None:
            self.fit_marginals()

        cols = ("Qmax", "Qmed", "Duracion")
        U = np.column_stack([
            _to_uniform(self.classified[c].values.astype(float),
                        self.marginals[c])
            for c in cols
        ])
        self.copula_corr = _fit_gaussian_copula(U)

        print("Gaussian copula correlation matrix (Qmax, Qmed, Duracion):")
        print(np.round(self.copula_corr, 3))
        return self.copula_corr

    # ------------------------------------------------------------------
    # Step 5 — Generate synthetic event ensemble
    # ------------------------------------------------------------------

    def generate_synthetic(self, seed: int | None = None) -> pd.DataFrame:
        """
        Sample *n_synthetic* events from the Gaussian copula and back-transform
        through the fitted marginals.

        Returns
        -------
        synthetic : DataFrame with columns Qmax, Qmed, Duracion, shape_type.
        Also sets ``self.synthetic``.
        """
        if self.copula_corr is None:
            self.fit_copula()

        rng = np.random.default_rng(seed)
        cols = ("Qmax", "Qmed", "Duracion")
        X = _sample_gaussian_copula(
            self.copula_corr,
            self.n_synthetic,
            [self.marginals[c] for c in cols],
            rng=rng,
        )

        # Assign shape type from a discrete empirical distribution
        type_probs = (
            self.classified["shape_type"]
            .value_counts(normalize=True)
            .sort_index()
        )
        shape_types = rng.choice(
            type_probs.index.values,
            size=self.n_synthetic,
            p=type_probs.values,
        )

        self.synthetic = pd.DataFrame(X, columns=list(cols))
        self.synthetic = self.synthetic.clip(lower=0)
        self.synthetic["shape_type"] = shape_types

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.synthetic.to_csv(self.output_dir / "synthetic_events.csv",
                                  index=False)
            self.classified.to_csv(self.output_dir / "observed_events.csv",
                                   index=False)

        if self.plot:
            self._plot_synthetic()

        return self.synthetic

    # ------------------------------------------------------------------
    # Convenience: run full pipeline in one call
    # ------------------------------------------------------------------

    def run(self, seed: int | None = None) -> pd.DataFrame:
        """
        Execute all steps in order:
        extract → classify → fit_marginals → fit_copula → generate_synthetic.

        Returns the synthetic event DataFrame.
        """
        self.extract_events()
        self.classify_events()
        self.fit_marginals()
        self.fit_copula()
        return self.generate_synthetic(seed=seed)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _plot_events(self):
        Q  = self.discharge.values
        t  = np.arange(len(Q))
        th = self.threshold

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(self.discharge.index, Q, "--k", lw=1.2, label="Caudal")
        ax.axhline(th, ls="--", lw=1.5, color="steelblue", label=f"Umbral {th:.0f} m³/s")

        for _, row in self.events_bounds.iterrows():
            a, b = int(row["Inicio_evento"]), int(row["Fin_evento"])
            ax.axvspan(self.discharge.index[a], self.discharge.index[min(b, len(Q)-1)],
                       alpha=0.15, color="orange")

        ax.set(ylabel="Q (m³/s)", title=f"Eventos seleccionados (N={len(self.events_bounds)})")
        ax.legend(fontsize=9)
        plt.tight_layout()
        if self.output_dir:
            plt.savefig(self.output_dir / "events_selection.png", dpi=150)
        plt.show()

    def _plot_characterization(self):
        df = self.classified
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        sc = axes[0].scatter(df["Qmax"], df["Qmed"], c=df["shape_type"],
                              cmap="tab20", s=30, edgecolors="k", lw=0.3)
        axes[0].set(xlabel="Qmax (m³/s)", ylabel="Qmed (m³/s)", title="Qmax vs Qmed")
        plt.colorbar(sc, ax=axes[0], label="Tipo")

        axes[1].scatter(df["Qmax"], df["Duracion"], c=df["shape_type"],
                        cmap="tab20", s=30, edgecolors="k", lw=0.3)
        axes[1].set(xlabel="Qmax (m³/s)", ylabel="Duración (días)", title="Qmax vs Duración")

        axes[2].scatter(df["Qmed"], df["Duracion"], c=df["shape_type"],
                        cmap="tab20", s=30, edgecolors="k", lw=0.3)
        axes[2].set(xlabel="Qmed (m³/s)", ylabel="Duración (días)", title="Qmed vs Duración")

        plt.suptitle("Caracterización de eventos observados", fontsize=12)
        plt.tight_layout()
        if self.output_dir:
            plt.savefig(self.output_dir / "events_characterization.png", dpi=150)
        plt.show()

    def _plot_marginals(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, col in zip(axes, ("Qmax", "Qmed", "Duracion")):
            data = self.classified[col].values.astype(float)
            data = data[np.isfinite(data) & (data > 0)]
            dist = self.marginals[col]
            x = np.linspace(data.min(), data.max(), 200)
            ax.hist(data, bins=20, density=True, alpha=0.5,
                    color="steelblue", edgecolor="k", label="Observado")
            ax.plot(x, dist.pdf(x), "r-", lw=2,
                    label=f"{dist.dist.name}")
            ax.set(xlabel=col, title=f"Ajuste marginal — {col}")
            ax.legend(fontsize=8)
        plt.tight_layout()
        if self.output_dir:
            plt.savefig(self.output_dir / "marginal_fits.png", dpi=150)
        plt.show()

    def _plot_synthetic(self):
        df_obs  = self.classified
        df_syn  = self.synthetic
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        pairs = [("Qmax", "Qmed"), ("Qmax", "Duracion"), ("Qmed", "Duracion")]
        for ax, (cx, cy) in zip(axes, pairs):
            ax.scatter(df_syn[cx], df_syn[cy],  s=4,  alpha=0.3,
                       color="steelblue", label="Sintético")
            ax.scatter(df_obs[cx], df_obs[cy],  s=30, alpha=0.9,
                       color="tomato", edgecolors="k", lw=0.5, label="Observado")
            ax.set(xlabel=cx, ylabel=cy, title=f"{cx} vs {cy}")
            ax.legend(fontsize=8)

        plt.suptitle("Ensamble sintético vs observado", fontsize=12)
        plt.tight_layout()
        if self.output_dir:
            plt.savefig(self.output_dir / "synthetic_vs_observed.png", dpi=150)
        plt.show()

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Return a summary table of observed vs synthetic event statistics."""
        rows = []
        for col in ("Qmax", "Qmed", "Duracion"):
            obs_data = self.classified[col].values if self.classified is not None else np.array([np.nan])
            syn_data = self.synthetic[col].values  if self.synthetic  is not None else np.array([np.nan])
            rows.append({
                "variable":   col,
                "obs_mean":   np.nanmean(obs_data),
                "obs_std":    np.nanstd(obs_data),
                "obs_max":    np.nanmax(obs_data),
                "syn_mean":   np.nanmean(syn_data),
                "syn_std":    np.nanstd(syn_data),
                "syn_max":    np.nanmax(syn_data),
            })
        return pd.DataFrame(rows).round(2)


# ---------------------------------------------------------------------------
# Copula family comparison for flood event variables
# ---------------------------------------------------------------------------

class FloodCopulaComparison:
    """
    Compare multiple copula families for trivariate flood event dependence.

    Fits Gaussian, Gumbel, Clayton and Frank copulas to the observed
    (Qmax, Qmed, Duracion) triplets and reports:

    * Kendall τ per variable pair
    * Copula parameter θ (Archimedean families)
    * Upper / lower tail dependence coefficients (λ_U, λ_L)
    * n_ok: number of successfully generated synthetic samples

    The **Gaussian copula** has no tail dependence (λ = 0), so it
    underestimates the co-occurrence probability of simultaneously large
    Qmax, Qmed and Duration — relevant for extreme flood events.

    **Gumbel** has upper tail dependence (λ_U > 0): it is the natural
    choice when floods are dominated by high joint extremes.

    **Clayton** has lower tail dependence (λ_L > 0): useful when
    dependence concentrates at low values.

    **Frank** is symmetric with no tail dependence, like Gaussian but
    with a different body shape.

    Parameters
    ----------
    vars : list of str
        Variable names in the events DataFrame (default: Qmax, Qmed, Duracion).
    families : list of str
        Copula families to compare.  Supported: 'gaussian', 'gumbel',
        'clayton', 'frank'.

    Example
    -------
    >>> cmp = FloodCopulaComparison()
    >>> cmp.fit(classified_events_df)
    >>> print(cmp.summary_table())
    >>> cmp.plot_comparison()
    >>> best = cmp.best_family()
    """

    _FAMILIES = ('gaussian', 'gumbel', 'clayton', 'frank')

    # Tail dependence coefficients (bivariate; trivariate is similar)
    # Gumbel:  λ_U = 2 − 2^{1/θ},  λ_L = 0
    # Clayton: λ_L = 2^{−1/θ},     λ_U = 0
    # Frank / Gaussian: λ_U = λ_L = 0
    @staticmethod
    def _tail_dep(family, theta):
        if family == 'gumbel':
            lam_U = 2.0 - 2.0 ** (1.0 / theta) if theta > 1 else 0.0
            return round(lam_U, 4), 0.0
        if family == 'clayton':
            lam_L = 2.0 ** (-1.0 / theta) if theta > 0 else 0.0
            return 0.0, round(lam_L, 4)
        return 0.0, 0.0   # gaussian / frank

    def __init__(self, vars=('Qmax', 'Qmed', 'Duracion'),
                 families=('gaussian', 'gumbel', 'clayton', 'frank')):
        self.vars     = list(vars)
        self.families = [f for f in families if f in self._FAMILIES]
        self._fitted  = {}   # family → TrivariateCopula or dict
        self._samples = {}   # family → (x, y, z) arrays
        self._tau     = {}   # pair → τ

    def fit(self, events_df):
        """
        Fit all requested copula families to *events_df*.

        Args:
            events_df: DataFrame with columns matching ``self.vars``.

        Returns:
            self
        """
        from pyhydra.climate.spatial_analysis.copulas import (
            TrivariateCopula, _tau_to_theta, _kendall_tau,
        )
        from scipy.stats import kendalltau as _kt

        x_col, y_col, z_col = self.vars
        x = events_df[x_col].values.astype(float)
        y = events_df[y_col].values.astype(float)
        z = events_df[z_col].values.astype(float)

        tau_xy = float(_kt(x, y).statistic)
        tau_xz = float(_kt(x, z).statistic)
        tau_yz = float(_kt(y, z).statistic)
        tau_mean = (tau_xy + tau_xz + tau_yz) / 3.0
        self._tau = {
            f'{x_col}–{y_col}': tau_xy,
            f'{x_col}–{z_col}': tau_xz,
            f'{y_col}–{z_col}': tau_yz,
            'mean': tau_mean,
        }
        print(f"Kendall τ: {x_col}–{y_col}={tau_xy:.3f}  "
              f"{x_col}–{z_col}={tau_xz:.3f}  {y_col}–{z_col}={tau_yz:.3f}")

        for family in self.families:
            print(f"\n[{family.upper()}]")
            if family == 'gaussian':
                # Gaussian copula via scipy (no extra deps)
                cols = (x_col, y_col, z_col)
                marginals = {}
                for col in cols:
                    d, name, bic = _best_marginal_bic(events_df[col].values.astype(float))
                    marginals[col] = d
                U = np.column_stack([
                    _to_uniform(events_df[c].values.astype(float), marginals[c])
                    for c in cols
                ])
                corr = _fit_gaussian_copula(U)
                self._fitted['gaussian'] = {
                    'corr': corr, 'marginals': marginals, 'theta': None,
                    'tau': tau_mean, 'lam_U': 0.0, 'lam_L': 0.0,
                }
                print(f"  Correlation matrix (normal scores):\n{np.round(corr, 3)}")
            else:
                try:
                    cop = TrivariateCopula(family=family)
                    cop.fit(x, y, z, labels=tuple(self.vars))
                    theta = cop._theta
                    lam_U, lam_L = self._tail_dep(family, theta)
                    self._fitted[family] = {
                        'model': cop, 'theta': theta,
                        'tau': tau_mean, 'lam_U': lam_U, 'lam_L': lam_L,
                    }
                    print(f"  θ={theta:.3f}  λ_U={lam_U:.3f}  λ_L={lam_L:.3f}")
                except Exception as e:
                    print(f"  WARN: {e}")
        return self

    def summary_table(self, n_samples=1000, seed=42):
        """
        Return a DataFrame comparing all fitted copula families.

        Columns: family, theta, tau_mean, lambda_U, lambda_L, sample_ok.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for family in self.families:
            info = self._fitted.get(family, {})
            ok = False
            try:
                if family == 'gaussian':
                    X = _sample_gaussian_copula(
                        info['corr'], n_samples,
                        [info['marginals'][c] for c in self.vars], rng=rng)
                    self._samples[family] = (X[:, 0], X[:, 1], X[:, 2])
                    ok = True
                else:
                    cop = info.get('model')
                    if cop is not None:
                        xs, ys, zs = cop.sample(n_samples, random_state=rng)
                        self._samples[family] = (xs, ys, zs)
                        ok = True
            except Exception as e:
                print(f"Sample warning [{family}]: {e}")

            rows.append({
                'copula':   family,
                'theta':    round(info.get('theta') or 0.0, 3),
                'tau_mean': round(info.get('tau', float('nan')), 3),
                'lambda_U': info.get('lam_U', 0.0),
                'lambda_L': info.get('lam_L', 0.0),
                'sample_ok': ok,
            })
        return pd.DataFrame(rows)

    def best_family(self, criterion='upper_tail'):
        """
        Return the name of the recommended copula family.

        criterion='upper_tail': Gumbel if τ > 0 (extremes are co-occurring),
            else Gaussian.
        criterion='frank': Frank (symmetric, no tail dependence).
        """
        tau_mean = self._tau.get('mean', 0.0)
        if criterion == 'upper_tail':
            if tau_mean > 0.1 and 'gumbel' in self._fitted:
                return 'gumbel'
            return 'gaussian'
        return criterion if criterion in self._fitted else 'gaussian'

    def plot_comparison(self, pair=(0, 1), n_samples=1000, figsize=(14, 4),
                        seed=42):
        """
        Grid of scatter plots comparing observed vs synthetic for each family.

        Args:
            pair:      Indices into ``self.vars`` for the plot axes.
            n_samples: Synthetic samples per family.
            figsize:   Figure size.
        """
        if not self._samples:
            self.summary_table(n_samples=n_samples, seed=seed)

        x_col = self.vars[pair[0]]
        y_col = self.vars[pair[1]]

        n_fam = len(self.families)
        fig, axes = plt.subplots(1, n_fam + 1, figsize=figsize, sharey=True)

        # Shared observed data (from the TrivariateCopula stored in gumbel/clayton/frank,
        # or reconstructed from gaussian's marginals)
        obs_x = obs_y = None
        for f in ['gumbel', 'clayton', 'frank']:
            if f in self._fitted and 'model' in self._fitted[f]:
                cop = self._fitted[f]['model']
                obs_x = cop._x if pair[0] == 0 else (cop._y if pair[0] == 1 else cop._z)
                obs_y = cop._x if pair[1] == 0 else (cop._y if pair[1] == 1 else cop._z)
                break

        # Observed panel
        ax0 = axes[0]
        if obs_x is not None:
            ax0.scatter(obs_x, obs_y, s=25, color='tomato', edgecolors='k',
                        lw=0.4, zorder=5)
        ax0.set(xlabel=x_col, ylabel=y_col, title='Observado')
        ax0.grid(alpha=0.3)

        # Synthetic panels per family
        for ax, family in zip(axes[1:], self.families):
            if family in self._samples:
                xs, ys, zs = self._samples[family]
                sx = xs if pair[0] == 0 else (ys if pair[0] == 1 else zs)
                sy = xs if pair[1] == 0 else (ys if pair[1] == 1 else zs)
                ax.scatter(sx, sy, s=4, alpha=0.4, color='steelblue')
                if obs_x is not None:
                    ax.scatter(obs_x, obs_y, s=25, color='tomato',
                               edgecolors='k', lw=0.4, zorder=5)
            lU = self._fitted.get(family, {}).get('lam_U', 0.0)
            lL = self._fitted.get(family, {}).get('lam_L', 0.0)
            ax.set(xlabel=x_col,
                   title=f'{family.capitalize()}\nλ_U={lU:.2f}  λ_L={lL:.2f}')
            ax.grid(alpha=0.3)

        fig.suptitle(f'Comparación de cópulas — {x_col} vs {y_col}', fontsize=12)
        plt.tight_layout()
        plt.show()
        return fig, axes

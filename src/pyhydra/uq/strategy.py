"""Choosing an uncertainty-propagation strategy from a pilot run.

A distribution over rainfall scenarios or model parameters has to be pushed
through an expensive hydraulic model. There are two usual routes:

``reduce_and_emulate``
    select a maximally dissimilar design, simulate it, fit an emulator and
    reconstruct the rest of the ensemble;
``monte_carlo``
    spend the whole budget on random draws and take the empirical quantile.

Which is right depends on whether the response can be emulated at the design
size the budget allows -- and that is measurable, from a pilot that is charged
against the same budget. :func:`select_strategy` measures it; :func:`propagate`
runs the whole thing end to end against a solver you supply.

The accounting is the part that is easy to get wrong. Both branches are held to
``n_max`` solver evaluations in total. Because :func:`~pyhydra.uq.maxdiss_order` is
nested, a run that goes on to emulate keeps its pilot points as the opening of
the final design and pays nothing for the diagnostic; a run that falls back to
Monte Carlo has spent them, and draws only ``n_max - n_pilot``. A comparison
that gave the rule ``n_pilot + n_max`` would be measuring a larger budget, not
a better decision.

Examples
--------
>>> import numpy as np
>>> from pyhydra.uq import propagate
>>> rng = np.random.default_rng(0)
>>> X = rng.lognormal(size=(4000, 3))
>>> solver = lambda x: 10 * np.sqrt(x.sum(1)) + np.log1p(x[:, 0])
>>> out = propagate(X, solver, n_max=200, quantile=0.95)
>>> out.strategy
'reduce_and_emulate'
>>> out.n_evaluations <= 200
True
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyhydra.uq.design import maxdiss_order, pilot_design
from pyhydra.uq.emulability import SELECTION_CAP, Emulability, emulability
from pyhydra.uq.emulators import emulator_families

__all__ = [
    "select_strategy",
    "propagate",
    "reduce_and_emulate",
    "StrategyDecision",
    "PropagationResult",
    "EMULABILITY_THRESHOLD",
]

#: Cross-validated skill at or above which reduction plus emulation is chosen.
#: Not a universal constant: it is the value swept and reported in the study
#: this module implements, where the loss curve is flat between roughly 0.4 and
#: 0.7. Re-sweep it if your loss function is not error in a design quantile.
EMULABILITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class StrategyDecision:
    """The decision, and everything it rested on.

    Attributes
    ----------
    strategy : str
        ``"reduce_and_emulate"`` or ``"monte_carlo"``.
    emulability : Emulability
        The pilot diagnostic, including which family won and by how much.
    threshold : float
        What it was compared against.
    n_pilot, n_max : int
        The pilot size and the total budget.
    n_remaining : int
        Evaluations left for the chosen route: ``n_max - n_pilot`` under Monte
        Carlo, since those points cannot be reused; the same number under
        reduction, where the pilot points are already in the design.
    margin : float
        ``emulability.r2 - threshold``. A decision taken on a margin of 0.01 is
        a coin toss and should be reported as one.
    """

    strategy: str
    emulability: Emulability
    threshold: float
    n_pilot: int
    n_max: int
    n_remaining: int

    @property
    def margin(self) -> float:
        return float(self.emulability.r2 - self.threshold)

    @property
    def is_marginal(self) -> bool:
        """True when the decision turned on less than 0.05 of skill."""
        return abs(self.margin) < 0.05

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"StrategyDecision({self.strategy!r}, R2={self.emulability.r2:.3f} "
            f"vs {self.threshold:.2f}, margin={self.margin:+.3f}"
            f"{', marginal' if self.is_marginal else ''})"
        )


@dataclass
class PropagationResult:
    """What :func:`propagate` did and what it produced."""

    strategy: str
    estimate: float
    quantile: float
    decision: StrategyDecision
    n_evaluations: int
    values: np.ndarray = field(repr=False, default=None)
    design_index: np.ndarray = field(repr=False, default=None)
    family: str | None = None
    design_emulability: Emulability | None = None

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"PropagationResult({self.strategy!r}, q{self.quantile:g}="
            f"{self.estimate:.4g}, n_evaluations={self.n_evaluations}, "
            f"family={self.family!r})"
        )


def select_strategy(
    X_pilot,
    y_pilot,
    n_max: int,
    threshold: float = EMULABILITY_THRESHOLD,
    families=None,
    random_state: int = 0,
) -> StrategyDecision:
    """Decide between reduction-plus-emulation and direct Monte Carlo.

    Call this after simulating the pilot returned by
    :func:`~pyhydra.uq.pilot_design`.

    Parameters
    ----------
    X_pilot : array of shape (n_pilot, n_features)
        Pilot inputs.
    y_pilot : array of shape (n_pilot,)
        The solver's response there.
    n_max : int
        Total solver budget, pilot included.
    threshold : float
        Emulability at or above which reduction is chosen; see
        :data:`EMULABILITY_THRESHOLD`.
    families : sequence of str or dict, optional
        The declared family set. Whatever is passed here must also be passed to
        :func:`reduce_and_emulate`, or the diagnostic will not describe the
        estimator.
    random_state : int
        Seed for the cross-validation shuffle.

    Returns
    -------
    StrategyDecision

    Notes
    -----
    The rule is a decision procedure, not an oracle. In the published test it
    selected the better of the two fixed policies in about 80% of 1,620
    synthetic problems; it fails on responses that vary faster than the design
    can resolve -- sharp ridges, high-frequency oscillation -- where a pilot
    reports respectable skill because it never samples the structure it is
    missing. When :attr:`StrategyDecision.is_marginal` is true, treat the choice
    as undetermined and, if the budget allows, run both.
    """
    X_pilot = np.asarray(X_pilot, float)
    y_pilot = np.asarray(y_pilot, float)
    n_pilot = len(y_pilot)
    if n_pilot >= n_max:
        raise ValueError(
            f"pilot of {n_pilot} leaves nothing of a budget of {n_max}; "
            "the decision would have no branch to take"
        )
    e = emulability(X_pilot, y_pilot, families=families, random_state=random_state)
    return StrategyDecision(
        strategy="reduce_and_emulate" if e.r2 >= threshold else "monte_carlo",
        emulability=e,
        threshold=float(threshold),
        n_pilot=int(n_pilot),
        n_max=int(n_max),
        n_remaining=int(n_max - n_pilot),
    )


def reduce_and_emulate(
    X,
    design_index,
    y_design,
    families=None,
    selection_cap: int = SELECTION_CAP,
    random_state: int = 0,
):
    """Fit the best emulator on a simulated design and reconstruct the ensemble.

    The family is selected on the design by the same cross-validated procedure
    that produced the pilot diagnostic, so the number the decision was taken on
    describes the model that ends up being used.

    Simulated members keep their simulated values; only the rest are emulated.
    The result is therefore exact wherever the solver was run, and all of the
    emulator's error sits at the members that were not.

    Parameters
    ----------
    X : array of shape (n_ensemble, n_features)
        The full ensemble to reconstruct.
    design_index : sequence of int
        Rows of *X* that were simulated.
    y_design : array of shape (len(design_index),)
        The solver's response at those rows.
    families, selection_cap, random_state
        As in :func:`~pyhydra.uq.emulability`.

    Returns
    -------
    values : array of shape (n_ensemble,)
        Reconstructed response over the whole ensemble.
    result : Emulability
        The selection made on the design.
    """
    X = np.asarray(X, float)
    sel = np.asarray(design_index, int)
    y_design = np.asarray(y_design, float)
    if len(sel) != len(y_design):
        raise ValueError(f"{len(sel)} design points but {len(y_design)} responses")

    e = emulability(
        X[sel], y_design, families=families,
        selection_cap=selection_cap, random_state=random_state,
    )
    models = families if isinstance(families, dict) else emulator_families(families)
    values = models[e.family].fit(X[sel], y_design).predict(X)
    values = np.asarray(values, float)
    values[sel] = y_design
    return values, e


def propagate(
    X,
    simulate,
    n_max: int,
    quantile: float = 0.95,
    n_pilot: int | None = None,
    threshold: float = EMULABILITY_THRESHOLD,
    families=None,
    random_state: int = 0,
    rng=None,
) -> PropagationResult:
    """Run the decision rule end to end against a solver.

    Draws the pilot, calls *simulate* on it, decides, and then either extends
    the design to ``n_max`` and emulates, or spends the remaining
    ``n_max - n_pilot`` evaluations on a random subsample of the ensemble. Total
    solver evaluations never exceed ``n_max`` on either branch, which is what
    makes the two routes comparable.

    Parameters
    ----------
    X : array of shape (n_ensemble, n_features)
        The synthetic ensemble to propagate. It must already exist: this layer
        decides how to *evaluate* it, not how to generate it.
    simulate : callable
        ``simulate(X_subset) -> array`` of responses, one per row. This is the
        expensive model. It is called at most twice.
    n_max : int
        Total solver budget.
    quantile : float
        The design quantile to estimate. The rule was tuned and tested against
        error in a high quantile; a different target may want a different
        threshold.
    n_pilot : int, optional
        Default ``min(30, n_max // 2)``.
    threshold, families, random_state
        As in :func:`select_strategy`.
    rng : numpy.random.Generator, optional
        Used only on the Monte Carlo branch. Supply one for reproducibility.

    Returns
    -------
    PropagationResult

    Raises
    ------
    ValueError
        If *simulate* returns the wrong number of values -- caught here rather
        than surfacing later as a silent misalignment between inputs and
        outputs, which is the failure mode this layer exists to prevent.
    """
    X = np.asarray(X, float)
    if n_max > len(X):
        raise ValueError(f"budget of {n_max} exceeds the ensemble of {len(X)}")
    rng = np.random.default_rng() if rng is None else rng

    p = pilot_design(X, n_max, n_pilot)
    pilot = np.asarray(p["pilot"], int)
    y_pilot = _simulate(simulate, X[pilot], len(pilot))

    decision = select_strategy(
        X[pilot], y_pilot, n_max=n_max, threshold=threshold,
        families=families, random_state=random_state,
    )

    if decision.strategy == "reduce_and_emulate":
        # The pilot is the opening of the MaxDiss order, so only the remainder
        # is simulated and the total is exactly n_max.
        order = np.asarray(p["order"], int)
        rest = order[decision.n_pilot:]
        y_rest = _simulate(simulate, X[rest], len(rest))
        y_design = np.concatenate([y_pilot, y_rest])
        values, e_design = reduce_and_emulate(
            X, order, y_design, families=families, random_state=random_state
        )
        return PropagationResult(
            strategy=decision.strategy,
            estimate=float(np.quantile(values, quantile)),
            quantile=quantile,
            decision=decision,
            n_evaluations=int(n_max),
            values=values,
            design_index=order,
            family=e_design.family,
            design_emulability=e_design,
        )

    # Monte Carlo: the pilot is spent, so only n_max - n_pilot draws remain.
    idx = rng.choice(len(X), decision.n_remaining, replace=False)
    y_mc = _simulate(simulate, X[idx], len(idx))
    return PropagationResult(
        strategy=decision.strategy,
        estimate=float(np.quantile(y_mc, quantile)),
        quantile=quantile,
        decision=decision,
        n_evaluations=int(decision.n_pilot + decision.n_remaining),
        values=y_mc,
        design_index=idx,
        family=None,
        design_emulability=None,
    )


def _simulate(simulate, X_subset, expected: int) -> np.ndarray:
    y = np.asarray(simulate(X_subset), float).ravel()
    if len(y) != expected:
        raise ValueError(
            f"simulate() returned {len(y)} values for {expected} inputs; "
            "responses must align with the rows they were computed from"
        )
    return y

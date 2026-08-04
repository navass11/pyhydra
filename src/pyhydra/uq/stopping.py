"""When to stop adding simulations to a design.

Given a nested MaxDiss order, the design can be grown a few points at a time
and the emulator's leave-one-out error tracked as it goes. That error costs no
extra solver runs -- it is computed on evaluations already bought -- so it is
the natural signal for a stopping rule.

The rule has to distinguish two things a naive "improvement below tau" test
conflates. An error curve that has flattened has converged; an error curve that
has turned upward has not, and stopping on it would report instability as
success. Each increment is therefore classified into one of three states, and
only sustained *convergence* stops the run:

=================  ==========================================
``improving``      relative improvement >= tau
``converged``      0 <= relative improvement < tau
``deteriorating``  relative improvement < 0
=================  ==========================================

A run of ``k`` consecutive ``deteriorating`` steps is recorded as an
instability flag rather than absorbed into a stop.

One caveat, and it is the reason this module reports rather than recommends: in
the published application the rule stopped at a design size whose *unweighted*
held-out error had indeed flattened, but whose population-weighted error was
still about 20% above its floor. Leave-one-out error over a MaxDiss design is
not the population-weighted error, and a rule tuned on one does not transfer to
the other. Pass ``population=`` to weight the internal signal by Voronoi cell
counts and close part of that gap; the weights are computable before any
simulation, so this costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import LeaveOneOut

from pyhydra.uq.design import voronoi_weights
from pyhydra.uq.emulators import emulator_families

__all__ = ["sequential_design", "loo_error", "StoppingResult"]


@dataclass
class StoppingResult:
    """The trace of a sequential design, and where it stopped.

    Attributes
    ----------
    history : list of dict
        One record per design size: ``n``, ``error``, ``relative_improvement``,
        ``state``, ``converged_run``.
    stopped_at : int or None
        Design size at which ``k`` consecutive converged increments were seen.
        ``None`` means the rule ran to ``n_max`` without converging, which is a
        result and not a failure to report.
    unstable_at : int or None
        First design size at which ``k`` consecutive increments deteriorated.
    n_evaluations : int
        Solver evaluations spent.
    weighted : bool
        Whether the internal signal was population-weighted.
    """

    history: list = field(default_factory=list)
    stopped_at: int | None = None
    unstable_at: int | None = None
    n_evaluations: int = 0
    weighted: bool = False

    def to_frame(self):
        """Return the history as a :class:`pandas.DataFrame`."""
        import pandas as pd

        return pd.DataFrame(self.history)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"StoppingResult(stopped_at={self.stopped_at}, "
            f"unstable_at={self.unstable_at}, "
            f"n_evaluations={self.n_evaluations}, weighted={self.weighted})"
        )


def loo_error(model, X, y, weights=None) -> float:
    """Leave-one-out RMSE on the simulated design. No extra solver runs.

    With *weights* supplied, each design point contributes in proportion to the
    number of ensemble members it represents, which is what makes the internal
    signal comparable with the population-weighted error the analysis is
    ultimately judged on.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    pred = np.empty(len(y))
    for tr, te in LeaveOneOut().split(X):
        pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
    e2 = (pred - y) ** 2
    if weights is None:
        return float(np.sqrt(np.mean(e2)))
    w = np.asarray(weights, float)
    return float(np.sqrt(np.sum(w * e2) / w.sum()))


def sequential_design(
    X,
    simulate,
    n_max: int,
    order=None,
    n0: int = 25,
    delta: int = 25,
    k: int = 3,
    tau: float = 0.05,
    emulator: str = "rbf_linear",
    population=None,
) -> StoppingResult:
    """Grow a MaxDiss design until its leave-one-out error stops improving.

    Parameters
    ----------
    X : array of shape (n_ensemble, n_features)
        The ensemble to design over.
    simulate : callable
        ``simulate(X_subset) -> array``. Called once per increment, on the new
        points only.
    n_max : int
        Hard cap on solver evaluations.
    order : sequence of int, optional
        A precomputed nested design order. Default: :func:`~pyhydra.uq.maxdiss_order`
        on *X*.
    n0, delta : int
        Initial design size and increment.
    k : int
        Consecutive converged increments required to stop.
    tau : float
        Relative improvement below which an increment counts as flat.
    emulator : str
        Family used for the internal signal, fixed a priori. Selecting it as the
        curve is walked would make the curve a record of the selection.
    population : array, optional
        The full ensemble, for Voronoi weighting of the internal error. See the
        module docstring: without it the rule optimizes an objective that is not
        the one you care about.

    Returns
    -------
    StoppingResult

    Examples
    --------
    >>> import numpy as np
    >>> from pyhydra.uq import sequential_design
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 2))
    >>> r = sequential_design(X, lambda x: x[:, 0] ** 2, n_max=150, n0=20, delta=20)
    >>> r.n_evaluations <= 150
    True
    """
    X = np.asarray(X, float)
    if order is None:
        from pyhydra.uq.design import maxdiss_order

        order = maxdiss_order(X, min(n_max, len(X)))
    order = np.asarray(order, int)
    if n_max > len(order):
        raise ValueError(f"budget of {n_max} exceeds the supplied order of {len(order)}")
    model = emulator_families((emulator,))[emulator]

    y = np.empty(0)
    history, states, errors = [], [], []
    stopped_at = unstable_at = None
    n = n0
    while n <= n_max:
        if len(y) < n:
            new = order[len(y):n]
            y_new = np.asarray(simulate(X[new]), float).ravel()
            if len(y_new) != len(new):
                raise ValueError(
                    f"simulate() returned {len(y_new)} values for {len(new)} inputs"
                )
            y = np.concatenate([y, y_new])
        sel = order[:n]
        w = None if population is None else voronoi_weights(X[sel], population)
        e = loo_error(model, X[sel], y[:n], w)

        rel = (errors[-1] - e) / errors[-1] if errors else np.nan
        errors.append(e)
        if np.isnan(rel):
            state = "initial"
        elif rel < 0:
            state = "deteriorating"
        elif rel < tau:
            state = "converged"
        else:
            state = "improving"
        states.append(state)

        converged_run = len(states) > k and all(s == "converged" for s in states[-k:])
        deteriorating_run = len(states) > k and all(s == "deteriorating" for s in states[-k:])
        history.append(
            {
                "n": int(n),
                "error": round(float(e), 6),
                "relative_improvement": None if np.isnan(rel) else round(float(rel), 6),
                "state": state,
                "converged_run": bool(converged_run),
            }
        )
        if deteriorating_run and unstable_at is None:
            unstable_at = int(n)
        if converged_run:
            stopped_at = int(n)
            break
        n += delta

    return StoppingResult(
        history=history,
        stopped_at=stopped_at,
        unstable_at=unstable_at,
        n_evaluations=int(len(y)),
        weighted=population is not None,
    )

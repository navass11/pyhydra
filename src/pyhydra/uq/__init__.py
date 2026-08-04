"""Deciding how to propagate uncertainty through an expensive model.

Flood-hazard estimates require pushing a distribution over rainfall scenarios or
model parameters through a hydraulic model. When each run costs hours, the whole
ensemble cannot be simulated, and practice has settled into two habits: reduce
the ensemble to a design, simulate it and emulate the rest; or spend the budget
on random draws. The choice is usually made by convention, because it depends on
whether the model's response can be emulated at all -- a property that is rarely
measured.

This subpackage measures it, and decides from the measurement.

.. code-block:: python

    import numpy as np
    from pyhydra.uq import propagate

    X = ...                      # (n_ensemble, n_features) synthetic ensemble
    def simulate(rows):          # the expensive hydraulic model
        return run_solver(rows)

    out = propagate(X, simulate, n_max=400, quantile=0.95)
    print(out.strategy, out.estimate, out.decision.emulability.r2)

For finer control the same three steps can be taken by hand:

.. code-block:: python

    from pyhydra.uq import pilot_design, select_strategy, reduce_and_emulate

    p = pilot_design(X, n_max=400)
    y_pilot = simulate(X[p["pilot"]])
    d = select_strategy(X[p["pilot"]], y_pilot, n_max=400)
    if d.strategy == "reduce_and_emulate":
        y = simulate(X[p["order"]])
        values, e = reduce_and_emulate(X, p["order"], y)

What the rule is, and is not
----------------------------
It is a decision procedure with a measured failure rate, not an oracle. Tested
prospectively on 1,620 synthetic problems against an independent reference, it
selected the better of the two fixed policies in about 80% of runs and reduced
mean error in the target quantile from 2.40% and 2.61% to 1.70%; an oracle would
have reached 1.09%. It fails on responses that vary faster than the design can
resolve, where a pilot reports respectable skill precisely because it never
samples the structure it is missing. :attr:`~pyhydra.uq.StrategyDecision.margin`
and :attr:`~pyhydra.uq.Emulability.scores` are there so that a decision taken on
a hair's breadth can be seen to have been.

Two things it will not do for you. It will not tell you the threshold for a loss
function other than error in a high quantile -- re-sweep it. And
:func:`sequential_design` optimizes leave-one-out error over the design, which
is not the population-weighted error: pass ``population=`` unless you mean the
former.
"""

from pyhydra.uq.design import maxdiss_order, pilot_design, standardize, voronoi_weights
from pyhydra.uq.emulability import SELECTION_CAP, Emulability, emulability
from pyhydra.uq.emulators import (
    DEFAULT_FAMILIES,
    RBF_SMOOTHING,
    EnsembleMean,
    PeakOnly1D,
    RBF,
    emulator_families,
)
from pyhydra.uq.metrics import error_metrics, weighted_r2, weighted_rmse
from pyhydra.uq.stopping import StoppingResult, loo_error, sequential_design
from pyhydra.uq.strategy import (
    EMULABILITY_THRESHOLD,
    PropagationResult,
    StrategyDecision,
    propagate,
    reduce_and_emulate,
    select_strategy,
)

__all__ = [
    # design
    "maxdiss_order",
    "pilot_design",
    "voronoi_weights",
    "standardize",
    # emulators
    "emulator_families",
    "DEFAULT_FAMILIES",
    "RBF",
    "RBF_SMOOTHING",
    "EnsembleMean",
    "PeakOnly1D",
    # emulability
    "emulability",
    "Emulability",
    "SELECTION_CAP",
    # strategy
    "select_strategy",
    "propagate",
    "reduce_and_emulate",
    "StrategyDecision",
    "PropagationResult",
    "EMULABILITY_THRESHOLD",
    # stopping
    "sequential_design",
    "loo_error",
    "StoppingResult",
    # metrics
    "error_metrics",
    "weighted_rmse",
    "weighted_r2",
]

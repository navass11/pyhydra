"""Measuring whether a solver's response can be emulated at all.

The quantity is the best cross-validated coefficient of determination attained
by any member of a declared family set on a given design. It is deliberately
*not* a property of the physical problem: it is a property of the triple
(problem, family set, design size), and every value carries all three.

The same function serves two roles, and it matters that they are the same
function. On a small pilot it is a *diagnostic* -- an estimate of how well a
surrogate would do if one were built. On the full design it is the *selection
step* of the surrogate that actually gets built. If the two were different
procedures, the diagnostic would be measuring the emulability of a model nobody
would go on to use, and the threshold it is compared against would not be
interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut

from pyhydra.uq.emulators import emulator_families

__all__ = ["emulability", "Emulability", "SELECTION_CAP"]

#: Above this design size, cross-validated selection is made on a fixed
#: evenly-spaced sub-design. Selection is the dominant cost on large designs and
#: this bounds it. It is a cost compromise, not a free choice: report it.
SELECTION_CAP = 400


@dataclass(frozen=True)
class Emulability:
    """What a cross-validated family selection found.

    Attributes
    ----------
    r2 : float
        Best cross-validated coefficient of determination over the families
        tried. May be negative, meaning no family beat the design mean.
    family : str
        The family that attained it -- the one a surrogate would be built from.
    scores : dict
        Every family's score, so the margin between them is visible. A winner
        that leads by 0.01 is a different situation from one that leads by 0.4.
    n_design : int
        Points the selection actually saw, after any capping.
    n_supplied : int
        Points supplied. Differs from ``n_design`` when the cap bound.
    scheme : str
        ``"leave-one-out"`` or ``"5-fold"``.
    families : tuple of str
        The declared family set. The value above is meaningless without it.
    """

    r2: float
    family: str
    scores: dict = field(default_factory=dict)
    n_design: int = 0
    n_supplied: int = 0
    scheme: str = ""
    families: tuple = ()

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"Emulability(r2={self.r2:.3f}, family={self.family!r}, "
            f"n_design={self.n_design}, scheme={self.scheme!r})"
        )


def emulability(
    X,
    y,
    families=None,
    selection_cap: int = SELECTION_CAP,
    random_state: int = 0,
    loo_below: int = 40,
) -> Emulability:
    """Best cross-validated skill over a declared set of emulator families.

    Leave-one-out below ``loo_below`` points, five-fold above: on a 30-point
    pilot a five-fold split leaves 24 points to fit an interpolant in several
    dimensions, which measures the split more than the problem.

    Parameters
    ----------
    X : array of shape (n_design, n_features)
        Inputs of the simulated design.
    y : array of shape (n_design,)
        The solver's response at those inputs.
    families : sequence of str or dict, optional
        Names from :func:`~pyhydra.uq.emulator_families`, or a ready-made
        ``{name: estimator}`` mapping. Default
        :data:`~pyhydra.uq.emulators.DEFAULT_FAMILIES`.
    selection_cap : int
        Cap on the number of points used for selection; see
        :data:`SELECTION_CAP`. Pass ``None`` to disable.
    random_state : int
        Seed for the five-fold shuffle. Recorded so the value is reproducible.
    loo_below : int
        Design size below which leave-one-out is used.

    Returns
    -------
    Emulability

    Notes
    -----
    A family that raises during fitting -- an RBF given fewer points than its
    polynomial tail needs, say -- is skipped rather than scored, and simply does
    not appear in ``scores``. If every family fails the result has ``r2 = -inf``
    and the caller should treat the problem as not emulable at this design size,
    which is the honest reading.

    Examples
    --------
    >>> import numpy as np
    >>> from pyhydra.uq import emulability
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 2))
    >>> e = emulability(X, X[:, 0] ** 2 + X[:, 1])       # smooth: emulable
    >>> e.r2 > 0.9
    True
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_design, n_features)")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")

    n_supplied = len(y)
    if selection_cap is not None and n_supplied > selection_cap:
        idx = np.linspace(0, n_supplied - 1, selection_cap).astype(int)
        X, y = X[idx], y[idx]
    n = len(y)
    if n < 3:
        raise ValueError(f"cannot cross-validate a design of {n} points")

    if isinstance(families, dict):
        models = families
    else:
        models = emulator_families(families)

    splitter = LeaveOneOut() if n < loo_below else KFold(5, shuffle=True, random_state=random_state)
    scheme = "leave-one-out" if n < loo_below else "5-fold"

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    scores: dict[str, float] = {}
    for name, model in models.items():
        pred = np.empty(n)
        try:
            for tr, te in splitter.split(X):
                pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
        except Exception:
            continue
        scores[name] = float(1 - np.sum((y - pred) ** 2) / ss_tot) if ss_tot > 0 else -np.inf

    if scores:
        family = max(scores, key=scores.get)
        best = scores[family]
    else:
        family, best = next(iter(models)), -np.inf

    return Emulability(
        r2=best,
        family=family,
        scores=dict(sorted(scores.items(), key=lambda kv: -kv[1])),
        n_design=n,
        n_supplied=n_supplied,
        scheme=scheme,
        families=tuple(models),
    )

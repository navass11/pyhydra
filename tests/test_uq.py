"""Tests for the uncertainty-propagation decision layer.

These check the properties the decision rule's validity rests on, not just that
the code runs: the design order is nested, the budget is never exceeded on
either branch, the diagnostic and the estimator are the same procedure, and the
rule routes emulable and non-emulable problems differently.
"""

import numpy as np
import pytest

from pyhydra.uq import (
    emulability,
    emulator_families,
    error_metrics,
    maxdiss_order,
    pilot_design,
    propagate,
    reduce_and_emulate,
    select_strategy,
    sequential_design,
    voronoi_weights,
    weighted_r2,
)


@pytest.fixture
def ensemble():
    rng = np.random.default_rng(20260803)
    return rng.lognormal(mean=0.0, sigma=0.5, size=(1500, 3))


def smooth(X):
    return 10.0 * np.sqrt(X.sum(1)) + 2.0 * np.log1p(X[:, 0])


def ridged(X):
    s = X.sum(1)
    return 9.0 * np.sqrt(s) + 5.0 * np.abs(np.sin(7.0 * s)) - 4.0 * np.abs(X[:, 1] - X[:, 2])


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------

def test_maxdiss_is_nested(ensemble):
    """The greedy order must be a prefix order, or a pilot cannot be reused."""
    order = maxdiss_order(ensemble, 60)
    assert order[:20] == maxdiss_order(ensemble, 20)


def test_maxdiss_is_deterministic(ensemble):
    assert maxdiss_order(ensemble, 40) == maxdiss_order(ensemble, 40)


def test_maxdiss_selects_distinct_members(ensemble):
    order = maxdiss_order(ensemble, 100)
    assert len(set(order)) == 100


def test_maxdiss_rejects_impossible_request(ensemble):
    with pytest.raises(ValueError, match="cannot select"):
        maxdiss_order(ensemble, len(ensemble) + 1)


def test_voronoi_weights_partition_the_population(ensemble):
    """Every ensemble member is assigned to exactly one design point."""
    sel = np.array(maxdiss_order(ensemble, 50))
    w = voronoi_weights(ensemble[sel], ensemble)
    assert w.sum() == len(ensemble)
    assert (w >= 0).all()


def test_pilot_leaves_the_declared_remainder(ensemble):
    p = pilot_design(ensemble, n_max=200)
    assert p["n_pilot"] == 30
    assert p["n_remaining"] == 170
    assert p["pilot"] == p["order"][:30]


# ---------------------------------------------------------------------------
# Emulability
# ---------------------------------------------------------------------------

def test_emulability_high_on_smooth_response(ensemble):
    sel = np.array(maxdiss_order(ensemble, 60))
    e = emulability(ensemble[sel], smooth(ensemble[sel]))
    assert e.r2 > 0.9
    assert e.family in e.scores
    assert e.scores[e.family] == e.r2


def test_emulability_low_on_noise(ensemble):
    """An unlearnable response must not be reported as emulable."""
    rng = np.random.default_rng(0)
    sel = np.array(maxdiss_order(ensemble, 60))
    e = emulability(ensemble[sel], rng.normal(size=60))
    assert e.r2 < 0.5


def test_emulability_records_its_own_conditions(ensemble):
    sel = np.array(maxdiss_order(ensemble, 30))
    e = emulability(ensemble[sel], smooth(ensemble[sel]))
    assert e.scheme == "leave-one-out"
    assert e.n_design == 30
    assert set(e.families) == {"rbf_linear", "rbf_cubic", "knn_5"}


def test_emulability_switches_to_kfold_above_the_cut(ensemble):
    sel = np.array(maxdiss_order(ensemble, 60))
    assert emulability(ensemble[sel], smooth(ensemble[sel])).scheme == "5-fold"


def test_selection_cap_bounds_the_selection_cost(ensemble):
    sel = np.array(maxdiss_order(ensemble, 900))
    e = emulability(ensemble[sel], smooth(ensemble[sel]), selection_cap=400)
    assert e.n_design == 400
    assert e.n_supplied == 900


def test_emulability_rejects_misaligned_inputs(ensemble):
    with pytest.raises(ValueError, match="rows but"):
        emulability(ensemble[:30], smooth(ensemble[:29]))


def test_emulability_accepts_a_custom_family_set(ensemble):
    sel = np.array(maxdiss_order(ensemble, 50))
    e = emulability(ensemble[sel], smooth(ensemble[sel]), families=("knn_5", "baseline_mean"))
    assert set(e.families) == {"knn_5", "baseline_mean"}
    assert e.family == "knn_5"          # must beat the null model


def test_unknown_family_is_an_error():
    with pytest.raises(KeyError, match="unknown emulator families"):
        emulator_families(("kriging_with_pixie_dust",))


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def test_rule_emulates_a_smooth_response(ensemble):
    p = pilot_design(ensemble, n_max=200)
    pilot = np.array(p["pilot"])
    d = select_strategy(ensemble[pilot], smooth(ensemble[pilot]), n_max=200)
    assert d.strategy == "reduce_and_emulate"
    assert d.margin > 0


def test_rule_declines_to_emulate_pure_noise(ensemble):
    rng = np.random.default_rng(1)
    p = pilot_design(ensemble, n_max=200)
    pilot = np.array(p["pilot"])
    d = select_strategy(ensemble[pilot], rng.normal(size=len(pilot)), n_max=200)
    assert d.strategy == "monte_carlo"
    assert d.n_remaining == 170


def test_marginal_decisions_are_flagged(ensemble):
    p = pilot_design(ensemble, n_max=200)
    pilot = np.array(p["pilot"])
    d = select_strategy(
        ensemble[pilot], smooth(ensemble[pilot]), n_max=200, threshold=0.999999
    )
    assert d.is_marginal


def test_pilot_cannot_swallow_the_budget(ensemble):
    p = pilot_design(ensemble, n_max=200)
    pilot = np.array(p["pilot"])
    with pytest.raises(ValueError, match="leaves nothing"):
        select_strategy(ensemble[pilot], smooth(ensemble[pilot]), n_max=30)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def test_reconstruction_is_exact_at_simulated_points(ensemble):
    order = maxdiss_order(ensemble, 150)
    sel = np.array(order)
    values, e = reduce_and_emulate(ensemble, order, smooth(ensemble[sel]))
    assert np.allclose(values[sel], smooth(ensemble[sel]))
    assert len(values) == len(ensemble)
    assert e.family in emulator_families()


def test_reconstruction_recovers_a_smooth_quantile(ensemble):
    order = maxdiss_order(ensemble, 200)
    sel = np.array(order)
    values, _ = reduce_and_emulate(ensemble, order, smooth(ensemble[sel]))
    truth = np.quantile(smooth(ensemble), 0.95)
    assert abs(np.quantile(values, 0.95) - truth) / truth < 0.05


def test_reconstruction_rejects_misaligned_responses(ensemble):
    order = maxdiss_order(ensemble, 50)
    with pytest.raises(ValueError, match="design points but"):
        reduce_and_emulate(ensemble, order, smooth(ensemble[np.array(order[:49])]))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_propagate_never_exceeds_the_budget(ensemble):
    """The accounting claim the whole comparison rests on."""
    for response in (smooth, ridged):
        calls = []

        def counted(rows, f=response):
            calls.append(len(rows))
            return f(rows)

        out = propagate(ensemble, counted, n_max=200, rng=np.random.default_rng(0))
        assert sum(calls) <= 200
        assert out.n_evaluations == sum(calls)


def test_monte_carlo_branch_pays_for_its_pilot(ensemble):
    rng = np.random.default_rng(2)
    noise = rng.normal(size=len(ensemble))
    out = propagate(
        ensemble, lambda rows: noise[: len(rows)], n_max=200, rng=np.random.default_rng(0)
    )
    assert out.strategy == "monte_carlo"
    assert out.n_evaluations == 200
    assert len(out.values) == 170        # n_max - n_pilot, not n_max


def test_propagate_emulates_and_estimates_a_smooth_quantile(ensemble):
    out = propagate(ensemble, smooth, n_max=200, quantile=0.95)
    truth = np.quantile(smooth(ensemble), 0.95)
    assert out.strategy == "reduce_and_emulate"
    assert out.family is not None
    assert abs(out.estimate - truth) / truth < 0.05


def test_diagnostic_and_estimator_are_the_same_procedure(ensemble):
    """The pilot must score the model that would actually be built."""
    order = maxdiss_order(ensemble, 60)
    sel = np.array(order)
    y = smooth(ensemble[sel])
    from_diagnostic = emulability(ensemble[sel], y)
    _, from_estimator = reduce_and_emulate(ensemble, order, y)
    assert from_diagnostic.family == from_estimator.family
    assert from_diagnostic.r2 == pytest.approx(from_estimator.r2)


def test_propagate_is_the_components_it_packages(ensemble):
    """propagate() must not be a second, quietly different procedure.

    The published results were computed by driving the components directly, so
    the convenience wrapper is only trustworthy if it is those components and
    nothing else. This asserts equality, not similarity.
    """
    p = pilot_design(ensemble, n_max=200)
    pilot = np.array(p["pilot"])
    y_pilot = smooth(ensemble[pilot])
    d = select_strategy(ensemble[pilot], y_pilot, n_max=200)
    assert d.strategy == "reduce_and_emulate"

    order = np.array(p["order"])
    values, e = reduce_and_emulate(ensemble, order, smooth(ensemble[order]))
    manual = float(np.quantile(values, 0.95))

    out = propagate(ensemble, smooth, n_max=200, quantile=0.95)
    assert out.estimate == manual
    assert out.family == e.family
    assert out.decision.emulability.r2 == d.emulability.r2
    assert np.array_equal(out.design_index, order)


def test_propagate_catches_a_solver_that_returns_the_wrong_length(ensemble):
    with pytest.raises(ValueError, match="responses must align"):
        propagate(ensemble, lambda rows: smooth(rows)[:-1], n_max=100)


def test_budget_larger_than_the_ensemble_is_an_error(ensemble):
    with pytest.raises(ValueError, match="exceeds the ensemble"):
        propagate(ensemble, smooth, n_max=len(ensemble) + 1)


# ---------------------------------------------------------------------------
# Sequential stopping
# ---------------------------------------------------------------------------

def test_sequential_design_stops_on_a_converging_response(ensemble):
    r = sequential_design(ensemble, smooth, n_max=400, n0=25, delta=25)
    assert r.stopped_at is not None
    assert r.n_evaluations == r.stopped_at
    assert r.history[-1]["converged_run"]
    # Stopping means "improvement fell below tau three times running", not
    # "the error stopped falling". The last increments here still improve by
    # about 4%; the rule trades that for the simulations it saves.
    assert [h["state"] for h in r.history[-3:]] == ["converged"] * 3


def test_sequential_design_respects_the_cap(ensemble):
    r = sequential_design(ensemble, ridged, n_max=100, n0=25, delta=25, k=99)
    assert r.stopped_at is None          # never converged: reported, not hidden
    assert r.n_evaluations <= 100


def test_deterioration_is_not_reported_as_convergence(ensemble):
    """A rising error curve must not satisfy a rule that means 'flattened'."""
    rng = np.random.default_rng(3)
    r = sequential_design(
        ensemble, lambda rows: rng.normal(size=len(rows)), n_max=200, n0=25, delta=25
    )
    for row, nxt in zip(r.history, r.history[1:]):
        if nxt["state"] == "deteriorating":
            assert nxt["relative_improvement"] < 0
    if r.stopped_at is not None:
        tail = [h["state"] for h in r.history[-3:]]
        assert all(s == "converged" for s in tail)


def test_population_weighting_changes_the_internal_signal(ensemble):
    unweighted = sequential_design(ensemble, smooth, n_max=125, n0=25, delta=25, k=99)
    weighted = sequential_design(
        ensemble, smooth, n_max=125, n0=25, delta=25, k=99, population=ensemble
    )
    assert weighted.weighted and not unweighted.weighted
    assert weighted.history[0]["error"] != unweighted.history[0]["error"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_weighted_r2_uses_the_weighted_mean_as_reference():
    y = np.array([1.0, 2.0, 10.0])
    w = np.array([10.0, 10.0, 1.0])
    assert weighted_r2(y, y, w) == 1.0
    assert weighted_r2(y, np.full(3, np.average(y, weights=w)), w) == pytest.approx(0.0)


def test_error_metrics_weighting_matters():
    y = np.array([0.0, 0.0, 100.0])
    p = np.array([0.0, 0.0, 0.0])
    flat = error_metrics(y, p)["rmse"]
    down = error_metrics(y, p, weights=[100.0, 100.0, 1.0])["rmse"]
    assert down < flat


def test_error_metrics_rejects_bad_weights():
    with pytest.raises(ValueError, match="non-negative"):
        error_metrics([1.0, 2.0], [1.0, 2.0], weights=[-1.0, 1.0])

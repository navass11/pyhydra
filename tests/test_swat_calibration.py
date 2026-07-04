"""Tests for the SWAT+ calibration helpers (SWATModel, calibrate_swat_sceua).

These mirror the ad-hoc pySWATPlus + spotpy workflow previously duplicated
inside notebooks/modeling/hydrology/SWAT.ipynb, now extracted into
pyhydra.modeling.hydrology.swat. pySWATPlus itself is not installed in the
test environment (it requires a real SWAT+ binary), so it is faked via
sys.modules injection — the same technique test_model_adapters.py uses for
hecdss.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyhydra.modeling.hydrology import swat

_CHANNEL_SD_HEADER = (
    " Test watershed             SWAT+ 2024-07-08\n"
    "   jday   mon   day    yr     unit   name         flo_stor      null    flo_out      null\n"
    "                                                       m^3/s               m^3/s\n"
)


def _write_channel_sd_mon(out_dir, monthly_values, units=(1, 2)):
    """Write a channel_sd_mon.txt with `monthly_values` on the last unit (outlet)."""
    out_dir = Path(out_dir)
    lines = [_CHANNEL_SD_HEADER]
    for mon, val in enumerate(monthly_values, start=1):
        for unit in units:
            flo = val if unit == max(units) else val * 0.1
            lines.append(
                f"{mon:7d}{mon:6d}{28:6d}{2020:6d}"
                f"{unit:9d} cha{unit:03d}     {0.0:10.3f}"
                f"{0.0:10.3f}{flo:10.3f}{0.0:10.3f}\n"
            )
    (out_dir / "channel_sd_mon.txt").write_text("".join(lines))


class _FakeTxtinoutReader:
    """Fakes pySWATPlus.TxtinoutReader: records calls, writes a synthetic output."""

    last_calls: list[dict] = []

    def __init__(self, txtinout_dir):
        self.txtinout_dir = txtinout_dir

    def run_swat(self, sim_dir, parameters):
        _FakeTxtinoutReader.last_calls.append(
            {"sim_dir": sim_dir, "parameters": parameters}
        )
        # Simulated discharge responds to the 'cn2' parameter so calibration
        # has a real optimum to search for.
        cn2 = next((p["value"] for p in parameters if p["name"] == "cn2"), 0.0)
        base = 10.0 + cn2 * 0.1
        monthly = [base + i * 0.2 for i in range(12)]
        _write_channel_sd_mon(sim_dir, monthly)


@pytest.fixture
def fake_pyswatplus(monkeypatch, tmp_path):
    _FakeTxtinoutReader.last_calls = []
    fake_module = types.SimpleNamespace(TxtinoutReader=_FakeTxtinoutReader)
    monkeypatch.setitem(sys.modules, "pySWATPlus", fake_module)
    (tmp_path / "sim").mkdir()
    yield fake_module


def _make_model(tmp_path, **kwargs):
    params = [
        swat.SWATCalibrationParameter("cn2", "pctchg", -15.0, 15.0),
        swat.SWATCalibrationParameter("esco", "absval", 0.5, 1.0),
    ]
    return swat.SWATModel(
        txtinout_dir=str(tmp_path / "template"),
        sim_dir=str(tmp_path / "sim"),
        parameters=params,
        **kwargs,
    )


# ── SWATModel.run_swat ───────────────────────────────────────────────────────

def test_run_swat_applies_parameters_and_reads_outlet_discharge(fake_pyswatplus, tmp_path):
    model = _make_model(tmp_path)

    result = model.run_swat(5.0, 0.75)

    assert _FakeTxtinoutReader.last_calls == [
        {
            "sim_dir": str(tmp_path / "sim"),
            "parameters": [
                {"name": "cn2", "change_type": "pctchg", "value": 5.0},
                {"name": "esco", "change_type": "absval", "value": 0.75},
            ],
        }
    ]
    assert isinstance(result, pd.Series)
    assert result.name == "sim"
    assert len(result) == 12
    # base = 10 + 5*0.1 = 10.5, then +0.2 per month
    assert result.iloc[0] == pytest.approx(10.5)
    assert result.iloc[-1] == pytest.approx(10.5 + 11 * 0.2)


def test_run_swat_autodetects_outlet_as_max_unit_id(fake_pyswatplus, tmp_path):
    model = _make_model(tmp_path)

    model.run_swat(0.0, 0.75)

    assert model.outlet_unit == 2  # max(units=(1, 2)) in the fixture


def test_run_swat_rejects_wrong_parameter_count(fake_pyswatplus, tmp_path):
    model = _make_model(tmp_path)

    with pytest.raises(ValueError, match="Expected 2 SWAT\\+ parameters"):
        model.run_swat(1.0)


def test_run_swat_raises_helpful_error_without_pyswatplus(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pySWATPlus", None)
    model = _make_model(tmp_path)

    with pytest.raises(ImportError, match="pySWATPlus"):
        model.run_swat(0.0, 0.75)


# ── _align_swat_series ───────────────────────────────────────────────────────

def test_align_swat_series_intersects_matching_dates():
    idx1 = pd.date_range("2020-01-31", periods=4, freq="ME")
    idx2 = pd.date_range("2020-02-29", periods=4, freq="ME")
    sim = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx1)
    obs = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx2)

    sim_arr, obs_arr = swat._align_swat_series(sim, obs)

    # Overlap is Feb, Mar, Apr -> sim[1:4], obs[0:3]
    assert sim_arr.tolist() == [2.0, 3.0, 4.0]
    assert obs_arr.tolist() == [10.0, 20.0, 30.0]


def test_align_swat_series_falls_back_to_positional_truncation():
    sim = np.zeros(12)
    obs = pd.Series(np.arange(10.0))

    sim_arr, obs_arr = swat._align_swat_series(sim, obs)

    assert len(sim_arr) == 10
    assert len(obs_arr) == 10


# ── validate_swat_parameter_sensitivity ─────────────────────────────────────

def test_validate_swat_parameter_sensitivity_detects_response(fake_pyswatplus, tmp_path):
    model = _make_model(tmp_path)

    report = swat.validate_swat_parameter_sensitivity(model)

    assert report["max_delta"] > 0
    assert report["perturbed_peak"] != report["baseline_peak"]


def test_validate_swat_parameter_sensitivity_raises_when_unresponsive(fake_pyswatplus, tmp_path):
    model = _make_model(tmp_path)
    flat = np.array([0.0, 0.75])

    with pytest.raises(RuntimeError, match="did not change the hydrograph"):
        swat.validate_swat_parameter_sensitivity(model, baseline=flat, perturbed=flat)


# ── calibrate_swat_sceua (end-to-end with a synthetic, duck-typed model) ────

class _ToyModel:
    """Deterministic stand-in for SWATModel: no pySWATPlus/SWAT+ involved.

    Exercises the real spotpy SCE-UA integration (spotpy is installed) to
    confirm calibrate_swat_sceua actually drives the search toward the
    known optimum, and that a simulation failure is absorbed instead of
    crashing the sampler.
    """

    TARGET_CN2 = 8.0

    def __init__(self):
        self.parameters = [swat.SWATCalibrationParameter("cn2", "pctchg", -15.0, 15.0)]
        self.calls = 0

    @property
    def parameter_bounds(self):
        return [(p.name, p.lower, p.upper) for p in self.parameters]

    def run_swat(self, *params):
        self.calls += 1
        # Deterministically fail on the very first evaluation (SCE-UA's
        # initial population), instead of an unlikely-to-be-sampled extreme
        # value, so the test actually exercises the crash-recovery path
        # rather than passing by luck when SCE-UA never visits it.
        if self.calls == 1:
            raise RuntimeError("simulated SWAT+ crash on first evaluation")
        cn2 = params[0]
        error = abs(cn2 - self.TARGET_CN2)
        months = np.arange(12, dtype=float)
        return pd.Series(10.0 + months - error, name="sim")


def test_calibrate_swat_sceua_converges_and_survives_simulation_failures(tmp_path):
    model = _ToyModel()
    observed = pd.Series(10.0 + np.arange(12, dtype=float))
    dbname = str(tmp_path / "swat_sceua")

    sampler = swat.calibrate_swat_sceua(model, observed, dbname=dbname, n_evals=60, ngs=2)

    # The first evaluation always raises; if calibrate_swat_sceua let that
    # exception propagate, sampler.sample() above would have crashed instead
    # of reaching this line — so calls > 1 proves the recovery path ran.
    assert model.calls > 1

    df = pd.read_csv(dbname + ".csv")
    best = df.loc[df["like1"].idxmin()]
    assert abs(best["parcn2"] - _ToyModel.TARGET_CN2) < 3.0  # SCE-UA found the optimum region
    assert sampler is not None

"""Tests for NSRPModel and STNSRPModel (instantiation and config logic, no NEOPRENE needed)."""

import pytest

from pyhydra.climate.stochastic_generation import (
    NSRPModel,
    STNSRPModel,
    _build_calibration_yaml,
    _build_simulation_yaml,
    _default_model_bounds,
    _default_statistics,
    _default_weights,
)


class TestNSRPModelInstantiation:
    def test_default_instantiation(self):
        m = NSRPModel()
        assert m.temporal_resolution == "d"
        assert m.seasonality == "monthly"
        assert m.process == "normal"

    def test_custom_resolution(self):
        m = NSRPModel(temporal_resolution="h")
        assert m.temporal_resolution == "h"

    def test_custom_seasonality(self):
        m = NSRPModel(seasonality="seasonal")
        assert m.seasonality == "seasonal"

    def test_custom_process(self):
        m = NSRPModel(process="storms")
        assert m.process == "storms"

    def test_default_statistics_populated(self):
        m = NSRPModel()
        assert len(m.statistics_name) > 0
        assert "mean" in m.statistics_name

    def test_custom_statistics(self):
        stats = ["mean", "var_h"]
        m = NSRPModel(statistics=stats)
        assert m.statistics_name == stats

    def test_default_weights_match_statistics(self):
        m = NSRPModel()
        assert set(m.weights.keys()) == set(m.statistics_name)

    def test_custom_weights(self):
        m = NSRPModel(statistics=["mean", "var_h"], weights={"mean": 2.0, "var_h": 0.5})
        assert m.weights["mean"] == 2.0

    def test_pso_params(self):
        m = NSRPModel(n_iterations=200, n_bees=50, n_initializations=3)
        assert m.n_iterations == 200
        assert m.n_bees == 50
        assert m.n_initializations == 3

    def test_model_bounds_default_keys(self):
        m = NSRPModel()
        expected = {
            "time_between_storms", "number_storm_cells",
            "cell_duration", "cell_intensity", "storm_cell_displacement",
        }
        assert set(m.model_bounds.keys()) == expected

    def test_custom_model_bounds(self):
        bounds = _default_model_bounds()
        bounds["cell_intensity"] = [0.5, 10.0]
        m = NSRPModel(model_bounds=bounds)
        assert m.model_bounds["cell_intensity"] == [0.5, 10.0]

    def test_simulate_before_fit_raises(self):
        m = NSRPModel()
        with pytest.raises(RuntimeError, match="fit"):
            m.simulate(2000, 2050)

    def test_calibrate_before_stats_raises(self):
        m = NSRPModel()
        with pytest.raises(RuntimeError):
            m.calibrate()

    def test_summary_before_fit_returns_none(self):
        m = NSRPModel()
        assert m.summary() is None

    def test_calibration_yaml_passthrough(self, tmp_path):
        yaml_file = tmp_path / "cal.yml"
        yaml_file.write_text("Seasonality_type: monthly\n")
        m = NSRPModel(calibration_yaml=str(yaml_file))
        assert m._calibration_yaml == str(yaml_file)


class TestSTNSRPModelInstantiation:
    def test_default_instantiation(self):
        m = STNSRPModel()
        assert m.temporal_resolution == "d"
        assert m.coordinates == "geographical"

    def test_utm_coordinates(self):
        m = STNSRPModel(coordinates="UTM")
        assert m.coordinates == "UTM"

    def test_cell_radius(self):
        m = STNSRPModel(cell_radius=50.0)
        assert m.cell_radius == 50.0

    def test_simulate_before_fit_raises(self):
        import pandas as pd
        m = STNSRPModel()
        with pytest.raises(RuntimeError, match="fit"):
            m.simulate(2000, 2050, attributes=pd.DataFrame())

    def test_calibrate_without_stats_raises(self):
        m = STNSRPModel()
        with pytest.raises(RuntimeError):
            m.calibrate()

    def test_calibrate_df_without_attrs_raises(self):
        import pandas as pd
        m = STNSRPModel()
        with pytest.raises(ValueError, match="attributes"):
            m.calibrate(pd.DataFrame({"A": [1, 2]}), attributes=None)

    def test_summary_before_fit_returns_none(self):
        m = STNSRPModel()
        assert m.summary() is None


class TestYamlBuilders:
    def test_calibration_yaml_keys(self):
        stats = _default_statistics()
        cfg = _build_calibration_yaml(
            temporal_resolution="d",
            seasonality="monthly",
            process="normal",
            statistics_name=stats,
            weights=_default_weights(stats),
            n_iterations=100,
            n_bees=20,
            n_initializations=1,
            model_bounds=_default_model_bounds(),
        )
        assert "temporal_resolution" in cfg
        assert "Seasonality_type" in cfg
        assert "statistics_name" in cfg
        assert "number_iterations" in cfg
        assert "time_between_storms" in cfg
        assert "cell_intensity" in cfg

    def test_simulation_yaml_keys(self):
        cfg = _build_simulation_yaml(
            temporal_resolution="d",
            seasonality="monthly",
            process="normal",
            statistics_name=_default_statistics(),
            year_ini=2000,
            year_fin=2050,
        )
        assert cfg["year_ini"] == 2000
        assert cfg["year_fin"] == 2050
        assert "temporal_resolution" in cfg

    def test_seasonality_user_included_when_provided(self):
        cfg = _build_calibration_yaml(
            temporal_resolution="d",
            seasonality="user_defined",
            process="normal",
            statistics_name=["mean"],
            weights={"mean": 1.0},
            n_iterations=10,
            n_bees=5,
            n_initializations=1,
            model_bounds=_default_model_bounds(),
            seasonality_user=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        )
        assert "Seasonality_user" in cfg

    def test_year_ini_is_int(self):
        cfg = _build_simulation_yaml(
            temporal_resolution="d", seasonality="monthly", process="normal",
            statistics_name=["mean"], year_ini=2000.0, year_fin=2050,
        )
        assert isinstance(cfg["year_ini"], int)

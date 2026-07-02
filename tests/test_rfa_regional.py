from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyhydra.climate.spatial_analysis import rfa


def test_regional_index_flood_normalises_each_station_by_mean():
    data = {
        "A": np.array([10.0, 20.0, 30.0]),
        "B": np.array([5.0, 10.0, 15.0]),
    }

    normalised, index_flood = rfa.regional_index_flood(data)

    assert index_flood.to_dict() == {"A": 20.0, "B": 10.0}
    assert np.allclose(normalised["A"], [0.5, 1.0, 1.5])
    assert np.allclose(normalised["B"], [0.5, 1.0, 1.5])


def test_fit_regional_gev_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        rfa.fit_regional_gev({"A": np.array([1.0, 2.0, 3.0])}, method="invalid")


def test_regional_return_levels_scale_regional_quantiles_by_station_index(monkeypatch):
    def fake_fit_regional_gev(data_dict, method="lmom"):
        return {"shape": 0.0}, pd.Series({"A": 10.0, "B": 20.0}, name="index_flood")

    def fake_return_level(params, return_periods):
        return np.asarray(return_periods, dtype=float) * 0.1

    monkeypatch.setattr(rfa, "fit_regional_gev", fake_fit_regional_gev)
    monkeypatch.setattr(rfa, "return_level", fake_return_level)

    result = rfa.regional_return_levels(
        {"A": np.array([1.0, 2.0]), "B": np.array([2.0, 4.0])},
        T_values=[10, 50],
    )

    assert list(result.index) == ["A", "B"]
    assert list(result.columns) == ["T10", "T50"]
    assert result.loc["A"].to_dict() == {"T10": 10.0, "T50": 50.0}
    assert result.loc["B"].to_dict() == {"T10": 20.0, "T50": 100.0}

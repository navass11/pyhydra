"""
Time Series Analysis — event extraction and extreme value analysis.
"""

from pyhydra.climate.time_series import events, extremes
from pyhydra.climate.time_series.events import (
    extract_events,
    extract_discharge_events,
    extract_precipitation_events,
    extract_precipitation_events_pot,
    extract_precipitation_events_nday,
    extract_concurrent_events,
)
from pyhydra.climate.time_series.extremes import (
    # extraction
    extract_block_maxima,
    extract_pot,
    threshold_stability_plot,
    # GEV fitting
    fit_gev,
    fit_gev_map,
    fit_gev_fisher,
    fit_gev_mcmc,
    # GPD fitting
    fit_gpd,
    # return levels
    return_level_gev,
    return_level_gpd,
    return_levels,
    return_level_ci,
    # diagnostics
    plot_return_levels,
    plot_diagnostic,
)

__all__ = [
    "events",
    "extremes",
    # event extraction
    "extract_events",
    "extract_discharge_events",
    "extract_precipitation_events",
    "extract_precipitation_events_pot",
    "extract_precipitation_events_nday",
    "extract_concurrent_events",
    # block maxima / POT extraction
    "extract_block_maxima",
    "extract_pot",
    "threshold_stability_plot",
    # GEV fitting
    "fit_gev",
    "fit_gev_map",
    "fit_gev_fisher",
    "fit_gev_mcmc",
    # GPD fitting
    "fit_gpd",
    # return levels
    "return_level_gev",
    "return_level_gpd",
    "return_levels",
    "return_level_ci",
    # diagnostics
    "plot_return_levels",
    "plot_diagnostic",
]

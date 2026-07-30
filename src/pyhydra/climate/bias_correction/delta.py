import numpy as np
import pandas as pd


def delta_method(serie_raw, serie_hist, serie_mod, var, stat):
    """
    Apply the Delta method to project a raw observed series to future conditions.

    Args:
        serie_raw: Observed series to be modified (pandas Series with DatetimeIndex)
        serie_hist: Historical model series (pandas Series with DatetimeIndex)
        serie_mod: Future model series (pandas Series with DatetimeIndex)
        var: Variable type — 'pr' for precipitation (multiplicative delta),
             any other string for additive delta (e.g. temperature)
        stat: How to summarise multi-model deltas — 'mean' or 'median'

    Returns:
        pandas Series with bias-corrected values and time index shifted to the future period
    """
    # Ensure inputs are 1-D Series regardless of whether caller passed DataFrames
    if isinstance(serie_hist, pd.DataFrame):
        serie_hist = serie_hist.squeeze()
    if isinstance(serie_mod, pd.DataFrame):
        serie_mod = serie_mod.squeeze()

    serie_modify = serie_raw.copy()
    dif_year = serie_mod.index.year[0] - serie_raw.index.year[0]

    if var == "pr":
        serie_hist_month = serie_hist.resample("ME").sum().dropna()
        serie_mod_month = serie_mod.resample("ME").sum().dropna()
    else:
        serie_hist_month = serie_hist.resample("ME").mean().dropna()
        serie_mod_month = serie_mod.resample("ME").mean().dropna()

    if var == "pr":
        delta = (
            serie_mod_month.groupby(serie_mod_month.index.month).mean()
            / serie_hist_month.groupby(serie_hist_month.index.month).mean()
        )
    else:
        delta = (
            serie_mod_month.groupby(serie_mod_month.index.month).mean()
            - serie_hist_month.groupby(serie_hist_month.index.month).mean()
        )

    # For multi-model DataFrames aggregate across columns; Series is already 1-D
    if isinstance(delta, pd.DataFrame):
        if stat == "mean":
            delta = delta.mean(axis=1)
        elif stat == "median":
            delta = delta.median(axis=1)

    for month in range(1, 13):
        mask = serie_modify.index.month == month
        raw_vals = serie_raw[serie_raw.index.month == month].values
        if var == "pr":
            serie_modify[mask] = raw_vals * delta.loc[month]
        else:
            serie_modify[mask] = raw_vals + delta.loc[month]

    serie_modify.index = serie_modify.index + pd.DateOffset(years=dif_year)
    return serie_modify

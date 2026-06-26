"""Time series performance statistics (NSE, KGE, PBIAS)."""

from __future__ import annotations
import numpy as np

def compute_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency (NSE)."""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    if len(o) == 0:
        return np.nan
    denominator = np.sum((o - o.mean())**2)
    if denominator == 0:
        return np.nan
    return float(1 - np.sum((o - s)**2) / denominator)

def compute_kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta Efficiency (KGE) — 2009 formulation."""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    if len(o) == 0:
        return np.nan
    std_obs = o.std()
    mean_obs = o.mean()
    if std_obs == 0 or mean_obs == 0:
        return np.nan
    r_matrix = np.corrcoef(o, s)
    if r_matrix.shape == (2, 2):
        r = r_matrix[0, 1]
    else:
        r = 0.0
    alpha = s.std() / std_obs
    beta = s.mean() / mean_obs
    return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

def compute_pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """Percent Bias (PBIAS)."""
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    if len(o) == 0 or o.sum() == 0:
        return np.nan
    return float(100.0 * np.sum(s - o) / np.sum(o))

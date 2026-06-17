"""
Example 03 — Stochastic Time Series Generation with CoSMoS_py
===============================================================
Demonstrates fitting a seasonal stochastic model to a discharge series and
generating synthetic realisations that preserve the observed statistics.

Requires: pip install -e /path/to/CoSMoS_py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyhydra.climate.stochastic_generation import (
    analyze_ts,
    fit_distribution,
    report_ts,
    simulate_ts,
)

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Synthetic "observed" discharge series (30 years daily)
# ---------------------------------------------------------------------------
dates = pd.date_range("1990-01-01", "2019-12-31", freq="D")
n = len(dates)

# Seasonal signal + lognormal noise
month = pd.Series(dates.month)
seasonal = 20 + 15 * np.cos((month - 1) * 2 * np.pi / 12).values
noise    = rng.lognormal(mean=0, sigma=0.6, size=n)
Q        = np.maximum(0, seasonal * noise)

# Add ~15% zero-flow periods (dry season)
dry_mask = (dates.month.isin([7, 8])) & (rng.random(n) < 0.5)
Q[dry_mask] = 0.0

series = pd.Series(Q, index=dates, name="discharge_m3s")

print(f"Observed series: {len(series)} days  mean={series.mean():.2f}  p0={( series==0).mean():.2%}")

# ---------------------------------------------------------------------------
# 1. Fit marginal distribution to non-zero values
# ---------------------------------------------------------------------------
nonzero = series[series > 0]
dist_fit = fit_distribution(nonzero, dist="gengamma")
print(f"\nFitted distribution: {dist_fit['dist']}")
print(f"  params: {dist_fit['params_dict']}")
print(f"  objective (fitting error): {dist_fit['objective']:.5f}")

# ---------------------------------------------------------------------------
# 2. Analyse full seasonal model (distribution + autocorrelation per month)
# ---------------------------------------------------------------------------
print("\nFitting seasonal stochastic model …")
model = analyze_ts(series, season="month", dist="gengamma", acs_id="weibull")

summary = report_ts(model, method="stat")
print("\nSeasonal statistics summary (first 4 months):")
print(summary.head(4).to_string())

# ---------------------------------------------------------------------------
# 3. Generate 5 synthetic realisations (same period as obs)
# ---------------------------------------------------------------------------
print("\nGenerating 5 synthetic realisations …")
synthetics = [simulate_ts(model) for _ in range(5)]

# ---------------------------------------------------------------------------
# Plot: observed vs ensemble of synthetic series
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Monthly mean comparison
ax = axes[0]
obs_monthly = series.resample("ME").mean()
ax.plot(obs_monthly.index, obs_monthly.values, "k-", lw=2, label="Observed", zorder=5)
for i, syn in enumerate(synthetics):
    syn_s = pd.Series(syn["value"].values, index=pd.to_datetime(syn["date"]))
    ax.plot(syn_s.resample("ME").mean(), alpha=0.4, lw=0.8, label=f"Sim {i+1}")
ax.set_title("Monthly mean discharge — observed vs synthetic ensemble")
ax.set_ylabel("Q (m³/s)")
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.3)

# Empirical CDF comparison
ax = axes[1]
sorted_obs = np.sort(series[series > 0].values)
p = np.linspace(0, 1, len(sorted_obs))
ax.plot(sorted_obs, p, "k-", lw=2, label="Observed", zorder=5)
for i, syn in enumerate(synthetics):
    v = syn["value"].values
    v_nz = np.sort(v[v > 0])
    ax.plot(v_nz, np.linspace(0, 1, len(v_nz)), alpha=0.5, lw=0.8, label=f"Sim {i+1}")
ax.set_xscale("log")
ax.set_title("Empirical CDF (non-zero values, log scale)")
ax.set_xlabel("Q (m³/s)")
ax.set_ylabel("P(X ≤ x)")
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("stochastic_cosmos.png", dpi=120)
print("\nFigure saved: stochastic_cosmos.png")

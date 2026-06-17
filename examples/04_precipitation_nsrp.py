"""
Example 04 — Stochastic Precipitation with NEOPRENE (NSRP)
============================================================
Demonstrates the single-site NSRP workflow using NSRPModel:

  1. Configure the model
  2. Compute observed statistics
  3. Calibrate NSRPM parameters via PSO
  4. Generate synthetic daily precipitation
  5. Validate statistics

Requires: pip install NEOPRENE
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyhydra.climate.stochastic_generation import NSRPModel

# ---------------------------------------------------------------------------
# Synthetic "observed" daily precipitation (30 years)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(99)
dates = pd.date_range("1990-01-01", "2019-12-31", freq="D")
n = len(dates)

# Seasonal occurrence probability
month = pd.Series(dates.month, index=dates)
p_wet = 0.30 + 0.20 * np.cos((month.values - 1) * 2 * np.pi / 12)

# Wet-day amounts (Gamma distributed)
wet = rng.random(n) < p_wet
amounts = rng.gamma(shape=1.5, scale=4.0, size=n)
PR = np.where(wet, amounts, 0.0)
rainfall = pd.Series(PR, index=dates, name="precipitation_mm")

print(f"Observed rainfall: {n} days  "
      f"mean={rainfall.mean():.2f} mm/d  "
      f"p0={(rainfall == 0).mean():.1%}")

# ---------------------------------------------------------------------------
# Configure NSRPModel
# ---------------------------------------------------------------------------
model = NSRPModel(
    temporal_resolution="d",
    seasonality="monthly",
    process="normal",
    statistics=["mean", "var_h", "autocorr_l_h", "fih_h", "fiWW_h", "fiDD_h"],
    n_iterations=150,
    n_bees=30,
    n_initializations=1,
    model_bounds={
        "time_between_storms":     [1.0, 72.0],
        "number_storm_cells":      [1.0, 15.0],
        "cell_duration":           [0.5, 24.0],
        "cell_intensity":          [0.1,  8.0],
        "storm_cell_displacement": [0.0,  0.0],
    },
)

print("\nModel configuration:")
print(f"  seasonality  : {model.seasonality}")
print(f"  statistics   : {model.statistics_name}")
print(f"  PSO bees     : {model.n_bees}")
print(f"  PSO iterations: {model.n_iterations}")

# ---------------------------------------------------------------------------
# Full workflow: statistics → calibration → simulation
# ---------------------------------------------------------------------------
print("\nStep 1: computing statistics …")
stats_obj = model.compute_statistics(rainfall)
print("  Done.")

print("Step 2: calibrating (PSO) …")
cal_result = model.calibrate(verbose=False)
print("  Done.")

print("\nCalibration summary (observed vs fitted):")
summary = model.summary()
if summary is not None:
    print(summary.head(8).to_string())

print("\nStep 3: simulating 2020–2069 (50 years) …")
sim_result = model.simulate(year_ini=2020, year_fin=2069)
daily_sim = sim_result.Daily_Simulation
print(f"  Generated: {len(daily_sim)} days")

# ---------------------------------------------------------------------------
# Validation plot
# ---------------------------------------------------------------------------
sim_series = pd.Series(
    daily_sim.values.flatten(),
    index=pd.date_range("2020-01-01", periods=len(daily_sim), freq="D"),
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Monthly mean
ax = axes[0]
obs_mm = rainfall.resample("ME").mean()
sim_mm = sim_series.resample("ME").mean()
ax.plot(obs_mm.index, obs_mm.values, "k-", lw=1.5, label="Observed")
ax.plot(sim_mm.index, sim_mm.values, "r-", lw=1.0, alpha=0.7, label="Synthetic")
ax.set_title("Monthly mean precipitation")
ax.set_ylabel("mm/day")
ax.legend()
ax.grid(True, alpha=0.3)

# Monthly wet-day fraction
ax = axes[1]
obs_p0 = rainfall.resample("ME").apply(lambda x: (x > 0).mean())
sim_p0 = sim_series.resample("ME").apply(lambda x: (x > 0).mean())
ax.plot(obs_p0.index, obs_p0.values, "k-", lw=1.5, label="Observed")
ax.plot(sim_p0.index, sim_p0.values, "r-", lw=1.0, alpha=0.7, label="Synthetic")
ax.set_title("Monthly wet-day fraction")
ax.set_ylabel("Fraction")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("NSRP validation — observed vs synthetic", fontsize=13)
plt.tight_layout()
plt.savefig("nsrp_validation.png", dpi=120)
print("\nFigure saved: nsrp_validation.png")

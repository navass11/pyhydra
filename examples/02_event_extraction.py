"""
Example 02 — Threshold-based Event Extraction
===============================================
Demonstrates extracting flood events from a discharge series and wet spells
from a precipitation series using pyhydra.climate.time_series.events.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyhydra.climate.time_series import (
    extract_discharge_events,
    extract_precipitation_events,
)

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Synthetic discharge series: background flow + 4 flood pulses
# ---------------------------------------------------------------------------
n = 730  # 2 years
dates = pd.date_range("2015-01-01", periods=n, freq="D")
Q = np.full(n, 8.0) + rng.uniform(0, 3, n)

events_def = [
    (40,  55,  180),   # start, peak_day, peak_Q
    (130, 145, 320),
    (350, 365, 250),
    (500, 510, 140),
]
for start, peak, peak_q in events_def:
    rise_len = peak - start
    fall_len = int(rise_len * 1.8)
    Q[start:peak] = np.linspace(Q[start], peak_q, rise_len)
    end = min(peak + fall_len, n)
    Q[peak:end] = np.linspace(peak_q, 8.0, end - peak)

series_q = pd.Series(Q, index=dates, name="discharge_m3s")

threshold_q = 50.0   # m³/s event threshold
threshold_q2 = 100.0  # minimum peak to retain

stats_q, bounds_q = extract_discharge_events(
    series_q, threshold=threshold_q, threshold2=threshold_q2
)

print("=== Discharge events ===")
print(stats_q.to_string(index=False))

# ---------------------------------------------------------------------------
# Synthetic precipitation series: alternating wet/dry spells
# ---------------------------------------------------------------------------
PR = np.zeros(n)
wet_spells = [
    (10, [1, 4, 12, 18, 9, 3]),
    (60, [0.5, 6, 22, 15, 8, 2, 1]),
    (200, [3, 11, 7]),
    (290, [0.2]),                       # trace: filtered if min_duration=2
    (400, [2, 5, 30, 25, 10, 4, 1, 0.5]),
]
for start, vals in wet_spells:
    PR[start:start + len(vals)] = vals

series_pr = pd.Series(PR, index=dates, name="precipitation_mm")

stats_pr, bounds_pr = extract_precipitation_events(
    series_pr, threshold=0.5, min_duration=2, min_gap=1
)

print("\n=== Precipitation events ===")
print(stats_pr.to_string(index=False))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# — Discharge
ax = axes[0]
ax.plot(dates, Q, color="steelblue", linewidth=1, label="Discharge")
ax.axhline(threshold_q, color="orange", linestyle="--", label=f"Threshold {threshold_q} m³/s")
ax.axhline(threshold_q2, color="red", linestyle=":", label=f"Peak filter {threshold_q2} m³/s")
for _, row in bounds_q.iterrows():
    ax.axvspan(dates[row["start"]], dates[row["end"]], alpha=0.15, color="red")
ax.set_ylabel("Q (m³/s)")
ax.set_title("Discharge events (shaded)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# — Precipitation
ax = axes[1]
ax.bar(dates, PR, color="royalblue", width=1, label="Precipitation")
for _, row in bounds_pr.iterrows():
    ax.axvspan(dates[row["start"]], dates[row["end"]], alpha=0.25, color="green")
ax.set_ylabel("P (mm)")
ax.set_title("Precipitation wet-spell events (shaded)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("event_extraction.png", dpi=120)
print("\nFigure saved: event_extraction.png")
print(f"\nDischarge: {len(stats_q)} events found (threshold_peak={threshold_q2})")
print(f"Precipitation: {len(stats_pr)} events found (min_duration=2, min_gap=1)")

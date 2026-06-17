"""
Example 01 — Bias Correction
==============================
Demonstrates all three bias correction methods available in pyhydra:

  - Quantile Mapping (QM)
  - Quantile Delta Mapping (QDM)
  - Scaled Distribution Mapping (SDM) — precipitation and temperature
  - Delta Method — additive (temperature) and multiplicative (precipitation)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyhydra.climate.bias_correction import BiasCorrection, delta_method

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Synthetic data: model has a +40 % wet bias and underestimates variability
# ---------------------------------------------------------------------------
n_obs = 1000
obs = rng.gamma(shape=2.0, scale=3.0, size=n_obs)          # "truth"
mod = rng.gamma(shape=2.0, scale=4.2, size=n_obs)          # biased historical
sce = rng.gamma(shape=2.2, scale=4.5, size=n_obs)          # biased future scenario

print("=== Raw statistics ===")
print(f"  obs  mean={obs.mean():.2f}  std={obs.std():.2f}")
print(f"  mod  mean={mod.mean():.2f}  std={mod.std():.2f}  (historical, biased)")
print(f"  sce  mean={sce.mean():.2f}  std={sce.std():.2f}  (future, biased)")

bc = BiasCorrection(obs=obs, mod=mod, sce=sce)

# ---------------------------------------------------------------------------
# 1. Quantile Mapping (additive correction)
# ---------------------------------------------------------------------------
qm = bc.quantile_mapping()
print(f"\n[QM]   mean={qm.mean():.2f}  std={qm.std():.2f}")

# ---------------------------------------------------------------------------
# 2. Quantile Delta Mapping (multiplicative — better for precipitation)
# ---------------------------------------------------------------------------
qdm = bc.quantile_deltamapping()
print(f"[QDM]  mean={qdm.mean():.2f}  std={qdm.std():.2f}")

# ---------------------------------------------------------------------------
# 3. Scaled Distribution Mapping — precipitation (gamma parametric fit)
# ---------------------------------------------------------------------------
sdm_pr = bc.scaled_distribution_mapping("precipitation")
print(f"[SDM-pr]  mean={sdm_pr.mean():.2f}  std={sdm_pr.std():.2f}")

# ---------------------------------------------------------------------------
# 4. SDM — temperature (normal parametric fit)
# ---------------------------------------------------------------------------
obs_t = rng.normal(15, 3, n_obs)
mod_t = rng.normal(17, 3, n_obs)     # +2°C bias
sce_t = rng.normal(20, 3, n_obs)
bc_t = BiasCorrection(obs_t, mod_t, sce_t)
sdm_t = bc_t.scaled_distribution_mapping("temperature")
print(f"[SDM-T]   mean={sdm_t.mean():.2f}  (obs mean={obs_t.mean():.2f})")

# ---------------------------------------------------------------------------
# 5. Delta Method
# ---------------------------------------------------------------------------
idx_obs  = pd.date_range("1980-01-01", "2009-12-31", freq="D")
idx_mod  = pd.date_range("2070-01-01", "2099-12-31", freq="D")

base = 15 + 8 * np.sin(np.linspace(0, 6 * np.pi, len(idx_obs)))
obs_s  = pd.Series(base + rng.normal(0, 1, len(idx_obs)),       index=idx_obs)
hist_s = pd.Series(base + rng.normal(0, 1, len(idx_obs)),       index=idx_obs)
fut_s  = pd.Series(base + 3.0 + rng.normal(0, 1, len(idx_mod)), index=idx_mod)

delta_result = delta_method(obs_s, hist_s, fut_s, var="tas", stat="mean")
print(f"\n[Delta-T]  obs mean={obs_s.mean():.2f}  future mean={delta_result.mean():.2f}")
print(f"           Index shifted to: {delta_result.index[0].year}–{delta_result.index[-1].year}")

# ---------------------------------------------------------------------------
# Plot: empirical CDF comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, (title, data_dict) in zip(axes, [
    ("Precipitation bias correction", {
        "Observed": obs, "Biased scenario": sce,
        "QM": qm, "QDM": qdm, "SDM": sdm_pr,
    }),
    ("Temperature bias correction", {
        "Observed": obs_t, "Biased scenario": sce_t, "SDM": sdm_t,
    }),
]):
    for label, arr in data_dict.items():
        sorted_arr = np.sort(arr)
        p = np.linspace(0, 1, len(sorted_arr))
        ax.plot(sorted_arr, p, label=label)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Empirical CDF")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("bias_correction_comparison.png", dpi=120)
print("\nFigure saved: bias_correction_comparison.png")

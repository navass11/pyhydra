"""
Correlated Manning robustness check via importance sampling.

Uses the existing 1000 SFINCS simulations (combinaciones_rugosidad.csv +
comparison_sfincs_hecras_clean.csv) to estimate regime probabilities under
a Gaussian copula with rho=0.5, without running new hydraulic simulations.

Method:
  For each simulation i with Manning vector n_i:
    1. Transform to uniform via empirical rank: u_ij = rank_ij / (N+1)
    2. Transform to standard normal: z_ij = Phi^{-1}(u_ij)
    3. Compute importance weight: w_i = N(z_i; 0, Sigma_rho) / N(z_i; 0, I)
    4. Normalize weights and compute weighted regime fractions.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.mixture import GaussianMixture

# ── Paths ─────────────────────────────────────────────────────────────────────
COMB_CSV = Path("/Volumes/My Passport 2/OneDrive/Scripts_Python/Paper_Rugosidades/combinaciones_rugosidad.csv")
RES_CSV  = Path("/Users/salvadornavasfernandez/Desktop/Github/HYDRA"
                "/notebooks/modeling/hydraulic/manning_sensitivity"
                "/comparison_sfincs_hecras_clean.csv")

# ── Load data ─────────────────────────────────────────────────────────────────
comb = pd.read_csv(COMB_CSV)          # 1000 × 9 Manning values
res  = pd.read_csv(RES_CSV, index_col=0)  # 995 rows (5 removed)

# Align: combinaciones has 1000 rows (index 0-999), res has 995 (some dropped)
# Use res index to select corresponding combinations
comb_aligned = comb.loc[res.index].values   # (995, 9)
print(f"Aligned: {comb_aligned.shape[0]} simulations, {comb_aligned.shape[1]} Manning classes")

N, p = comb_aligned.shape

# ── Regime classification (GMM on HEC-RAS flooded area, matching paper) ──────
area = res["hecras_area_km2"].values
gmm = GaussianMixture(n_components=2, random_state=42, n_init=10)
lab = gmm.fit_predict(area.reshape(-1, 1))
if gmm.means_.flatten()[0] > gmm.means_.flatten()[1]:
    lab = 1 - lab   # 0=low, 1=high

n_low  = (lab == 0).sum()
n_high = (lab == 1).sum()
pct_high_indep = n_high / N * 100
print(f"\nIndependent sampling (original):")
print(f"  Low regime:  N={n_low}  ({n_low/N*100:.1f}%)")
print(f"  High regime: N={n_high} ({pct_high_indep:.1f}%)")

# ── Empirical PIT (probability integral transform) ───────────────────────────
# u_ij = rank of n_ij among all 995 values for class j, scaled to (0,1)
from scipy.stats import rankdata
U = np.zeros_like(comb_aligned, dtype=float)
for j in range(p):
    U[:, j] = rankdata(comb_aligned[:, j]) / (N + 1)

# Transform to standard normal
Z = stats.norm.ppf(U)   # (N, p), clip extreme values
Z = np.clip(Z, -4, 4)

# ── Importance weights for equicorrelation Gaussian copula ───────────────────
def log_weight_equicorr(Z, rho):
    """
    Log importance weight: log[N(z; 0, Sigma_rho) / N(z; 0, I)]
    For equicorrelation matrix Sigma_rho = (1-rho)*I + rho*11^T.

    Closed-form log density ratio:
      -0.5 * [z^T (Sigma_rho^{-1} - I) z] - 0.5 * log|Sigma_rho|

    Equicorrelation inverse:
      Sigma_rho^{-1} = 1/(1-rho) * I - rho/((1-rho)*(1+(p-1)*rho)) * 11^T

    Log determinant:
      log|Sigma_rho| = (p-1)*log(1-rho) + log(1+(p-1)*rho)
    """
    p = Z.shape[1]
    a = 1.0 / (1 - rho)
    b = rho / ((1 - rho) * (1 + (p - 1) * rho))

    # z^T (Sigma^{-1} - I) z = (a-1)*||z||^2 - b*(sum z)^2
    zz  = np.sum(Z ** 2, axis=1)          # (N,)
    zs  = np.sum(Z, axis=1) ** 2          # (N,)  (sum of z_j)^2
    quad = (a - 1) * zz - b * zs

    log_det = (p - 1) * np.log(1 - rho) + np.log(1 + (p - 1) * rho)
    return -0.5 * quad - 0.5 * log_det

def weighted_regime(Z, lab, rho, label=""):
    lw = log_weight_equicorr(Z, rho)
    lw -= lw.max()          # numerical stability
    w  = np.exp(lw)
    w /= w.sum()            # normalize

    eff_n = 1.0 / np.sum(w ** 2)   # effective sample size

    w_high = w[lab == 1].sum() * 100
    w_low  = w[lab == 0].sum() * 100

    print(f"\nCorrelated sampling rho={rho} {label}:")
    print(f"  Low regime:  {w_low:.1f}%")
    print(f"  High regime: {w_high:.1f}%")
    print(f"  Effective N: {eff_n:.0f} / {N}")
    return w_low, w_high, eff_n

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
for rho in [0.3, 0.5, 0.8]:
    weighted_regime(Z, lab, rho)

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("Summary for paper:")
print(f"  Independent (rho=0): High = {pct_high_indep:.1f}%")
for rho in [0.3, 0.5, 0.8]:
    wl, wh, en = weighted_regime(Z, lab, rho)
    print(f"  Correlated  rho={rho:.1f}: High = {wh:.1f}%  (eff. N={en:.0f})")

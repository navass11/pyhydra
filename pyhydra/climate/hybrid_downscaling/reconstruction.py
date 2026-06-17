"""
Representative event selection (MaxDiss) and synthetic hydrograph reconstruction.

Given a synthetic event matrix (Qmax, Qmed, Duration, shape_type) produced by
the copula sampling step, this module:

1. Selects the most dissimilar representative events using the **MaxDiss**
   algorithm so that the full ensemble can be reconstructed from a small
   number of hydraulic simulations.
2. Scales the observed hydrograph shape of each type to match the synthetic
   (Qmax, Qmed, Duration) parameters, producing one hydrograph CSV per centroid.

The representative centroids and updated synthetic matrix are saved to
``output_dir`` for use in the downstream flood-map interpolation step.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# MaxDiss helpers
# ---------------------------------------------------------------------------

def _scalar_distance(data, s, k, aux_d2):
    """Mixed scalar + circular distance from row *s* to all rows of *data*."""
    d = data[s, :] - data
    circ = np.abs(d[:, k:])
    d[:, k:] = np.minimum(circ, aux_d2 - circ) / math.pi
    return np.sum(d ** 2, axis=1)


def maxdiss(data, n_select, scalar_cols, seed_positions):
    """
    Select *n_select* maximally dissimilar rows from *data*.

    The first ``len(seed_positions)`` selections are pre-seeded (one per
    hydrograph type); subsequent selections maximise the minimum distance to
    already-selected rows.

    Args:
        data:            2-D array (n_samples × n_features), normalised.
        n_select:        Total number of representative cases to choose.
        scalar_cols:     Column indices treated as scalar (vs. circular).
        seed_positions:  Pre-seeded row indices (one per shape type).

    Returns:
        subset:    (n_select × n_features) array of selected rows.
        positions: list of selected row indices into *data*.
    """
    nx, ny = data.shape
    k = len(scalar_cols)
    aux_d2 = 2 * math.pi * np.ones((nx, ny - k))

    seed = int(np.argmax(data[:, 0]))
    subset = np.zeros((n_select, ny))
    subset[0] = data[seed]
    positions = [seed]

    dist_last = _scalar_distance(data, seed, k, aux_d2)
    pos_max = int(dist_last.argmax())
    subset[1] = data[pos_max]
    dist_last[pos_max] = 0.0
    dist_last[seed] = 0.0
    positions.append(pos_max)

    n_seeds = len(seed_positions)
    for n in range(2, n_select):
        if n < n_seeds + 2:
            positions.append(int(seed_positions[n - 2]))
        else:
            dist_new = _scalar_distance(data, pos_max, k, aux_d2)
            dist_last = np.minimum(dist_new, dist_last)
            pos_max = int(dist_last.argmax())
            subset[n] = data[pos_max]
            dist_last[pos_max] = 0.0
            positions.append(pos_max)

    return subset, positions


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class HydrographReconstructor:
    """
    Select representative synthetic events and build scaled hydrographs.

    Parameters
    ----------
    discharge : pd.Series
        Observed discharge series (DatetimeIndex).
    synthetic_matrix : pd.DataFrame
        Synthetic event table with columns ``Qmax``, ``Qmed``, ``Duracion``,
        ``shape_type`` (output of the copula sampling step).
    classified_events : pd.DataFrame
        Observed events with columns ``Qmax``, ``Qmed``, ``Duracion``,
        ``shape_type``, ``Inicio_evento``, ``Fin_evento`` (output of
        :class:`~pyhydra.climate.hybrid_downscaling.classification.HydrographClassifier`).
    n_types : int
        Number of hydrograph shape types.
    n_representatives : int
        Total number of representative centroids to pass to hydraulic
        simulations (default 400 → 20 × 20 grid).
    output_dir : str or Path
        Where to write ``Hidrograma_*.csv`` and ``centroids.csv``.
    plot : bool
        Plot synthetic vs. observed hydrographs per type (default False).
    """

    def __init__(
        self,
        discharge,
        synthetic_matrix,
        classified_events,
        n_types,
        n_representatives=400,
        output_dir=".",
        plot=False,
    ):
        self.discharge = discharge
        self.synthetic_matrix = synthetic_matrix.reset_index(drop=True)
        self.classified = classified_events.reset_index(drop=True)
        self.n_types = n_types
        self.n_repr = n_representatives
        self.output_dir = Path(output_dir)
        self.plot = plot
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — MaxDiss representative selection
    # ------------------------------------------------------------------

    def select_representatives(self):
        """
        Apply MaxDiss to the normalised synthetic matrix and return centroids.

        Returns
        -------
        centroids : pd.DataFrame
            (n_types + n_representatives) rows with columns
            ``Qmax``, ``Qmed``, ``Duracion``, ``shape_type``.
        synthetic_matrix : pd.DataFrame
            Copy of the synthetic matrix with an added ``cluster`` column
            mapping each event to its nearest centroid.
        """
        M = self.synthetic_matrix.copy()
        cols = ["Qmax", "Qmed", "Duracion"]

        # Seed positions: index of the highest-Qmax event for each type
        seed_positions = []
        for t in range(self.n_types):
            subset = M[M["shape_type"] == t]
            if subset.empty:
                seed_positions.append(0)
            else:
                seed_positions.append(int(np.where(M["Qmax"] == subset["Qmax"].max())[0][0]))

        norm = (M[cols] - M[cols].mean()) / M[cols].std()
        selected, sel_idx = maxdiss(
            norm.values, self.n_repr, list(range(len(cols))), seed_positions
        )

        # De-normalise centroids
        selected_df = pd.DataFrame(selected, columns=cols)
        selected_denorm = selected_df * M[cols].std().values + M[cols].mean().values

        # Build centroids table: first n_types rows = type seeds, rest = MaxDiss
        centroids = pd.DataFrame(
            index=range(self.n_types + self.n_repr),
            columns=cols + ["shape_type"],
        )
        for k in range(self.n_types):
            idx = seed_positions[k]
            centroids.loc[k, cols] = M.loc[idx, cols].values
            centroids.loc[k, "shape_type"] = M.loc[idx, "shape_type"]

        centroids.loc[self.n_types:, cols] = selected_denorm.values
        for x, orig_idx in enumerate(sel_idx):
            centroids.loc[self.n_types + x, "shape_type"] = M.loc[orig_idx, "shape_type"]

        # Assign each synthetic event to its nearest centroid (normalised space)
        cent_norm = (centroids[cols].astype(float) - M[cols].mean().values) / M[cols].std().values
        cluster_labels = []
        for i in range(len(norm)):
            d = ((cent_norm["Qmax"] - norm["Qmax"].iloc[i]) ** 2 +
                 (cent_norm["Qmed"] - norm["Qmed"].iloc[i]) ** 2 +
                 (cent_norm["Duracion"] - norm["Duracion"].iloc[i]) ** 2) ** 0.5
            cluster_labels.append(int(d.argmin()))
        M["cluster"] = cluster_labels

        self._centroids = centroids
        self._synthetic_matrix = M
        return centroids, M

    # ------------------------------------------------------------------
    # Step 2 — Hydrograph reconstruction
    # ------------------------------------------------------------------

    def build_hydrographs(self):
        """
        Scale the representative observed shape to each centroid's parameters
        and write one ``Hidrograma_{j}.csv`` per centroid.

        Returns
        -------
        centroids : pd.DataFrame
        synthetic_matrix : pd.DataFrame
        """
        if not hasattr(self, "_centroids"):
            self.select_representatives()

        centroids = self._centroids
        cols = ["Qmax", "Qmed", "Duracion"]
        Q_vals = self.discharge.values

        if self.plot:
            side = int(self.n_types ** 0.5)
            fig, axes = plt.subplots(side, side, figsize=(24, 24))

        for j, row in centroids.iterrows():
            tipo = int(row["shape_type"])

            # Find the observed event with maximum Qmax for this type
            obs_type = self.classified[self.classified["shape_type"] == tipo]
            if obs_type.empty:
                continue
            peak_idx = obs_type["Qmax"].idxmax()
            x1 = int(self.classified.loc[peak_idx, "Inicio_evento"])
            x2 = int(self.classified.loc[peak_idx, "Fin_evento"])
            segment = Q_vals[x1: x2 + 1]

            # Upsample to hourly resolution
            t_orig = np.arange(len(segment), dtype=float)
            t_fine = np.linspace(t_orig[0], t_orig[-1], len(t_orig) * 24)
            Q_fine = np.interp(t_fine, t_orig, segment)

            Qmax_j = float(row["Qmax"])
            Qmed_j = float(row["Qmed"])
            Dur_j = float(row["Duracion"])

            # Quadratic scaling: a*Q² + b*Q matches Qmax_j and Qmed_j
            denom = (Q_fine.sum() ** 2 * Q_fine.max()
                     - Q_fine.max() ** 2 * Q_fine.sum())
            if denom != 0:
                a = (len(Q_fine) * Qmed_j * Q_fine.max()
                     - Qmax_j * Q_fine.sum()) / denom
            else:
                a = 0.0
            b = (Qmax_j - a * Q_fine.max() ** 2) / Q_fine.max()

            Q_synth = np.maximum(a * Q_fine ** 2 + b * Q_fine, 0.0)
            t_out = t_fine / t_fine.max() * Dur_j

            out = pd.DataFrame({"Q_m3/s": Q_synth}, index=t_out * 24 * 3600)
            out.to_csv(self.output_dir / f"Hidrograma_{j}.csv")

            if self.plot:
                r, c = tipo // int(self.n_types ** 0.5), tipo % int(self.n_types ** 0.5)
                axes[r, c].plot(t_out, Q_synth, "b-", alpha=0.4)
                axes[r, c].plot(t_fine, Q_fine, "r-", alpha=0.4)

        centroids.to_csv(self.output_dir / "centroids.csv")
        return centroids, self._synthetic_matrix

    # ------------------------------------------------------------------
    # Convenience: run both steps at once
    # ------------------------------------------------------------------

    def run(self):
        """Select representatives and build all hydrographs in one call."""
        self.select_representatives()
        return self.build_hydrographs()

"""
Threshold-based event extraction for discharge and precipitation time series.

Provides standalone functions extracted from the discretization workflow:
- extract_discharge_events(): inflection-point method (generalised from generacion_eventos_sinteticos)
- extract_precipitation_events(): consecutive wet-spell method
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Discharge events
# ---------------------------------------------------------------------------

def extract_discharge_events(series, threshold, threshold2=None, plot=False):
    """
    Identify flood events in a discharge series using an inflection-point method.

    Events begin just before the rising limb crosses `threshold` and end when
    the falling limb drops back below it.  Only events whose peak exceeds
    `threshold2` are retained.

    Args:
        series: pd.Series of discharge values with a DatetimeIndex (or any
                monotonic index).  Values must be non-negative.
        threshold: Minimum discharge (same units as series) to define an event.
        threshold2: Minimum peak discharge to retain an event.  Defaults to
                    `threshold`.
        plot: If True, plot the largest event with start/end markers.

    Returns:
        Tuple (stats, bounds):
            stats  – DataFrame with columns [peak, mean, duration, volume, date_peak].
                     volume is in m³ (daily timestep assumed: Q × 86 400 s/day).
            bounds – DataFrame with columns [start, end] (dates from series.index).
    """
    if threshold2 is None:
        threshold2 = threshold

    values = np.asarray(series.values, dtype=float)
    n = len(values)
    x = np.arange(n)

    # ── slope sign series ──────────────────────────────────────────────────
    slope = np.sign(np.diff(values))  # +1 rising, -1 falling, 0 flat

    # ── find rising crossings (x_start) and falling crossings (x_end) ──────
    x_start, x_end = [], []
    for i in range(n - 1):
        m = values[i + 1] - values[i]
        if m > 0 and values[i] <= threshold < values[i + 1]:
            x_start.append(i)
        if m < 0 and values[i] >= threshold > values[i + 1]:
            x_end.append(i + 1)

    if not x_start or not x_end:
        return pd.DataFrame(columns=["peak", "mean", "duration", "volume", "date_peak"]), \
               pd.DataFrame(columns=["start", "end"])

    x_end_arr = np.array(x_end)

    # For each rising crossing, find the first falling crossing that comes AFTER it.
    # This matches the original eventos_caudal() pairing logic.
    x_end_paired = []
    valid_starts = []
    for xs in x_start:
        after = x_end_arr[x_end_arr > xs]
        if len(after) == 0:
            continue
        x_end_paired.append(int(after[0]))
        valid_starts.append(xs)
    x_start = valid_starts

    if not x_start:
        return pd.DataFrame(columns=["peak", "mean", "duration", "volume", "date_peak"]), \
               pd.DataFrame(columns=["start", "end"])

    # ── find inflection points (local minima) for event boundaries ──────────
    # slope changes from -1 → +1 mark the foot of recessions (local minima)
    diff_slope = np.diff(slope)
    inflex_min_idx = np.where(diff_slope == 2)[0] + 1  # indices in `values`

    def _nearest_inflex_before(xref):
        """Last local minimum at or before xref."""
        candidates = inflex_min_idx[inflex_min_idx <= xref]
        return int(candidates[-1]) if len(candidates) else max(0, xref - 1)

    def _nearest_inflex_after(xref):
        """First local minimum at or after xref."""
        candidates = inflex_min_idx[inflex_min_idx >= xref]
        return int(candidates[0]) if len(candidates) else min(n - 1, xref + 1)

    # ── collect event windows aligned to inflection points ─────────────────
    peaks, means, durations, volumes, date_peaks = [], [], [], [], []
    starts_out, ends_out = [], []

    index = series.index
    for xs, xe in zip(x_start, x_end_paired):
        evt_start = _nearest_inflex_before(xs)
        evt_end = _nearest_inflex_after(xe)

        segment = values[evt_start:evt_end + 1]
        if len(segment) == 0 or segment.max() < threshold2:
            continue

        peaks.append(float(segment.max()))
        means.append(float(segment.mean()))
        durations.append(evt_end - evt_start)
        volumes.append(float(segment.sum()) * 86400)  # m³ (assuming daily timestep)
        date_peaks.append(index[evt_start + int(np.argmax(segment))])
        starts_out.append(index[evt_start])
        ends_out.append(index[evt_end])

    stats = pd.DataFrame({
        "peak": peaks,
        "mean": means,
        "duration": durations,
        "volume": volumes,
        "date_peak": date_peaks,
    })
    bounds = pd.DataFrame({"start": starts_out, "end": ends_out})

    if plot and len(peaks):
        import matplotlib.pyplot as plt
        pos_max = int(np.argmax(peaks))
        # Use positional integer indices for start/end (stored as dates — look them up)
        evt_start_pos = list(index).index(starts_out[pos_max])
        evt_end_pos   = list(index).index(ends_out[pos_max])
        margin = max(30, (evt_end_pos - evt_start_pos) // 2)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(series.index, values, "--k", linewidth=1.5, label="Discharge")
        ax.axhline(threshold, linestyle="--", color="orange", label=f"Threshold = {threshold}")
        ax.axvline(starts_out[pos_max], color="green", linewidth=1, label="Event start")
        ax.axvline(ends_out[pos_max],   color="red",   linewidth=1, label="Event end")
        i0 = max(0, evt_start_pos - margin)
        i1 = min(n - 1, evt_end_pos + margin)
        ax.set_xlim(index[i0], index[i1])
        ax.set_ylim(0, max(peaks) * 1.1)
        ax.set_xlabel("Date"); ax.set_ylabel("Discharge")
        ax.legend(); ax.grid(True)
        plt.tight_layout()

    return stats, bounds


# ---------------------------------------------------------------------------
# Precipitation events
# ---------------------------------------------------------------------------

def extract_precipitation_events(series, threshold=0.0, min_duration=1, min_gap=1):
    """
    Extract wet spells from a precipitation series.

    Consecutive time steps with precipitation > `threshold` form an event.
    Gaps shorter than `min_gap` steps are bridged (events merged).
    Events shorter than `min_duration` steps after merging are discarded.

    Args:
        series: pd.Series of precipitation values with a DatetimeIndex.
        threshold: Minimum precipitation to consider a step wet (default 0).
        min_duration: Minimum event length in time steps (default 1).
        min_gap: Dry steps between two wet periods that are bridged into one
                 event (default 1, i.e. single dry days are bridged).

    Returns:
        Tuple (stats, bounds):
            stats  – DataFrame with [peak, total, duration, mean_intensity, date_start]
            bounds – DataFrame with [start, end] (positional integer indices)
    """
    values = np.asarray(series.values, dtype=float)
    wet = (values > threshold).astype(int)
    n = len(wet)

    # ── find raw wet runs ──────────────────────────────────────────────────
    changes = np.diff(np.concatenate([[0], wet, [0]]))
    run_starts = np.where(changes == 1)[0]
    run_ends = np.where(changes == -1)[0] - 1

    if len(run_starts) == 0:
        return (
            pd.DataFrame(columns=["peak", "total", "duration", "mean_intensity", "date_start"]),
            pd.DataFrame(columns=["start", "end"]),
        )

    # ── merge runs separated by short dry gaps ─────────────────────────────
    merged_starts = [run_starts[0]]
    merged_ends = [run_ends[0]]
    for rs, re in zip(run_starts[1:], run_ends[1:]):
        if rs - merged_ends[-1] - 1 <= min_gap:
            merged_ends[-1] = re
        else:
            merged_starts.append(rs)
            merged_ends.append(re)

    # ── compute event statistics ───────────────────────────────────────────
    index = series.index
    peaks, totals, durations, intensities, date_starts = [], [], [], [], []
    starts_out, ends_out = [], []

    for s0, s1 in zip(merged_starts, merged_ends):
        dur = s1 - s0 + 1
        if dur < min_duration:
            continue
        segment = values[s0:s1 + 1]
        peaks.append(float(segment.max()))
        totals.append(float(segment.sum()))
        durations.append(dur)
        intensities.append(float(segment[segment > threshold].mean()) if (segment > threshold).any() else 0.0)
        date_starts.append(index[s0])
        starts_out.append(index[s0])
        ends_out.append(index[s1])

    stats = pd.DataFrame({
        "peak": peaks,
        "total": totals,
        "duration": durations,
        "mean_intensity": intensities,
        "date_start": date_starts,
    })
    bounds = pd.DataFrame({"start": starts_out, "end": ends_out})
    return stats, bounds


# ---------------------------------------------------------------------------
# Multi-station concurrent event analysis
# ---------------------------------------------------------------------------

def extract_concurrent_events(event_bounds, series_dict, buffer_days=0,
                               stats=("max", "mean", "total")):
    """
    Extract statistics at multiple stations during the windows of events
    detected at a target station.

    For each event window [start, end] in event_bounds, computes summary
    statistics for every series in series_dict over the same window
    (optionally extended by buffer_days on each side to capture lagged
    downstream responses).

    Typical use cases
    -----------------
    - **Compound / concurrent extremes**: was a flood at station A also extreme
      at stations B, C, D?
    - **Spatial coherence**: do all catchments respond simultaneously?
    - **Multivariate extreme-value analysis**: build a joint dataset of maxima
      across stations for copula or MEVD fitting.

    Args:
        event_bounds: DataFrame with ``start`` and ``end`` columns — output of
                      any ``extract_*_events`` function.
        series_dict:  dict mapping station name → pd.Series, all sharing the
                      same DatetimeIndex.
        buffer_days:  Extend each event window by this many days on each side
                      (default 0 — exact event window only).
        stats:        Tuple of statistics to compute for each window.
                      Supported: ``'max'``, ``'mean'``, ``'total'``.

    Returns:
        pd.DataFrame with one row per event and columns:
        ``event_start``, ``event_end``,
        ``<name>_max``, ``<name>_mean``, ``<name>_total`` for each station.
    """
    if event_bounds.empty:
        return pd.DataFrame()

    rows = []
    for _, row in event_bounds.iterrows():
        t0 = row["start"] - pd.Timedelta(days=int(buffer_days))
        t1 = row["end"]   + pd.Timedelta(days=int(buffer_days))
        record = {"event_start": row["start"], "event_end": row["end"]}
        for name, series in series_dict.items():
            seg = series.loc[t0:t1]
            if "max"   in stats:
                record[f"{name}_max"]   = float(seg.max())
            if "mean"  in stats:
                record[f"{name}_mean"]  = float(seg.mean())
            if "total" in stats:
                record[f"{name}_total"] = float(seg.sum())
        rows.append(record)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Precipitation — POT with declustering
# ---------------------------------------------------------------------------

def extract_precipitation_events_pot(series, threshold, min_sep=7):
    """
    Extract extreme precipitation events using Peak-over-Threshold (POT).

    All days with P > threshold are identified as exceedances.  Exceedances
    within min_sep days of each other are merged into one cluster; only the
    peak day of each cluster is retained (declustering).  Each retained peak
    is then expanded to its surrounding contiguous wet spell (P > 0) to
    capture the full event.

    Produces approximately independent events — suitable for frequency
    analysis and GPD / GEV fitting.

    Args:
        series:    pd.Series of daily precipitation with DatetimeIndex.
        threshold: Minimum daily rainfall for a POT exceedance (mm).
                   Typically the 90th–95th percentile of non-zero rainfall.
        min_sep:   Maximum gap (days) between two exceedances that still
                   belong to the same cluster (default 7).

    Returns:
        Tuple (stats, bounds):
            stats  – DataFrame [peak, total, duration, mean_intensity, date_peak].
            bounds – DataFrame [start, end].
    """
    values = np.asarray(series.values, dtype=float)
    n = len(values)
    index = series.index

    exceed_idx = np.where(values > threshold)[0]
    if len(exceed_idx) == 0:
        return (
            pd.DataFrame(columns=["peak", "total", "duration", "mean_intensity", "date_peak"]),
            pd.DataFrame(columns=["start", "end"]),
        )

    # ── cluster nearby exceedances ────────────────────────────────────────
    clusters = [[exceed_idx[0]]]
    for idx in exceed_idx[1:]:
        if idx - clusters[-1][-1] <= min_sep:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])

    # ── keep peak per cluster, expand to contiguous wet spell ─────────────
    peaks, totals, durations, intensities, date_peaks = [], [], [], [], []
    starts_out, ends_out = [], []

    for cluster in clusters:
        peak_pos = cluster[int(np.argmax(values[cluster]))]

        s = peak_pos
        while s > 0 and values[s - 1] > 0:
            s -= 1
        e = peak_pos
        while e < n - 1 and values[e + 1] > 0:
            e += 1

        segment = values[s:e + 1]
        wet_vals = segment[segment > 0]
        peaks.append(float(values[peak_pos]))
        totals.append(float(segment.sum()))
        durations.append(e - s + 1)
        intensities.append(float(wet_vals.mean()) if len(wet_vals) else 0.0)
        date_peaks.append(index[peak_pos])
        starts_out.append(index[s])
        ends_out.append(index[e])

    stats = pd.DataFrame({
        "peak":           peaks,
        "total":          totals,
        "duration":       durations,
        "mean_intensity": intensities,
        "date_peak":      date_peaks,
    })
    bounds = pd.DataFrame({"start": starts_out, "end": ends_out})
    return stats, bounds


# ---------------------------------------------------------------------------
# Precipitation — N-day rolling accumulation
# ---------------------------------------------------------------------------

def extract_precipitation_events_nday(series, n_days=3, threshold=None, min_sep=None):
    """
    Extract events based on N-day rolling accumulation exceeding a threshold.

    Computes the rolling N-day sum, identifies windows where it exceeds
    `threshold`, and clusters overlapping windows into events.  Within each
    cluster the period centred on the maximum rolling-sum value is retained.

    Useful for catchments where multi-day accumulation drives flooding, not
    single-day peaks.

    Args:
        series:    pd.Series of daily precipitation with DatetimeIndex.
        n_days:    Accumulation window length in days (default 3).
        threshold: Minimum N-day total (mm) to qualify as an event.
                   Defaults to the 90th percentile of the rolling sum.
        min_sep:   Minimum days between retained event peaks (default n_days).

    Returns:
        Tuple (stats, bounds):
            stats  – DataFrame [peak_nday, total, duration, date_peak].
            bounds – DataFrame [start, end].
    """
    values = np.asarray(series.values, dtype=float)
    n = len(values)
    index = series.index

    if min_sep is None:
        min_sep = n_days

    # Rolling N-day sum (aligned so position i = sum ending on day i)
    rolling = np.convolve(values, np.ones(n_days), mode="full")[:n]
    # Shift so that rolling[i] = sum of values[i-n_days+1 : i+1]
    rolling[:n_days - 1] = np.nan

    if threshold is None:
        threshold = float(np.nanpercentile(rolling[rolling > 0], 90))

    exceed_idx = np.where(rolling > threshold)[0]
    if len(exceed_idx) == 0:
        return (
            pd.DataFrame(columns=["peak_nday", "total", "duration", "date_peak"]),
            pd.DataFrame(columns=["start", "end"]),
        )

    # ── cluster contiguous / nearby exceedance positions ──────────────────
    clusters = [[exceed_idx[0]]]
    for idx in exceed_idx[1:]:
        if idx - clusters[-1][-1] <= min_sep:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])

    # ── for each cluster, the event window is centred on max rolling sum ──
    peak_ndays, totals, durations, date_peaks = [], [], [], []
    starts_out, ends_out = [], []

    for cluster in clusters:
        # position with highest rolling sum in cluster
        best = cluster[int(np.argmax(rolling[cluster]))]
        # event window: the n_days ending at `best`
        s = max(0, best - n_days + 1)
        e = best

        segment = values[s:e + 1]
        peak_ndays.append(float(rolling[best]))
        totals.append(float(segment.sum()))
        durations.append(e - s + 1)
        date_peaks.append(index[best])
        starts_out.append(index[s])
        ends_out.append(index[e])

    stats = pd.DataFrame({
        "peak_nday":  peak_ndays,
        "total":      totals,
        "duration":   durations,
        "date_peak":  date_peaks,
    })
    bounds = pd.DataFrame({"start": starts_out, "end": ends_out})
    return stats, bounds


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def extract_events(series, threshold, variable="discharge", method="spell",
                   threshold2=None, min_duration=1, min_gap=1,
                   min_sep=7, n_days=3, plot=False):
    """
    Unified entry point for event extraction.

    Args:
        series:       pd.Series with a time index.
        threshold:    Event detection threshold.
        variable:     ``'discharge'`` or ``'precipitation'``.
        method:       Precipitation method — ``'spell'`` (wet-spell, default),
                      ``'pot'`` (Peak-over-Threshold with declustering), or
                      ``'nday'`` (N-day rolling accumulation).
        threshold2:   Discharge only — minimum peak to retain an event.
        min_duration: Wet-spell only — minimum event length (days).
        min_gap:      Wet-spell only — dry steps bridged between events.
        min_sep:      POT only — max gap between exceedances in the same cluster
                      (days, default 7).
        n_days:       N-day method only — accumulation window (days, default 3).
        plot:         Discharge only — plot the largest event.

    Returns:
        Tuple (stats, bounds) — see the specific functions for column details.
    """
    if variable == "discharge":
        return extract_discharge_events(series, threshold, threshold2=threshold2, plot=plot)
    elif variable == "precipitation":
        if method == "spell":
            return extract_precipitation_events(series, threshold=threshold,
                                                min_duration=min_duration, min_gap=min_gap)
        elif method == "pot":
            return extract_precipitation_events_pot(series, threshold=threshold,
                                                    min_sep=min_sep)
        elif method == "nday":
            return extract_precipitation_events_nday(series, n_days=n_days,
                                                     threshold=threshold)
        else:
            raise ValueError(
                f"Unknown precipitation method '{method}'. Use 'spell', 'pot', or 'nday'."
            )
    else:
        raise ValueError(f"Unknown variable '{variable}'. Use 'discharge' or 'precipitation'.")

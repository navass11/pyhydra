"""
Pixel-based return period computation for flood maps.

Given an ensemble of flood-depth maps (one per synthetic event), derives the
T-year flood depth at each pixel independently using empirical order statistics
calibrated with the historical event rate (Poisson process assumption).

Formula
-------
For a Poisson process with annual rate λ (events/year) and N synthetic events
sorted in ascending order:

    idx(T) = round(N * (1 - 1 / (T * λ))) - 1

The value at that rank is the flood depth with annual exceedance probability 1/T.

References
----------
Solari S. et al. (2017). A unified statistical model for hydrological variables
including the selection of threshold for the peak over threshold method.
Water Resources Research.
"""

from pathlib import Path

import numpy as np


DEFAULT_RETURN_PERIODS = (5, 10, 25, 50, 100, 200, 500, 1000)


def pixel_return_period(events_block, landa, return_periods=DEFAULT_RETURN_PERIODS):
    """
    Compute return-period flood depths for a spatial block of synthetic events.

    Args:
        events_block: ndarray of shape (nrows, ncols, n_events).
                      The ensemble dimension (axis=2) must already be sorted
                      in ascending order.
        landa:        Annual event rate (events / year) estimated from the
                      historical discharge record.
        return_periods: Iterable of return periods in years (default: standard
                        set 5, 10, 25, 50, 100, 200, 500, 1000).

    Returns:
        dict mapping each T (int) → ndarray of shape (nrows, ncols).
    """
    n_events = events_block.shape[2]
    result = {}
    for T in return_periods:
        idx = round(n_events * (1 - 1.0 / (T * landa))) - 1
        idx = max(0, min(idx, n_events - 1))
        result[T] = events_block[:, :, idx].copy()
    return result


def save_return_period_geotiffs(calados, template_tif, output_dir):
    """
    Write one GeoTIFF per return period, copying spatial reference from a template.

    Args:
        calados:      dict {T: ndarray(nrows, ncols)} as returned by
                      :func:`pixel_return_period` (accumulated over all blocks).
        template_tif: Path to any existing GeoTIFF with the target spatial
                      reference and dimensions (e.g. one of the simulation files).
        output_dir:   Directory where ``Calado_T{T}.tif`` files will be written.

    Returns:
        list of Path objects for the written files.
    """
    import rasterio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(str(template_tif)) as src:
        profile = src.profile.copy()
    profile.update(dtype="float32", count=1, nodata=0)

    written = []
    for T, data in calados.items():
        out_path = output_dir / f"Calado_T{T}.tif"
        with rasterio.open(str(out_path), "w", **profile) as dst:
            dst.write(data.astype("float32"), 1)
        written.append(out_path)

    return written

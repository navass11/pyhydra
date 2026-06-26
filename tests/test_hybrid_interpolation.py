"""Tests for hybrid-downscaling interpolation helpers."""

import numpy as np
import pandas as pd
import pytest

from pyhydra.climate.hybrid_downscaling.interpolation import (
    FloodMapInterpolatorCC,
    _knn_weights,
    _open_tif_array,
    _rbf_weights,
    _read_tif_block,
)


def _feature_frame(rows):
    return pd.DataFrame(rows, columns=["Qmax", "Qmed", "Duracion"])


class TestWeights:
    def test_rbf_weights_are_normalized_for_far_tail_events(self):
        centroids = _feature_frame([(0.0, 0.0, 1.0), (10.0, 0.0, 1.0), (20.0, 0.0, 1.0)])
        synthetic = _feature_frame([(1_000_000.0, 1_000_000.0, 365.0)])

        weights = _rbf_weights(synthetic, centroids, sigma=0.1)

        assert weights.shape == (1, 3)
        assert np.isfinite(weights).all()
        assert weights.sum(axis=1)[0] == pytest.approx(1.0)

    def test_knn_exact_match_gets_nearly_all_weight(self):
        centroids = _feature_frame([(1.0, 1.0, 1.0), (5.0, 5.0, 5.0)])
        synthetic = _feature_frame([(1.0, 1.0, 1.0)])

        positions, weights = _knn_weights(synthetic, centroids, k=2)

        exact_pos = int(np.where(positions[0] == 0)[0][0])
        assert weights[0, exact_pos] > 0.999999
        assert weights.sum(axis=1)[0] == pytest.approx(1.0)


class TestGeoTiffHelpers:
    def test_open_tif_array_replaces_nodata_and_negative_values(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        path = tmp_path / "depth.tif"
        data = np.array([[1.0, -9999.0], [-2.0, 4.0]], dtype=np.float32)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=2,
            width=2,
            count=1,
            dtype="float32",
            nodata=-9999.0,
            transform=from_origin(0, 2, 1, 1),
        ) as dst:
            dst.write(data, 1)

        arr = _open_tif_array(path)

        assert arr.tolist() == [[1.0, 0.0], [0.0, 4.0]]

    def test_read_tif_block_returns_requested_window(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin

        path = tmp_path / "depth.tif"
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=4,
            width=4,
            count=1,
            dtype="float32",
            transform=from_origin(0, 4, 1, 1),
        ) as dst:
            dst.write(data, 1)

        block = _read_tif_block(path, 1, 3, 1, 4, total_nrows=4, total_ncols=4)

        np.testing.assert_array_equal(block, data[1:3, 1:4])


class TestFloodMapInterpolatorCC:
    def test_constructor_forwards_method_and_rbf_sigma(self, tmp_path):
        synthetic = _feature_frame([(1.0, 1.0, 1.0)])
        centroids = pd.DataFrame(
            {
                "sim_id": [1, 2],
                "Qmax": [1.0, 2.0],
                "Qmed": [1.0, 2.0],
                "Duracion": [1.0, 2.0],
            }
        )

        interp = FloodMapInterpolatorCC(
            synthetic,
            centroids,
            tmp_path / "hist",
            tmp_path / "cc",
            n_simulations_hist=1,
            n_simulations_cc=1,
            output_dir=tmp_path / "out",
            method="rbf",
            rbf_sigma=2.5,
        )

        assert interp.method == "rbf"
        assert interp.rbf_sigma == pytest.approx(2.5)
        assert len(interp.centroids) == 2
        assert interp.n_sim_hist == 1
        assert interp.n_sim_cc == 1

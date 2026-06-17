"""Tests for pyhydra.data_sources.soils.soilgrids — pure-numpy functions only."""

import numpy as np
import pytest

from pyhydra.data_sources.soils.soilgrids import find_usda_soilclass


# USDA class indices (1-based as defined in soilgrids.py)
CLASS = {
    "clay": 1, "clay_loam": 2, "loam": 3, "loamy_sand": 4, "sand": 5,
    "sandy_clay": 6, "sandy_clay_loam": 7, "sandy_loam": 8,
    "silt": 9, "silty_clay": 10, "silty_clay_loam": 11, "silt_loam": 12,
}


def _pixel(sand_val, silt_val, clay_val):
    s = np.array([[sand_val]], dtype=np.uint8)
    sl = np.array([[silt_val]], dtype=np.uint8)
    c = np.array([[clay_val]], dtype=np.uint8)
    return find_usda_soilclass(s, sl, c)[0][0, 0]


class TestUSDAClasses:
    def test_sand(self):
        # sand: silt + 1.5*clay < 15  →  class 5
        assert _pixel(93, 5, 2) == CLASS["sand"]

    def test_loamy_sand(self):
        # 15 <= silt + 1.5*clay < 30  →  class 4
        assert _pixel(80, 12, 8) == CLASS["loamy_sand"]

    def test_sandy_loam(self):
        # 7<=clay<20, sand>52, (silt+2*clay)>=30  →  class 8
        assert _pixel(60, 25, 15) == CLASS["sandy_loam"]

    def test_loam(self):
        # 7<=clay<27, 28<=silt<50, sand<52  →  class 3
        assert _pixel(40, 40, 20) == CLASS["loam"]

    def test_silt_loam(self):
        # silt>=50, clay>=12, clay<27  →  class 12
        assert _pixel(20, 65, 15) == CLASS["silt_loam"]

    def test_silt(self):
        # silt>=80, clay<12  →  class 9
        assert _pixel(10, 85, 5) == CLASS["silt"]

    def test_clay(self):
        # clay>=40, sand<=45, silt<40  →  class 1
        assert _pixel(30, 25, 45) == CLASS["clay"]

    def test_silty_clay(self):
        # clay>=40, silt>=40  →  class 10
        assert _pixel(5, 50, 45) == CLASS["silty_clay"]

    def test_sandy_clay(self):
        # clay>=35, sand>45  →  class 6
        assert _pixel(50, 10, 40) == CLASS["sandy_clay"]

    def test_clay_loam(self):
        # 27<=clay<40, sand>20, sand<=45  →  class 2
        assert _pixel(35, 35, 30) == CLASS["clay_loam"]

    def test_silty_clay_loam(self):
        # 27<=clay<40, sand<=20  →  class 11
        assert _pixel(10, 60, 30) == CLASS["silty_clay_loam"]


class TestNoDataHandling:
    def test_nodata_pixel_returns_zero(self):
        sand = np.array([[255, 50]], dtype=np.uint8)
        silt = np.array([[10,  30]], dtype=np.uint8)
        clay = np.array([[10,  20]], dtype=np.uint8)
        result, _ = find_usda_soilclass(sand, silt, clay, no_data_value=255)
        assert result[0, 0] == 0
        assert result[0, 1] != 0

    def test_all_nodata_returns_zeros(self):
        sand = np.full((3, 3), 255, dtype=np.uint8)
        silt = np.full((3, 3), 255, dtype=np.uint8)
        clay = np.full((3, 3), 255, dtype=np.uint8)
        result, _ = find_usda_soilclass(sand, silt, clay)
        assert (result == 0).all()

    def test_custom_nodata_value(self):
        sand = np.array([[0]], dtype=np.uint8)
        silt = np.array([[0]], dtype=np.uint8)
        clay = np.array([[0]], dtype=np.uint8)
        result, _ = find_usda_soilclass(sand, silt, clay, no_data_value=0)
        assert result[0, 0] == 0


class TestOutputShape:
    def test_shape_preserved(self, texture_arrays):
        sand, silt, clay = texture_arrays
        result, names = find_usda_soilclass(sand, silt, clay)
        assert result.shape == sand.shape

    def test_soiltype_list_has_12_classes(self, texture_arrays):
        sand, silt, clay = texture_arrays
        _, names = find_usda_soilclass(sand, silt, clay)
        assert len(names) == 12

    def test_classes_in_valid_range(self, texture_arrays):
        sand, silt, clay = texture_arrays
        result, _ = find_usda_soilclass(sand, silt, clay)
        assert result.min() >= 0
        assert result.max() <= 12

    def test_dtype_is_uint8(self, texture_arrays):
        sand, silt, clay = texture_arrays
        result, _ = find_usda_soilclass(sand, silt, clay)
        assert result.dtype == np.uint8

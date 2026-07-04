"""Tests for HEC-HMS runtime safeguards."""

import platform

import pytest

from pyhydra.modeling.hydrology import hec_hms


def test_run_hms_script_rejects_native_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hec_hms, "generate_py", lambda *args, **kwargs: None)

    with pytest.raises(OSError, match="not supported natively on macOS"):
        hec_hms.run_hms_script(str(tmp_path), "demo", ["Run 1"])


def test_hms_available_false_on_macos_even_if_sh_file_exists(monkeypatch, tmp_path):
    # Regression: a hec-hms.sh copied from a Docker volume can exist on disk
    # on a Mac dev machine without being runnable there — file existence
    # alone must not be read as "HEC-HMS is usable".
    (tmp_path / "hec-hms.sh").touch()
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    assert hec_hms.hms_available(str(tmp_path)) is False


def test_hms_available_checks_sh_file_on_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Linux")

    assert hec_hms.hms_available(str(tmp_path)) is False
    (tmp_path / "hec-hms.sh").touch()
    assert hec_hms.hms_available(str(tmp_path)) is True


def test_hms_available_checks_jython_api_importable_on_windows(monkeypatch, tmp_path):
    import sys
    import types

    monkeypatch.setattr(platform, "system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "hms", None)
    monkeypatch.setitem(sys.modules, "hms.model", None)
    assert hec_hms.hms_available(str(tmp_path)) is False

    fake_hms = types.ModuleType("hms")
    fake_model = types.ModuleType("hms.model")
    fake_hms.model = fake_model
    monkeypatch.setitem(sys.modules, "hms", fake_hms)
    monkeypatch.setitem(sys.modules, "hms.model", fake_model)
    assert hec_hms.hms_available(str(tmp_path)) is True


def test_estimate_muskingum_k_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        hec_hms.estimate_muskingum_k(8.0, 0.002, method="chow")


def test_estimate_muskingum_k_usace_uses_fixed_x_regardless_of_slope():
    # 'usace' must ignore the slope-based heuristic entirely (X always 0.2),
    # unlike 'custom' which switches to X=0.3 for steeper slopes.
    _, x_mild = hec_hms.estimate_muskingum_k(8.0, 0.0005, method="usace")
    _, x_steep = hec_hms.estimate_muskingum_k(8.0, 0.01, method="usace")

    assert x_mild == 0.2
    assert x_steep == 0.2


def test_estimate_muskingum_k_custom_switches_x_by_slope():
    _, x_mild = hec_hms.estimate_muskingum_k(8.0, 0.0005, method="custom")
    _, x_steep = hec_hms.estimate_muskingum_k(8.0, 0.01, method="custom")

    assert x_mild == 0.2
    assert x_steep == 0.3


def test_estimate_muskingum_k_same_travel_time_for_both_methods():
    k_custom, _ = hec_hms.estimate_muskingum_k(8.0, 0.002, velocity_mps=1.5, method="custom")
    k_usace, _ = hec_hms.estimate_muskingum_k(8.0, 0.002, velocity_mps=1.5, method="usace")

    assert k_custom == k_usace == pytest.approx((8000.0 / 1.5) / 3600.0, abs=1e-3)

"""Tests for HEC-HMS runtime safeguards."""

import platform

import pytest

from pyhydra.modeling.hydrology import hec_hms


def test_run_hms_script_rejects_native_macos(monkeypatch, tmp_path):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hec_hms, "generate_py", lambda *args, **kwargs: None)

    with pytest.raises(OSError, match="not supported natively on macOS"):
        hec_hms.run_hms_script(str(tmp_path), "demo", ["Run 1"])

"""Tests for pyhydra-get-data (Zenodo backend), without network access."""
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from pyhydra.data import download as dl


def test_index_matches_the_archived_datasets():
    idx = dl._load_index()
    assert idx["doi"] == "10.5281/zenodo.23366599"
    assert dl.list_pilot_cases() == [
        "los_corrales_buelna", "m30_manzanares", "manning_rugosidades", "valencia_dana",
    ]
    # every dataset file points to a zip that is listed with a size and an MD5
    for case, by_zip in idx["datasets"].items():
        for zip_name, files in by_zip.items():
            assert zip_name in idx["zips"], (case, zip_name)
            assert files and all(isinstance(s, int) for _, s in files)
    for info in idx["zips"].values():
        assert len(info["md5"]) == 32 and info["size"] > 0
    counts = {c: sum(len(v) for v in z.values()) for c, z in idx["datasets"].items()}
    assert counts["manning_rugosidades"] == 3009
    assert counts["m30_manzanares"] == 5 and counts["valencia_dana"] == 5


@pytest.fixture
def fake_zenodo(tmp_path, monkeypatch):
    """A tiny archive with the same structure, served from local files."""
    src = tmp_path / "src"
    (src / "pilot_cases" / "caso" / "sub").mkdir(parents=True)
    (src / "pilot_cases" / "caso" / "a.csv").write_text("a,b\n1,2\n")
    (src / "pilot_cases" / "caso" / "sub" / "b.txt").write_text("hello")
    (src / "pilot_cases" / "otro").mkdir()
    (src / "pilot_cases" / "otro" / "c.txt").write_text("not for caso")
    zpath = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(src).as_posix())
    md5 = hashlib.md5(zpath.read_bytes()).hexdigest()
    index = {
        "record": "1", "doi": "10.5281/zenodo.1", "version": "1.0.0",
        "zips": {"bundle.zip": {"size": zpath.stat().st_size, "md5": md5}},
        "datasets": {
            "caso": {"bundle.zip": [["a.csv", 8], ["sub/b.txt", 5]]},
            "otro": {"bundle.zip": [["c.txt", 12]]},
        },
    }
    monkeypatch.setattr(dl, "_load_index", lambda: index)
    calls = []

    def fake_download(url, out, size, md5, desc, retries=6):
        calls.append(desc)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(zpath.read_bytes())

    monkeypatch.setattr(dl, "_download_zip", fake_download)
    return calls


def test_download_places_only_the_requested_dataset(tmp_path, fake_zenodo):
    dest = tmp_path / "data"
    out = dl.download_pilot_case("caso", dest=dest)
    assert out == dest / "pilot_cases" / "caso"
    assert (out / "a.csv").read_text() == "a,b\n1,2\n"
    assert (out / "sub" / "b.txt").read_text() == "hello"
    assert not (dest / "pilot_cases" / "otro").exists()      # other datasets untouched
    assert not (dest / dl._TMP_DIRNAME).exists()              # temporary zips cleaned up
    assert fake_zenodo == ["bundle.zip"]


def test_existing_files_are_skipped_unless_overwrite(tmp_path, fake_zenodo, capsys):
    dest = tmp_path / "data"
    dl.download_pilot_case("caso", dest=dest)
    (dest / "pilot_cases" / "caso" / "a.csv").write_text("changed")
    dl.download_pilot_case("caso", dest=dest)                  # nothing missing: no download
    assert fake_zenodo == ["bundle.zip"]
    assert (dest / "pilot_cases" / "caso" / "a.csv").read_text() == "changed"
    assert "2 files already present" in capsys.readouterr().out
    dl.download_pilot_case("caso", dest=dest, overwrite=True)
    assert (dest / "pilot_cases" / "caso" / "a.csv").read_text() == "a,b\n1,2\n"
    assert fake_zenodo == ["bundle.zip", "bundle.zip"]


def test_unknown_dataset_lists_the_available_ones(tmp_path, fake_zenodo):
    with pytest.raises(ValueError, match="not found"):
        dl.download_pilot_case("nope", dest=tmp_path)


def test_default_destination_uses_hydra_data_dir(tmp_path, fake_zenodo, monkeypatch):
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path / "env"))
    out = dl.download_pilot_case("caso")
    assert out == tmp_path / "env" / "pilot_cases" / "caso"


def test_extraction_refuses_paths_outside_the_destination(tmp_path):
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("pilot_cases/caso/ok.txt", "x")
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(RuntimeError, match="Unsafe path"):
        dl._extract_members(zpath, {"pilot_cases/caso/ok.txt": tmp_path / "outside.txt"}, root)

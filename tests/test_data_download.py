from __future__ import annotations

from pathlib import Path

import pytest
import requests

from pyhydra.data import download as dl


def test_azure_url_encodes_path_and_query_parameters():
    url = dl._azure_url("pilot_cases/demo case/file.txt", restype="directory", comp="list")

    assert "pilot_cases/demo%20case/file.txt" in url
    assert "restype=directory" in url
    assert "comp=list" in url
    assert "sig=" in url


def test_list_pilot_cases_filters_hidden_entries(monkeypatch):
    xml = """<?xml version="1.0" encoding="utf-8"?>
    <EnumerationResults>
      <Entries>
        <Directory><Name>manning_rugosidades</Name></Directory>
        <Directory><Name>.DS_Store</Name></Directory>
        <Directory><Name>.hidden</Name></Directory>
        <Directory><Name>m30_manzanares</Name></Directory>
      </Entries>
    </EnumerationResults>
    """

    class Response:
        text = xml

        def raise_for_status(self):
            return None

    monkeypatch.setattr(dl.requests, "get", lambda url, timeout: Response())

    assert dl.list_pilot_cases() == ["manning_rugosidades", "m30_manzanares"]


def test_collect_files_recurses_from_azure_listing(monkeypatch):
    tree = {
        "pilot_cases/demo": (["nested"], [("root.txt", 4)]),
        "pilot_cases/demo/nested": ([], [("child.bin", 2)]),
    }

    monkeypatch.setattr(dl, "_list_dir", lambda path: tree[path])

    assert dl._collect_files("pilot_cases/demo") == [
        ("root.txt", 4),
        ("nested/child.bin", 2),
    ]


def test_download_pilot_case_writes_files_and_preserves_tree(monkeypatch, tmp_path):
    payloads = {
        "root.txt": b"abc",
        "nested/child.bin": b"xy",
    }

    class StreamResponse:
        def __init__(self, data: bytes):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size: int):
            yield self.data

    def fake_get(url, stream=False, timeout=None):
        rel_path = next(key for key in payloads if key in url)
        return StreamResponse(payloads[rel_path])

    monkeypatch.setattr(dl, "_collect_files", lambda path: [(k, len(v)) for k, v in payloads.items()])
    monkeypatch.setattr(dl.requests, "get", fake_get)

    out_dir = dl.download_pilot_case("demo", dest=tmp_path)

    assert out_dir == tmp_path / "pilot_cases" / "demo"
    assert (out_dir / "root.txt").read_bytes() == b"abc"
    assert (out_dir / "nested" / "child.bin").read_bytes() == b"xy"


def test_download_pilot_case_reports_available_cases_on_404(monkeypatch, tmp_path):
    response = requests.Response()
    response.status_code = 404
    error = requests.HTTPError(response=response)

    def raise_404(path):
        raise error

    monkeypatch.setattr(dl, "_collect_files", raise_404)
    monkeypatch.setattr(dl, "list_pilot_cases", lambda: ["demo"])

    with pytest.raises(ValueError, match="Dataset 'missing' not found"):
        dl.download_pilot_case("missing", dest=tmp_path)

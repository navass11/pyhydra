"""
Download HYDRA pilot-case data from Azure File Share.

Usage (CLI):
    pyhydra-get-data                         # list available datasets
    pyhydra-get-data manning_rugosidades     # download to HYDRA_DATA_DIR
    pyhydra-get-data manning_rugosidades --dest /path/to/data

Usage (Python):
    from pyhydra.data.download import download_pilot_case
    download_pilot_case("manning_rugosidades", dest="/path/to/data")
"""

from __future__ import annotations

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode, quote

try:
    import requests
    from tqdm import tqdm
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

# ── Azure File Share connection (read-only, expires 2027-12-31) ──────────────
_ACCOUNT   = "hydratools"
_SHARE     = "dockerdata"
_BASE_PATH = "pilot_cases"
_SAS       = (
    "se=2027-12-31T23%3A59%3A59Z"
    "&sp=rl"
    "&sv=2026-04-06"
    "&sr=s"
    "&sig=9dqbAIJEOZJknYeSnKzCUIZ09oOdU0d5GUDfiE6cHds%3D"
)
_BASE_URL  = f"https://{_ACCOUNT}.file.core.windows.net/{_SHARE}"

# Files/dirs to skip when downloading
_SKIP_NAMES = {".DS_Store", "._DS_Store", "Thumbs.db"}


def _azure_url(path: str, **params) -> str:
    """Build an authenticated Azure File Share REST URL."""
    encoded = quote(path, safe="/")
    qs = _SAS + ("&" + urlencode(params) if params else "")
    return f"{_BASE_URL}/{encoded}?{qs}"


def list_pilot_cases() -> list[str]:
    """Return the names of available pilot-case datasets."""
    url = _azure_url(_BASE_PATH, restype="directory", comp="list")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    return [
        e.text for e in root.iter("Name")
        if e.text and e.text not in _SKIP_NAMES
        and not e.text.startswith(".")
    ]


def _list_dir(azure_path: str) -> tuple[list[str], list[str]]:
    """Return (subdirs, files) at *azure_path* in the File Share."""
    url = _azure_url(azure_path, restype="directory", comp="list")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    root = ET.fromstring(r.text)

    dirs, files = [], []
    for entry in root.findall(".//Entries/Directory/Name"):
        if entry.text and entry.text not in _SKIP_NAMES:
            dirs.append(entry.text)
    for entry in root.findall(".//Entries/File"):
        name_el = entry.find("Name")
        size_el = entry.find("Properties/Content-Length")
        if name_el is not None and name_el.text not in _SKIP_NAMES:
            size = int(size_el.text) if size_el is not None else 0
            files.append((name_el.text, size))
    return dirs, files


def _collect_files(azure_path: str) -> list[tuple[str, int]]:
    """Recursively list all (relative_path, size_bytes) under *azure_path*."""
    result: list[tuple[str, int]] = []

    def _walk(path: str, prefix: str) -> None:
        dirs, files = _list_dir(path)
        for name, size in files:
            result.append((f"{prefix}{name}", size))
        for d in dirs:
            _walk(f"{path}/{d}", f"{prefix}{d}/")

    _walk(azure_path, "")
    return result


def download_pilot_case(
    name: str,
    dest: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a pilot-case dataset from Azure to *dest*.

    Args:
        name: Dataset name, e.g. ``"manning_rugosidades"``.
        dest: Root destination directory.  Defaults to the value of the
            ``HYDRA_DATA_DIR`` environment variable, or ``./data`` if unset.
            The dataset is placed at ``<dest>/pilot_cases/<name>/``.
        overwrite: If False (default) skip files that already exist locally.

    Returns:
        Path to the downloaded dataset directory.
    """
    if not _DEPS_OK:
        raise ImportError("requests and tqdm are required: pip install requests tqdm")

    if dest is None:
        dest = Path(os.environ.get("HYDRA_DATA_DIR", "data"))
    dest = Path(dest)

    out_dir = dest / "pilot_cases" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    azure_path = f"{_BASE_PATH}/{name}"

    print(f"Listing files in '{name}'…")
    try:
        all_files = _collect_files(azure_path)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            available = list_pilot_cases()
            raise ValueError(
                f"Dataset '{name}' not found. Available: {available}"
            ) from exc
        raise

    total_bytes = sum(s for _, s in all_files)
    print(f"  {len(all_files)} files  ·  {total_bytes / 1e6:.1f} MB")

    skipped = 0
    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=name) as pbar:
        for rel_path, size in all_files:
            local = out_dir / rel_path
            if local.exists() and not overwrite:
                pbar.update(size)
                skipped += 1
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            url = _azure_url(f"{azure_path}/{rel_path}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(local, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        pbar.update(len(chunk))

    if skipped:
        print(f"  {skipped} files already present, skipped (use --overwrite to re-download)")
    print(f"\nData ready at: {out_dir}")
    return out_dir


# ── CLI ──────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(
        prog="pyhydra-get-data",
        description="Download HYDRA pilot-case data from Azure.",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Pilot-case name to download (omit to list available datasets).",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help=(
            "Destination root directory. "
            "Defaults to $HYDRA_DATA_DIR or ./data. "
            "Data lands at <dest>/pilot_cases/<dataset>/."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist locally.",
    )
    args = parser.parse_args()

    if not _DEPS_OK:
        sys.exit("Error: install missing dependencies:  pip install requests tqdm")

    if args.dataset is None:
        cases = list_pilot_cases()
        print("Available pilot-case datasets:")
        for c in cases:
            print(f"  • {c}")
        print("\nUsage:  pyhydra-get-data <dataset> [--dest /path/to/data]")
        return

    try:
        download_pilot_case(args.dataset, dest=args.dest, overwrite=args.overwrite)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    _main()

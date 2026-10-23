"""
Download pyhydra pilot-case data from the Zenodo record that archives it.

The data are archived in the open Zenodo record 10.5281/zenodo.23599126
(CC-BY-4.0).  ``pyhydra-get-data`` downloads the zip files that hold the
requested dataset, verifies their checksums and unpacks the dataset into
``<dest>/pilot_cases/<dataset>/``.  The layout, the dataset names and the
behaviour (files already present are skipped unless ``--overwrite`` is used)
are the ones the notebooks expect.

Usage (CLI):
    pyhydra-get-data                         # list available datasets
    pyhydra-get-data manning_rugosidades     # download to HYDRA_DATA_DIR
    pyhydra-get-data manning_rugosidades --dest /path/to/data

Usage (Python):
    from pyhydra.data.download import download_pilot_case
    download_pilot_case("manning_rugosidades", dest="/path/to/data")

Some files in the archive are, or derive from, records of third parties
(AEMET, CHC, SIAR, AVAMET, CORDEX) and keep the terms of use of their
providers; see the README of the Zenodo record.
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import argparse
import zipfile
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

_INDEX_PATH = Path(__file__).with_name("zenodo_index.json")
_CHUNK = 1 << 20
_TMP_DIRNAME = ".pyhydra_download"


def _load_index() -> dict:
    """Index of the Zenodo record: its zip files and the files of each dataset."""
    with open(_INDEX_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _zip_url(record: str, name: str) -> str:
    return f"https://zenodo.org/records/{record}/files/{name}?download=1"


def list_pilot_cases() -> list[str]:
    """Return the names of available pilot-case datasets."""
    return sorted(_load_index()["datasets"])


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_zip(url: str, out: Path, size: int, md5: str, desc: str,
                  retries: int = 6) -> None:
    """Download *url* to *out*, resuming a partial file, and verify size and MD5."""
    part = out.with_name(out.name + ".part")
    out.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    with tqdm(total=size, unit="B", unit_scale=True, desc=desc) as pbar:
        for _attempt in range(retries):
            have = part.stat().st_size if part.exists() else 0
            if have > size:
                part.unlink()
                have = 0
            pbar.n = have
            pbar.refresh()
            try:
                if have < size:
                    headers = {"Range": f"bytes={have}-"} if have else {}
                    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        if have and r.status_code != 206:      # server ignored Range
                            have = 0
                            pbar.n = 0
                            pbar.refresh()
                        with open(part, "ab" if have else "wb") as f:
                            for chunk in r.iter_content(chunk_size=_CHUNK):
                                f.write(chunk)
                                pbar.update(len(chunk))
                if part.stat().st_size != size:
                    raise IOError(f"incomplete download of {out.name}")
                break
            except (requests.RequestException, IOError) as exc:
                last_error = exc
        else:
            raise RuntimeError(f"Could not download {out.name}: {last_error}")
    if _md5(part) != md5:
        part.unlink()
        raise RuntimeError(f"Checksum mismatch for {out.name}; please run the command again.")
    part.replace(out)


def _extract_members(zip_path: Path, members: dict[str, Path], root: Path) -> None:
    """Extract the given zip members to their local paths (never outside *root*)."""
    root = root.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member, local in members.items():
            target = local.resolve()
            if root not in target.parents:
                raise RuntimeError(f"Unsafe path in archive: {member}")
            local.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(local, "wb") as dst:
                while True:
                    chunk = src.read(_CHUNK)
                    if not chunk:
                        break
                    dst.write(chunk)


def download_pilot_case(
    name: str,
    dest: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a pilot-case dataset from Zenodo to *dest*.

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

    index = _load_index()
    if name not in index["datasets"]:
        raise ValueError(
            f"Dataset '{name}' not found. Available: {list_pilot_cases()}"
        )

    if dest is None:
        dest = Path(os.environ.get("HYDRA_DATA_DIR", "data"))
    dest = Path(dest)

    out_dir = dest / "pilot_cases" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing files in '{name}'…")
    by_zip = index["datasets"][name]
    n_files = sum(len(v) for v in by_zip.values())
    total_bytes = sum(s for v in by_zip.values() for _, s in v)
    print(f"  {n_files} files  ·  {total_bytes / 1e6:.1f} MB")

    skipped = 0
    to_fetch: dict[str, dict[str, Path]] = {}
    for zip_name, files in by_zip.items():
        for rel_path, _size in files:
            local = out_dir / rel_path
            if local.exists() and not overwrite:
                skipped += 1
                continue
            member = f"pilot_cases/{name}/{rel_path}"
            to_fetch.setdefault(zip_name, {})[member] = local

    tmp_dir = dest / _TMP_DIRNAME
    for zip_name, members in to_fetch.items():
        info = index["zips"][zip_name]
        zip_path = tmp_dir / zip_name
        _download_zip(_zip_url(index["record"], zip_name), zip_path,
                      info["size"], info["md5"], zip_name)
        try:
            _extract_members(zip_path, members, dest)
        finally:
            zip_path.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    if skipped:
        print(f"  {skipped} files already present, skipped (use --overwrite to re-download)")
    print(f"\nData ready at: {out_dir}")
    return out_dir


# ── CLI ──────────────────────────────────────────────────────────────────────

def _main() -> None:
    parser = argparse.ArgumentParser(
        prog="pyhydra-get-data",
        description="Download pyhydra pilot-case data from Zenodo.",
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
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    _main()

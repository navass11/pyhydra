"""
CMIP6 downloader via the Earth System Grid Federation (ESGF) nodes.

Supports both OPeNDAP (lazy remote access) and HTTPServer (full download) protocols.
"""

import gc
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import xarray as xr
from tqdm.auto import tqdm


ESGF_NODES = [
    "https://esgf.nci.org.au/esg-search/search",
    "https://esgf-ui.ceda.ac.uk/esg-search/search",
    "https://esgf-node.llnl.gov/esg-search/search",
    "https://esgf-data.dkrz.de/esg-search/search",
    "https://esgf-node.ipsl.upmc.fr/esg-search/search",
    "https://esgf-node.ornl.gov/esg-search/search",
]

_SCENARIO_RANGES = {
    "historical": (1950, 2014),
    "ssp126": (2015, 2100),
    "ssp245": (2015, 2100),
    "ssp370": (2015, 2100),
    "ssp585": (2015, 2100),
}

_SCENARIO_MAP = {
    "ssp245": "ssp2_4_5",
    "ssp585": "ssp5_8_5",
    "ssp126": "ssp1_2_6",
    "ssp370": "ssp3_7_0",
    "historical": "historical",
}


def _esgf_request(payload):
    headers = {"Accept": "application/solr+json"}
    for node in ESGF_NODES:
        try:
            r = requests.get(node, params=payload, headers=headers, timeout=(5, 10))
            r.raise_for_status()
            return r.json()["response"]["docs"]
        except requests.RequestException:
            continue
    raise RuntimeError("Could not reach any ESGF node.")


def get_dataset_metadata(filters, limit=5000):
    """
    Query ESGF for dataset metadata matching the given filters.

    Args:
        filters: Dict of ESGF search facets (e.g. project, source_id, experiment_id)
        limit: Maximum number of datasets to return

    Returns:
        DataFrame with columns: dataset_id, model, experiment, variable, variant, table, start, end
    """
    payload = {
        "type": "Dataset",
        "format": "application/solr+json",
        "distrib": "true",
        "limit": 1000,
    }
    for key, val in filters.items():
        payload[key] = val if isinstance(val, list) else [val]

    all_datasets = []
    offset = 0
    total = 1

    while offset < total and len(all_datasets) < limit:
        payload["offset"] = offset
        docs = _esgf_request(payload)
        offset += len(docs)
        total = offset + 1 if len(docs) == 1000 else offset

        for d in docs:
            if "id" not in d:
                continue
            all_datasets.append({
                "dataset_id": d["id"].split("|")[0],
                "model": d.get("source_id", [""])[0],
                "experiment": d.get("experiment_id", [""])[0],
                "variable": d.get("variable_id", [""])[0],
                "variant": d.get("variant_label", [""])[0],
                "table": d.get("table_id", [""])[0],
                "start": d.get("time_coverage_start", ""),
                "end": d.get("time_coverage_end", ""),
                "version": d.get("version", [""]),
            })

    if not all_datasets:
        return pd.DataFrame(columns=["dataset_id", "model", "experiment", "variable", "variant", "table", "start", "end"])
    return pd.DataFrame(all_datasets)


def get_all_urls(dataset_id):
    """
    Return OPeNDAP and HTTPServer URLs for all files in a dataset.

    Args:
        dataset_id: Full ESGF dataset ID string

    Returns:
        List of dicts with keys 'url' and 'url_type'
    """
    urls = []
    parts = dataset_id.split(".")
    if len(parts) < 8:
        print(f"Invalid dataset ID: {dataset_id}")
        return urls

    _, _, _, model, experiment, variant, table, variable = parts[:8]
    payload = {
        "type": "File",
        "format": "application/solr+json",
        "limit": 500,
        "variable_id": variable,
        "source_id": model,
        "experiment_id": experiment,
        "variant_label": variant,
        "table_id": table,
    }

    try:
        docs = _esgf_request(payload)
        for f in docs:
            if f.get("dataset_id", "").split("|")[0] != dataset_id:
                continue
            if variable not in f.get("title", ""):
                continue
            for u in f.get("url", []):
                parts_u = u.split("|")
                if len(parts_u) == 3:
                    url, _, url_type = parts_u
                elif len(parts_u) == 2:
                    url, url_type = parts_u
                else:
                    continue
                url_type = url_type.strip().upper()
                if url_type == "OPENDAP":
                    url = url.strip().replace(".html", "")
                urls.append({"url": url.strip(), "url_type": url_type})
    except Exception as exc:
        print(f"Error getting URLs for {dataset_id}: {exc}")

    return urls


def get_combination_if_complete(model, experiment, variant, variables):
    """
    Return dataset rows only if ALL requested variables are available for this model/experiment/variant.

    Args:
        model: CMIP6 source_id
        experiment: experiment_id (e.g. 'historical', 'ssp245')
        variant: variant_label (e.g. 'r1i1p1f1')
        variables: list of variable_id strings

    Returns:
        List of DataFrame rows (one per variable), or empty list if incomplete
    """
    filters = {
        "project": "CMIP6",
        "table_id": "day",
        "source_id": model,
        "experiment_id": experiment,
        "variant_label": variant,
    }
    df = get_dataset_metadata(filters, limit=200)
    df = df[df["variable"].isin(variables)]
    if df.empty:
        return []

    found = df["variable"].unique().tolist()
    if not all(v in found for v in variables):
        return []

    return [df[df["variable"] == v].iloc[0] for v in variables]


def _extract_years(filename):
    m = re.search(r"(\d{4})\d{4}-(\d{4})\d{4}", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d{4})-(\d{4})", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _valid_year_range(scenario, year_ini, year_fin):
    if scenario not in _SCENARIO_RANGES:
        return True
    req_start, req_end = _SCENARIO_RANGES[scenario]
    return not (year_fin < req_start or year_ini > req_end)


def download_file(url, local_path, chunk_size=1024 * 1024, max_retries=3):
    """
    Download a file from a URL with retries.

    Args:
        url: Source URL
        local_path: Destination file path
        chunk_size: Download chunk size in bytes
        max_retries: Number of retry attempts

    Returns:
        True on success, False on failure
    """
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            print(f"Downloaded: {os.path.basename(local_path)}")
            return True
        except Exception as exc:
            wait = 2 ** attempt
            print(f"Retry {attempt + 1}/{max_retries}: {url} → {exc} (waiting {wait}s)")
            time.sleep(wait)
    print(f"Failed after {max_retries} attempts: {url}")
    return False


def process_file(url, path_output, lat_min, lat_max, lon_min, lon_max, url_type="OPENDAP"):
    """
    Download (or access via OPeNDAP) a CMIP6 file and save a spatial subset.

    Args:
        url: File URL (OPeNDAP endpoint or HTTPServer URL)
        path_output: Root output directory; files are saved under <variable>/<scenario>/
        lat_min, lat_max, lon_min, lon_max: Bounding box in degrees
        url_type: 'OPENDAP' or 'HTTPSERVER'

    Returns:
        Status string ('[OK]', '[SKIPPED]', or error message)
    """
    name = url.split("/")[-1]
    year_ini, year_fin = _extract_years(name)
    if year_ini is None:
        return f"[SKIPPED] Unexpected filename: {name}"

    try:
        scenario_raw = name.split("_")[-4]
    except IndexError:
        return f"[SKIPPED] Cannot extract scenario: {name}"

    if not _valid_year_range(scenario_raw, year_ini, year_fin):
        return f"[SKIPPED] Invalid range for {scenario_raw}: {year_ini}-{year_fin}"

    scenario = _SCENARIO_MAP.get(scenario_raw, scenario_raw)
    variable = url.split("/")[-4]
    output_dir = os.path.join(path_output, variable, scenario)
    output_file = os.path.join(output_dir, name)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_file):
        return f"[SKIPPED] Already exists: {output_file}"

    temp_file = None
    try:
        if url_type == "HTTPSERVER":
            temp_file = os.path.join(output_dir, f"__temp__{name}")
            if not download_file(url, temp_file):
                return f"[FAIL] Could not download: {url}"
            open_path = temp_file
        else:
            open_path = url

        with xr.open_dataset(open_path, decode_times=False) as ds:
            ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
            ds_sel = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

            if all(v.count().item() == 0 for v in ds_sel.data_vars.values()):
                return f"[SKIPPED] No data in domain: {name}"

            for coord, attrs in [
                ("lat", {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"}),
                ("lon", {"units": "degrees_east", "standard_name": "longitude", "axis": "X"}),
            ]:
                ds_sel[coord].attrs.update(attrs)

            ds_sel.to_netcdf(output_file)

        if url_type == "HTTPSERVER" and temp_file and os.path.exists(temp_file):
            try:
                time.sleep(1)
                gc.collect()
                os.remove(temp_file)
            except PermissionError:
                print(f"Warning: could not remove temp file: {temp_file}")

        return f"[OK] Saved: {name}"

    except Exception as exc:
        if url_type == "HTTPSERVER" and temp_file and os.path.exists(temp_file):
            try:
                time.sleep(1)
                gc.collect()
                os.remove(temp_file)
            except PermissionError:
                pass
        return f"[ERROR] {name}: {exc}"

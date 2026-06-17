"""
River Discharge data sources.

Submodules:
    grdc  — GRDC file reader (daily/monthly .day/.mon text files)
    usgs  — USGS NWIS REST API (daily streamflow for US stations)
    glofas — GloFAS via Copernicus CDS API (global reanalysis/forecast)
"""

from pyhydra.data_sources.river_discharge import grdc, usgs, glofas
from pyhydra.data_sources.river_discharge.grdc import (
    read_grdc,
    read_grdc_metadata,
    read_grdc_folder,
    analyze_grdc_quality,
)
from pyhydra.data_sources.river_discharge.usgs import (
    download_usgs,
    get_usgs_site_info,
    search_usgs_sites,
)
from pyhydra.data_sources.river_discharge.glofas import (
    download_glofas,
    download_glofas_by_year,
    read_glofas_nc,
)

__all__ = [
    "grdc",
    "usgs",
    "glofas",
    "read_grdc",
    "read_grdc_metadata",
    "read_grdc_folder",
    "analyze_grdc_quality",
    "download_usgs",
    "get_usgs_site_info",
    "search_usgs_sites",
    "download_glofas",
    "download_glofas_by_year",
    "read_glofas_nc",
]

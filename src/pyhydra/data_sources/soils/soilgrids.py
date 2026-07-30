"""
SoilGrids pipeline: download → USDA classification → modal depth aggregation.

Reference: Hengl et al. (2017), SoilGrids250m. PLoS ONE 12(2): e0169748.
"""

from pathlib import Path

import numpy as np
import requests
import scipy.stats as sc


_BASE_URL = "https://files.isric.org/soilgrids/former/2017-03-10/data/"
_VARIABLES = ("SNDPPT", "SLTPPT", "CLYPPT")
_DEPTHS = [f"sl{i}" for i in range(1, 8)]


def download_soilgrids(output_dir, max_retries=10, include_metadata=False):
    """
    Download SoilGrids sand/silt/clay GeoTIFFs (ISRIC 2017 release, 250 m).

    Args:
        output_dir: Destination directory (created if absent)
        max_retries: Download attempts per file before giving up
        include_metadata: Also download XML and CSV metadata files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f"{v}_M_{d}_250m_ll.tif" for v in _VARIABLES for d in _DEPTHS]
    if include_metadata:
        files += [f + ".xml" for f in files]
        files += ["README.md", "META_GEOTIFF_1B.csv"]

    for filename in files:
        dest = output_dir / filename
        if dest.exists():
            continue

        url = _BASE_URL + filename
        for attempt in range(1, max_retries + 1):
            try:
                with open(dest, "wb") as f:
                    r = requests.get(url, stream=True)
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)
                print(f"Downloaded {filename}")
                break
            except Exception as exc:
                print(f"Attempt {attempt}/{max_retries} failed for {filename}: {exc}")
                if attempt == max_retries:
                    print(f"Could not download {filename} after {max_retries} attempts")


def open_soilgrids_geotif(file):
    """
    Open a SoilGrids GeoTIFF and return its data array and bounding-box coordinates.

    Returns:
        (data, coords): numpy uint8 array and 5×2 coordinate array (corners + centre)
    """
    from osgeo import gdal

    ds = gdal.Open(str(file))
    data = ds.GetRasterBand(1).ReadAsArray()
    width, height = ds.RasterXSize, ds.RasterYSize
    gt = ds.GetGeoTransform()

    coords = np.zeros((5, 2))
    coords[0, 0] = coords[1, 0] = gt[0]
    coords[0, 1] = coords[2, 1] = gt[3]
    coords[2, 0] = coords[3, 0] = gt[0] + width * gt[1]
    coords[1, 1] = coords[3, 1] = gt[3] + height * gt[5]
    coords[4, 0] = gt[0] + (width / 2) * gt[1]
    coords[4, 1] = gt[3] + (height / 2) * gt[5]

    return data, coords


def find_usda_soilclass(sand, silt, clay, no_data_value=255):
    """
    Classify each pixel into one of 12 USDA soil texture classes.

    Classes (1–12): Clay, Clay Loam, Loam, Loamy Sand, Sand, Sandy Clay,
    Sandy Clay Loam, Sandy Loam, Silt, Silty Clay, Silty Clay Loam, Silt Loam.

    Args:
        sand, silt, clay: Percentage arrays from SoilGrids (same shape)
        no_data_value: Pixels with this value in any component are set to 0

    Returns:
        (soilclass, soiltype): integer array and list of class names
    """
    soilclass = np.zeros(sand.shape, dtype=np.uint8)
    soiltype = [
        "clay", "clay loam", "loam", "loamy sand", "sand",
        "sandy clay", "sandy clay loam", "sandy loam",
        "silt", "silty clay", "silty clay loam", "silt loam",
    ]

    soilclass[(clay >= 40) & (sand <= 45) & (silt < 40)] = 1
    soilclass[(clay >= 27) & (clay < 40) & (sand > 20) & (sand <= 45)] = 2
    soilclass[(clay >= 7) & (clay < 27) & (silt >= 28) & (silt < 50) & (sand < 52)] = 3
    soilclass[((silt + 1.5 * clay) >= 15) & ((silt + 2 * clay) < 30)] = 4
    soilclass[(silt + 1.5 * clay) < 15] = 5
    soilclass[(clay >= 35) & (sand > 45)] = 6
    soilclass[(clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45)] = 7
    soilclass[
        ((clay >= 7) & (clay < 20) & (sand > 52) & ((silt + 2 * clay) >= 30))
        | ((clay < 7) & (silt < 50) & ((silt + 2 * clay) >= 30))
    ] = 8
    soilclass[(silt >= 80) & (clay < 12)] = 9
    soilclass[(clay >= 40) & (silt >= 40)] = 10
    soilclass[(clay >= 27) & (clay < 40) & (sand <= 20)] = 11
    soilclass[
        ((silt >= 50) & (clay >= 12) & (clay < 27))
        | ((silt >= 50) & (silt < 80) & (clay < 12))
    ] = 12
    soilclass[(sand == no_data_value) | (silt == no_data_value) | (clay == no_data_value)] = 0

    return soilclass, soiltype


def create_new_tif(template_file, data, output_file):
    """
    Write a numpy array to a GeoTIFF, copying spatial reference from a template.

    Args:
        template_file: Existing GeoTIFF used as spatial reference
        data: 2-D numpy array to write
        output_file: Path for the new GeoTIFF
    """
    from osgeo import gdal

    ds = gdal.Open(str(template_file))
    cols, rows = data.shape

    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(str(output_file), rows, cols, 1, gdal.GDT_UInt16, options=["COMPRESS=DEFLATE"])
    out.SetGeoTransform(ds.GetGeoTransform())
    out.SetProjection(ds.GetProjection())
    out.GetRasterBand(1).WriteArray(data)
    out.GetRasterBand(1).SetNoDataValue(-1)
    out.FlushCache()


def convert_depth_to_soilclass(input_dir, output_dir, no_data_value=255):
    """
    Convert SoilGrids texture GeoTIFFs to USDA soil-class GeoTIFFs for all 7 depths.

    Args:
        input_dir: Directory with SNDPPT/SLTPPT/CLYPPT_M_sl*_250m_ll.tif files
        output_dir: Directory to write usda_soilclass_sl*_250m_ll.tif files
        no_data_value: No-data value used by SoilGrids (default 255)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sl in _DEPTHS:
        sand, sc_ = open_soilgrids_geotif(input_dir / f"SNDPPT_M_{sl}_250m_ll.tif")
        silt, sl_ = open_soilgrids_geotif(input_dir / f"SLTPPT_M_{sl}_250m_ll.tif")
        clay, cl_ = open_soilgrids_geotif(input_dir / f"CLYPPT_M_{sl}_250m_ll.tif")

        if not (np.allclose(cl_, sc_) and np.allclose(cl_, sl_)):
            raise ValueError(f"Coordinate mismatch between texture layers at depth {sl}")

        soilclass, _ = find_usda_soilclass(sand, silt, clay, no_data_value)

        out_path = output_dir / f"usda_soilclass_{sl}_250m_ll.tif"
        template = input_dir / f"CLYPPT_M_{sl}_250m_ll.tif"
        create_new_tif(template, soilclass, out_path)
        print(f"Saved {out_path.name}")


def extract_mode_soilclass(input_dir, output_path):
    """
    Compute the modal USDA soil class across the 7 SoilGrids depths and save as GeoTIFF.

    Args:
        input_dir: Directory containing usda_soilclass_sl*_250m_ll.tif files
        output_path: Output file path for the mode GeoTIFF
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    layers = [
        open_soilgrids_geotif(input_dir / f"usda_soilclass_{sl}_250m_ll.tif")[0]
        for sl in _DEPTHS
    ]
    soilclasses = np.dstack(layers)
    mode = sc.mode(soilclasses, axis=2)[0].squeeze()

    template = str(input_dir / f"usda_soilclass_{_DEPTHS[0]}_250m_ll.tif")
    create_new_tif(template, mode, output_path)
    print(f"Mode soil class saved to {output_path}")

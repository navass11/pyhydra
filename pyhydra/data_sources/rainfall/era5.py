"""
ERA5 downloader via the Copernicus Climate Data Store (CDS) API.

Requires cdsapi and a valid ~/.cdsapirc configuration file.
"""

import datetime
import glob
import os
import shutil
import zipfile
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_era5(
    folder_out,
    api_key,
    url,
    area,
    variables_list,
    years,
    months=range(1, 13),
    max_workers=5,
    file_format="netcdf",
    frequency="hourly",
):
    """
    Download ERA5 data from the Copernicus CDS for a range of years and months.

    Args:
        folder_out: Output directory path
        api_key: Copernicus CDS API key
        url: Copernicus CDS API URL
        area: Bounding box [N, W, S, E] in degrees
        variables_list: List of CDS variable names
        years: List or range of years to download
        months: List or range of months (default: 1–12)
        max_workers: Parallel download threads (default: 5)
        file_format: Output format (default: 'netcdf')
        frequency: 'hourly' or 'monthly'
    """
    import xarray as xr

    today_minus_10 = datetime.date.today() - datetime.timedelta(days=10)
    os.makedirs(folder_out, exist_ok=True)

    if frequency == "hourly":
        times = [f"{h:02d}:00" for h in range(24)]
    elif frequency == "monthly":
        times = ["00:00"]
    else:
        raise ValueError("frequency must be 'hourly' or 'monthly'.")

    def download_task(year, month, day=None):
        import cdsapi

        if frequency == "hourly":
            task_date = datetime.date(year, month, day)
            if task_date > today_minus_10:
                print(f"Skipped: {task_date} is too recent (within last 10 days).")
                return
            base_filename = f"ERA5_hourly_{year}_{month:02d}_{day:02d}"
            date_str = f"{year}-{month:02d}-{day:02d}"
        else:
            base_filename = f"ERA5_monthly_{year}_{month:02d}"
            date_str = f"{year}-{month:02d}"

        combined_path = os.path.join(folder_out, base_filename + "_combined.nc")
        temp_folder = os.path.join(folder_out, base_filename + "_tmp")
        zip_path = os.path.join(temp_folder, base_filename + ".zip")

        if os.path.exists(combined_path):
            print(f"Already exists: {combined_path}")
            return

        os.makedirs(temp_folder, exist_ok=True)
        print(f"Downloading {date_str} ({frequency}) ...")
        start = datetime.datetime.now()

        try:
            c = cdsapi.Client(url=url, key=api_key)
            request_dict = {
                "variable": variables_list,
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "area": area,
                "format": file_format,
            }

            if frequency == "hourly":
                request_dict.update({
                    "product_type": ["reanalysis"],
                    "day": [f"{day:02d}"],
                    "time": times,
                })
                dataset = "reanalysis-era5-single-levels"
            else:
                request_dict.update({
                    "product_type": ["monthly_averaged_reanalysis"],
                    "time": times,
                })
                dataset = "reanalysis-era5-single-levels-monthly-means"

            c.retrieve(dataset, request_dict, zip_path)

            if zipfile.is_zipfile(zip_path):
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(temp_folder)
                os.remove(zip_path)

                nc_files = glob.glob(os.path.join(temp_folder, "*.nc"))
                datasets = [xr.open_dataset(f) for f in nc_files]
                try:
                    combined_ds = xr.merge(datasets)
                    combined_ds.to_netcdf(combined_path)
                finally:
                    for ds in datasets:
                        ds.close()
            else:
                # The current CDS API may return a single NetCDF file directly
                # even when the request format is "netcdf".
                shutil.move(zip_path, combined_path)

            shutil.rmtree(temp_folder, ignore_errors=True)
            print(f"Saved: {combined_path} (elapsed: {datetime.datetime.now() - start})")

        except Exception as exc:
            print(f"Error on {date_str}: {exc}")
            with open(os.path.join(folder_out, "download_errors.log"), "a") as log:
                log.write(f"{datetime.datetime.now()}: Error {date_str} - {exc}\n")
            shutil.rmtree(temp_folder, ignore_errors=True)

    tasks = []
    if frequency == "hourly":
        for year in years:
            for month in months:
                for day in range(1, monthrange(year, month)[1] + 1):
                    tasks.append((year, month, day))
    else:
        for year in years:
            for month in months:
                tasks.append((year, month, None))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_task, *t) for t in tasks]
        for future in as_completed(futures):
            future.result()

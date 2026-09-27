# ================================
# IMPORT REQUIRED LIBRARIES
# ================================

import argparse   # Used to read parameters from command line (very useful for SLURM jobs)
import os         # Used for file and directory operations (create folders, check files)
import calendar   # Helps to get number of days in a month (handles leap years correctly)
import cdsapi     # Official API client to download ERA5 data from Copernicus Climate Data Store
import xarray as xr


NZ_BBOX = [-30.2525, 160.542, -48.95, 184.7475] # Bounding box for New Zealand region (North, West, South, East)

# The code below is a script to download ERA5 single-levels hourly data in
#  monthly chunks from the Copernicus Climate Data Store (CDS). 
# It includes functions to generate a range of months, download data for
#  each month, and handle command-line arguments for flexibility.
#  The script also includes checks for existing files and applies spatial 
# filters based on a bounding box for New Zealand.
def parse_bool(value, default=False): 
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def file_matches_requested_domain(file_path, whole_region):
    if whole_region:
        return True

    with xr.open_dataset(file_path) as ds:
        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            return False

        north = float(ds.latitude.max())
        south = float(ds.latitude.min())
        west = float(ds.longitude.min())
        east = float(ds.longitude.max())

    return (
        north <= NZ_BBOX[0]
        and south >= NZ_BBOX[2]
        and west >= NZ_BBOX[1]
        and east <= NZ_BBOX[3]
    )


# ================================
# FUNCTION 1: GENERATE YYYYMM RANGE
# ================================

def yyyymm_iter(start_ym: str, end_ym: str):
    """
    Purpose:
    --------
    Generate a sequence of months between start_ym and end_ym (inclusive).

    Example:
    --------
    Input:  start_ym = "202201", end_ym = "202203"
    Output: 202201, 202202, 202203

    Why needed?
    -----------
    ERA5 downloads are done monthly (not all at once).
    So we loop month-by-month to download data efficiently.
    """

    # Extract year and month from input strings
    sy, sm = int(start_ym[:4]), int(start_ym[4:])
    ey, em = int(end_ym[:4]), int(end_ym[4:])

    y, m = sy, sm

    # Loop until we reach the end month
    while (y < ey) or (y == ey and m <= em):

        # Yield month in YYYYMM format
        yield f"{y}{m:02d}"

        # Move to next month
        m += 1

        # If month exceeds December, move to next year
        if m == 13:
            y += 1
            m = 1


# ================================
# FUNCTION 2: DOWNLOAD DATA FOR ONE MONTH
# ================================

def download_month(c, dataset, product_type, ym, L0_dir, var_name, whole_region):
    """
    Purpose:
    --------
    Download ERA5 data for a SINGLE month.

    Steps:
    ------
    1. Prepare request parameters (year, month, days, time)
    2. Define spatial region (bounding box)
    3. Create output directory
    4. Check if file already exists (skip if yes)
    5. Send request to CDS API and download data
    """

    # Extract year and month from YYYYMM
    year, month = int(ym[:4]), int(ym[4:])

    # Get number of days in the month (handles leap years correctly)
    ndays = calendar.monthlen(year, month) if hasattr(calendar, "monthlen") else calendar.monthrange(year, month)[1]

    # Convert to required string format for API
    years = [f"{year}"]
    months = [f"{month:02d}"]

    # Create list of all days in the month
    days = [f"{d:02d}" for d in range(1, ndays + 1)]

    # Create hourly timestamps (00:00 → 23:00)
    times = [f"{h:02d}:00" for h in range(24)]


    # ================================
    # BUILD API REQUEST (HOURLY DATA)
    # ================================

    if dataset == "reanalysis-era5-single-levels":

        # This request will download HOURLY data
        req = {
            "product_type": [product_type],   # e.g., reanalysis
            "variable": [var_name],           # e.g., 2m_temperature
            "year": years,
            "month": months,
            "day": days,
            "time": times,
            "data_format": "netcdf",          # Output format
            "download_format": "unarchived"
        }


    # ================================
    # BUILD API REQUEST (MONTHLY MEAN)
    # ================================

    if dataset == "reanalysis-era5-single-levels-monthly-means":

        # This request will download monthly aggregated data
        req = {
            "product_type": [product_type],
            "variable": [var_name],
            "year": years,
            "month": months,
            "time": ["00:00"],                # Only one time step needed
            "data_format": "netcdf",
            "download_format": "unarchived"
        }


    # ================================
    # APPLY SPATIAL FILTER (BOUNDING BOX)
    # ================================

    if not whole_region:

        """
        Format: [North, West, South, East]

        This bounding box roughly covers New Zealand region.
        This is VERY IMPORTANT because:
        - Reduces data size
        - Speeds up download
        - Focuses only on relevant region
        """

        new_row = {"area": NZ_BBOX}
        req.update(new_row)


    # Print request for debugging
    print(f"req = {req}")


    # ================================
    # CREATE OUTPUT DIRECTORY
    # ================================

    # Structure: L0_dir/variable/YYYYMM/
    out_dir = os.path.join(L0_dir, f"{var_name}/{ym}")

    # Create folder if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Output file path
    out_file = os.path.join(out_dir, f"ERA5_single_levels_{ym}.nc")


    # ================================
    # CHECK IF FILE ALREADY EXISTS
    # ================================

    # (This avoids re-downloading same data)
    if os.path.isfile(out_file):
        if not file_matches_requested_domain(out_file, whole_region):
            print(f"[RETRY] {ym} exists but does not match requested domain: {out_file}")
            os.remove(out_file)
        else:
            print(f"[SKIP] {ym} already exists: {out_file}")
            return


    # ================================
    # DOWNLOAD DATA FROM CDS API
    # ================================

    print(f"[REQ ] {ym} -> {out_file}")

    # Send request to Copernicus server
    c.retrieve(dataset, req, target=out_file)

    print(f"[DONE] {ym}")


# ================================
# FUNCTION 3: LOOP THROUGH ALL MONTHS
# ================================

def download_forecast_data(L0_dir, var_name, start_ym, end_ym, dataset, product_type, whole_region):
    """
    Purpose:
    --------
    Controls full download process.

    Steps:
    ------
    1. Connect to CDS API
    2. Loop through months
    3. Call download_month() for each
    """

    # API endpoint and authentication key
    URL = "https://cds.climate.copernicus.eu/api"
    KEY = "5d8531d9-e8ac-4ded-9e71-479d2780ab19"  # ⚠️ Better to use .cdsapirc instead

    # Create API client
    c = cdsapi.Client(url=URL, key=KEY)

    # Loop through all months
    for ym in yyyymm_iter(start_ym, end_ym):
        try:
            download_month(c, dataset, product_type, ym, L0_dir, var_name, whole_region)
        except Exception as e:
            print(f"[FAIL] {ym}: {e}")


# ================================
# FUNCTION 4: READ INPUT ARGUMENTS
# ================================

def parse_args():
    """
    Purpose:
    --------
    Read parameters from command line (important for SLURM jobs)

    Example:
    --------
    python script.py --var_name 2m_temperature --start_ym 202201
    """

    p = argparse.ArgumentParser(
        description="Download ERA5 single-levels hourly data in monthly chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset type
    p.add_argument("--dataset", type=str, default="reanalysis-era5-single-levels")

    # Product type
    p.add_argument("--product_type", type=str, default="reanalysis")

    # Region flag
    p.add_argument("--whole_region", type=str, default="false",
                   help="If true, download the full ERA5 global grid. If false, apply the NZ-region bounding box.")

    # Output directory
    p.add_argument("--L0_dir", type=str, default="/nesi/project/massey04632/data/ERA5/L0_CDS",
                   help="Root folder to store data")

    # Variable selection
    p.add_argument("--var_name", type=str, default="2m_temperature",
                   help="e.g., 10m_u_component_of_wind, 10m_v_component_of_wind, 2m_temperature, sea_surface_temperature, mean_sea_level_pressure, total_precipitation")

    # Time range
    p.add_argument("--start_ym", type=str, default="2015501")
    p.add_argument("--end_ym", type=str, default="202512")

    return p.parse_args()


# ================================
# MAIN EXECUTION
# ================================

if __name__ == "__main__":

    # Read arguments
    args = parse_args()
    whole_region = parse_bool(args.whole_region)

    # Start download process
    download_forecast_data(
        args.L0_dir,
        args.var_name,
        args.start_ym,
        args.end_ym,
        args.dataset,
        args.product_type,
        whole_region
    )

    print("Step one on downloading ERA5 data completed.")
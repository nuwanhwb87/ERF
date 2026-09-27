import argparse
import json
import os
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr


NZ_BBOX = [-30.2525, 160.542, -48.95, 184.7475]


def parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_request(year: str, month: str, day: str, time: str, whole_region: bool) -> dict:
    request = {
        "product_type": ["reanalysis"],
        "variable": ["land_sea_mask", "geopotential", "orography"],
        "year": [year],
        "month": [month],
        "day": [day],
        "time": [time],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if not whole_region:
        request["area"] = NZ_BBOX
    return request


def download_static(output_path: str, year: str, month: str, day: str, time: str, whole_region: bool) -> None:
    client = cdsapi.Client()
    request = build_request(year=year, month=month, day=day, time=time, whole_region=whole_region)
    print(f"Submitting CDS request with area={request.get('area', 'global')}...")
    client.retrieve("reanalysis-era5-single-levels", request, target=output_path)
    print(f"Static file saved to: {output_path}")


def _to_numpy(data_array: xr.DataArray) -> np.ndarray:
    values = data_array.values
    if values.ndim == 3 and values.shape[0] == 1:
        return values[0]
    return values


def extract_static_variables(nc_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    ds = xr.open_dataset(nc_path)
    try:
        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            raise ValueError("Expected latitude/longitude coordinates in ERA5 static file.")

        lat = ds["latitude"].values
        lon = ds["longitude"].values
        np.savez(os.path.join(output_dir, "latlon_static.npz"), lat=lat, lon=lon)

        # ERA5 static files may expose either long names (land_sea_mask/geopotential)
        # or short names (lsm/z). Resolve whichever is present and save canonical outputs.
        alias_map = {
            "land_sea_mask": ["land_sea_mask", "lsm"],
            "geopotential": ["geopotential", "z"],
            "orography": ["orography"],
        }

        resolved = {}
        for canonical, candidates in alias_map.items():
            found = next((name for name in candidates if name in ds), None)
            if found is not None:
                resolved[canonical] = found

        if "land_sea_mask" in resolved:
            lsm = _to_numpy(ds[resolved["land_sea_mask"]]).astype(np.float32)
            np.save(os.path.join(output_dir, "lsm.npy"), lsm)
        else:
            print("Warning: could not find land-sea mask variable (expected one of: land_sea_mask, lsm).")

        if "geopotential" in resolved:
            z = _to_numpy(ds[resolved["geopotential"]]).astype(np.float32)
            np.save(os.path.join(output_dir, "geopotential.npy"), z)
        else:
            z = None
            print("Warning: could not find geopotential variable (expected one of: geopotential, z).")

        if "orography" in resolved:
            oro = _to_numpy(ds[resolved["orography"]]).astype(np.float32)
            np.save(os.path.join(output_dir, "orography.npy"), oro)
        elif z is not None:
            # Convert geopotential [m^2 s^-2] to geopotential height [m].
            np.save(os.path.join(output_dir, "orography.npy"), z / np.float32(9.80665))
        else:
            print("Warning: could not produce orography.npy (no orography/z variable available).")

        metadata = {
            "source_nc": nc_path,
            "variables": {k: v for k, v in resolved.items()},
            "shape": {
                "latitude": int(ds.sizes["latitude"]),
                "longitude": int(ds.sizes["longitude"]),
            },
            "lat_min": float(np.min(lat)),
            "lat_max": float(np.max(lat)),
            "lon_min": float(np.min(lon)),
            "lon_max": float(np.max(lon)),
        }
        with open(os.path.join(output_dir, "static_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Extracted static variables to: {output_dir}")
    finally:
        ds.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract ERA5 static variables (lsm, orography, geopotential).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_nc", type=str, default="/nesi/project/massey04632/data/ERA5/static/era5_static.nc")
    parser.add_argument("--extract_dir", type=str, default="/nesi/project/massey04632/data/ERA5/static")
    parser.add_argument("--year", type=str, default="2021")
    parser.add_argument("--month", type=str, default="01")
    parser.add_argument("--day", type=str, default="01")
    parser.add_argument("--time", type=str, default="00:00")
    parser.add_argument("--whole_region", type=str, default="False", help="If true, request global domain; otherwise use NZ bounding box.")
    parser.add_argument("--skip_download", type=str, default="False", help="If true, skip CDS download and only extract from --output_nc.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_nc = Path(args.output_nc)
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    Path(args.extract_dir).mkdir(parents=True, exist_ok=True)

    whole_region = parse_bool(args.whole_region)
    skip_download = parse_bool(args.skip_download)

    if not skip_download:
        download_static(
            output_path=str(output_nc),
            year=args.year,
            month=args.month,
            day=args.day,
            time=args.time,
            whole_region=whole_region,
        )
    elif not output_nc.exists():
        raise FileNotFoundError(
            f"--skip_download=True but file not found: {output_nc}"
        )

    extract_static_variables(nc_path=str(output_nc), output_dir=args.extract_dir)


if __name__ == "__main__":
    main()
import argparse
import glob
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Union, Optional

import fsspec
import pandas as pd
import xarray as xr
import zarr


# ================================
# GENERIC HELPERS
# ================================


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_bool(x: Optional[str], default: bool = False) -> bool:
    if x is None:
        return default
    return str(x).lower() in {"1", "true", "yes", "y"}


def parse_var_names(value: str) -> list[str]:
    try:
        parsed = json.loads(value.replace("'", '"'))
    except json.JSONDecodeError:
        return [value]

    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def parse_chunks(spec: str) -> dict:
    chunks = {}
    for part in spec.split(","):
        key, value = part.split(":")
        chunks[key] = int(value)
    return chunks


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate monthly ERA5 NetCDFs (L0) into a consolidated Zarr (L1). If Zarr exists, append new months.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--L0_dir", type=str, default="/nesi/project/massey04632/data/ERA5/L0_CDS",
                   help="Root folder containing monthly ERA5 NetCDFs.")
    p.add_argument("--var_name", type=str, default="['total_precipitation', '2m_temperature','10m_u_component_of_wind', '10m_v_component_of_wind','mean_sea_level_pressure']",
                   help="ERA5 variable name or JSON-style list of variable names.")
    p.add_argument("--L1_dir", type=str, default="/nesi/project/massey04632/data/ERA5/L1",
                   help="L1 output directory.")
    p.add_argument("--chunks", type=str, default="valid_time:24,latitude:41,longitude:41",
                   help="Chunk spec, e.g. valid_time:24,latitude:41,longitude:41")
    p.add_argument("--skip", type=str, default="False", help="Skip this step.")
    return p.parse_args()


# ================================
# ZARR CREATION LOGIC
# ================================


def generate_zarr(nc_files: list[str], output_dir: str, chunks_spec: str):
    # Build one time-ordered dataset from monthly NetCDF files.
    ds = xr.open_mfdataset(
        nc_files,
        engine="netcdf4",
        combine="nested",
        concat_dim="valid_time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
        parallel=False,
        chunks={},
    ).sortby("valid_time")

    ds = ds.drop_vars(["expver", "number"], errors="ignore")
    ds = ds.chunk(parse_chunks(chunks_spec))

    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, "LATEST_ERA5.zarr")
    ds.to_zarr(output_path, mode="w", consolidated=True)
    print(f"Generated metadata at {output_path}.")


# ================================
# GRID COMPATIBILITY HELPERS
# ================================


def _dataset_grid_signature(ds: xr.Dataset) -> tuple:
    # Represent non-time grid structure so files can be grouped by compatibility.
    signature = []
    for dim in sorted(dim for dim in ds.sizes if dim != "valid_time"):
        signature.extend([dim, int(ds.sizes[dim])])
        if dim in ds.coords:
            coord = ds[dim].values
            signature.extend([float(coord[0]), float(coord[-1])])
    return tuple(signature)


def _filter_files_to_dominant_grid(nc_files: list[str]) -> list[str]:
    # Keep only files matching the most common grid to avoid concat/append conflicts.
    counts: Counter[tuple] = Counter()
    file_signatures: dict[str, tuple] = {}

    for file in nc_files:
        ds = xr.open_dataset(file)
        signature = _dataset_grid_signature(ds)
        ds.close()
        file_signatures[file] = signature
        counts[signature] += 1

    dominant_signature, dominant_count = counts.most_common(1)[0]
    compatible_files = [file for file in nc_files if file_signatures[file] == dominant_signature]
    skipped_files = [file for file in nc_files if file_signatures[file] != dominant_signature]

    if skipped_files:
        print(
            f"Skipping {len(skipped_files)} file(s) that do not match the dominant grid "
            f"({dominant_count}/{len(nc_files)} files):"
        )
        for file in skipped_files:
            print(f" - {file}")

    return compatible_files


def _non_append_dim_sizes(ds: xr.Dataset) -> dict[str, int]:
    return {dim: int(size) for dim, size in ds.sizes.items() if dim != "valid_time"}


def _datasets_are_append_compatible(existing: xr.Dataset, incoming: xr.Dataset) -> tuple[bool, str]:
    # Validate non-time dimensions/coords before appending to an existing store.
    existing_dims = _non_append_dim_sizes(existing)
    incoming_dims = _non_append_dim_sizes(incoming)
    if existing_dims != incoming_dims:
        return False, f"dimension sizes differ: {existing_dims} != {incoming_dims}"

    for dim in existing_dims:
        if dim in existing.coords and dim in incoming.coords and not existing[dim].equals(incoming[dim]):
            return False, f"coordinate values differ for dimension '{dim}'"

    return True, ""


def _archive_incompatible_store(zarr_path: str) -> str:
    # Preserve old store instead of deleting it when schema/grid has changed.
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    archive_path = f"{zarr_path}.incompatible_{timestamp}"
    shutil.move(zarr_path, archive_path)
    print(f"Archived incompatible zarr store to {archive_path}.")
    return archive_path


def append_zarr(file: str, zarr_path: str, existing_months: set[str]):
    # Append exactly one monthly file if its YYYYMM is not already present.
    m = re.search(r"(\d{6})", str(Path(file)))
    if not m:
        raise ValueError(f"Could not find YYYYMM in: {file}")
    yyyymm = m.group(1)

    if yyyymm in existing_months:
        print(f"{yyyymm} has been included in zarr.")
        return

    print(f"{yyyymm} missing; appending from {file}...")
    ds_new = xr.open_dataset(file).sortby("valid_time")
    ds_new = ds_new.drop_vars(["expver", "number"], errors="ignore")
    store = fsspec.get_mapper(zarr_path)
    ds_new.to_zarr(store, mode="a", append_dim="valid_time", consolidated=True)
    zarr.consolidate_metadata(store)
    existing_months.add(yyyymm)
    print(f"Appended {yyyymm} and re-consolidated metadata at {zarr_path}.")


# ================================
# L0 -> L1 CONVERSION PIPELINE
# ================================


def _convert_L0(args, var_name: str):
    # Discover all monthly L0 NetCDF files for one variable.
    var_dir = os.path.join(args.L0_dir, var_name)
    nc_path = f"{var_dir}/**/*.nc"
    nc_files = sorted(glob.glob(nc_path, recursive=True))
    print(f"Found {len(nc_files)} files for {var_name}.")
    print(nc_files)
    if not nc_files:
        print(f"No new NetCDF files found under {nc_path}; skipping L1 update for {var_name}.")
        return False

    nc_files = _filter_files_to_dominant_grid(nc_files)
    if not nc_files:
        print(f"No compatible NetCDF files remained after grid filtering for {var_name}.")
        return False

    output_dir = os.path.join(args.L1_dir, var_name)
    zarr_path = os.path.join(output_dir, "LATEST_ERA5.zarr")

    # If zarr does not exist, build it from all compatible files.
    if not os.path.exists(zarr_path):
        print(f"Creating new zarr at {zarr_path}")
        generate_zarr(nc_files, output_dir=output_dir, chunks_spec=args.chunks)
        return True

    # If zarr exists, append only missing months.
    print(f"Zarr exists at {zarr_path}; appending new files...")
    ds_z = xr.open_zarr(zarr_path, consolidated=True)
    ds_first = xr.open_dataset(nc_files[0]).sortby("valid_time")
    ds_first = ds_first.drop_vars(["expver", "number"], errors="ignore")

    compatible, reason = _datasets_are_append_compatible(ds_z, ds_first)
    ds_first.close()
    if not compatible:
        ds_z.close()
        print(f"Existing zarr is incompatible with incoming NetCDF files: {reason}")
        _archive_incompatible_store(zarr_path)
        print(f"Rebuilding zarr at {zarr_path} from {len(nc_files)} NetCDF files...")
        generate_zarr(nc_files, output_dir=output_dir, chunks_spec=args.chunks)
        return True

    # Track existing YYYYMM to avoid duplicate appends.
    existing_months = set(pd.DatetimeIndex(ds_z.valid_time.values).strftime("%Y%m"))
    ds_z.close()
    print("Existing months: ", existing_months)
    for file in nc_files:
        append_zarr(file, zarr_path, existing_months)
    return True


# ================================
# ENTRYPOINT
# ================================


def main(args):
    if parse_bool(args.skip):
        print("Skipping L1 conversion because --skip=True")
        return

    var_list = parse_var_names(args.var_name)
    print(f"total variables: {var_list}")
    updated_any = False
    for var_name in var_list:
        print(f"Converting variable: {var_name}")
        updated_any = _convert_L0(args, var_name) or updated_any
    if not updated_any:
        print("No new L0 files were staged for any variable; exiting without L1 changes.")


if __name__ == "__main__":
    main(parse_args())
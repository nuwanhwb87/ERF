import xarray as xr
import numpy as np
import os
import sys
import json
from pathlib import Path
import argparse
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configclass.dataset_dataclass import DatasetConfig

def create_dataset(cfg: DatasetConfig):
    os.makedirs(cfg.save_directory, exist_ok=True)

    # Parse variable configurations
    if isinstance(cfg.folders, str):
        cfg.folders = eval(cfg.folders)
    if isinstance(cfg.long_names, str):
        cfg.long_names = eval(cfg.long_names)
    if isinstance(cfg.short_names, str):
        cfg.short_names = eval(cfg.short_names)

    if not (len(cfg.folders) == len(cfg.long_names) == len(cfg.short_names)):
        raise ValueError("folders, long_names, short_names must have equal length.")

    var_names = {f: (L, s) for f, L, s in zip(cfg.folders, cfg.long_names, cfg.short_names)}
    cfg.num_variables = len(var_names)
    
    # Load ERA5 L1 data (zarr format)
    print(f"Loading ERA5 data from: {cfg.file_directory}", flush=True)
    
    # Get time index
    start_datetime = pd.to_datetime(cfg.start_datetime)
    end_datetime = pd.to_datetime(cfg.end_datetime)
    if start_datetime > end_datetime:
        raise ValueError("start_datetime must be earlier than or equal to end_datetime.")
    suffix = f"{start_datetime.strftime('%Y')}-{end_datetime.strftime('%Y')}"
    
    # Load lat/lon from the zarr file
    try:
        file_pattern = f"{cfg.file_directory}/total_precipitation/LATEST_ERA5.zarr"
        ds = xr.open_zarr(file_pattern)
        print(f"Opened zarr: {file_pattern}", flush=True)
        print(f"Dataset dims: {ds.dims}", flush=True)
        print(f"Dataset vars: {list(ds.data_vars)}", flush=True)

        # Enforce the requested time window with robust handling of time dtype/order.
        start_dt = pd.to_datetime(cfg.start_datetime)
        end_dt = pd.to_datetime(cfg.end_datetime)
        time_coord = None
        if "valid_time" in ds.coords:
            time_coord = "valid_time"
        elif "time" in ds.coords:
            time_coord = "time"

        if time_coord is not None:
            t_raw = ds[time_coord].values
            t_idx = pd.to_datetime(t_raw, errors="coerce", utc=False)

            if getattr(t_idx, "tz", None) is not None:
                t_idx = t_idx.tz_localize(None)

            if t_idx.isna().all():
                raise ValueError(
                    f"Could not parse any values from dataset coordinate '{time_coord}' as datetimes."
                )

            valid_count = int((~t_idx.isna()).sum())
            print(
                f"Time coord '{time_coord}': parsed {valid_count}/{len(t_idx)} values; "
                f"min={t_idx.min()} max={t_idx.max()}"
                , flush=True
            )

            not_na = ~t_idx.isna()
            if valid_count >= 2:
                monotonic_inc = bool((t_idx[not_na][1:] >= t_idx[not_na][:-1]).all())
                monotonic_dec = bool((t_idx[not_na][1:] <= t_idx[not_na][:-1]).all())
            else:
                monotonic_inc = False
                monotonic_dec = False

            if monotonic_inc:
                ds = ds.sel({time_coord: slice(np.datetime64(start_dt), np.datetime64(end_dt))})
            elif monotonic_dec:
                ds = ds.sel({time_coord: slice(np.datetime64(end_dt), np.datetime64(start_dt))})
            else:
                # Fallback for non-monotonic/irregular time arrays.
                mask = not_na & (t_idx >= start_dt) & (t_idx <= end_dt)
                ds = ds.isel({time_coord: np.where(mask)[0]})

            # Final fallback if direct .sel produced empty unexpectedly.
            if ds["tp"].shape[0] == 0:
                mask = not_na & (t_idx >= start_dt) & (t_idx <= end_dt)
                ds = ds.isel({time_coord: np.where(mask)[0]})
        else:
            print("Warning: no valid_time/time coordinate found; using full dataset range.")

        if ds["tp"].shape[0] == 0:
            raise ValueError(
                f"No samples found for requested time range {cfg.start_datetime} to {cfg.end_datetime}."
            )

        print(f"Dataset dims after time slicing: {ds.dims}", flush=True)
    except Exception as e:
        print(f"Error loading zarr: {e}")
        raise

    # Use actual dataset dimensions
    actual_height = ds['latitude'].sizes['latitude']
    actual_width = ds['longitude'].sizes['longitude']
    cfg.height = actual_height
    cfg.width = actual_width
    
    lat = ds['latitude'].values
    lon = ds['longitude'].values
    
    # Save lat/lon
    np.savez(f'{cfg.save_directory}/latlon_{suffix}.npz', lat=lat, lon=lon)
    print(f"Saved latlon to {cfg.save_directory}/latlon_{suffix}.npz", flush=True)

    # Prepare memmap for storing data
    combined_shape = (ds['tp'].shape[0], cfg.num_variables, cfg.height, cfg.width)
    print(f"Combined shape: {combined_shape}", flush=True)

    save_name = '_'.join([var_name[1] for var_name in var_names.values()])
    memmap_file_path = f'{cfg.save_directory}/{save_name}_{suffix}.npy'
    memmap_array = np.memmap(memmap_file_path, dtype='float32', mode='w+', shape=combined_shape)

    # Process each variable
    statistics = {}
    i = 0
    for file_prefix, names in var_names.items():
        var_name = names[0]
        short_name = names[1]
        print(f"Processing variable: {var_name} (short_name: {short_name})", flush=True)

        if file_prefix == "total_precipitation":
            variable_ds = ds
        else:
            file_pattern = f"{cfg.file_directory}/{file_prefix}/LATEST_ERA5.zarr"
            variable_ds = xr.open_zarr(file_pattern)
            if time_coord not in variable_ds.coords:
                raise ValueError(f"Zarr store {file_pattern} does not contain '{time_coord}'.")
            try:
                variable_ds = variable_ds.sel({time_coord: ds[time_coord]})
                _, variable_ds = xr.align(ds, variable_ds, join="exact", copy=False)
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"Zarr store {file_pattern} does not exactly match the precipitation time/grid coordinates."
                ) from error

        if short_name not in variable_ds:
            raise KeyError(f"Variable '{short_name}' not found in {file_prefix} Zarr store.")

        array = variable_ds[short_name]
        print(f"Dataset shape: {array.shape}", flush=True)

        mean_sum = 0.0
        sq_sum = 0.0
        count = 0
        
        # Process in chunks
        for j in range(0, array.shape[0], cfg.chunk_size):
            end_idx = min(j + cfg.chunk_size, array.shape[0])

            chunk = array[j:end_idx].compute() if hasattr(array[j:end_idx], 'compute') else array[j:end_idx]
            chunk = np.nan_to_num(chunk, nan=0.0)
            memmap_array[j:end_idx, i, :, :] = chunk

            # Calculate statistics
            c = chunk if isinstance(chunk, np.ndarray) else chunk.values
            mean_sum += c.sum()
            sq_sum += (c ** 2).sum()
            count += c.size

        # Calculate final statistics
        mean_value = mean_sum / count if count > 0 else 0
        std_value = np.sqrt(sq_sum / count - mean_value ** 2) if count > 0 else 1
        statistics[var_name] = {"mean": float(mean_value), "std": float(std_value)}
        with open(f'{cfg.save_directory}/norm_factors.json', 'w') as f:
            json.dump(statistics, f, indent=4)
        print(f"{var_name}: Mean = {mean_value}, Std = {std_value}", flush=True)
        if variable_ds is not ds:
            variable_ds.close()
        i += 1

    memmap_array.flush()
    print(f"Combined data saved as: {memmap_file_path}")
    
    # Save normalization factors
    json_file = f'{cfg.save_directory}/norm_factors.json'
    with open(json_file, 'w') as f:
        json.dump(statistics, f, indent=4)
    print(f"Normalization factors saved to {json_file}")
    
    for var_name, stats in statistics.items():
        print(f"{var_name}: Mean = {stats['mean']}, Std = {stats['std']}")

# --------------------------- CLI ---------------------------
def parse_args() -> DatasetConfig:
    ap = argparse.ArgumentParser(
        description="Dataset creation from Zarr files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # dynamic variables - ERA5 total_precipitation
    # 'total_precipitation', '2m_temperature','10m_u_component_of_wind', '10m_v_component_of_wind','mean_sea_level_pressure'
    ap.add_argument("--folders", type=str, default="['total_precipitation','2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind','mean_sea_level_pressure']")
    ap.add_argument("--long_names",  type=str, default="['total_precipitation','2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind','mean_sea_level_pressure']")
    ap.add_argument("--short_names",  type=str, default="['tp','t2m','u10','v10','msl']")
    # static fields (not used for ERA5)
    ap.add_argument("--field_folders", type=str, default="[]", help="Static field variable names (not used for ERA5).")
    ap.add_argument("--field_shorts", type=str, default="[]", help="Short names for static fields (not used for ERA5).")
    ap.add_argument("--height", type=int, default=74, help="Grid height for ERA5 data.")
    ap.add_argument("--width", type=int, default=96, help="Grid width for ERA5 data.")
    ap.add_argument("--chunk_size", type=int, default=50, help="chunk_size for loading data.")
    ap.add_argument("--num_variables", type=int, default=5, help="Number of dynamic ERA5 variables.")
    ap.add_argument("--num_static_fields", type=int, default=0, help="num_static_fields (0 for ERA5).")
    ap.add_argument("--max_horizon", type=int, default=180, help="Prediction frames (T_lead).")

    # Optional: input/output dirs for ERA5 data
    ap.add_argument("--file_directory", type=str, default="/nesi/project/massey04632/data/ERA5/L1", help="Path to ERA5 L1 zarr files")
    ap.add_argument("--save_directory", type=str, default="/nesi/project/massey04632/data/ERA5/L2/2015_2025_5V", help="Output path to L2 files for ML.")

    # --- dataset time config ---
    ap.add_argument("--start_datetime", type=str, default="2015-01-01T00:00:00")
    ap.add_argument("--end_datetime", type=str, default="2025-12-31T00:00:00")
    ap.add_argument("--time_freq", type=str, default="1d", help="Pandas offset alias, e.g. 1h, 6h, 7D, W, W-TUE")
    # --- split config (years/dates only) ---
    ap.add_argument("--split_mode", type=str, default="dates", choices=["years", "dates"])
    ap.add_argument("--train_year_end", type=str, default="2023", help="Used when split_mode=years")
    ap.add_argument("--val_year_end", type=str, default="2024", help="Used when split_mode=years")
    ap.add_argument("--train_until", type=str, default="2023-12-31T00:00:00", help="Used when split_mode=dates (inclusive)")
    ap.add_argument("--val_until", type=str, default="2024-12-31T00:00:00", help="Used when split_mode=dates (inclusive)")
    ap.add_argument("--skip", type=str, default="False", help="Skip this step")

    args = ap.parse_args()
    return DatasetConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    print(f'value of skip {cfg.skip}, type of skip {type(cfg.skip)}')
    if cfg.skip == "False":
        print("We are performing the step on dataset creation.", flush=True)
        create_dataset(cfg)
    else:
        print("We are skipping the step on preparing datasets")
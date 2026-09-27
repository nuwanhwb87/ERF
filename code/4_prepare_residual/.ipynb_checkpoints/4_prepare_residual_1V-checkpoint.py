import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configclass.dataset_dataclass import DatasetConfig
from tools.utils import make_time_index, split_time_index


def _parse_list(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return list(value)


def _normalize_config_lists(cfg: DatasetConfig) -> None:
    cfg.folders = _parse_list(cfg.folders)
    cfg.long_names = _parse_list(cfg.long_names)
    cfg.short_names = _parse_list(cfg.short_names)
    cfg.field_folders = _parse_list(cfg.field_folders)
    cfg.field_shorts = _parse_list(cfg.field_shorts)


def _resolve_time_index(cfg: DatasetConfig, var_names: dict[str, tuple[str, str]], expected_samples: int) -> pd.DatetimeIndex:
    primary_var = next(iter(var_names.keys()))
    zarr_path = Path(cfg.file_directory) / primary_var / "LATEST_ERA5.zarr"
    if zarr_path.exists():
        ds = xr.open_zarr(zarr_path, consolidated=True)
        try:
            if "valid_time" in ds.coords:
                time_values = ds.valid_time.values
            elif "time" in ds.coords:
                time_values = ds.time.values
            else:
                raise ValueError("No valid_time/time coordinate found in source zarr.")

            ti = pd.DatetimeIndex(time_values)
            start_dt = pd.Timestamp(cfg.start_datetime)
            end_dt = pd.Timestamp(cfg.end_datetime)
            ti = ti[(ti >= start_dt) & (ti <= end_dt)]
        finally:
            ds.close()
        if len(ti) != expected_samples:
            raise ValueError(
                f"Zarr time axis length ({len(ti)}) does not match dataset samples ({expected_samples})."
            )
        return ti

    ti = make_time_index(cfg.start_datetime, cfg.end_datetime, cfg.time_freq)
    if len(ti) != expected_samples:
        raise ValueError(
            f"Configured time index length ({len(ti)}) does not match dataset samples ({expected_samples}). "
            f"Provide the matching L1 zarr under {cfg.file_directory} or update the time settings."
        )
    return ti


def _resolve_dataset_path(save_directory: str, save_name: str, suffix: str) -> Path:
    candidate_names = [
        f"{save_name}_{suffix}.npy",
        f"{save_name}_{suffix}_5.625deg.npy",
    ]
    for candidate in candidate_names:
        path = Path(save_directory) / candidate
        if path.exists():
            return path

    matches = sorted(Path(save_directory).glob(f"{save_name}_{suffix}*.npy"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find dataset file for prefix '{save_name}_{suffix}' in {save_directory}."
        )
    raise FileNotFoundError(
        f"Found multiple dataset files for prefix '{save_name}_{suffix}' in {save_directory}: {matches}"
    )


def _sample_training_indices(train_stop: int, max_horizon: int, conditioning_times: list[int], sample_size: int) -> np.ndarray:
    spinup = max(0, -min(conditioning_times, default=0))
    start = spinup
    stop = train_stop - max_horizon
    if stop <= start:
        raise ValueError(
            f"Not enough training samples to compute residuals: start={start}, stop={stop}, max_horizon={max_horizon}."
        )

    population = np.arange(start, stop, dtype=np.int64)
    if len(population) <= sample_size:
        return population

    rng = np.random.default_rng(0)
    return np.sort(rng.choice(population, size=sample_size, replace=False))


def _compute_residual_std_for_lead_time(
    dataset: np.memmap,
    indices: np.ndarray,
    lead_time: int,
    std_scale: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    sum_diff = np.zeros(std_scale.shape[0], dtype=np.float64)
    sum_sq_diff = np.zeros(std_scale.shape[0], dtype=np.float64)
    count = 0

    for start in range(0, len(indices), chunk_size):
        batch_indices = indices[start:start + chunk_size]
        current = dataset[batch_indices]
        future = dataset[batch_indices + lead_time]
        diff = (future - current) / std_scale[None, :, None, None]

        sum_diff += diff.sum(axis=(0, 2, 3), dtype=np.float64)
        sum_sq_diff += np.square(diff, dtype=np.float64).sum(axis=(0, 2, 3), dtype=np.float64)
        count += diff.shape[0] * diff.shape[2] * diff.shape[3]

    mean_diff = sum_diff / count
    variance = np.maximum(sum_sq_diff / count - mean_diff ** 2, 0.0)
    return np.sqrt(variance)


def calculate_residuals(cfg: DatasetConfig):
    os.makedirs(cfg.save_directory, exist_ok=True)
    _normalize_config_lists(cfg)

    if not (len(cfg.folders) == len(cfg.long_names) == len(cfg.short_names)):
        raise ValueError("folders, long_names, short_names must have equal length.")
    if not (len(cfg.field_folders) == len(cfg.field_shorts)):
        raise ValueError("field_folders and field_shorts must have equal length.")

    var_names = {
        folder: (long_name, short_name)
        for folder, long_name, short_name in zip(cfg.folders, cfg.long_names, cfg.short_names)
    }
    variable_names = [names[0] for names in var_names.values()]
    conditioning_times = [0]

    with open(Path(cfg.save_directory) / "norm_factors.json", "r") as f:
        statistics = json.load(f)

    mean_data = np.array([statistics[name]["mean"] for name in variable_names], dtype=np.float32)
    std_data = np.array([statistics[name]["std"] for name in variable_names], dtype=np.float32)

    save_name = "_".join(variable_names)
    expected_suffix = f"{cfg.start_datetime[:4]}-{cfg.end_datetime[:4]}"
    dataset_path = _resolve_dataset_path(cfg.save_directory, save_name, expected_suffix)

    sample_shape = (cfg.num_variables, cfg.height, cfg.width)
    item_count = int(np.prod(sample_shape))
    total_values = dataset_path.stat().st_size // np.dtype(np.float32).itemsize
    if total_values % item_count != 0:
        raise ValueError(
            f"Dataset file {dataset_path} size is not divisible by sample shape {sample_shape}."
        )
    n_samples = total_values // item_count

    ti = _resolve_time_index(cfg, var_names, n_samples)
    suffix = f"{ti[0].strftime('%Y')}-{ti[-1].strftime('%Y')}"

    dataset = np.memmap(
        dataset_path,
        dtype=np.float32,
        mode="r",
        shape=(n_samples, cfg.num_variables, cfg.height, cfg.width),
    )

    _, n_train, _ = split_time_index(
        ti,
        split_mode=cfg.split_mode,
        train_year_end=int(cfg.train_year_end) if cfg.train_year_end is not None else None,
        val_year_end=int(cfg.val_year_end) if cfg.val_year_end is not None else None,
        train_until=cfg.train_until,
        val_until=cfg.val_until,
    )

    sample_indices = _sample_training_indices(
        train_stop=n_train,
        max_horizon=cfg.max_horizon,
        conditioning_times=conditioning_times,
        sample_size=cfg.residual_sample_size,
    )

    stds_directory = Path(cfg.save_directory) / "residual_stds"
    stds_directory.mkdir(parents=True, exist_ok=True)

    stds_dict = {var_name: [] for var_name in variable_names}
    for lead_time in range(1, cfg.max_horizon + 1):
        std_t = _compute_residual_std_for_lead_time(
            dataset=dataset,
            indices=sample_indices,
            lead_time=lead_time,
            std_scale=std_data,
            chunk_size=cfg.chunk_size,
        )
        print(f"lead time: {lead_time}, std is {std_t}", flush=True)
        for index, var_name in enumerate(stds_dict):
            stds_dict[var_name].append(float(std_t[index]))

    for var_name, stds in stds_dict.items():
        file_path = stds_directory / f"WB_{var_name}.txt"
        content = "\n".join(f"{lead_time} {std}" for lead_time, std in enumerate(stds, start=1))
        with open(file_path, "w") as file:
            file.write(content)
        print(f"Standard deviations for {var_name} saved to {file_path}")

    print(f"Residual calculation completed for {dataset_path.name} ({suffix}).")


def parse_args() -> DatasetConfig:
    ap = argparse.ArgumentParser(
        description="Residual standard deviation calculation for the ERA5 rainfall study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--folders", type=str, default="['total_precipitation']")
    ap.add_argument("--long_names", type=str, default="['total_precipitation']")
    ap.add_argument("--short_names", type=str, default="['tp']")
    ap.add_argument("--field_folders", type=str, default="[]", help="Static field variable names.")
    ap.add_argument("--field_shorts", type=str, default="[]", help="Short names for static fields.")
    ap.add_argument("--height", type=int, default=74, help="Grid height.")
    ap.add_argument("--width", type=int, default=96, help="Grid width.")
    ap.add_argument("--chunk_size", type=int, default=256, help="Number of sampled times per residual chunk.")
    ap.add_argument("--num_variables", type=int, default=2, help="Number of dynamic variables.")
    ap.add_argument("--num_static_fields", type=int, default=0, help="Number of static fields.")
    ap.add_argument("--max_horizon", type=int, default=180, help="Prediction frames (T_lead).")
    ap.add_argument("--residual_sample_size", type=int, default=10000, help="Number of training timestamps sampled per lead time.")

    ap.add_argument("--file_directory", type=str, default="/nesi/project/massey04632/data/ERA5/L1", help="Path to ERA5 L1 zarr files for exact timestamps.")
    ap.add_argument("--save_directory", type=str, default="/nesi/project/massey04632/data/ERA5/L2/2015_2025_1V", help="Path to ERA5 L2 memmap data and residual outputs.")

    ap.add_argument("--start_datetime", type=str, default="2015-01-01T00:00:00")
    ap.add_argument("--end_datetime", type=str, default="2025-12-31T00:00:00")
    ap.add_argument("--time_freq", type=str, default="1h", help="Pandas offset alias, e.g. 1h, 6h, 7D, W, W-TUE")
    ap.add_argument("--split_mode", type=str, default="dates", choices=["years", "dates"])
    ap.add_argument("--train_year_end", type=str, default=None, help="Used when split_mode=years")
    ap.add_argument("--val_year_end", type=str, default=None, help="Used when split_mode=years")
    ap.add_argument("--train_until", type=str, default="2023-12-31T23:00:00", help="Used when split_mode=dates (inclusive)")
    ap.add_argument("--val_until", type=str, default="2024-12-31T23:00:00", help="Used when split_mode=dates (inclusive)")
    ap.add_argument("--skip", type=str, default="False", help="Skip this step")

    args = ap.parse_args()
    cfg_values = vars(args).copy()
    residual_sample_size = cfg_values.pop("residual_sample_size")
    cfg = DatasetConfig(**cfg_values)
    setattr(cfg, "residual_sample_size", residual_sample_size)
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    print(f"value of skip {cfg.skip}, type of skip {type(cfg.skip)}")
    if cfg.skip == "False":
        print("We are performing the step on residual calculation.", flush=True)
        calculate_residuals(cfg)
    else:
        print("We are skipping the step on residual calculation.", flush=True)

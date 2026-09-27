#!/usr/bin/env python3
"""Standalone deterministic CNN trainer for ERA5 L2 rainfall data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loss.loss import WMSELoss
from tools.utils import make_time_index as shared_make_time_index, split_time_index as shared_split_time_index

try:
    import xarray as xr
except Exception:
    xr = None


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_time_freq_hours(time_freq: str) -> int:
    tf = time_freq.strip().lower()
    if not tf.endswith("h"):
        raise ValueError(f"Only hourly time_freq is supported, got: {time_freq}")
    step = int(tf[:-1])
    if step <= 0:
        raise ValueError(f"time_freq must be positive, got: {time_freq}")
    return step


def resolve_data_directory(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def infer_n_samples_from_memmap(dataset_path: Path, num_variables: int, n_lat: int, n_lon: int) -> int:
    bytes_per_value = np.dtype(np.float32).itemsize
    total_values = os.path.getsize(dataset_path) // bytes_per_value
    values_per_sample = num_variables * n_lat * n_lon
    if total_values % values_per_sample != 0:
        raise ValueError(
            f"Dataset file size is incompatible with dimensions: {dataset_path}, "
            f"num_variables={num_variables}, n_lat={n_lat}, n_lon={n_lon}."
        )
    return total_values // values_per_sample


def _resolve_dataset_file(data_directory: Path, variable_names: list[str], suffix: str) -> Path:
    prefix = "_".join(variable_names)
    candidates = [
        data_directory / f"{prefix}_{suffix}.npy",
        data_directory / f"{prefix}_{suffix}_5.625deg.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(data_directory.glob(f"{prefix}_{suffix}*.npy"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No dataset file found for prefix '{prefix}_{suffix}' in {data_directory}")
    raise FileNotFoundError(f"Multiple dataset files found for prefix '{prefix}_{suffix}': {matches}")


def _resolve_latlon_file(data_directory: Path, suffix: str) -> Path:
    candidates = [
        data_directory / f"latlon_{suffix}.npz",
        data_directory / f"latlon_{suffix}_5.625deg.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No lat/lon file found for suffix '{suffix}' in {data_directory}")


def split_counts_from_dates(
    n_samples: int,
    start_datetime: str,
    time_freq: str,
    train_until: str,
    val_until: str,
) -> tuple[int, int]:
    step_hours = parse_time_freq_hours(time_freq)

    start = datetime.fromisoformat(start_datetime)
    tr_end = datetime.fromisoformat(train_until)
    va_end = datetime.fromisoformat(val_until)

    def count_inclusive(end_dt: datetime) -> int:
        delta_hours = (end_dt - start).total_seconds() / 3600.0
        steps = int(np.floor(delta_hours / step_hours)) + 1
        return max(0, min(n_samples, steps))

    n_train = count_inclusive(tr_end)
    n_train_val = count_inclusive(va_end)
    n_val = max(0, n_train_val - n_train)
    return n_train, n_val


def make_time_index(start_datetime: str, end_datetime: str, time_freq: str) -> pd.DatetimeIndex:
    start = pd.to_datetime(start_datetime)
    end = pd.to_datetime(end_datetime)
    try:
        return pd.date_range(start, end, freq=time_freq, inclusive="both")
    except TypeError:
        ti = pd.date_range(start, end, freq=time_freq)
        if len(ti) and ti[-1] == end:
            return ti
        offset = pd.tseries.frequencies.to_offset(time_freq)
        if len(ti) and (end - ti[-1]) % offset == pd.Timedelta(0):
            return ti.append(pd.DatetimeIndex([end]))
        return ti


def split_time_index_dates(ti: pd.DatetimeIndex, train_until: str, val_until: str) -> tuple[int, int]:
    tr_end = pd.to_datetime(train_until)
    va_end = pd.to_datetime(val_until)
    train_mask = ti <= tr_end
    val_mask = (ti > tr_end) & (ti <= va_end)
    return int(train_mask.sum()), int(val_mask.sum())


def resolve_time_index_for_training(
    data_dir: Path,
    n_samples: int,
    start_datetime: str,
    end_datetime: str,
    time_freq: str,
) -> pd.DatetimeIndex:
    """Resolve a sample-aligned DatetimeIndex using L1 valid_time when available."""
    l1_zarr = data_dir.parent / "L1" / "total_precipitation" / "LATEST_ERA5.zarr"
    if l1_zarr.exists() and xr is not None:
        ds = xr.open_zarr(str(l1_zarr), consolidated=True)
        try:
            ti = pd.DatetimeIndex(ds.valid_time.values)
        finally:
            ds.close()
    else:
        ti = make_time_index(start_datetime, end_datetime, time_freq)

    if len(ti) != n_samples:
        print(
            f"Warning: time index length ({len(ti)}) != dataset samples ({n_samples}); "
            "using fallback index with dataset sample count.",
            flush=True,
        )
        ti = pd.date_range(start=pd.to_datetime(start_datetime), periods=n_samples, freq=time_freq)

    return ti


def comp_area_weights_simple(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    area_weights = np.cos(lat * np.pi / 180.0).reshape(-1, 1)
    area_weights = np.repeat(area_weights, lon.size, axis=1)
    area_weights *= (lat.size * lon.size / np.sum(area_weights))
    return area_weights


class LocalERA5Dataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        dataset_mode: str,
        n_samples: int,
        n_train: int,
        n_val: int,
        num_variables: int,
        n_lat: int,
        n_lon: int,
        lead_times: list[int],
        max_horizon: int,
        conditioning_times: list[int],
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        spacing: int = 1,
        spinup: int = 24,
    ) -> None:
        self.n_samples = n_samples
        self.n_train = n_train
        self.n_val = n_val
        self.num_variables = num_variables
        self.n_lat = n_lat
        self.n_lon = n_lon
        self.lead_times = list(lead_times)
        self.max_horizon = int(max_horizon)
        self.conditioning_times = np.asarray(conditioning_times, dtype=np.int64)
        self.spacing = int(spacing)
        self.spinup = int(spinup)

        self.norm_mean = norm_mean.astype(np.float32)
        self.norm_std = norm_std.astype(np.float32)

        self.mmap = np.memmap(
            str(dataset_path),
            dtype=np.float32,
            mode="r",
            shape=(n_samples, num_variables, n_lat, n_lon),
        )

        if dataset_mode == "train":
            start, stop = self.spinup, self.n_train
        elif dataset_mode == "val":
            start, stop = self.spinup + self.n_train, self.n_train + self.n_val
        else:
            raise ValueError(f"Unknown dataset_mode: {dataset_mode}")

        self.index_array = np.arange(start, stop - self.max_horizon, self.spacing, dtype=np.int64)
        if self.index_array.size == 0:
            raise ValueError(
                f"No valid indices for dataset_mode={dataset_mode}. "
                f"start={start}, stop={stop}, max_horizon={self.max_horizon}, spacing={self.spacing}."
            )

        self.input_channels = self.num_variables * len(self.conditioning_times)

    def __len__(self) -> int:
        return int(self.index_array.shape[0])

    def __getitem__(self, idx: int):
        start_index = int(self.index_array[idx])
        lead = int(np.random.choice(self.lead_times))

        x_index = start_index + self.conditioning_times
        y_index = start_index + lead

        x_sample = self.mmap[x_index, :].astype(np.float32)
        y_sample = self.mmap[y_index, :].astype(np.float32)

        x_sample = (x_sample - self.norm_mean[None, :, None, None]) / self.norm_std[None, :, None, None]
        y_sample = (y_sample - self.norm_mean[None, :, None, None]) / self.norm_std[None, :, None, None]

        x_sample = torch.from_numpy(x_sample).view(self.input_channels, self.n_lat, self.n_lon)
        y_sample = torch.from_numpy(y_sample).view(self.num_variables, self.n_lat, self.n_lon)
        lead_tensor = torch.tensor(lead, dtype=torch.float32)
        return x_sample, y_sample, lead_tensor


def _resolve_lsm_file(data_directory: Path, lsm_path: str | None = None) -> Path | None:
    candidates = []
    if lsm_path:
        candidates.append(Path(lsm_path))
    candidates.extend([
        data_directory / "lsm.npy",
        data_directory / "static" / "lsm.npy",
        data_directory.parent / "static" / "lsm.npy",
        Path("/nesi/project/massey04632/data/ERA5/static/lsm.npy"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _build_land_mask(data_directory: Path, height: int, width: int, device: torch.device, lsm_path: str | None = None) -> torch.Tensor | None:
    mask_path = _resolve_lsm_file(data_directory, lsm_path=lsm_path)
    if mask_path is None:
        raise FileNotFoundError(
            "Land-only CNN training requires lsm.npy, but no mask file was found. "
            "Provide --lsm_path or place lsm.npy under the data/static directory."
        )

    lsm = np.load(mask_path).astype(np.float32)
    if lsm.ndim == 3 and lsm.shape[0] == 1:
        lsm = lsm[0]
    if lsm.ndim != 2:
        raise ValueError(f"Expected 2D lsm array, got shape {lsm.shape} from {mask_path}")
    if lsm.shape != (height, width):
        raise ValueError(f"LSM shape {lsm.shape} does not match training grid {(height, width)}. File: {mask_path}")

    land_mask = (lsm > 0.5).astype(np.float32)
    print(f"Using land mask from {mask_path}; land fraction={land_mask.mean():.4f}", flush=True)
    return torch.tensor(land_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)


class DeterministicCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, filters: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 1, filters, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, filters), num_channels=filters),
            nn.GELU(),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, filters), num_channels=filters),
            nn.GELU(),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, filters), num_channels=filters),
            nn.GELU(),
            nn.Conv2d(filters, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, time_labels_norm: torch.Tensor) -> torch.Tensor:
        t_map = time_labels_norm.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, t_map], dim=1))


def calculate_gradient_stats(model: nn.Module) -> dict[str, float]:
    max_grad = -float("inf")
    min_grad = float("inf")
    total_params = 0
    param_inf_norms = []
    all_grad_values = []

    for param in model.parameters():
        total_params += param.numel()
        if param.grad is None:
            continue
        pabs = param.grad.data.abs()
        inf_norm = pabs.max().item()
        param_inf_norms.append(inf_norm)
        max_grad = max(max_grad, inf_norm)
        min_grad = min(min_grad, pabs.min().item())
        all_grad_values.append(pabs.flatten().cpu().numpy())

    avg_inf = float(np.mean(param_inf_norms)) if param_inf_norms else 0.0
    if all_grad_values:
        g = np.concatenate(all_grad_values)
        grad_mean = float(np.mean(g))
        grad_median = float(np.median(g))
        grad_p95 = float(np.percentile(g, 95))
        grad_p99 = float(np.percentile(g, 99))
    else:
        grad_mean = grad_median = grad_p95 = grad_p99 = 0.0

    return {
        "max_grad": 0.0 if max_grad == -float("inf") else float(max_grad),
        "avg_inf_norm": avg_inf,
        "min_grad": 0.0 if min_grad == float("inf") else float(min_grad),
        "grad_mean": grad_mean,
        "grad_median": grad_median,
        "grad_p95": grad_p95,
        "grad_p99": grad_p99,
        "total_params": int(total_params),
    }


@dataclass
class CNNTrainConfig:
    name: str
    batch_size: int
    filters: int
    weight_decay: float
    lr: float
    epochs: int
    spacing: int
    t_min: int
    t_max: int
    delta_t: int
    conditioning_times: list[int]
    model: str
    seed: int
    variable_names: list[str]
    num_variables: int
    height: int
    width: int
    data_directory: str
    result_directory: str
    save_every: int
    max_horizon: int
    start_datetime: str
    end_datetime: str
    split_mode: str
    train_year_end: int | None
    val_year_end: int | None
    train_until: str
    val_until: str
    test_start: str
    test_until: str
    time_freq: str
    land_only: bool = True
    lsm_path: str | None = None


def ensure_cnn_prefix(name: str) -> str:
    """Force CNN-prefixed run names so outputs are isolated from UNet runs."""
    return name if name.startswith("CNN") else f"CNN_{name}"


def parse_args() -> CNNTrainConfig:
    ap = argparse.ArgumentParser(
        description="Train deterministic CNN on ERA5 L2 rainfall memmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--name", type=str, default="CNN_rainfall_tp_model_cnn_12to36_dt6")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--spacing", type=int, default=1)
    ap.add_argument("--t_min", type=int, default=12)
    ap.add_argument("--t_max", type=int, default=36)
    ap.add_argument("--delta_t", type=int, default=6)
    ap.add_argument(
        "--conditioning_times",
        type=lambda s: [int(x.strip()) for x in s.split(",")],
        default=[0, -6],
    )
    ap.add_argument("--model", type=str, default="deterministic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variable_names", type=lambda s: [x.strip() for x in s.split(",")], default="total_precipitation")
    ap.add_argument("--num_variables", type=int, default=1)
    ap.add_argument("--height", type=int, default=74)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--data_directory", type=str, default="/nesi/project/massey04632/data/ERA5/L2/2015_2025_1V")
    ap.add_argument("--result_directory", type=str, default="/nesi/project/massey04632/ERF/models")
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--max_horizon", type=int, default=180)
    ap.add_argument("--start_datetime", type=str, default="2015-01-01T00:00:00")
    ap.add_argument("--end_datetime", type=str, default="2025-12-31T00:00:00")
    ap.add_argument("--split_mode", type=str, default="dates", choices=["years", "dates"])
    ap.add_argument("--train_year_end", type=int, default=None)
    ap.add_argument("--val_year_end", type=int, default=None)
    ap.add_argument("--train_until", type=str, default="2023-12-31T00:00:00")
    ap.add_argument("--val_until", type=str, default="2024-12-31T00:00:00")
    ap.add_argument("--test_start", type=str, default="2025-01-01T00:00:00")
    ap.add_argument("--test_until", type=str, default="2025-12-31T00:00:00")
    ap.add_argument("--time_freq", type=str, default="1h")
    ap.add_argument("--land_only", type=parse_bool, default=True, help="Mask ocean data using lsm.npy during training/validation.")
    ap.add_argument("--lsm_path", type=str, default=None, help="Optional explicit path to lsm.npy.")

    args = ap.parse_args()
    return CNNTrainConfig(**vars(args))


def load_norm_and_residuals(cfg: CNNTrainConfig, data_dir: Path, device: torch.device):
    with open(data_dir / "norm_factors.json", "r") as f:
        statistics = json.load(f)

    means = []
    stds = []
    residual_cols = []
    for var_name in cfg.variable_names:
        if var_name not in statistics:
            raise KeyError(f"Variable '{var_name}' not found in norm_factors.json")
        means.append(float(statistics[var_name]["mean"]))
        stds.append(float(statistics[var_name]["std"]))

        residual_path = data_dir / "residual_stds" / f"WB_{var_name}.txt"
        residual = np.loadtxt(residual_path, delimiter=" ")[:, 1].astype(np.float32)
        residual_cols.append(torch.tensor(residual, dtype=torch.float32, device=device))

    mean = np.asarray(means, dtype=np.float32)
    std = np.asarray(stds, dtype=np.float32)
    residual_stds = torch.stack(residual_cols, dim=1)[: cfg.max_horizon]
    return mean, std, residual_stds


def train_loop(cfg: CNNTrainConfig) -> None:
    if cfg.model != "deterministic":
        raise ValueError("This script supports only --model deterministic.")
    if cfg.variable_names != ["total_precipitation"] or cfg.num_variables != 1:
        raise ValueError(
            "This one-variable CNN trainer requires --variable_names total_precipitation "
            "and --num_variables 1."
        )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = resolve_data_directory(cfg.data_directory)
    land_mask = None
    if cfg.land_only:
        land_mask = _build_land_mask(data_dir, cfg.height, cfg.width, device, lsm_path=cfg.lsm_path)
    else:
        print("Land-only mode disabled by config; ocean points are not masked.", flush=True)

    suffix = f"{cfg.start_datetime[:4]}-{cfg.end_datetime[:4]}"
    dataset_path = _resolve_dataset_file(data_dir, cfg.variable_names, suffix)
    latlon = np.load(_resolve_latlon_file(data_dir, suffix))
    lat = latlon["lat"]
    lon = latlon["lon"]

    n_samples = infer_n_samples_from_memmap(dataset_path, cfg.num_variables, len(lat), len(lon))

    # Keep this CNN run aligned with the UNet comparison protocol.
    required_leads = [12, 18, 24, 30, 36]
    lead_times = list(range(cfg.t_min, cfg.t_max + 1, cfg.delta_t))
    if lead_times != required_leads:
        raise ValueError(
            "Lead times must be [12, 18, 24, 30, 36] for comparison runs. "
            f"Got {lead_times}. Set --t_min 12 --t_max 36 --delta_t 6."
        )

    ti = shared_make_time_index(cfg.start_datetime, cfg.end_datetime, cfg.time_freq)
    expected_samples, n_train, n_val = shared_split_time_index(
        ti,
        split_mode=cfg.split_mode,
        train_year_end=int(cfg.train_year_end) if cfg.train_year_end is not None else None,
        val_year_end=int(cfg.val_year_end) if cfg.val_year_end is not None else None,
        train_until=cfg.train_until,
        val_until=cfg.val_until,
    )
    if expected_samples != n_samples:
        raise ValueError(
            f"Dataset sample count ({n_samples}) does not match time-index sample count ({expected_samples}) "
            f"for start={cfg.start_datetime}, end={cfg.end_datetime}, freq={cfg.time_freq}."
        )
    n_test = n_samples - n_train - n_val

    if n_train <= 0 or n_val <= 0:
        raise ValueError(
            f"Invalid split sizes: n_samples={n_samples}, n_train={n_train}, n_val={n_val}."
        )

    mean, std, residual_stds = load_norm_and_residuals(cfg, data_dir, device)

    train_dataset = LocalERA5Dataset(
        dataset_path=dataset_path,
        dataset_mode="train",
        n_samples=n_samples,
        n_train=n_train,
        n_val=n_val,
        num_variables=cfg.num_variables,
        n_lat=len(lat),
        n_lon=len(lon),
        lead_times=lead_times,
        max_horizon=cfg.max_horizon,
        conditioning_times=cfg.conditioning_times,
        norm_mean=mean,
        norm_std=std,
        spacing=cfg.spacing,
        spinup=24,
    )
    val_dataset = LocalERA5Dataset(
        dataset_path=dataset_path,
        dataset_mode="val",
        n_samples=n_samples,
        n_train=n_train,
        n_val=n_val,
        num_variables=cfg.num_variables,
        n_lat=len(lat),
        n_lon=len(lon),
        lead_times=lead_times,
        max_horizon=cfg.max_horizon,
        conditioning_times=cfg.conditioning_times,
        norm_mean=mean,
        norm_std=std,
        spacing=cfg.spacing,
        spinup=24,
    )

    nw = min(4, max(1, os.cpu_count() // 2))
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=(len(train_dataset) >= cfg.batch_size),
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
    )

    input_channels = len(cfg.conditioning_times) * cfg.num_variables
    model = DeterministicCNN(in_channels=input_channels, out_channels=cfg.num_variables, filters=cfg.filters).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    loss_fn = WMSELoss(lat, lon, device, precomputed_std=residual_stds)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    run_name = ensure_cnn_prefix(cfg.name)
    if run_name != cfg.name:
        print(f"Adjusted run name to '{run_name}' to keep CNN outputs separate.", flush=True)
    result_path = Path(cfg.result_directory) / run_name
    result_path.mkdir(parents=True, exist_ok=True)
    (result_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Persist the effective run name used on disk for reproducibility.
    cfg_to_save = asdict(cfg)
    cfg_to_save["name"] = run_name
    with open(result_path / "config.json", "w") as f:
        json.dump(cfg_to_save, f, indent=2, sort_keys=True)

    training_log_path = result_path / "training_log.csv"
    gradient_log_path = result_path / "gradient_log.csv"

    with open(training_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Average Training Loss", "Validation Loss", "Alpha Value"])

    with open(gradient_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "Batch", "Max Gradient (L∞)", "Avg Inf Norm", "Min Gradient",
            "Grad Mean", "Grad Median", "Grad P95", "Grad P99", "Total Params"
        ])

    print(f"Using deterministic CNN model | lead_times={lead_times}", flush=True)
    print(f"Num params: {sum(p.numel() for p in model.parameters())}", flush=True)
    print(
        f"DataLoader: train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
        f"batch_size={cfg.batch_size}, num_workers={nw}",
        flush=True,
    )
    print(
        f"Split summary: train={n_train}, val={n_val}, test={n_test} | "
        f"window={cfg.start_datetime}..{cfg.end_datetime} freq={cfg.time_freq}",
        flush=True,
    )
    print(f"Land-only mode: {cfg.land_only}", flush=True)

    best_val_loss = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (x, y, lead) in enumerate(tqdm(train_loader)):
            x = x.to(device)
            y = y.to(device)
            lead = lead.to(device)
            if land_mask is not None:
                x = x * land_mask
                y = y * land_mask

            optimizer.zero_grad()
            loss = loss_fn(model, y, x, lead.float() / cfg.max_horizon)
            loss.backward()

            grad_stats = calculate_gradient_stats(model.module if hasattr(model, "module") else model)
            with open(gradient_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    batch_idx,
                    f"{grad_stats['max_grad']:.6f}",
                    f"{grad_stats['avg_inf_norm']:.6f}",
                    f"{grad_stats['min_grad']:.6f}",
                    f"{grad_stats['grad_mean']:.6f}",
                    f"{grad_stats['grad_median']:.6f}",
                    f"{grad_stats['grad_p95']:.6f}",
                    f"{grad_stats['grad_p99']:.6f}",
                    grad_stats["total_params"],
                ])

            optimizer.step()
            total_train_loss += float(loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y, lead in val_loader:
                x = x.to(device)
                y = y.to(device)
                lead = lead.to(device)
                if land_mask is not None:
                    x = x * land_mask
                    y = y * land_mask
                loss = loss_fn(model, y, x, lead.float() / cfg.max_horizon)
                total_val_loss += float(loss.item())

        avg_val_loss = total_val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), result_path / "best_model.pth")

        scheduler.step()

        with open(training_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, 0])

        print(
            f"Epoch [{epoch + 1}/{cfg.epochs}] train_loss={avg_train_loss:.6f} val_loss={avg_val_loss:.6f}",
            flush=True,
        )
        if torch.cuda.is_available():
            print(
                f"cuda.max_memory_allocated.GiB={torch.cuda.max_memory_allocated() / 1024**3:.3f}",
                flush=True,
            )

        if cfg.save_every and (epoch % cfg.save_every == 0):
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "model": (model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": asdict(cfg),
                },
                result_path / "checkpoints" / f"epoch_{epoch}.pth",
            )


def main() -> None:
    cfg = parse_args()
    train_loop(cfg)


if __name__ == "__main__":
    main()

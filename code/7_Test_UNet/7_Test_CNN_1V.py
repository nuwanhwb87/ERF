from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import gc
import zarr
import sys
import os
import numpy as np
import xarray as xr
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.utils import ERA5Dataset, pretty_print_kwargs, make_time_index
from loss.loss import calculate_AreaWeightedRMSE


def parse_time_freq_hours(time_freq: str) -> int:
    tf = time_freq.strip().lower()
    if not tf.endswith("h"):
        raise ValueError(f"Only hourly time_freq is supported, got: {time_freq}")
    step = int(tf[:-1])
    if step <= 0:
        raise ValueError(f"time_freq must be positive, got: {time_freq}")
    return step


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


class DeterministicCNN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, filters: int = 64) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels + 1, filters, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(num_groups=min(8, filters), num_channels=filters),
            torch.nn.GELU(),
            torch.nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(num_groups=min(8, filters), num_channels=filters),
            torch.nn.GELU(),
            torch.nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(num_groups=min(8, filters), num_channels=filters),
            torch.nn.GELU(),
            torch.nn.Conv2d(filters, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, time_labels_norm: torch.Tensor) -> torch.Tensor:
        t_map = time_labels_norm.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, t_map], dim=1))


@dataclass
class TestConfig:
    name: str
    model: str
    eval_name: str | None
    batch_size: int
    spacing: int
    t_min: int
    t_max: int
    t_iter: int
    t_direct: int
    n_ens: int
    event_thresholds: list[float]
    land_only: bool
    static_nc_path: str
    land_threshold: float
    test_start: str
    test_end: str
    data_directory: str
    model_directory: str
    result_directory: str


@dataclass
class TrainedCNNConfig:
    name: str
    model: str
    filters: int
    num_variables: int
    variable_names: list[str]
    conditioning_times: list[int]
    max_horizon: int
    t_max: int
    delta_t: int
    width: int
    data_directory: str
    start_datetime: str
    end_datetime: str
    train_until: str
    val_until: str
    time_freq: str
    problem_name: str = 'rainfall'
    num_static_fields: int = 0

    @staticmethod
    def from_json(path: Path) -> "TrainedCNNConfig":
        with path.open("r") as f:
            d = json.load(f)
        # Keep compatibility with historical configs that may omit some fields.
        d.setdefault("num_static_fields", 0)
        d.setdefault("problem_name", "rainfall")
        d.setdefault("end_datetime", d.get("test_until", d.get("val_until", d.get("start_datetime"))))
        return TrainedCNNConfig(**{k: d[k] for k in TrainedCNNConfig.__dataclass_fields__.keys()})


def ensure_cnn_prefix(name: str) -> str:
    return name if name.startswith("CNN") else f"CNN_{name}"


def resolve_model_run_name(model_directory: str, requested_name: str) -> str:
    """Resolve a model folder name without forcing a rename on disk.

    Preference order:
      1) requested_name (as passed)
      2) CNN-prefixed variant
      3) de-prefixed variant (strip leading 'CNN_')
    """
    root = Path(model_directory)
    candidates = [requested_name]
    prefixed = ensure_cnn_prefix(requested_name)
    if prefixed not in candidates:
        candidates.append(prefixed)
    if requested_name.startswith("CNN_"):
        deprefixed = requested_name.replace("CNN_", "", 1)
        if deprefixed not in candidates:
            candidates.append(deprefixed)

    for name in candidates:
        if (root / name / "config.json").exists():
            return name

    raise FileNotFoundError(
        "Could not find model config in any expected folder: "
        + ", ".join(str(root / c / "config.json") for c in candidates)
    )


def _parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _resolve_dataset_file(data_directory: str, variable_names: list[str], suffix: str) -> Path:
    prefix = "_".join(variable_names)
    candidates = [
        Path(data_directory) / f"{prefix}_{suffix}.npy",
        Path(data_directory) / f"{prefix}_{suffix}_5.625deg.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(Path(data_directory).glob(f"{prefix}_{suffix}*.npy"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No dataset file found for prefix '{prefix}_{suffix}' in {data_directory}")
    raise FileNotFoundError(f"Multiple dataset files found for prefix '{prefix}_{suffix}': {matches}")


def _resolve_latlon_file(data_directory: str, suffix: str) -> Path:
    candidates = [
        Path(data_directory) / f"latlon_{suffix}.npz",
        Path(data_directory) / f"latlon_{suffix}_5.625deg.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No lat/lon file found for suffix '{suffix}' in {data_directory}")


def _assert_test_window(cfg: TestConfig, ti, n_train: int, n_val: int) -> None:
    test_start_idx = n_train + n_val
    if test_start_idx >= len(ti):
        raise ValueError("No test samples available: train+val spans all timesteps.")

    actual_start = ti[test_start_idx]
    actual_end = ti[-1]
    expected_start = np.datetime64(cfg.test_start)
    expected_end = np.datetime64(cfg.test_end)

    if actual_start != expected_start or actual_end != expected_end:
        raise ValueError(
            "Test window mismatch. "
            f"Expected {expected_start}..{expected_end}, "
            f"but split yields {actual_start}..{actual_end}."
        )
    print(f"Confirmed test window: {actual_start}..{actual_end}", flush=True)


def _load_land_mask(lat, lon, static_nc_path: str, threshold: float, device):
    static_path = Path(static_nc_path)
    if not static_path.exists():
        raise FileNotFoundError(f"Static file not found: {static_path}")

    with xr.open_dataset(static_path) as ds:
        if "lsm" in ds:
            lsm = ds["lsm"].values
        elif "land_sea_mask" in ds:
            lsm = ds["land_sea_mask"].values
        else:
            raise ValueError("No land-sea mask found in static file (expected 'lsm' or 'land_sea_mask').")
        lat_static = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon_static = np.asarray(ds["longitude"].values, dtype=np.float64)

    if lsm.ndim == 3:
        lsm = lsm[0]
    elif lsm.ndim == 4:
        lsm = lsm[0, 0]

    lsm = np.asarray(lsm, dtype=np.float64)
    lat_ref = np.asarray(lat, dtype=np.float64)
    lon_ref = np.asarray(lon, dtype=np.float64)
    if lsm.shape != (len(lat_ref), len(lon_ref)):
        raise ValueError(f"Land mask shape {lsm.shape} does not match rainfall grid {(len(lat_ref), len(lon_ref))}")

    if np.allclose(lat_static[::-1], lat_ref, atol=1e-6, rtol=0.0):
        lsm = lsm[::-1, :]
    elif not np.allclose(lat_static, lat_ref, atol=1e-6, rtol=0.0):
        raise ValueError("Latitude coordinates in static mask do not align with rainfall grid.")

    if np.allclose(lon_static[::-1], lon_ref, atol=1e-6, rtol=0.0):
        lsm = lsm[:, ::-1]
    elif not np.allclose(lon_static, lon_ref, atol=1e-6, rtol=0.0):
        raise ValueError("Longitude coordinates in static mask do not align with rainfall grid.")

    land_mask_np = (lsm >= float(threshold)).astype(np.float32)
    land_mask_t = torch.tensor(land_mask_np, device=device, dtype=torch.float32)
    return land_mask_t, land_mask_np

def renormalize(x, cfg_train, device):
    with open(f'{cfg_train.data_directory}/norm_factors.json', 'r') as f:
        statistics = json.load(f)
    mean_data = torch.tensor([statistics[name]["mean"] for name in cfg_train.variable_names])
    std_data = torch.tensor([statistics[name]["std"] for name in cfg_train.variable_names])
    mean_data = mean_data.to(device)
    std_data = std_data.to(device)
    x = x * std_data[None, :, None, None] + mean_data[None, :, None, None]
    return x


def _safe_divide(numerator, denominator):
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def init_metric_accumulators(n_times, n_vars, thresholds):
    shape = (n_times, n_vars)
    event_shape = (len(thresholds), n_times, n_vars)
    return {
        'sum_abs': np.zeros(shape, dtype=np.float64),
        'sum_sq': np.zeros(shape, dtype=np.float64),
        'sum_diff': np.zeros(shape, dtype=np.float64),
        'sum_w': np.zeros(shape, dtype=np.float64),
        'sum_wx': np.zeros(shape, dtype=np.float64),
        'sum_wy': np.zeros(shape, dtype=np.float64),
        'sum_wxx': np.zeros(shape, dtype=np.float64),
        'sum_wyy': np.zeros(shape, dtype=np.float64),
        'sum_wxy': np.zeros(shape, dtype=np.float64),
        'tp': np.zeros(event_shape, dtype=np.float64),
        'fp': np.zeros(event_shape, dtype=np.float64),
        'fn': np.zeros(event_shape, dtype=np.float64),
        'tn': np.zeros(event_shape, dtype=np.float64),
    }


def update_metric_accumulators(acc, forecast, truth, area_weights, thresholds):
    # forecast: (B, E, T, V, H, W), truth: (B, T, V, H, W)
    forecast_mean = forecast.mean(dim=1)
    diff = forecast_mean - truth

    weighted = area_weights[None, None, None, :, :]
    sum_dims = (0, -1, -2)

    acc['sum_abs'] += (weighted * diff.abs()).sum(dim=sum_dims).cpu().numpy()
    acc['sum_sq'] += (weighted * diff.square()).sum(dim=sum_dims).cpu().numpy()
    acc['sum_diff'] += (weighted * diff).sum(dim=sum_dims).cpu().numpy()

    sum_w_batch = weighted.sum(dim=(-1, -2)).cpu().numpy()[0, 0, 0] * forecast_mean.shape[0]
    acc['sum_w'] += np.full(acc['sum_w'].shape, sum_w_batch, dtype=np.float64)
    acc['sum_wx'] += (weighted * forecast_mean).sum(dim=sum_dims).cpu().numpy()
    acc['sum_wy'] += (weighted * truth).sum(dim=sum_dims).cpu().numpy()
    acc['sum_wxx'] += (weighted * forecast_mean.square()).sum(dim=sum_dims).cpu().numpy()
    acc['sum_wyy'] += (weighted * truth.square()).sum(dim=sum_dims).cpu().numpy()
    acc['sum_wxy'] += (weighted * forecast_mean * truth).sum(dim=sum_dims).cpu().numpy()

    for threshold_idx, threshold in enumerate(thresholds):
        event_forecast = forecast_mean >= threshold
        event_truth = truth >= threshold

        acc['tp'][threshold_idx] += (weighted * (event_forecast & event_truth)).sum(dim=sum_dims).cpu().numpy()
        acc['fp'][threshold_idx] += (weighted * (event_forecast & ~event_truth)).sum(dim=sum_dims).cpu().numpy()
        acc['fn'][threshold_idx] += (weighted * (~event_forecast & event_truth)).sum(dim=sum_dims).cpu().numpy()
        acc['tn'][threshold_idx] += (weighted * (~event_forecast & ~event_truth)).sum(dim=sum_dims).cpu().numpy()


def finalize_metric_accumulators(acc):
    mae = _safe_divide(acc['sum_abs'], acc['sum_w'])
    mse = _safe_divide(acc['sum_sq'], acc['sum_w'])
    bias = _safe_divide(acc['sum_diff'], acc['sum_w'])

    cov = acc['sum_wxy'] - _safe_divide(acc['sum_wx'] * acc['sum_wy'], acc['sum_w'])
    var_x = acc['sum_wxx'] - _safe_divide(acc['sum_wx'] ** 2, acc['sum_w'])
    var_y = acc['sum_wyy'] - _safe_divide(acc['sum_wy'] ** 2, acc['sum_w'])
    corr = _safe_divide(cov, np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0)))

    precision = _safe_divide(acc['tp'], acc['tp'] + acc['fp'])
    recall = _safe_divide(acc['tp'], acc['tp'] + acc['fn'])
    pod = recall.copy()
    specificity = _safe_divide(acc['tn'], acc['tn'] + acc['fp'])
    accuracy = _safe_divide(acc['tp'] + acc['tn'], acc['tp'] + acc['tn'] + acc['fp'] + acc['fn'])
    balanced_accuracy = 0.5 * (recall + specificity)
    far = _safe_divide(acc['fp'], acc['tp'] + acc['fp'])
    csi = _safe_divide(acc['tp'], acc['tp'] + acc['fp'] + acc['fn'])
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    return {
        'mae': mae,
        'mse': mse,
        'bias': bias,
        'correlation': corr,
        'precision': precision,
        'recall': recall,
        'POD': pod,
        'specificity': specificity,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'FAR': far,
        'CSI': csi,
        'F1': f1,
    }


def init_roc_accumulators(n_times, n_vars, n_event_thresholds, n_score_thresholds):
    shape = (n_event_thresholds, n_score_thresholds, n_times, n_vars)
    return {
        'tp': np.zeros(shape, dtype=np.float64),
        'fp': np.zeros(shape, dtype=np.float64),
        'fn': np.zeros(shape, dtype=np.float64),
        'tn': np.zeros(shape, dtype=np.float64),
    }


def update_roc_accumulators(roc_acc, forecast, truth, area_weights, event_thresholds, score_thresholds):
    # forecast: (B, E, T, V, H, W), truth: (B, T, V, H, W)
    forecast_mean = forecast.mean(dim=1)
    weighted = area_weights[None, None, None, :, :]
    sum_dims = (1, -1, -2)

    score_thresholds_t = torch.as_tensor(score_thresholds, device=forecast.device, dtype=forecast_mean.dtype)
    pred_events = forecast_mean.unsqueeze(0) >= score_thresholds_t[:, None, None, None, None, None]

    for event_idx, event_threshold in enumerate(event_thresholds):
        truth_event = truth >= event_threshold

        tp = (weighted.unsqueeze(0) * (pred_events & truth_event.unsqueeze(0))).sum(dim=sum_dims)
        fp = (weighted.unsqueeze(0) * (pred_events & ~truth_event.unsqueeze(0))).sum(dim=sum_dims)
        fn = (weighted.unsqueeze(0) * (~pred_events & truth_event.unsqueeze(0))).sum(dim=sum_dims)
        tn = (weighted.unsqueeze(0) * (~pred_events & ~truth_event.unsqueeze(0))).sum(dim=sum_dims)

        roc_acc['tp'][event_idx] += tp.cpu().numpy()
        roc_acc['fp'][event_idx] += fp.cpu().numpy()
        roc_acc['fn'][event_idx] += fn.cpu().numpy()
        roc_acc['tn'][event_idx] += tn.cpu().numpy()


def finalize_roc_accumulators(roc_acc):
    tpr = _safe_divide(roc_acc['tp'], roc_acc['tp'] + roc_acc['fn'])
    fpr = _safe_divide(roc_acc['fp'], roc_acc['fp'] + roc_acc['tn'])

    # Reverse threshold axis so the curve is traversed from strict to permissive threshold.
    tpr_curve = tpr[:, ::-1, :, :]
    fpr_curve = fpr[:, ::-1, :, :]

    auc = np.full((tpr.shape[0], tpr.shape[2], tpr.shape[3]), np.nan, dtype=np.float64)
    for event_idx in range(tpr.shape[0]):
        for t_idx in range(tpr.shape[2]):
            for v_idx in range(tpr.shape[3]):
                x = fpr_curve[event_idx, :, t_idx, v_idx]
                y = tpr_curve[event_idx, :, t_idx, v_idx]
                valid = np.isfinite(x) & np.isfinite(y)
                if np.count_nonzero(valid) < 2:
                    continue
                auc[event_idx, t_idx, v_idx] = np.trapz(y[valid], x[valid])

    return {
        'roc_tpr': tpr,
        'roc_fpr': fpr,
        'roc_auc': auc,
    }


def summarize_sample_metric(samples: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.nanmean(samples, axis=0)
    std = np.nanstd(samples, axis=0, ddof=1)
    n = np.sum(~np.isnan(samples), axis=0)
    se = np.divide(std, np.sqrt(np.maximum(n, 1)), out=np.zeros_like(std), where=n > 0)
    delta = 1.96 * se
    return {
        'mean': mean,
        'ci95_lower': mean - delta,
        'ci95_upper': mean + delta,
    }

def build_dataloaders(cfg, cfg_train, device):
    suffix = f"{cfg_train.start_datetime[:4]}-{cfg_train.end_datetime[:4]}"
    latlon = np.load(_resolve_latlon_file(cfg.data_directory, suffix))
    lat = latlon["lat"]
    lon = latlon["lon"]

    # Get sample counts directly from the memmap file to avoid datetime/file-size mismatch.
    dataset_path = str(_resolve_dataset_file(cfg.data_directory, cfg_train.variable_names, suffix))
    values_per_sample = cfg_train.num_variables * len(lat) * len(lon)
    n_samples = os.path.getsize(dataset_path) // (np.dtype(np.float32).itemsize * values_per_sample)
    if n_samples <= 0:
        raise ValueError(f"No samples found in dataset file: {dataset_path}")

    # Keep split boundaries identical to train_CNN.py for fair CNN-vs-UNet comparison.
    n_train, n_val = split_counts_from_dates(
        n_samples=n_samples,
        start_datetime=cfg_train.start_datetime,
        time_freq=cfg_train.time_freq,
        train_until=cfg_train.train_until,
        val_until=cfg_train.val_until,
    )
    ti = make_time_index(cfg_train.start_datetime, cfg_train.end_datetime, cfg_train.time_freq)
    if len(ti) != n_samples:
        raise ValueError(
            f"Dataset sample count ({n_samples}) does not match time-index sample count ({len(ti)}) "
            f"for start={cfg_train.start_datetime}, end={cfg_train.end_datetime}, freq={cfg_train.time_freq}."
        )
    _assert_test_window(cfg, ti, n_train, n_val)
    # ---------------- Load normalization factors (ordered) -------------------
    with open(f'{cfg.data_directory}/norm_factors.json', 'r') as f:
        statistics = json.load(f)
    mean_data = torch.tensor([statistics[name]["mean"] for name in cfg_train.variable_names])
    std_data = torch.tensor([statistics[name]["std"] for name in cfg_train.variable_names])
    norm_factors = np.stack([mean_data, std_data], axis=0)

    if cfg.t_iter > cfg_train.t_max:
        print(f"The iterative lead time {cfg.t_iter} is larger than the maximum trained lead time {cfg_train.t_max}")
    if cfg.t_direct < cfg_train.delta_t:
        print(f"The direct lead time {cfg.t_direct} is smaller than the trained dt {cfg_train.delta_t}")

    # Use the fixed 36-hour evaluation horizon for the rainfall test window.
    eval_max_horizon = 36
    print(f"Using evaluation max_horizon={eval_max_horizon}", flush=True)

    static_data_path = None
    kwargs = {
                'dataset_path':     dataset_path,
                'sample_counts':    (n_samples, n_train, n_val),
                'dimensions':       (cfg_train.num_variables, len(lat), len(lon)),
                'max_horizon':      eval_max_horizon,
                'norm_factors':     norm_factors,
                'device':           device,
                'spacing':          cfg.spacing,
                'dtype':            'float32',
                'conditioning_times':    cfg_train.conditioning_times,
                'lead_time_range':  [cfg.t_min, cfg.t_max, cfg.t_direct],
                'static_data_path': static_data_path,
                'random_lead_time': 0,
                }
    pretty_print_kwargs(kwargs)

    # Define the batch samplers
    forecasting_times = cfg.t_min + cfg.t_direct * np.arange(0, 1 + (cfg.t_max - cfg.t_min) // cfg.t_direct)
    dataset = ERA5Dataset(lead_time=forecasting_times, dataset_mode='test', **kwargs)

    test_time_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    print(f"Datset contains {len(dataset)} samples",  flush=True)
    print(f"We do {len(test_time_loader)} batches",  flush=True)

    return test_time_loader, dataset, forecasting_times, lat, lon

def _load_state_dict_flexible(model, ckpt_path: Path, device):
    state = torch.load(str(ckpt_path), map_location=device)
    # Handle DataParallel save formats gracefully.
    try:
        model.load_state_dict(state)
        return
    except RuntimeError:
        pass

    if any(k.startswith("module.") for k in state.keys()):
        stripped = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(stripped)
    else:
        prefixed = {f"module.{k}": v for k, v in state.items()}
        model.load_state_dict(prefixed)


def build_model(cfg, cfg_train, device):
    if cfg.n_ens > 1:
        raise ValueError("CNN deterministic model requires n_ens=1.")
    if cfg.model != 'deterministic':
        raise ValueError("test_CNN.py currently supports only --model deterministic.")

    input_times = len(cfg_train.conditioning_times) * cfg_train.num_variables
    model = DeterministicCNN(
        in_channels=input_times,
        out_channels=cfg_train.num_variables,
        filters=cfg_train.filters,
    )

    ckpt_path = Path(cfg.model_directory) / cfg.name / 'best_model.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    _load_state_dict_flexible(model, ckpt_path, device)
    model.to(device)

    print("Using deterministic CNN model", flush=True)
    print(f"Loaded model {cfg.name}, {cfg_train.model}", flush=True)
    print("Num params: ", sum(p.numel() for p in model.parameters()), flush=True)
    return model

def test_loop(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(cfg.name, flush=True)
    print("[t_direct, t_iter, t_max]", [cfg.t_direct, cfg.t_iter, cfg.t_max],  flush=True)
    print("n_ens:", cfg.n_ens,  flush=True)

    # Resolve the actual on-disk model folder (supports both prefixed and legacy names).
    resolved_model_name = resolve_model_run_name(cfg.model_directory, cfg.name)
    if resolved_model_name != cfg.name:
        print(f"Resolved model directory to '{resolved_model_name}' from requested '{cfg.name}'.", flush=True)
    cfg.name = resolved_model_name

    # Save each evaluation in a dedicated CNN folder so diffusion outputs are never overwritten.
    default_eval = f"{cfg.name}_test_{cfg.t_min}to{cfg.t_max}_dt{cfg.t_direct}_iter{cfg.t_iter}_ens{cfg.n_ens}"
    run_name = ensure_cnn_prefix(cfg.eval_name if cfg.eval_name else default_eval)
    result_path = Path(f'{cfg.result_directory}/{run_name}')
    result_path.mkdir(parents=True, exist_ok=True)
    config_path = result_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)
    print(f"Saved config to {config_path}", flush=True)
    # ---------------- Load training config data ------------------------------------
    cfg_train = TrainedCNNConfig.from_json(Path(cfg.model_directory) / cfg.name / "config.json")
    if cfg_train.variable_names != ["total_precipitation"] or cfg_train.num_variables != 1:
        raise ValueError(
            "This one-variable CNN test script requires a total_precipitation checkpoint "
            "with num_variables=1."
        )
    # ---------------- Build data loaders ------------------------------------
    test_time_loader, dataset, forecasting_times, lat, lon = build_dataloaders(cfg, cfg_train,  device)
    land_mask_t = None
    land_mask_np = None
    if cfg.land_only:
        land_mask_t, land_mask_np = _load_land_mask(lat, lon, cfg.static_nc_path, cfg.land_threshold, device)
        print(f"Land-only evaluation enabled with threshold {cfg.land_threshold}.", flush=True)
    # ---------------- Build model ------------------------------------
    model = build_model(cfg, cfg_train, device)
    # ---------------- Testing loop -----------------------------------------
    model.eval()
    # Initialize the dimensions based on the first batch
    previous, current, time_labels = next(iter(test_time_loader))
    n_times = time_labels.shape[1]
    n_conditions = previous.shape[1]
    dx = current.shape[2]
    dy = current.shape[3]
    # prepare output zarr
    predictions = zarr.open_array(f'{result_path}/{cfg.name}.zarr', mode='w', shape=(len(dataset), cfg.n_ens, n_times, cfg_train.num_variables, dx, dy),
                                    chunks = (1, cfg.n_ens, n_times, cfg_train.num_variables, dx, dy),
                                    dtype='float32')

    write_idx = 0  # Track index for where to write in output zarr
    # Combined implementation of Algorithm 1, 2 and 3 in the paper
    for previous, current, time_labels in tqdm(test_time_loader):
        n_samples = current.shape[0]

        with torch.no_grad():
            previous = previous.to(device) # (B, Ncond*vars+2, H, W)
            current = current.view(-1, cfg_train.num_variables, dx, dy).to(device)

            direct_time_labels = torch.tensor(np.array([x for x in time_labels[0] if x <= cfg.t_iter]), device=device)
            n_iter = time_labels.shape[1] // direct_time_labels.shape[0] # how many steps towards the full horizon
            n_direct = direct_time_labels.shape[0] #how many steps towards direct prediction

            class_labels = previous.repeat_interleave(n_direct * cfg.n_ens, dim=0) # Can not be changed if batchsz > 1

            direct_time_labels_repeated = direct_time_labels.repeat(cfg.n_ens * n_samples) # Can not be changed if n_direct > 1

            # Test
            predicted_all = torch.zeros((n_samples, cfg.n_ens, n_times, cfg_train.num_variables, dx, dy), device=device)

            for i in tqdm(range(n_iter)):
                predicted = model(class_labels, direct_time_labels_repeated / cfg_train.max_horizon)

                predicted_all[:, :, i*n_direct:(i+1)*n_direct] = predicted.view(n_samples, cfg.n_ens, n_direct, cfg_train.num_variables, dx, dy)

                predicted = predicted.view(n_samples*cfg.n_ens, n_direct, cfg_train.num_variables, dx, dy)
                class_labels = class_labels.view(n_samples*cfg.n_ens, n_direct, n_conditions, dx, dy)[:, 0]

                # Build class_labels for next iteration with all conditioning_times
                # For deterministic models, we need all conditioning_times variables
                conditioning_vars_list = []
                
                # The first conditioning time (usually 0) is the latest prediction
                initial_condition = predicted[:,-1]  # t=0 (latest)
                conditioning_vars_list.append(initial_condition)
                
                # For other conditioning times, extract from predicted or previous class_labels
                for cond_idx, cond_time in enumerate(cfg_train.conditioning_times[1:], start=1):
                    # Find where this conditioning time appears in direct_time_labels
                    idx = np.argwhere(direct_time_labels.cpu().numpy() == -cond_time)
                    if idx.size == 0:
                        # This conditioning time is not in current direct_time_labels
                        # It must be from a previous iteration, so get it from class_labels
                        # class_labels structure: [cond_time[0] vars, cond_time[1] vars, ..., cond_time[n] vars, static_fields]
                        earlier_condition = class_labels[:, cond_idx * cfg_train.num_variables:(cond_idx + 1) * cfg_train.num_variables]
                    else:
                        idx = idx[0][0] + 1
                        # Extract the variable at this time step
                        if idx == len(direct_time_labels):
                            # This time step is in the previous class_labels (before current iteration)
                            earlier_condition = class_labels[:, cond_idx * cfg_train.num_variables:(cond_idx + 1) * cfg_train.num_variables]
                        else:
                            # This time step is in the current predicted output
                            earlier_condition = predicted[:, -(idx+1)]
                    
                    conditioning_vars_list.append(earlier_condition)
                
                # Concatenate all conditioning variables
                class_labels = torch.cat(conditioning_vars_list, dim=1).repeat_interleave(n_direct, dim=0)

            # Save predictions incrementally to zarr file
            pred_np = renormalize(predicted_all, cfg_train, device).view(n_samples, cfg.n_ens, n_times, cfg_train.num_variables, dx, dy).cpu().numpy()
            if land_mask_np is not None:
                pred_np = pred_np * land_mask_np[None, None, None, None, :, :]
            predictions[write_idx:write_idx + n_samples, :, :, :, :, :] = pred_np

            write_idx += n_samples

        gc.collect()
        torch.cuda.empty_cache()

    metrics_cal(cfg, cfg_train, test_time_loader, lat, lon, predictions, forecasting_times, result_path, device, land_mask_t=land_mask_t)

def metrics_cal(cfg, cfg_train, test_time_loader, lat, lon, predictions, forecasting_times, result_path, device, land_mask_t=None):
    # Calculate metrics
    metrics = zarr.open_group(result_path / 'evaluation_metrics.zarr', mode='a')

    calc = calculate_AreaWeightedRMSE(lat, lon, device)
    calculate_WCRPS = calc.CRPS
    calculate_WScores = calc.skill_and_spread
    calculate_WMAE = calc.mae
    calculate_rmse_per_sample = calc.rmse_per_sample
    calculate_CRPS_per_sample = calc.CRPS_per_sample

    skill_list = []
    spread_list = []
    ssr_list = []
    CRPS_list = []
    dx_same_list = []
    dx_different_list = []
    dx_truth_list = []
    rmse_per_sample_list = []
    CRPS_per_sample_list = []

    i = 0
    with torch.no_grad():
        for previous, current, time_labels in tqdm(test_time_loader):
            n_times = time_labels.shape[1]
            n_samples, _, dx, dy = current.shape

            truth = renormalize(current.to(device).view(n_samples, n_times, cfg_train.num_variables, dx, dy), cfg_train, device)
            if land_mask_t is not None:
                truth = truth * land_mask_t[None, None, None, :, :]

            forecast = predictions[i:i + truth.shape[0]]
            forecast = torch.tensor(forecast, device=device)
            if land_mask_t is not None:
                forecast = forecast * land_mask_t[None, None, None, None, :, :]
            i = i + truth.shape[0]

            # Calculate metrics
            skill, spread, ssr = calculate_WScores(forecast, truth)
            CRPS = calculate_WCRPS(forecast, truth)
            dx_same = calculate_WMAE(forecast[:, :, 1:, :], forecast[:, :, :-1, :])
            dx_different = calculate_WMAE(forecast[:, 1:, 1:, :], forecast[:, :-1, :-1, :])
            dx_truth = calculate_WMAE(truth[:, 1:, :].unsqueeze(1), truth[:, :-1, :].unsqueeze(1))

            # Per-sample metrics for statistical testing
            rmse_per_sample_list.append(calculate_rmse_per_sample(forecast, truth))
            CRPS_per_sample_list.append(calculate_CRPS_per_sample(forecast, truth))

            # Append to list
            skill_list.append(skill)
            spread_list.append(spread)
            ssr_list.append(ssr)
            CRPS_list.append(CRPS)
            dx_same_list.append(dx_same)
            dx_different_list.append(dx_different)
            dx_truth_list.append(dx_truth)


    skill = torch.tensor(np.array(skill_list)).mean(axis=0).cpu().numpy()
    spread = torch.tensor(np.array(spread_list)).mean(axis=0).cpu().numpy()
    ssr = torch.tensor(np.array(ssr_list)).mean(axis=0).cpu().numpy()
    CRPS = torch.tensor(np.array(CRPS_list)).mean(axis=0).cpu().numpy()
    dx_same = torch.tensor(np.array(dx_same_list)).mean(axis=0).cpu().numpy()
    dx_different = torch.tensor(np.array(dx_different_list)).mean(axis=0).cpu().numpy()
    dx_truth = torch.tensor(np.array(dx_truth_list)).mean(axis=0).cpu().numpy()

    # Check if group for eval_name exists, else create it
    if cfg_train.name not in metrics:
        metrics.create_group(cfg_train.name)

    def _write_dataset(g, name, data):
        if name in g:
            del g[name]
        g.create_dataset(name=name, data=data)

    # Store the metrics in the corresponding group
    group = metrics[cfg_train.name]

    # Keep CNN outputs identical to UNet for strict apple-to-apple comparison.
    allowed_keys = {
        'skill',
        'spread',
        'SSR',
        'CRPS',
        'dx_same',
        'dx_different',
        'dx_truth',
        'times',
        'rmse_per_sample',
        'CRPS_per_sample',
    }
    for key in list(group.keys()):
        if key not in allowed_keys:
            del group[key]

    _write_dataset(group, 'skill', skill)
    _write_dataset(group, 'spread', spread)
    _write_dataset(group, 'SSR', ssr)
    _write_dataset(group, 'CRPS', CRPS)
    _write_dataset(group, 'dx_same', dx_same)
    _write_dataset(group, 'dx_different', dx_different)
    _write_dataset(group, 'dx_truth', dx_truth)
    _write_dataset(group, 'times', forecasting_times)

    # Per-sample RMSE/CRPS for statistical testing: shape (n_samples, n_times, n_vars)
    rmse_per_sample = np.concatenate(rmse_per_sample_list, axis=0)
    CRPS_per_sample = np.concatenate(CRPS_per_sample_list, axis=0)
    _write_dataset(group, 'rmse_per_sample', rmse_per_sample)
    _write_dataset(group, 'CRPS_per_sample', CRPS_per_sample)

    print(f"Metrics saved under {cfg_train.name}")

# --------------------------- CLI ---------------------------
def parse_args() -> TestConfig:
    ap = argparse.ArgumentParser(
        description="Test deterministic CNN spatiotemporal forecaster on NPZ shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Common usage:
    # --name CNN_rainfall_tp_model_cnn_12to36_dt6 --spacing 1 --batch_size 1 --n_ens 1
    ap.add_argument("--name", type=str, default='CNN_rainfall_tp_model_cnn_12to36_dt6', help="Trained CNN run name under model_directory (CNN prefix will be enforced).")
    ap.add_argument("--eval_name", type=str, default=None, help="Optional output folder name under result_directory. If omitted, a CNN-prefixed name is auto-generated.")
    ap.add_argument('--model', type=str, default='deterministic', help='CNN test supports only deterministic mode.')
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    ap.add_argument("--spacing", type=int, default=1, help="Use a larger number to subsample the data in time.")
    ap.add_argument("--t_min", type=int, default=12, help="Minimum lead time in hours.")
    ap.add_argument("--t_max", type=int, default=36, help="Maximum lead time in hours.")
    ap.add_argument("--t_iter", type=int, default=36, help="Iterative horizon in hours. Set to t_max to evaluate the full lead-time range.")
    ap.add_argument("--t_direct", type=int, default=6, help="Lead-time step in hours. With t_min=12 and t_max=36, this gives 12,18,24,30,36.")
    ap.add_argument("--n_ens", type=int, default=1, help="Number of ensemble members.")
    ap.add_argument("--event_thresholds", type=lambda s: [float(x.strip()) for x in s.split(',')], default="0.02", help="Comma-separated rainfall thresholds in native target units for event metrics (e.g. '0.02').")
    ap.add_argument("--land_only", type=str, default="True", help="If true, mask ocean pixels and evaluate rainfall over land only.")
    ap.add_argument("--static_nc_path", type=str, default="/nesi/project/massey04632/data/ERA5/static/era5_static.nc", help="Path to ERA5 static file containing land-sea mask.")
    ap.add_argument("--land_threshold", type=float, default=0.5, help="Land-sea mask threshold; >= threshold is treated as land.")
    ap.add_argument("--test_start", type=str, default="2025-01-01T00:00:00", help="Expected inclusive start time for the test split.")
    ap.add_argument("--test_end", type=str, default="2025-12-31T00:00:00", help="Expected inclusive end time for the test split.")


    # Optional: local overrides for input dirs (useful for local testing)
    ap.add_argument("--data_directory", type=str, default="/nesi/project/massey04632/data/ERA5/L2/2015_2025_1V", help="Path containing rainfall test arrays and normalization factors.")
    ap.add_argument("--model_directory", type=str, default="/nesi/project/massey04632/ERF/models", help="Path containing trained model directories.")
    ap.add_argument("--result_directory", type=str, default="/nesi/project/massey04632/ERF/results", help="Path where test outputs and metrics are written.")

    args = ap.parse_args()
    parsed = vars(args)
    parsed["land_only"] = _parse_bool(parsed["land_only"], default=True)
    return TestConfig(**parsed)

if __name__ == "__main__":
    cfg = parse_args()
    test_loop(cfg)
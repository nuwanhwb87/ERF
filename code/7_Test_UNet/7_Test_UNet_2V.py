from dataclasses import asdict
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import gc
import zarr
import sys
import numpy as np
from typing import Optional
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.utils import ERA5Dataset, pretty_print_kwargs, make_time_index, split_time_index
from loss.loss import calculate_AreaWeightedRMSE
from tools.sampler import heun_sampler
from configclass.test_dataclass import TestConfig
from configclass.train_dataclass import TrainConfig


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BasicUNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = torch.nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.up3 = torch.nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = torch.nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = torch.nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.head = torch.nn.Conv2d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _match_spatial(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        _, _, h, w = ref.shape
        sh, sw = src.shape[-2], src.shape[-1]

        if sh < h or sw < w:
            pad_h = max(h - sh, 0)
            pad_w = max(w - sw, 0)
            src = torch.nn.functional.pad(src, (0, pad_w, 0, pad_h))
        if src.shape[-2] > h or src.shape[-1] > w:
            src = src[:, :, :h, :w]
        return src

    def forward(self, x: torch.Tensor, _time_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self._match_spatial(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match_spatial(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match_spatial(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)

def load_config(json_file):
    with open(json_file, 'r') as file:
        config = json.load(file)
    return config

def renormalize(x, cfg_train, device):
    with open(f'{cfg_train.data_directory}/norm_factors.json', 'r') as f:
        statistics = json.load(f)
    mean_data = torch.tensor([statistics[name]["mean"] for name in cfg_train.variable_names])
    std_data = torch.tensor([statistics[name]["std"] for name in cfg_train.variable_names])
    mean_data = mean_data.to(device)
    std_data = std_data.to(device)
    x = x * std_data[None, :, None, None] + mean_data[None, :, None, None]
    return x


def parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
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


def _resolve_lsm_file(data_directory: str, lsm_path: Optional[str] = None) -> Optional[Path]:
    candidates = []
    if lsm_path:
        candidates.append(Path(lsm_path))
    data_dir = Path(data_directory)
    candidates.extend([
        data_dir / "lsm.npy",
        data_dir / "static" / "lsm.npy",
        data_dir.parent / "static" / "lsm.npy",
        Path("/nesi/project/massey04632/data/ERA5/static/lsm.npy"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _build_land_mask(data_directory: str, height: int, width: int, device: torch.device, lsm_path: Optional[str] = None) -> torch.Tensor:
    mask_path = _resolve_lsm_file(data_directory, lsm_path=lsm_path)
    if mask_path is None:
        raise FileNotFoundError(
            "Land-only testing requires lsm.npy, but no mask file was found. "
            "Provide --lsm_path or place lsm.npy under data/static."
        )

    lsm = np.load(mask_path).astype(np.float32)
    if lsm.ndim == 3 and lsm.shape[0] == 1:
        lsm = lsm[0]
    if lsm.ndim != 2:
        raise ValueError(f"Expected 2D lsm array, got shape {lsm.shape} from {mask_path}")
    if lsm.shape != (height, width):
        raise ValueError(f"LSM shape {lsm.shape} does not match grid {(height, width)}. File: {mask_path}")

    land_mask = (lsm > 0.5).astype(np.float32)
    print(f"Using land mask from {mask_path}; land fraction={land_mask.mean():.4f}", flush=True)
    return torch.tensor(land_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)


def _assert_test_window(cfg: TestConfig, ti: pd.DatetimeIndex, n_train: int, n_val: int) -> None:
    test_start_idx = n_train + n_val
    if test_start_idx >= len(ti):
        raise ValueError("No test samples available: train+val spans all timesteps.")

    actual_start = pd.Timestamp(ti[test_start_idx])
    actual_end = pd.Timestamp(ti[-1])
    expected_start = pd.Timestamp(cfg.test_start)
    expected_end = pd.Timestamp(cfg.test_end)

    if actual_start != expected_start or actual_end != expected_end:
        raise ValueError(
            "Test window mismatch. "
            f"Expected {expected_start}..{expected_end}, "
            f"but split yields {actual_start}..{actual_end}. "
            "Update train split boundaries or test_start/test_end."
        )
    print(f"Confirmed test window: {actual_start}..{actual_end}", flush=True)

def build_dataloaders(cfg, cfg_train, device):
    suffix = f"{cfg_train.start_datetime[:4]}-{cfg_train.end_datetime[:4]}"
    latlon = np.load(_resolve_latlon_file(cfg.data_directory, suffix))
    lat = latlon["lat"]
    lon = latlon["lon"]

    # Get the number of samples, training and validation samples
    ti = make_time_index(cfg_train.start_datetime, cfg_train.end_datetime, cfg_train.time_freq)
    n_samples, n_train, n_val = split_time_index(
        ti,
        split_mode=cfg_train.split_mode,
        train_year_end=cfg_train.train_year_end,
        val_year_end=cfg_train.val_year_end,
        train_until=cfg_train.train_until,
        val_until=cfg_train.val_until,
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

    dataset_path = str(_resolve_dataset_file(cfg.data_directory, cfg_train.variable_names, suffix))
    static_data_path = None
    if cfg_train.num_static_fields > 0:
        static_data_path = f'{cfg.data_directory}/static_fields.npy'
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

    return test_time_loader, dataset, forecasting_times,lat, lon, kwargs

def build_model (cfg, cfg_train, device):
    input_times = (1 + len(cfg_train.conditioning_times))*cfg_train.num_variables + cfg_train.num_static_fields

    deterministic = False
    if 'autoregressive' in cfg_train.model:
        raise ValueError("This testing script is configured for deterministic U-Net checkpoints only.")
    elif 'continuous' in cfg_train.model:
        raise ValueError("This testing script is configured for deterministic U-Net checkpoints only.")
    elif 'deterministic' in cfg_train.model:
        if cfg.n_ens > 1:
            raise ValueError("Deterministic model can not be used with n_ens > 1. Use n_ens = 1 for deterministic models.")

        deterministic = True
        input_times = (len(cfg_train.conditioning_times))*cfg_train.num_variables + cfg_train.num_static_fields

        model = BasicUNet(
            in_channels=input_times,
            out_channels=cfg_train.num_variables,
            base_channels=cfg_train.filters,
        )
        print("Using deterministic model", flush=True)
        if cfg_train.use_baseline_fno:
            print("Using baseline FNO model", flush=True)
    else:
        raise ValueError(f"Model choice {cfg_train.model} not recognized.")

    model.load_state_dict(torch.load(f'{cfg.model_directory}/{cfg.name}/best_model.pth', map_location=device))
    model.to(device)

    print(f"Loaded model {cfg.name}, {cfg_train.model}",  flush=True)
    print("Num params: ", sum(p.numel() for p in model.parameters()), flush=True)

    return model, deterministic

def get_latents(latent_shape, n_direct, alpha=1.0, device='cpu'):
    """
    The noise correlation done in Algorithm 2 reparameterized with alpha instead of rho.
    Note that this will only affect the direct forecasting, not the iterative timesteps.
    Variance preserving function for the noise z.
    alpha=1.0 means fixed noise,
    alpha=0.0 means uncorrelated noise
    """
    B, C, H, W = latent_shape # latent_shape: (n_samples * n_ens, cfg_train.num_variables, dx, dy)

    z = torch.zeros((n_direct, B, C, H, W), device=device)
    z[0] = torch.randn((B, C, H, W), device=device)
    alpha = torch.tensor(alpha, device=device)

    for t in range(1, n_direct):
        noise = torch.randn((B, C, H, W), device=device)
        z[t] = (alpha).sqrt() * z[t - 1] + (1 - alpha).sqrt() * noise

    z = z.transpose(0, 1).reshape(n_direct * B, C, H, W) # Transposing makes sure the order is preserved.

    return z

def test_loop(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(cfg.name, flush=True)
    print("[t_direct, t_iter, t_max]", [cfg.t_direct, cfg.t_iter, cfg.t_max],  flush=True)
    print("n_ens:", cfg.n_ens,  flush=True)
    print("alpha:", cfg.alpha,  flush=True)

    # Saving testing scripy params
    result_path = Path(f'{cfg.result_directory}/{cfg.name}')
    result_path.mkdir(parents=True, exist_ok=True)
    config_path = result_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)
    print(f"Saved config to {config_path}", flush=True)
    # ---------------- Load training config data ------------------------------------
    cfg_train = TrainConfig.from_json(Path(cfg.model_directory) / cfg.name / "config.json")
    if cfg_train.variable_names != ["total_precipitation", "2m_temperature"] or cfg_train.num_variables != 2:
        raise ValueError(
            "This two-variable test script requires a total_precipitation,2m_temperature "
            "checkpoint with num_variables=2."
        )
    cfg_train.data_directory = cfg.data_directory
    # ---------------- Build data loaders ------------------------------------
    test_time_loader, dataset, forecasting_times, lat, lon, _ = build_dataloaders(cfg, cfg_train, device)
    land_mask = None
    if cfg.land_only:
        land_mask = _build_land_mask(cfg.data_directory, cfg_train.height, cfg_train.width, device, lsm_path=cfg.lsm_path)
    else:
        print("Land-only mode disabled; ocean points are not masked.", flush=True)
    # ---------------- Build model ------------------------------------
    model, deterministic = build_model(cfg, cfg_train, device)
    # ---------------- Build sampler ------------------------------------
    sampler_fn = heun_sampler
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

            if land_mask is not None:
                n_dyn = len(cfg_train.conditioning_times) * cfg_train.num_variables
                previous[:, :n_dyn] = previous[:, :n_dyn] * land_mask
                current = current * land_mask

            direct_time_labels = torch.tensor(np.array([x for x in time_labels[0] if x <= cfg.t_iter]), device=device)
            n_iter = time_labels.shape[1] // direct_time_labels.shape[0] # how many steps towards the full horizon
            n_direct = direct_time_labels.shape[0] #how many steps towards direct prediction

            class_labels = previous.repeat_interleave(n_direct * cfg.n_ens, dim=0) # Can not be changed if batchsz > 1

            static_fields = class_labels[:, -cfg_train.num_static_fields:]

            latent_shape = (n_samples * cfg.n_ens, cfg_train.num_variables, dx, dy)

            direct_time_labels_repeated = direct_time_labels.repeat(cfg.n_ens * n_samples) # Can not be changed if n_direct > 1

            # Test
            predicted_all = torch.zeros((n_samples, cfg.n_ens, n_times, cfg_train.num_variables, dx, dy), device=device)

            for i in tqdm(range(n_iter)):
                # Control the correlation of the noise with alpha (Algorithm 2)
                latents = get_latents(latent_shape, n_direct, alpha=cfg.alpha, device=device)

                if deterministic:
                    predicted = model(class_labels, direct_time_labels_repeated / cfg_train.max_horizon)
                else:
                    predicted = sampler_fn(model, latents, class_labels, direct_time_labels_repeated / cfg_train.max_horizon,
                                        sigma_max=80, sigma_min=0.03, rho=7, num_steps=20, S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05)

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

                if cfg_train.num_static_fields != 0:
                    class_labels = torch.cat((class_labels, static_fields), dim=1)

            # Save predictions incrementally to zarr file
            output = renormalize(predicted_all, cfg_train, device).view(n_samples, cfg.n_ens, n_times, cfg_train.num_variables, dx, dy)
            if land_mask is not None:
                output = output * land_mask.view(1, 1, 1, 1, dx, dy)
            predictions[write_idx:write_idx + n_samples, :, :, :, :, :] = output.cpu().numpy()

            write_idx += n_samples

        gc.collect()
        torch.cuda.empty_cache()

    metrics_cal(cfg_train, test_time_loader, lat, lon, predictions, forecasting_times, result_path, device, land_mask=land_mask)

def metrics_cal(cfg_train, test_time_loader, lat, lon, predictions, forecasting_times, result_path, device, land_mask=None):
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

            forecast = predictions[i:i + truth.shape[0]]
            forecast = torch.tensor(forecast, device=device)
            i = i + truth.shape[0]

            if land_mask is not None:
                truth = truth * land_mask.view(1, 1, 1, dx, dy)
                forecast = forecast * land_mask.view(1, 1, 1, 1, dx, dy)

            # Add windspeed (only for weather problem with u10 and v10 at indices 3 and 4)
            if cfg_train.problem_name == 'weather' and cfg_train.num_variables >= 5:
                w_truth = (truth[:,:,3]**2 + truth[:,:,4]**2).sqrt().unsqueeze(2)
                truth = torch.cat((truth, w_truth), dim=2)
                w_forecast = (forecast[:,:,:,3]**2 + forecast[:,:,:,4]**2).sqrt().unsqueeze(3)
                forecast = torch.cat((forecast, w_forecast), dim=3)

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
        description="Test Diffusion spatiotemporal forecaster on NPZ shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # often requires modifications
    # e.g., --name "iterative-6h" --spacing 1 --model "autoregressive" --batch_size 128 --n_ens 50
    ap.add_argument("--name", type=str, default='deterministic-iterative-6h', help="method name.")
    ap.add_argument('--model', type=str, default='deterministic-iterative-6h', help='options are continuous or autoregressive.')
    ap.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    ap.add_argument("--spacing", type=int, default=1, help="Use a larger number to subsample the data in time.")
    ap.add_argument("--t_min", type=int, default=6, help="Minimum time the model forecasts.")
    ap.add_argument("--t_max", type=int, default=240, help="Maximum time the model forecasts.")
    ap.add_argument("--t_iter", type=int, default=6, help="Time to iterate on.")
    ap.add_argument("--t_direct", type=int, default=6, help="Shortest time the model forecasts.")
    ap.add_argument("--n_ens", type=int, default=1, help="Number of ensemble members.")
    ap.add_argument("--alpha", type=int, default=1, help="Correlation of the noise with alpha (1.0 means fixed noise, 0.0 means uncorrelated noise)")
    ap.add_argument("--test_start", type=str, default="2025-01-01T00:00:00", help="Expected inclusive start time for the test split.")
    ap.add_argument("--test_end", type=str, default="2025-12-31T00:00:00", help="Expected inclusive end time for the test split.")
    ap.add_argument("--land_only", type=parse_bool, default=True, help="Mask ocean points using lsm.npy during testing and metric calculation.")
    ap.add_argument("--lsm_path", type=str, default="/nesi/project/massey04632/data/ERA5/static/lsm.npy", help="Optional explicit path to lsm.npy.")


    # Optional: local overrides for input dirs (useful for local testing)
    ap.add_argument("--data_directory", type=str, default="/nesi/project/massey04632/data/ERA5/L2/2015_2025_2V", help="Path with prepared ERA5 L2 data files.")
    ap.add_argument("--model_directory", type=str, default="/nesi/project/massey04632/ERF/models", help="Path containing trained model run directories.")
    ap.add_argument("--result_directory", type=str, default="/nesi/project/massey04632/ERF/results", help="Path where test outputs/metrics are saved.")

    args = ap.parse_args()
    return TestConfig(**vars(args))

if __name__ == "__main__":
    cfg = parse_args()
    test_loop(cfg)
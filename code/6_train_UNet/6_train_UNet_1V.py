from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import csv, sys
import json
import argparse
from tqdm import tqdm
import os, random
from dataclasses import asdict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.utils import ERA5Dataset, DynamicKBatchSampler, get_uniform_t_dist_fn, pretty_print_kwargs, make_time_index, split_time_index, save_checkpoint, load_checkpoint
from loss.loss import WMSELoss
from configclass.train_dataclass import TrainConfig


def set_seed(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Some ops don't have deterministic kernels; warn_only=True avoids hard errors.
        torch.use_deterministic_algorithms(True, warn_only=True)


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
    """Basic deterministic U-Net with explicit encoder/decoder paths."""

    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = torch.nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = torch.nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = torch.nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = torch.nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.head = torch.nn.Conv2d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _match_spatial(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Pad/crop to match spatial shape of ref tensor for skip connections."""
        _, _, h, w = ref.shape
        sh, sw = src.shape[-2], src.shape[-1]

        if sh < h or sw < w:
            pad_h = max(h - sh, 0)
            pad_w = max(w - sw, 0)
            src = torch.nn.functional.pad(src, (0, pad_w, 0, pad_h))
        if src.shape[-2] > h or src.shape[-1] > w:
            src = src[:, :, :h, :w]
        return src

    def forward(self, x: torch.Tensor, _time_labels: torch.Tensor | None = None) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with skip connections
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

def calculate_gradient_stats(model):
    """
    Compute summary statistics for model gradients.
    
    Returns:
        dict: Gradient statistics including max/min values and distribution metrics.
    """
    # Access the wrapped model when using DataParallel.
    actual_model = model.module if hasattr(model, 'module') else model
    
    max_grad = -float('inf')
    min_grad = float('inf')
    total_params = 0  # Total number of scalar parameters.
    
    # Collect gradient stats from all parameters.
    param_inf_norms = []  # Per-parameter L-infinity norm (max absolute gradient value).
    all_grad_values = []  # All gradient values for distribution statistics.
    
    for param in actual_model.parameters():
        param_numel = param.numel()  # Number of elements in this parameter tensor.
        total_params += param_numel
        
        if param.grad is not None:
            # Gradient tensor has the same shape as the parameter tensor.
            # abs().max() is the largest absolute gradient element for this tensor.
            param_inf_norm = param.grad.data.abs().max().item()  # L-infinity norm for this tensor.
            param_inf_norms.append(param_inf_norm)
            max_grad = max(max_grad, param_inf_norm)  # Global max absolute gradient element.
            
            # Store all absolute gradient values for distribution metrics.
            param_grad_abs = param.grad.data.abs().flatten()  # Flatten to 1D.
            all_grad_values.append(param_grad_abs.cpu().numpy())
            
            # Smallest absolute gradient element across all parameters.
            min_grad = min(min_grad, param.grad.data.abs().min().item())
    
    # max_grad is the global L-infinity norm over all gradient elements.
    avg_inf_norm = np.mean(param_inf_norms) if param_inf_norms else 0.0
    
    # Compute distribution statistics across all gradient elements.
    if all_grad_values:
        all_grads_flat = np.concatenate(all_grad_values)  # Merge all gradient elements into one array.
        grad_median = np.median(all_grads_flat)
        grad_p95 = np.percentile(all_grads_flat, 95)  # 95th percentile.
        grad_p99 = np.percentile(all_grads_flat, 99)  # 99th percentile.
        grad_mean = np.mean(all_grads_flat)
    else:
        grad_median = 0.0
        grad_p95 = 0.0
        grad_p99 = 0.0
        grad_mean = 0.0
    
    return {
        'max_grad': max_grad if max_grad != -float('inf') else 0.0,  # Largest absolute gradient value.
        'avg_inf_norm': avg_inf_norm,  # Mean of per-parameter L-infinity norms.
        'min_grad': min_grad if min_grad != float('inf') else 0.0,  # Smallest absolute gradient value.
        'grad_mean': grad_mean,  # Mean of all gradient elements.
        'grad_median': grad_median,  # Median of all gradient elements.
        'grad_p95': grad_p95,  # 95th percentile of all gradient elements.
        'grad_p99': grad_p99,  # 99th percentile of all gradient elements.
        'total_params': total_params  # Total number of scalar parameters.
    }

def load_residual_stds(cfg, device):
    residual_stds = []
    for var_name in cfg.variable_names:
        std_values = torch.tensor(np.loadtxt(f'{cfg.data_directory}/residual_stds/WB_{var_name}.txt', delimiter=' ')[:, 1],
                                  dtype=torch.float32).to(device)
        residual_stds.append(std_values)
    residual_stds = torch.stack([res_std for res_std in residual_stds], axis=1)[:cfg.t_max]
    return residual_stds


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


def _resolve_lsm_file(data_directory: str, lsm_path: str | None = None) -> Path | None:
    data_dir = Path(data_directory)
    candidates = []
    if lsm_path:
        candidates.append(Path(lsm_path))
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


def _build_land_mask(data_directory: str, height: int, width: int, device: torch.device, lsm_path_override: str | None = None) -> torch.Tensor | None:
    lsm_path = _resolve_lsm_file(data_directory, lsm_path=lsm_path_override)
    if lsm_path is None:
        raise FileNotFoundError(
            "Land-only training requires lsm.npy, but no mask file was found. "
            "Place lsm.npy under the data directory, data_directory/static, "
            "or /nesi/project/massey04632/data/ERA5/static/."
        )

    lsm = np.load(lsm_path).astype(np.float32)
    if lsm.ndim == 3 and lsm.shape[0] == 1:
        lsm = lsm[0]
    if lsm.ndim != 2:
        raise ValueError(f"Expected 2D lsm array, got shape {lsm.shape} from {lsm_path}")
    if lsm.shape != (height, width):
        raise ValueError(
            f"LSM shape {lsm.shape} does not match training grid {(height, width)}. File: {lsm_path}"
        )

    land_mask = (lsm > 0.5).astype(np.float32)
    print(f"Using land mask from {lsm_path}; land fraction={land_mask.mean():.4f}", flush=True)
    return torch.tensor(land_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)


def build_dataloaders(cfg, device):
    suffix = f"{cfg.start_datetime[:4]}-{cfg.end_datetime[:4]}"
    dataset_path = _resolve_dataset_file(cfg.data_directory, cfg.variable_names, suffix)
    latlon_path = _resolve_latlon_file(cfg.data_directory, suffix)

    # ---------------- Load lat/lon in correct order --------------------------
    lat, lon = np.load(latlon_path).values()

    # Get the number of samples, training and validation samples
    # ti = pd.date_range(datetime.datetime(1979, 1, 1, 0), datetime.datetime(2018, 12, 31, 23), freq='1h')
    # n_samples, n_train, n_val = len(ti), sum(ti.year <= 2015), sum((ti.year >= 2016) & (ti.year <= 2017))
    ti = make_time_index(cfg.start_datetime, cfg.end_datetime, cfg.time_freq)
    n_samples, n_train, n_val = split_time_index(
        ti,
        split_mode=cfg.split_mode,
        train_year_end=int(cfg.train_year_end) if cfg.train_year_end is not None else None,
        val_year_end=int(cfg.val_year_end) if cfg.val_year_end is not None else None,
        train_until=cfg.train_until,
        val_until=cfg.val_until,
    )
    # ---------------- Load normalization factors (ordered) -------------------
    with open(f'{cfg.data_directory}/norm_factors.json', 'r') as f:
        statistics = json.load(f)
    mean_data = np.array([statistics[name]["mean"] for name in cfg.variable_names], dtype=np.float32)
    std_data = np.array([statistics[name]["std"] for name in cfg.variable_names], dtype=np.float32)
    norm_factors = np.stack([mean_data, std_data], axis=0)

    static_data_path = None
    if cfg.num_static_fields > 0:
        static_data_path = f'{cfg.data_directory}/static_fields.npy'

    kwargs = {
        'dataset_path': str(dataset_path),
        'sample_counts': (n_samples, n_train, n_val),
        'dimensions': (cfg.num_variables, len(lat), len(lon)),
        'max_horizon': cfg.max_horizon,  # For scaling the time embedding
        'norm_factors': norm_factors,
        'device': device,
        'spacing': cfg.spacing,
        'dtype': 'float32',
        'conditioning_times': cfg.conditioning_times,
        'lead_time_range': [cfg.t_min, cfg.t_max, cfg.delta_t],
        'static_data_path': static_data_path,
        'random_lead_time': 1,
    }

    pretty_print_kwargs(kwargs)

    # Define the batch samplers
    update_t_per_batch = get_uniform_t_dist_fn(t_min=cfg.t_min, t_max=cfg.t_max, delta_t=cfg.delta_t)

    train_time_dataset = ERA5Dataset(lead_time=cfg.t_max, dataset_mode='train', **kwargs)
    # model generalises to any lead you’ll ask for later using DynamicKBatchSampler
    train_batch_sampler = DynamicKBatchSampler(train_time_dataset, batch_size=cfg.batch_size, drop_last=True,
                                               t_update_callback=update_t_per_batch, shuffle=True)
    train_time_loader = DataLoader(train_time_dataset, batch_sampler=train_batch_sampler)

    val_time_dataset = ERA5Dataset(lead_time=cfg.t_max, dataset_mode='val', **kwargs)
    val_batch_sampler = DynamicKBatchSampler(val_time_dataset, batch_size=cfg.batch_size, drop_last=False,
                                             t_update_callback=update_t_per_batch, shuffle=True)
    val_time_loader = DataLoader(val_time_dataset, batch_sampler=val_batch_sampler)

    return train_time_loader, val_time_loader, lat, lon, kwargs

def build_model_and_loss (cfg, device, lat, lon, residual_stds):

    if 'deterministic' not in cfg.model:
        raise ValueError("This training script is configured for deterministic U-Net only. Set --model deterministic.")

    input_times = (len(cfg.conditioning_times)) * cfg.num_variables + cfg.num_static_fields
    model = BasicUNet(
        in_channels=input_times,
        out_channels=cfg.num_variables,
        base_channels=cfg.filters,
    )
    loss_fn = WMSELoss(lat, lon, device, precomputed_std=residual_stds)
    print("Using deterministic U-Net model", flush=True)

    print(cfg.name, flush=True)
    print(cfg.model, flush=True)
    print("Num params: ", sum(p.numel() for p in model.parameters()), flush=True)
    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model, loss_fn


def train_loop(cfg: TrainConfig):

    # ---------------- Load config from checkpoint if resuming ----------------
    resume_from_checkpoint = cfg.resume_from_checkpoint  # Save original checkpoint path
    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Infer result_path from checkpoint path (checkpoints/epoch_X.pth -> result_path)
        result_path = checkpoint_path.parent.parent
        config_path = result_path / 'config.json'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}. Cannot resume training without original config.")
        
        # Load config from saved config.json
        print(f"Loading config from {config_path} for resume training", flush=True)
        cfg = TrainConfig.from_json(config_path)
        # Restore resume_from_checkpoint path (not saved in JSON)
        cfg.resume_from_checkpoint = resume_from_checkpoint
        print(f"Resuming training with config: name={cfg.name}, epochs={cfg.epochs}, lr={cfg.lr}, etc.", flush=True)
    else:
        # Determine result_path from config
        result_path = Path(f'{cfg.result_directory}/{cfg.name}')
        config_path = result_path / 'config.json'

    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    land_mask = None
    if cfg.land_only:
        land_mask = _build_land_mask(cfg.data_directory, cfg.height, cfg.width, device, lsm_path_override=cfg.lsm_path)
    else:
        print("Land-only mode disabled by config; ocean points are not masked.", flush=True)


    # ---------------- Load residual stds (per-lead, per-var) ----------------
    residual_stds = load_residual_stds(cfg, device)
    # ---------------- Build data loaders ------------------------------------
    train_time_loader, val_time_loader, lat, lon, kwargs = build_dataloaders(cfg, device)
    # ---------------- Build model + loss ------------------------------------
    model, loss_fn = build_model_and_loss(cfg, device, lat, lon, residual_stds)
    print(
        f"Training setup: device={device}, cuda_device_count={torch.cuda.device_count()}, "
        f"data_parallel={isinstance(model, torch.nn.DataParallel)}",
        flush=True,
    )
    # ---------------- Optimizer + Scheduler (warmup -> cosine) --------------
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=1000)

    # ---------------- Logging / Checkpoints ---------------------------------
    # saving results path configuration
    result_path.mkdir(parents=True, exist_ok=True)
    (result_path / 'checkpoints').mkdir(parents=True, exist_ok=True)
    # saving training scripy params (only if not resuming)
    if not resume_from_checkpoint:
        with open(config_path, 'w') as f:
            json.dump(asdict(cfg), f, indent=2, sort_keys=True)
        print(f"Saved config to {config_path}", flush=True)
    # saving log path  configuration
    log_file_path = result_path / f'training_log.csv'
    gradient_log_path = result_path / f'gradient_log.csv'
    
    # ---------------- Load checkpoint if specified -------------------------
    start_epoch = 0
    best_val_loss = float('inf')
    if resume_from_checkpoint:
        checkpoint_path = Path(resume_from_checkpoint)
        checkpoint_info = load_checkpoint(checkpoint_path, model, optimizer, scheduler, warmup_scheduler, device)
        start_epoch = checkpoint_info["epoch"] + 1  # Resume from next epoch
        best_val_loss = checkpoint_info["best_val_loss"]
        print(f"Resuming training from epoch {start_epoch}/{cfg.epochs}", flush=True)
        # Ensure log file exists when resuming (in case it was deleted)
        if not log_file_path.exists():
            with open(log_file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Average Training Loss', 'Validation Loss', 'Alpha Value'])
        if not gradient_log_path.exists():
            with open(gradient_log_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Batch', 'Max Gradient (L∞)', 'Avg Inf Norm', 'Min Gradient', 'Grad Mean', 'Grad Median', 'Grad P95', 'Grad P99', 'Total Params'])
    else:
        # Only create new log file if not resuming
        with open(log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Average Training Loss', 'Validation Loss', 'Alpha Value'])
        # Create gradient log file
        with open(gradient_log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Batch', 'Max Gradient (L∞)', 'Avg Inf Norm', 'Min Gradient', 'Grad Mean', 'Grad Median', 'Grad P95', 'Grad P99', 'Total Params'])
    
    # ---------------- Training loop -----------------------------------------
    loss_values = []
    val_loss_values = []
    for epoch in range(start_epoch, cfg.epochs):

        # Training phase
        model.train()
        total_train_loss = 0
        epoch_inf_norms = []  # Track all batch L-infinity gradient norms for this epoch.
        epoch_max_grads = []
        epoch_min_grads = []
        
        for batch_idx, (previous, current, time_label) in enumerate(tqdm(train_time_loader)):
            current = current.to(device)
            previous = previous.to(device)
            time_label = time_label.to(device)
            if land_mask is not None:
                current = current * land_mask
                previous = previous * land_mask

            optimizer.zero_grad()
            loss = loss_fn(model, current, previous, time_label.float() / cfg.max_horizon)
            total_train_loss += loss.item()

            loss.backward()
            
            # Compute and record gradient statistics.
            grad_stats = calculate_gradient_stats(model)
            epoch_inf_norms.append(grad_stats['max_grad'])  # max_grad is the global L-infinity norm.
            epoch_max_grads.append(grad_stats['max_grad'])
            epoch_min_grads.append(grad_stats['min_grad'])
            
            # Print parameter/gradient diagnostics for the first batch only.
            if epoch == start_epoch and batch_idx == 0:
                print(f"Model Parameter Stats: Total={grad_stats['total_params']:,}", flush=True)
                print(f"Gradient Stats Explanation:", flush=True)
                print(f"  - max_grad: Largest single gradient element (abs) across all parameters = {grad_stats['max_grad']:.4f}", flush=True)
                print(f"  - min_grad: Smallest single gradient element (abs) across all parameters = {grad_stats['min_grad']:.4f}", flush=True)
                print(f"  - grad_mean: Mean of all gradient elements = {grad_stats['grad_mean']:.4f}", flush=True)
                print(f"  - grad_median: Median of all gradient elements = {grad_stats['grad_median']:.4f}", flush=True)
                print(f"  - grad_p95: 95th percentile = {grad_stats['grad_p95']:.4f}", flush=True)
                print(f"  - grad_p99: 99th percentile = {grad_stats['grad_p99']:.4f}", flush=True)
            
            # Append batch-level gradient statistics to CSV.
            with open(gradient_log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    epoch + 1,
                    batch_idx,
                    f"{grad_stats['max_grad']:.6f}",  # Largest absolute gradient value.
                    f"{grad_stats['avg_inf_norm']:.6f}",  # Mean per-parameter L-infinity norm.
                    f"{grad_stats['min_grad']:.6f}",  # Smallest absolute gradient value.
                    f"{grad_stats['grad_mean']:.6f}",  # Mean gradient value.
                    f"{grad_stats['grad_median']:.6f}",  # Median gradient value.
                    f"{grad_stats['grad_p95']:.6f}",  # 95th percentile.
                    f"{grad_stats['grad_p99']:.6f}",  # 99th percentile.
                    grad_stats['total_params']  # Total number of scalar parameters.
                ])
            
            # Optional warning hook for unusually large gradients.
            # if grad_stats['max_grad'] > 100.0:
                #print(f"⚠️  WARNING: Batch {batch_idx}, Max Gradient (L∞ norm) = {grad_stats['max_grad']:.2f} (exceeds 100.0)", flush=True)
            
            optimizer.step()
            warmup_scheduler.step()

        avg_train_loss = total_train_loss / len(train_time_loader)
        
        # Epoch-level gradient summary.
        if epoch_inf_norms:
            avg_inf_norm = np.mean(epoch_inf_norms)
            max_inf_norm = np.max(epoch_inf_norms)
            min_inf_norm = np.min(epoch_inf_norms)
            avg_max_grad = np.mean(epoch_max_grads)
            avg_min_grad = np.mean(epoch_min_grads)
            
            print(f"Epoch {epoch + 1} Gradient Stats: L∞_norm(avg={avg_inf_norm:.4f}, max={max_inf_norm:.4f}, min={min_inf_norm:.4f})", flush=True)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for previous, current, time_label in (val_time_loader):
                current = current.to(device)
                previous = previous.to(device)
                time_label = time_label.to(device)
                if land_mask is not None:
                    current = current * land_mask
                    previous = previous * land_mask

                loss = loss_fn(model, current, previous, time_label.float() / cfg.max_horizon)
                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_time_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(best_model_state, result_path / 'best_model.pth')

        scheduler.step()

        alpha_value = 0
        #if cfg.use_fno_bottleneck:
            #alpha_value = model.model.fno_alpha.tanh().item()
            #alpha_value = 0.05
        loss_values.append([avg_train_loss])
        val_loss_values.append(avg_val_loss)

        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, alpha_value])

        print(f'Epoch [{epoch + 1}/{cfg.epochs}], Average Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}',
              flush=True)
        print("torch.cuda.max_memory_allocated GiB:", torch.cuda.max_memory_allocated() / 1024**3, flush=True)
        if cfg.save_every and (epoch % cfg.save_every == 0):
            save_checkpoint(result_path / f'checkpoints/epoch_{epoch}.pth', model, optimizer, scheduler, warmup_scheduler, epoch=epoch, best_val_loss=best_val_loss)

# --------------------------- CLI ---------------------------
def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser(
        description="Train Diffusion spatiotemporal forecaster on NPZ shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # often requires modifications
    # e.g., --name "iterative-6h" --spacing 1 --model "autoregressive"
    ap.add_argument("--problem_name", type=str, choices=["weather"], default="weather", help="ERA5 workflow uses weather mode.")
    ap.add_argument("--name", type=str, default='deterministic-iterative-6h', help="method name.")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    ap.add_argument("--filters", type=int, default=32, help="Filter scaling in UNet，a.k.a model’s base channel width")
    ap.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay.")
    ap.add_argument("--lr", type=float, default=0.0005, help="Learning rate.")
    ap.add_argument("--epochs", type=int, default=300, help="Max training epochs.")
    ap.add_argument("--spacing", type=int, default=100, help="Use a larger number to subsample the data in time.")
    ap.add_argument("--t_min", type=int, default=6, help="Minimum time the model forecasts.")
    ap.add_argument("--t_max", type=int, default=6, help="Maximum time the model forecasts.")
    ap.add_argument("--delta_t", type=int, default=6, help="Timestep size of model, ")
    ap.add_argument("--conditioning_times", type=lambda s: [int(x.strip()) for x in s.split(",")], default="0,-6", help="Previous timesteps to condition on.")
    ap.add_argument('--model', type=str, choices=['deterministic'], default='deterministic', help='Use deterministic U-Net training only.')


    # not often requires modifications
    ap.add_argument("--variable_names", type=lambda s: [x.strip() for x in s.split(",")], default="total_precipitation", help="Comma-separated variable names matching L2 files.")
    ap.add_argument("--num_variables", type=int, default=1, help="Number of dynamic ERA5 variables.")
    ap.add_argument("--num_static_fields", type=int, default=0, help="Set >0 only if static_fields.npy exists.")
    ap.add_argument("--max_horizon", type=int, default=240, help="Prediction frames (T_lead).")
    ap.add_argument("--height", type=int, default=74, help="Grid height for prepared ERA5 data.")
    ap.add_argument("--width", type=int, default=96, help="Grid width for prepared ERA5 data.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Optional: local overrides for input dirs (useful for local testing)
    ap.add_argument("--data_directory", type=str, default="/nesi/project/massey04632/data/ERA5/L2", help="Path with prepared ERA5 L2 memmap files.")
    ap.add_argument("--result_directory", type=str, default="/nesi/project/massey04632/ERF/models", help="Output path for model checkpoints/logs.")
    # checkpoints settings
    ap.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs (0 = disable).")
    ap.add_argument("--ckpt_dir", type=str, default="../../../models/ERA5",help="Directory to store epoch checkpoints.")
    ap.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint file to resume training from.")

    # --- dataset time config ---
    ap.add_argument("--start_datetime", type=str, default="2015-01-01T00:00:00")
    ap.add_argument("--end_datetime", type=str, default="2025-12-31T00:00:00")
    ap.add_argument("--time_freq", type=str, default="1h", help="Pandas offset alias, e.g. 1h, 6h, 1d, 7D")
    # --- split config (years/dates only) ---
    ap.add_argument("--split_mode", type=str, default="dates", choices=["years", "dates"])
    ap.add_argument("--train_year_end", type=int, default=None, help="Used when split_mode=years")
    ap.add_argument("--val_year_end", type=int, default=None, help="Used when split_mode=years")
    ap.add_argument("--train_until", type=str, default="2023-12-31T00:00:00", help="Used when split_mode=dates (inclusive)")
    ap.add_argument("--val_until", type=str, default="2024-12-31T00:00:00", help="Used when split_mode=dates (inclusive)")
    ap.add_argument("--skip", type=str, default="False", help="Skip this step")
    ap.add_argument("--use_fno_bottleneck", type=lambda x: x.lower() == "true", default=False, help="Use FNO bottleneck")
    ap.add_argument("--use_baseline_fno", type=lambda x: x.lower() == "true", default=False, help="Use baseline FNO")
    ap.add_argument("--method", type=str, choices=["unet"], default="unet", help="Use the basic deterministic U-Net preset.")
    ap.add_argument("--land_only", type=lambda x: x.lower() == "true", default=True, help="Mask ocean data using lsm.npy during training/validation.")
    ap.add_argument("--lsm_path", type=str, default=None, help="Optional explicit path to lsm.npy.")
    args = ap.parse_args()
    
    # Convert string "None" to Python None (handles shell script passing "None" as string)
    if args.resume_from_checkpoint == "None" or args.resume_from_checkpoint == "none" or args.resume_from_checkpoint == "":
        args.resume_from_checkpoint = None
    
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train_loop(cfg)
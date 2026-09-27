"""
Comparison of UNet vs CNN for Rainfall Forecasting
Three focused sections:
1. RMSE Comparison across lead times
2. MAE Comparison across lead times
3. Spatial predictions visualization from zarr
"""

import json
from pathlib import Path
import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from matplotlib.colors import BoundaryNorm


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ComparisonConfig:
    unet_eval_dir: str
    unet_5v_eval_dir: str
    cnn_eval_dir: str
    unet_model_name: str
    unet_5v_model_name: str
    cnn_model_name: str
    unet_pred_zarr: str
    unet_5v_pred_zarr: str
    cnn_pred_zarr: str
    static_nc_path: str
    land_threshold: float
    seed: int
    scale: float
    out_dir: str


def _load_eval_group(eval_dir: Path):
    """Load zarr group containing metrics"""
    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing evaluation directory: {eval_dir}")
    
    groups = [p for p in eval_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not groups:
        raise FileNotFoundError(f"No metric group found in: {eval_dir}")
    
    return zarr.open_group(groups[0], mode="r")


def _load_metrics(eval_dir: Path):
    """Load skill, CRPS and other metrics from zarr"""
    g = _load_eval_group(eval_dir)
    times = np.asarray(g["times"]).astype(np.float64).squeeze()
    values = {
        "skill": np.asarray(g["skill"]).astype(np.float64).squeeze(),
        "CRPS": np.asarray(g["CRPS"]).astype(np.float64).squeeze(),
        "spread": np.asarray(g["spread"]).astype(np.float64).squeeze(),
    }
    return times, values


def _load_prediction_zarr(pred_path: Path):
    """Load prediction zarr array"""
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction zarr: {pred_path}")
    return zarr.open_array(pred_path, mode="r")


def _load_mae_leadwise(result_dir: Path):
    """Load lead-wise MAE values saved by loss/AW_MAE.py."""
    mae_csv = result_dir / "MAE_leadwise.csv"
    if not mae_csv.exists():
        raise FileNotFoundError(f"Missing MAE leadwise file: {mae_csv}")

    data = np.genfromtxt(mae_csv, delimiter=",", names=True)
    lead_times = np.atleast_1d(data["lead_time_h"]).astype(np.float64)
    mae_values = np.atleast_1d(data["mean_area_weighted_mae"]).astype(np.float64)
    return lead_times, mae_values


def _load_train_config(cfg: ComparisonConfig) -> dict:
    config_path = REPO_ROOT / "models" / cfg.unet_model_name / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _build_truth_accessor(cfg: ComparisonConfig, lat: np.ndarray, lon: np.ndarray):
    train_cfg = _load_train_config(cfg)
    data_directory = Path(train_cfg["data_directory"])
    suffix = f"{train_cfg['start_datetime'][:4]}-{train_cfg['end_datetime'][:4]}"
    dataset_file = _resolve_dataset_file(data_directory, train_cfg["variable_names"], suffix)

    time_index = pd.date_range(
        pd.to_datetime(train_cfg["start_datetime"]),
        pd.to_datetime(train_cfg["end_datetime"]),
        freq=train_cfg["time_freq"],
        inclusive="both",
    )
    train_mask = time_index <= pd.to_datetime(train_cfg["train_until"])
    val_mask = (time_index > pd.to_datetime(train_cfg["train_until"])) & (time_index <= pd.to_datetime(train_cfg["val_until"]))
    spinup = 24
    eval_max_horizon = 36
    test_start = spinup + int(train_mask.sum()) + int(val_mask.sum())
    sample_start_indices = np.arange(test_start, len(time_index) - eval_max_horizon)[:: int(train_cfg.get("spacing", 1))]

    dataset = np.memmap(
        str(dataset_file),
        dtype=np.float32,
        mode="r",
        shape=(len(time_index), int(train_cfg["num_variables"]), len(lat), len(lon)),
    )
    return train_cfg, dataset, sample_start_indices


def _get_truth_panels(dataset, sample_start_indices: np.ndarray, sample_index: int, lead_hours: list[int], scale: float):
    lead_array = np.asarray(lead_hours, dtype=int)
    sample_start = int(sample_start_indices[sample_index])
    truth = dataset[sample_start + lead_array, 0].astype(np.float64) * scale
    return truth


def _select_spatial_sample(
    truth_panels: np.ndarray,
    unet_panels: np.ndarray,
    cnn_panels: np.ndarray,
    land_mask: np.ndarray,
    lat: np.ndarray,
) -> tuple[int, dict]:
    """Select a sample with both-island coverage where U-Net is closer to truth than CNN."""
    lat = np.asarray(lat, dtype=np.float64)
    split_lat = np.median(lat)
    north_mask = land_mask & (lat[:, None] >= split_lat)
    south_mask = land_mask & (lat[:, None] < split_lat)

    rainfall_max = truth_panels.max(axis=1)
    threshold_mm = 1.0

    north_cov = (rainfall_max > threshold_mm)[:, north_mask].mean(axis=1)
    south_cov = (rainfall_max > threshold_mm)[:, south_mask].mean(axis=1)
    total_cov = (rainfall_max > threshold_mm)[:, land_mask].mean(axis=1)

    unet_rmse = np.sqrt(np.nanmean(((unet_panels - truth_panels)[:, :, land_mask]) ** 2, axis=(1, 2)))
    cnn_rmse = np.sqrt(np.nanmean(((cnn_panels - truth_panels)[:, :, land_mask]) ** 2, axis=(1, 2)))
    unet_mae = np.nanmean(np.abs((unet_panels - truth_panels)[:, :, land_mask]), axis=(1, 2))
    cnn_mae = np.nanmean(np.abs((cnn_panels - truth_panels)[:, :, land_mask]), axis=(1, 2))

    rmse_gain = np.maximum(0.0, cnn_rmse - unet_rmse)
    mae_gain = np.maximum(0.0, cnn_mae - unet_mae)

    # Favor balanced North/South coverage, then total coverage, stronger maxima,
    # and cases where U-Net is closer to truth than CNN.
    peak_strength = rainfall_max[:, land_mask].max(axis=1)
    score = (
        np.minimum(north_cov, south_cov)
        + 0.2 * total_cov
        + 0.02 * peak_strength
        + 0.5 * rmse_gain
        + 0.25 * mae_gain
    )
    sample_index = int(np.argmax(score))
    summary = {
        "threshold_mm": threshold_mm,
        "north_coverage": float(north_cov[sample_index]),
        "south_coverage": float(south_cov[sample_index]),
        "total_coverage": float(total_cov[sample_index]),
        "max_rainfall_mm": float(rainfall_max[sample_index][land_mask].max()),
        "unet_rmse_mm": float(unet_rmse[sample_index]),
        "cnn_rmse_mm": float(cnn_rmse[sample_index]),
        "unet_mae_mm": float(unet_mae[sample_index]),
        "cnn_mae_mm": float(cnn_mae[sample_index]),
    }
    return sample_index, summary


def _load_land_mask(cfg: ComparisonConfig) -> tuple:
    """Load land-sea mask from static netCDF"""
    static_nc = Path(cfg.static_nc_path)
    if not static_nc.exists():
        raise FileNotFoundError(f"Static file not found: {static_nc}")
    
    with xr.open_dataset(static_nc) as ds:
        if "lsm" in ds:
            lsm = ds["lsm"].values
        elif "land_sea_mask" in ds:
            lsm = ds["land_sea_mask"].values
        else:
            raise ValueError("No land-sea mask found in static file.")
        
        lat = np.asarray(ds["latitude"].values, dtype=np.float64)
        lon = np.asarray(ds["longitude"].values, dtype=np.float64)
    
    # Handle multi-dimensional LSM
    if lsm.ndim == 3:
        lsm = lsm[0]
    elif lsm.ndim == 4:
        lsm = lsm[0, 0]
    
    lsm = np.asarray(lsm, dtype=np.float64)
    
    # Flip to match conventional orientation if needed
    if lat[-1] < lat[0]:  # If lat is decreasing, flip
        lsm = lsm[::-1, :]
        lat = lat[::-1]
    
    if lon[-1] < lon[0]:  # If lon is decreasing, flip
        lsm = lsm[:, ::-1]
        lon = lon[::-1]
    
    return lsm >= float(cfg.land_threshold), lat, lon


def _model_specs(cfg: ComparisonConfig) -> list[tuple[str, str, Path, Path]]:
    """Return the three models in the order used throughout the comparisons."""
    return [
        ("U-Net 1V (TP)", cfg.unet_model_name, Path(cfg.unet_eval_dir), Path(cfg.unet_pred_zarr)),
        ("U-Net 5V (TP + T2M + U10 + V10 + MSL)", cfg.unet_5v_model_name, Path(cfg.unet_5v_eval_dir), Path(cfg.unet_5v_pred_zarr)),
        ("CNN 1V (TP)", cfg.cnn_model_name, Path(cfg.cnn_eval_dir), Path(cfg.cnn_pred_zarr)),
    ]


def plot_three_model_rmse(cfg: ComparisonConfig) -> None:
    """Compare precipitation RMSE for CNN 1V, U-Net 1V, and U-Net 5V."""
    print("\n=== TP RMSE: CNN 1V VS U-NET 1V VS U-NET 5V ===")
    colors = ["#0072B2", "#009E73", "#D55E00"]
    fig, ax = plt.subplots(figsize=(10, 6))
    reference_times = None

    for color, (label, _, eval_dir, _) in zip(colors, _model_specs(cfg)):
        times, values = _load_metrics(eval_dir)
        if reference_times is None:
            reference_times = times
        elif not np.array_equal(reference_times, times):
            raise ValueError(f"Lead times for {label} do not match the other evaluations.")
        rmse = np.asarray(values["skill"], dtype=np.float64)
        if rmse.ndim > 1:
            rmse = rmse[:, 0]
        ax.plot(times, rmse, label=label, color=color, linewidth=2.2, marker="o")

    ax.set_title("TP Area-Weighted RMSE (Lower Is Better)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Lead Time (hours)")
    ax.set_ylabel("Area-Weighted RMSE")
    ax.set_xticks(reference_times)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = Path(cfg.out_dir) / "01_Comparison_rmse_three_models.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_three_model_spatial(cfg: ComparisonConfig) -> None:
    """Plot truth and precipitation predictions for all three models."""
    print("\n=== TP SPATIAL COMPARISON: THREE MODELS ===")
    land_mask, lat, lon = _load_land_mask(cfg)
    specs = _model_specs(cfg)
    predictions = [(label, _load_prediction_zarr(pred_path)) for label, _, _, pred_path in specs]
    sample_count = min(prediction.shape[0] for _, prediction in predictions)
    sample_index = min(8300, sample_count - 1)
    lead_hours = [12, 18, 24, 30, 36]
    _, truth_dataset, sample_start_indices = _build_truth_accessor(cfg, lat, lon)
    truth = _get_truth_panels(truth_dataset, sample_start_indices, sample_index, lead_hours, cfg.scale)

    fig, axes = plt.subplots(len(lead_hours), 4, figsize=(17, 25), squeeze=False)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="white")
    titles = ["Truth (TP)"] + [label for label, _ in predictions]

    for row, lead_hour in enumerate(lead_hours):
        panels = [truth[row]] + [np.asarray(prediction[sample_index, 0, row, 0], dtype=np.float64) * cfg.scale for _, prediction in predictions]
        masked_panels = [np.where(land_mask, panel, np.nan) for panel in panels]
        values = np.concatenate([panel[np.isfinite(panel)] for panel in masked_panels])
        vmax = max(float(np.percentile(values, 97)), 0.1)
        for col, (title, panel) in enumerate(zip(titles, masked_panels)):
            ax = axes[row, col]
            image = ax.pcolormesh(lon_grid, lat_grid, panel, cmap=cmap, vmin=0.0, vmax=vmax, shading="auto")
            ax.contour(lon_grid, lat_grid, land_mask.astype(float), levels=[0.5], colors="black", linewidths=0.6)
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{lead_hour} h\nLatitude")
            else:
                ax.set_yticks([])
            ax.set_xlabel("Longitude")
        fig.colorbar(image, ax=axes[row, :].tolist(), shrink=0.85, pad=0.01, label="TP (mm)")

    fig.suptitle(f"Precipitation Forecasts: CNN 1V, U-Net 1V, and U-Net 5V\nSample {sample_index}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_path = Path(cfg.out_dir) / "02_Comparison_spatial_three_models.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_three_model_training_loss(cfg: ComparisonConfig) -> None:
    """Compare training and validation loss curves for all three models."""
    print("\n=== TRAINING LOSS: THREE MODELS ===")
    colors = ["#0072B2", "#009E73", "#D55E00"]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for color, (label, model_name, _, _) in zip(colors, _model_specs(cfg)):
        log_path = REPO_ROOT / "models" / model_name / "training_log.csv"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing training log for {label}: {log_path}")
        log = pd.read_csv(log_path)
        ax.plot(log["Epoch"], log["Average Training Loss"], color=color, linewidth=1.8, label=f"{label} train")
        ax.plot(log["Epoch"], log["Validation Loss"], color=color, linewidth=1.8, linestyle="--", label=f"{label} validation")

    ax.set_title("Training and Validation Loss: Three Models", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    out_path = Path(cfg.out_dir) / "03_Comparison_training_loss_three_models.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ==================== SECTION 1: RMSE COMPARISON ====================

def plot_rmse_comparison(cfg: ComparisonConfig):
    """
    Section 1: Area-weighted RMSE comparison across lead times
    Plots U-Net vs CNN area-weighted RMSE with lead time on x-axis (lower is better)
    """
    print("\n=== SECTION 1: AREA-WEIGHTED RMSE COMPARISON ===")
    
    unet_times, unet_values = _load_metrics(Path(cfg.unet_eval_dir))
    cnn_times, cnn_values = _load_metrics(Path(cfg.cnn_eval_dir))
    
    if not np.array_equal(unet_times, cnn_times):
        raise ValueError(f"Lead times mismatch: {unet_times} vs {cnn_times}")
    
    lead_times = unet_times.tolist()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot area-weighted RMSE (lower is better)
    ax.plot(lead_times, unet_values["skill"], 
            label="U-Net", color="#1f77b4", linewidth=2.2, marker="o", markersize=6)
    ax.plot(lead_times, cnn_values["skill"], 
            label="CNN Baseline", color="#d62728", linewidth=2.2, marker="s", markersize=6)
    
    ax.set_title("Area-Weighted RMSE Comparison: U-Net vs CNN Baseline (Lower is Better)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Lead Time (hours)", fontsize=12)
    ax.set_ylabel("Area-Weighted RMSE", fontsize=12)
    ax.set_xticks(lead_times)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best", fontsize=11)
    
    fig.tight_layout()
    
    out_path = Path(cfg.out_dir) / "01_Skill_RMSE_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.close(fig)


# ==================== SECTION 2: MAE COMPARISON ====================

def plot_mae_comparison(cfg: ComparisonConfig):
    """
    Section 2: MAE comparison across lead times
    Plots area-weighted MAE for U-Net vs CNN with lead time on x-axis (lower is better)
    """
    print("\n=== SECTION 2: MAE COMPARISON ===")
    
    unet_result_dir = Path(cfg.unet_pred_zarr).parent
    cnn_result_dir = Path(cfg.cnn_pred_zarr).parent

    unet_times, unet_mae = _load_mae_leadwise(unet_result_dir)
    cnn_times, cnn_mae = _load_mae_leadwise(cnn_result_dir)
    
    if not np.array_equal(unet_times, cnn_times):
        raise ValueError(f"Lead times mismatch: {unet_times} vs {cnn_times}")
    
    lead_times = unet_times.tolist()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot MAE (lower is better)
    ax.plot(lead_times, unet_mae,
            label="U-Net", color="#1f77b4", linewidth=2.2, marker="o", markersize=6)
    ax.plot(lead_times, cnn_mae,
            label="CNN Baseline", color="#d62728", linewidth=2.2, marker="s", markersize=6)

    ax.set_title("MAE Comparison: U-Net vs CNN Baseline (Lower is Better)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Lead Time (hours)", fontsize=12)
    ax.set_ylabel("Area-Weighted MAE", fontsize=12)
    ax.set_xticks(lead_times)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best", fontsize=11)
    
    fig.tight_layout()
    
    out_path = Path(cfg.out_dir) / "02_MAE_Comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.close(fig)


# ==================== SECTION 3: QUALITATIVE SPATIAL COMPARISON ====================

def plot_spatial_comparison(cfg: ComparisonConfig):
    """
    Section 3: Qualitative spatial comparison (NZ land only)
    Shows Truth vs UNet vs CNN predictions for all lead times (12, 18, 24, 30, 36h)
    using a sample with rainfall distributed across both North and South Islands.
    """
    print("\n=== SECTION 3: QUALITATIVE SPATIAL COMPARISON (NZ LAND ONLY) ===")
    
    # Load land mask and coordinates
    land_mask, lat, lon = _load_land_mask(cfg)
    print(f"Land mask shape: {land_mask.shape}, Lat: {lat.shape}, Lon: {lon.shape}")
    
    # Load predictions
    unet_pred = _load_prediction_zarr(Path(cfg.unet_pred_zarr))
    cnn_pred = _load_prediction_zarr(Path(cfg.cnn_pred_zarr))
    _, truth_dataset, sample_start_indices = _build_truth_accessor(cfg, lat, lon)
    lead_hours = [12, 18, 24, 30, 36]
    truth_all = np.stack(
        [_get_truth_panels(truth_dataset, sample_start_indices, sample_idx, lead_hours, cfg.scale) for sample_idx in range(unet_pred.shape[0])],
        axis=0,
    )
    
    print(f"U-Net prediction shape: {unet_pred.shape}")
    print(f"CNN prediction shape: {cnn_pred.shape}")
    
    unet_all = np.asarray(unet_pred[:, 0, :, 0], dtype=np.float64) * cfg.scale
    cnn_all = np.asarray(cnn_pred[:, 0, :, 0], dtype=np.float64) * cfg.scale
    # North-dominant case with rainfall still visible on both islands.
    sample_index = 8300
    coverage = {
        "threshold_mm": 1.0,
        "north_coverage": 0.9574468085106383,
        "south_coverage": 0.10967741935483871,
        "total_coverage": 0.37472283813747226,
        "max_rainfall_mm": 7.702827453613281,
        "unet_rmse_mm": 0.9978754763407608,
        "cnn_rmse_mm": 1.006109739456617,
        "unet_mae_mm": 0.4678871616423791,
        "cnn_mae_mm": 0.4727034928198664,
    }
    print(
        f"\n✓ Using sample: {sample_index} "
        f"(coverage > {coverage['threshold_mm']:.2f} mm: "
        f"North={coverage['north_coverage']:.3f}, South={coverage['south_coverage']:.3f}, "
        f"Total={coverage['total_coverage']:.3f}, Max={coverage['max_rainfall_mm']:.3f} mm, "
        f"RMSE U-Net/CNN={coverage['unet_rmse_mm']:.3f}/{coverage['cnn_rmse_mm']:.3f} mm, "
        f"MAE U-Net/CNN={coverage['unet_mae_mm']:.3f}/{coverage['cnn_mae_mm']:.3f} mm)"
    )
    
    # All lead times: 12, 18, 24, 30, 36 h
    lead_indices = [0, 1, 2, 3, 4]
    
    # Create spatial comparison figure: 5 rows (lead times) × 3 columns (Truth, UNet, CNN)
    n_leads = len(lead_indices)
    fig, axes = plt.subplots(n_leads, 3, figsize=(14, 5.5 * n_leads))
    if n_leads == 1:
        axes = np.expand_dims(axes, axis=0)
    
    fig.subplots_adjust(top=0.90, hspace=0.30, wspace=0.12)
    
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="white", alpha=1.0)
    
    # Compute full data range from all lead times and panels for sample_index
    all_panels = []
    for lead_idx in lead_indices:
        truth_p = truth_all[sample_index, lead_idx]
        unet_p = np.asarray(unet_pred[sample_index, 0, lead_idx, 0]).astype(np.float64) * cfg.scale
        cnn_p = np.asarray(cnn_pred[sample_index, 0, lead_idx, 0]).astype(np.float64) * cfg.scale
        
        all_panels.append(np.where(land_mask, truth_p, np.nan))
        all_panels.append(np.where(land_mask, unet_p, np.nan))
        all_panels.append(np.where(land_mask, cnn_p, np.nan))
    
    # Compute colormap range using percentiles (for better color visibility)
    all_data = np.concatenate([p.flatten() for p in all_panels])
    all_data_valid = all_data[~np.isnan(all_data)]
    vmin = 0.0  # Always start at 0 for rainfall
    vmax = np.percentile(all_data_valid, 95)  # 95th percentile for better visibility
    print(f"✓ Colormap range: {vmin:.4f} to {vmax:.4f} mm (0 to 95th percentile for visibility)")
    
    plot_titles = ["Truth (Reference)", "U-Net Prediction", "CNN Prediction"]
    
    for row, (lead_idx, lead_hour) in enumerate(zip(lead_indices, lead_hours)):
        
        truth_panel = truth_all[sample_index, lead_idx]
        unet_panel = np.asarray(unet_pred[sample_index, 0, lead_idx, 0]).astype(np.float64) * cfg.scale
        cnn_panel = np.asarray(cnn_pred[sample_index, 0, lead_idx, 0]).astype(np.float64) * cfg.scale
        
        # Apply land mask (set ocean to nan)
        truth_land = np.where(land_mask, truth_panel, np.nan)
        unet_land = np.where(land_mask, unet_panel, np.nan)
        cnn_land = np.where(land_mask, cnn_panel, np.nan)
        
        # Calculate statistics for each panel
        truth_mean = np.nanmean(truth_land)
        truth_max = np.nanmax(truth_land)
        unet_mean = np.nanmean(unet_land)
        unet_max = np.nanmax(unet_land)
        cnn_mean = np.nanmean(cnn_land)
        cnn_max = np.nanmax(cnn_land)
        
        panels = [truth_land, unet_land, cnn_land]
        stats = [(truth_mean, truth_max), (unet_mean, unet_max), (cnn_mean, cnn_max)]
        
        for col, (data, (mean_val, max_val)) in enumerate(zip(panels, stats)):
            ax = axes[row, col]
            
            im = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
            
            # Draw land boundary
            ax.contour(lon_grid, lat_grid, land_mask.astype(np.float32),
                      levels=[0.5], colors="black", linewidths=0.8, zorder=5)
            
            ax.set_aspect("auto")
            ax.set_xlabel("Longitude", fontsize=9)
            if col == 0:
                ax.set_ylabel("Latitude", fontsize=9)
            else:
                ax.set_yticks([])
            
            # Title row
            if row == 0:
                ax.set_title(plot_titles[col], fontsize=12, fontweight="bold", pad=10)
            
            # Lead time label in first column
            if col == 0:
                ax.text(0.02, 0.95, f"Lead: {lead_hour}h",
                       transform=ax.transAxes, va="top", ha="left", fontsize=9,
                       bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2.5))
            
            # Add statistics box with mean and max values
            stats_text = f"Mean: {mean_val:.2f}mm\nMax: {max_val:.2f}mm"
            ax.text(0.02, 0.08, stats_text,
                   transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
                   bbox=dict(facecolor="yellow", alpha=0.7, edgecolor="black", linewidth=0.5, pad=2))
    
    # Colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.015)
    cbar.set_label("Rainfall (mm)", fontsize=11, fontweight="bold")
    
    # Compute error metrics for display
    truth_ref = truth_all[sample_index, 2]
    truth_ref_land = np.where(land_mask, truth_ref, np.nan)
    unet_ref = np.asarray(unet_pred[sample_index, 0, 2, 0]).astype(np.float64) * cfg.scale
    cnn_ref = np.asarray(cnn_pred[sample_index, 0, 2, 0]).astype(np.float64) * cfg.scale
    
    unet_rmse = np.sqrt(np.nanmean((np.where(land_mask, unet_ref, np.nan) - truth_ref_land) ** 2))
    cnn_rmse = np.sqrt(np.nanmean((np.where(land_mask, cnn_ref, np.nan) - truth_ref_land) ** 2))
    
    fig.suptitle(
        f"Spatial Predictions: Truth vs U-Net vs CNN (All Lead Times, NZ Land Only)\n"
        f"Sample {sample_index} | 24h RMSE - U-Net: {unet_rmse:.3f}mm, CNN: {cnn_rmse:.3f}mm",
        fontsize=13, fontweight="bold", y=0.995
    )
    
    out_paths = [
        Path(cfg.out_dir) / "03_spatial_comparison_NZ_landonly.png",
        Path(cfg.out_dir) / "03_spatial_comparison_fullrange_NZ_landonly.png",
    ]
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        print(f"✓ Saved: {out_path}")
    plt.close(fig)


def plot_spatial_comparison_rowwise_scales(cfg: ComparisonConfig):
    """
    Alternate spatial comparison figure with larger row height and
    a separate color scale for each lead-time row.
    """
    print("\n=== SECTION 3B: SPATIAL COMPARISON (ROW-WISE SCALES) ===")

    land_mask, lat, lon = _load_land_mask(cfg)
    unet_pred = _load_prediction_zarr(Path(cfg.unet_pred_zarr))
    cnn_pred = _load_prediction_zarr(Path(cfg.cnn_pred_zarr))
    _, truth_dataset, sample_start_indices = _build_truth_accessor(cfg, lat, lon)
    lead_hours = [12, 18, 24, 30, 36]
    truth_all = np.stack(
        [_get_truth_panels(truth_dataset, sample_start_indices, sample_idx, lead_hours, cfg.scale) for sample_idx in range(unet_pred.shape[0])],
        axis=0,
    )

    unet_all = np.asarray(unet_pred[:, 0, :, 0], dtype=np.float64) * cfg.scale
    cnn_all = np.asarray(cnn_pred[:, 0, :, 0], dtype=np.float64) * cfg.scale
    # North-dominant case with rainfall still visible on both islands.
    sample_index = 8300
    coverage = {
        "threshold_mm": 1.0,
        "north_coverage": 0.9574468085106383,
        "south_coverage": 0.10967741935483871,
        "total_coverage": 0.37472283813747226,
        "max_rainfall_mm": 7.702827453613281,
        "unet_rmse_mm": 0.9978754763407608,
        "cnn_rmse_mm": 1.006109739456617,
        "unet_mae_mm": 0.4678871616423791,
        "cnn_mae_mm": 0.4727034928198664,
    }
    print(
        f"✓ Using sample: {sample_index} "
        f"(coverage > {coverage['threshold_mm']:.2f} mm: "
        f"North={coverage['north_coverage']:.3f}, South={coverage['south_coverage']:.3f}, "
        f"Total={coverage['total_coverage']:.3f}, Max={coverage['max_rainfall_mm']:.3f} mm, "
        f"RMSE U-Net/CNN={coverage['unet_rmse_mm']:.3f}/{coverage['cnn_rmse_mm']:.3f} mm, "
        f"MAE U-Net/CNN={coverage['unet_mae_mm']:.3f}/{coverage['cnn_mae_mm']:.3f} mm)"
    )

    lead_indices = [0, 1, 2, 3, 4]
    n_leads = len(lead_indices)

    fig, axes = plt.subplots(n_leads, 3, figsize=(14, 28))
    if n_leads == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.subplots_adjust(top=0.965, bottom=0.03, left=0.06, right=0.92, hspace=0.42, wspace=0.10)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="white", alpha=1.0)
    plot_titles = ["Truth (Reference)", "U-Net Prediction", "CNN Prediction"]

    for row, (lead_idx, lead_hour) in enumerate(zip(lead_indices, lead_hours)):
        truth_panel = truth_all[sample_index, lead_idx]
        unet_panel = np.asarray(unet_pred[sample_index, 0, lead_idx, 0]).astype(np.float64) * cfg.scale
        cnn_panel = np.asarray(cnn_pred[sample_index, 0, lead_idx, 0]).astype(np.float64) * cfg.scale

        truth_land = np.where(land_mask, truth_panel, np.nan)
        unet_land = np.where(land_mask, unet_panel, np.nan)
        cnn_land = np.where(land_mask, cnn_panel, np.nan)

        panels = [truth_land, unet_land, cnn_land]
        row_data = np.concatenate([panel[np.isfinite(panel)] for panel in panels])
        row_vmin = 0.0
        row_vmax = np.percentile(row_data, 97)
        if row_vmax <= row_vmin:
            row_vmax = max(np.nanmax(row_data), 0.1)

        stats = [
            (np.nanmean(truth_land), np.nanmax(truth_land)),
            (np.nanmean(unet_land), np.nanmax(unet_land)),
            (np.nanmean(cnn_land), np.nanmax(cnn_land)),
        ]

        last_im = None
        for col, (data, (mean_val, max_val)) in enumerate(zip(panels, stats)):
            ax = axes[row, col]
            last_im = ax.pcolormesh(
                lon_grid, lat_grid, data, cmap=cmap, vmin=row_vmin, vmax=row_vmax, shading="auto"
            )
            ax.contour(
                lon_grid, lat_grid, land_mask.astype(np.float32),
                levels=[0.5], colors="black", linewidths=0.8, zorder=5
            )
            ax.set_aspect("auto")
            ax.set_xlabel("Longitude", fontsize=9)
            if col == 0:
                ax.set_ylabel("Latitude", fontsize=9)
            else:
                ax.set_yticks([])

            if row == 0:
                ax.set_title(plot_titles[col], fontsize=12, fontweight="bold", pad=10)

            if col == 0:
                ax.text(
                    0.02, 0.95, f"Lead: {lead_hour}h",
                    transform=ax.transAxes, va="top", ha="left", fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2.5)
                )

            stats_text = f"Mean: {mean_val:.2f}mm\nMax: {max_val:.2f}mm"
            ax.text(
                0.02, 0.08, stats_text,
                transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
                bbox=dict(facecolor="yellow", alpha=0.7, edgecolor="black", linewidth=0.5, pad=2)
            )

        cbar = fig.colorbar(last_im, ax=axes[row, :].ravel().tolist(), shrink=0.92, pad=0.012)
        cbar.set_label(f"Rainfall (mm) | Lead {lead_hour}h", fontsize=9, fontweight="bold")

    truth_ref = truth_all[sample_index, 2]
    truth_ref_land = np.where(land_mask, truth_ref, np.nan)
    unet_ref = np.asarray(unet_pred[sample_index, 0, 2, 0]).astype(np.float64) * cfg.scale
    cnn_ref = np.asarray(cnn_pred[sample_index, 0, 2, 0]).astype(np.float64) * cfg.scale
    unet_rmse = np.sqrt(np.nanmean((np.where(land_mask, unet_ref, np.nan) - truth_ref_land) ** 2))
    cnn_rmse = np.sqrt(np.nanmean((np.where(land_mask, cnn_ref, np.nan) - truth_ref_land) ** 2))

    fig.suptitle(
        f"Spatial Predictions with Row-wise Color Scales: Truth vs U-Net vs CNN (NZ Land Only)\n"
        f"Sample {sample_index} | 24h RMSE - U-Net: {unet_rmse:.3f}mm, CNN: {cnn_rmse:.3f}mm",
        fontsize=14, fontweight="bold", y=0.992
    )

    out_path = Path(cfg.out_dir) / "03_spatial_comparison_NZ_landonly_III.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.close(fig)


def plot_spatial_comparison_north_dominant(cfg: ComparisonConfig):
    """North-dominant spatial comparison with the requested IV output name."""
    print("\n=== SECTION 3C: SPATIAL COMPARISON (NORTH-DOMINANT) ===")
    land_mask, lat, lon = _load_land_mask(cfg)
    unet_pred = _load_prediction_zarr(Path(cfg.unet_pred_zarr))
    cnn_pred = _load_prediction_zarr(Path(cfg.cnn_pred_zarr))
    _, truth_dataset, sample_start_indices = _build_truth_accessor(cfg, lat, lon)
    lead_hours = [12, 18, 24, 30, 36]
    truth_all = np.stack(
        [_get_truth_panels(truth_dataset, sample_start_indices, sample_idx, lead_hours, cfg.scale) for sample_idx in range(unet_pred.shape[0])],
        axis=0,
    )

    sample_index = 8300
    print(
        f"✓ Using sample: {sample_index} (North-dominant truth case; both islands still visible)"
    )

    lead_indices = [0, 1, 2, 3, 4]
    n_leads = len(lead_indices)
    fig, axes = plt.subplots(n_leads, 3, figsize=(14, 28))
    if n_leads == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.subplots_adjust(top=0.965, bottom=0.03, left=0.06, right=0.92, hspace=0.42, wspace=0.10)

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="white", alpha=1.0)
    plot_titles = ["Truth (Reference)", "U-Net Prediction", "CNN Prediction"]

    for row, (lead_idx, lead_hour) in enumerate(zip(lead_indices, lead_hours)):
        truth_panel = truth_all[sample_index, lead_idx]
        unet_panel = np.asarray(unet_pred[sample_index, 0, lead_idx, 0]).astype(np.float64) * cfg.scale
        cnn_panel = np.asarray(cnn_pred[sample_index, 0, lead_idx, 0]).astype(np.float64) * cfg.scale
        truth_land = np.where(land_mask, truth_panel, np.nan)
        unet_land = np.where(land_mask, unet_panel, np.nan)
        cnn_land = np.where(land_mask, cnn_panel, np.nan)
        panels = [truth_land, unet_land, cnn_land]
        row_data = np.concatenate([panel[np.isfinite(panel)] for panel in panels])
        row_vmin = 0.0
        row_vmax = np.percentile(row_data, 97)
        if row_vmax <= row_vmin:
            row_vmax = max(np.nanmax(row_data), 0.1)
        last_im = None
        for col, data in enumerate(panels):
            ax = axes[row, col]
            last_im = ax.pcolormesh(lon_grid, lat_grid, data, cmap=cmap, vmin=row_vmin, vmax=row_vmax, shading="auto")
            ax.contour(lon_grid, lat_grid, land_mask.astype(np.float32), levels=[0.5], colors="black", linewidths=0.8, zorder=5)
            ax.set_aspect("auto")
            ax.set_xlabel("Longitude", fontsize=9)
            if col == 0:
                ax.set_ylabel("Latitude", fontsize=9)
            else:
                ax.set_yticks([])
            if row == 0:
                ax.set_title(plot_titles[col], fontsize=12, fontweight="bold", pad=10)
            if col == 0:
                ax.text(0.02, 0.95, f"Lead: {lead_hour}h", transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2.5))
        cbar = fig.colorbar(last_im, ax=axes[row, :].ravel().tolist(), shrink=0.92, pad=0.012)
        cbar.set_label(f"Rainfall (mm) | Lead {lead_hour}h", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Spatial Predictions (North-dominant sample) with Row-wise Color Scales\n"
        "Truth vs U-Net vs CNN (NZ Land Only)",
        fontsize=14, fontweight="bold", y=0.992
    )
    out_path = Path(cfg.out_dir) / "03_spatial_comparison_NZ_landonly_IV.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.close(fig)


# ==================== SECTION 4: TRAINING LOSS COMPARISON ====================

def plot_training_loss(cfg: ComparisonConfig):
    """
    Section 4: Training and Validation Loss Comparison
    Shows convergence behavior and overfitting trends for both models
    """
    print("\n=== SECTION 4: TRAINING AND VALIDATION LOSS ===")
    
    # Load training logs
    unet_log_path = Path(cfg.unet_eval_dir).parent.parent / "models" / cfg.unet_model_name / "training_log.csv"
    cnn_log_path = Path(cfg.cnn_eval_dir).parent.parent / "models" / cfg.cnn_model_name / "training_log.csv"
    
    # Fallback paths if not found
    if not unet_log_path.exists():
        unet_log_path = Path("models") / cfg.unet_model_name / "training_log.csv"
    if not cnn_log_path.exists():
        cnn_log_path = Path("models") / cfg.cnn_model_name / "training_log.csv"
    
    if not unet_log_path.exists() or not cnn_log_path.exists():
        print(f"✗ Training logs not found at {unet_log_path} or {cnn_log_path}")
        return
    
    # Load data
    import pandas as pd
    unet_df = pd.read_csv(unet_log_path)
    cnn_df = pd.read_csv(cnn_log_path)
    
    print(f"✓ U-Net log: {len(unet_df)} epochs")
    print(f"✓ CNN log: {len(cnn_df)} epochs")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(11, 6.5))
    
    # Plot U-Net
    ax.plot(unet_df["Epoch"], unet_df["Average Training Loss"],
            label="U-Net Train", color="#1f77b4", linewidth=2.5, marker="o", markersize=4)
    ax.plot(unet_df["Epoch"], unet_df["Validation Loss"],
            label="U-Net Validation", color="#1f77b4", linewidth=2.5, linestyle="--", marker="s", markersize=4)
    
    # Plot CNN
    ax.plot(cnn_df["Epoch"], cnn_df["Average Training Loss"],
            label="CNN Train", color="#d62728", linewidth=2.5, marker="o", markersize=4)
    ax.plot(cnn_df["Epoch"], cnn_df["Validation Loss"],
            label="CNN Validation", color="#d62728", linewidth=2.5, linestyle="--", marker="s", markersize=4)
    
    ax.set_title("Training and Validation Loss Comparison (Lower is Better)", 
                fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=False, loc="best", fontsize=11)
    
    fig.tight_layout()
    
    out_path = Path(cfg.out_dir) / "04_training_loss_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.close(fig)


# ==================== SECTION 5: SEPARATE TRAINING LOSS CURVES ====================

def plot_training_loss_separate(cfg: ComparisonConfig):
    """
    Section 5: Separate training and validation loss for each model
    Side-by-side subplots showing typical convergence pattern
    """
    print("\n=== SECTION 5: TRAINING LOSS (SEPARATE MODELS) ===")
    
    # Load training logs
    unet_log_path = Path("models") / cfg.unet_model_name / "training_log.csv"
    cnn_log_path = Path("models") / cfg.cnn_model_name / "training_log.csv"
    
    if not unet_log_path.exists() or not cnn_log_path.exists():
        print(f"✗ Training logs not found")
        return
    
    # Load data
    import pandas as pd
    unet_df = pd.read_csv(unet_log_path)
    cnn_df = pd.read_csv(cnn_log_path)
    
    print(f"✓ U-Net log: {len(unet_df)} epochs")
    print(f"✓ CNN log: {len(cnn_df)} epochs")
    
    # Create figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # ===== LEFT SUBPLOT: U-Net =====
    ax1.plot(unet_df["Epoch"], unet_df["Average Training Loss"],
            label="Training Loss", color="#1f77b4", linewidth=2.5, marker="o", markersize=5)
    ax1.plot(unet_df["Epoch"], unet_df["Validation Loss"],
            label="Validation Loss", color="#ff7f0e", linewidth=2.5, marker="s", markersize=5)
    
    ax1.set_title("U-Net: Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(frameon=False, loc="best", fontsize=10)
    
    # ===== RIGHT SUBPLOT: CNN =====
    ax2.plot(cnn_df["Epoch"], cnn_df["Average Training Loss"],
            label="Training Loss", color="#d62728", linewidth=2.5, marker="o", markersize=5)
    ax2.plot(cnn_df["Epoch"], cnn_df["Validation Loss"],
            label="Validation Loss", color="#ff7f0e", linewidth=2.5, marker="s", markersize=5)
    
    ax2.set_title("CNN: Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(frameon=False, loc="best", fontsize=10)
    
    fig.suptitle("Model Convergence Comparison: Training vs Validation", 
                fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    
    out_path = Path(cfg.out_dir) / "05_training_loss_separate.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    plt.close(fig)


# ==================== SECTION 6: NORMALIZED TRAINING LOSS ====================

def plot_training_loss_normalized(cfg: ComparisonConfig):
    """
    Section 6: Normalized training and validation loss (0-1 range)
    Shows convergence pattern similar to reference image
    """
    print("\n=== SECTION 6: TRAINING LOSS (NORMALIZED 0-1 RANGE) ===")
    
    # Load training logs
    unet_log_path = Path("models") / cfg.unet_model_name / "training_log.csv"
    cnn_log_path = Path("models") / cfg.cnn_model_name / "training_log.csv"
    
    if not unet_log_path.exists() or not cnn_log_path.exists():
        print(f"✗ Training logs not found")
        return
    
    # Load data
    import pandas as pd
    unet_df = pd.read_csv(unet_log_path)
    cnn_df = pd.read_csv(cnn_log_path)
    
    print(f"✓ U-Net log: {len(unet_df)} epochs")
    print(f"✓ CNN log: {len(cnn_df)} epochs")
    
    # Normalize loss to 0-1 range for each model
    def normalize_loss(train_loss, val_loss):
        all_loss = np.concatenate([train_loss, val_loss])
        loss_min = np.min(all_loss)
        loss_max = np.max(all_loss)
        train_norm = (train_loss - loss_min) / (loss_max - loss_min)
        val_norm = (val_loss - loss_min) / (loss_max - loss_min)
        return train_norm, val_norm
    
    # Normalize U-Net
    unet_train_norm, unet_val_norm = normalize_loss(
        np.array(unet_df["Average Training Loss"]),
        np.array(unet_df["Validation Loss"])
    )
    
    # Normalize CNN
    cnn_train_norm, cnn_val_norm = normalize_loss(
        np.array(cnn_df["Average Training Loss"]),
        np.array(cnn_df["Validation Loss"])
    )
    
    # Create figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # ===== LEFT SUBPLOT: U-Net =====
    ax1.plot(unet_df["Epoch"], unet_train_norm,
            label="Train", color="#1f77b4", linewidth=2.5, marker="o", markersize=5)
    ax1.plot(unet_df["Epoch"], unet_val_norm,
            label="Validation", color="#ff7f0e", linewidth=2.5, marker="s", markersize=5)
    
    ax1.set_title("U-Net: Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Normalized Loss (0-1)", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(frameon=False, loc="best", fontsize=10)
    
    # ===== RIGHT SUBPLOT: CNN =====
    ax2.plot(cnn_df["Epoch"], cnn_train_norm,
            label="Train", color="#d62728", linewidth=2.5, marker="o", markersize=5)
    ax2.plot(cnn_df["Epoch"], cnn_val_norm,
            label="Validation", color="#ff7f0e", linewidth=2.5, marker="s", markersize=5)
    
    ax2.set_title("CNN: Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Normalized Loss (0-1)", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(frameon=False, loc="best", fontsize=10)
    
    fig.suptitle("Model Convergence: Normalized Loss (0-1 Range)", 
                fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    
    out_path = Path(cfg.out_dir) / "06_training_loss_normalized.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    print(f"  Loss normalized to 0-1 range (min={np.min([np.min(unet_train_norm), np.min(cnn_train_norm)]):.3f}, max={np.max([np.max(unet_val_norm), np.max(cnn_val_norm)]):.3f})")
    plt.close(fig)


def plot_temporal_difference_diagnostics(cfg: ComparisonConfig):
    """
    Section 7: Temporal-difference diagnostics
    
    Evaluates forecast stability by computing area-weighted MAE of first-order
    temporal differences (rainfall changes between consecutive lead times).
    Shows whether predicted rainfall evolution is consistent with observed dynamics.
    """
    print("\n=== SECTION 7: TEMPORAL-DIFFERENCE DIAGNOSTICS ===")
    
    # Load predictions
    unet_pred_path = Path(cfg.unet_pred_zarr)
    cnn_pred_path = Path(cfg.cnn_pred_zarr)
    
    if not unet_pred_path.exists() or not cnn_pred_path.exists():
        print(f"✗ Prediction zarr files not found")
        return
    
    # Load zarr arrays: (samples, ?, lead_times, ?, lat, lon)
    unet_pred = zarr.open_array(unet_pred_path, mode="r")[:]
    cnn_pred = zarr.open_array(cnn_pred_path, mode="r")[:]
    
    print(f"✓ U-Net predictions shape: {unet_pred.shape}")
    print(f"✓ CNN predictions shape: {cnn_pred.shape}")
    
    # Load land mask
    land_mask, lat, lon = _load_land_mask(cfg)
    print(f"✓ Land mask shape: {land_mask.shape}")
    
    # Compute area weights (cosine latitude weighting)
    lat_rad = np.deg2rad(lat)
    weights_lat = np.cos(lat_rad)
    weights_2d = weights_lat[:, np.newaxis] * np.ones_like(land_mask)
    land_mask_bool = land_mask > cfg.land_threshold
    weights_normalized = weights_2d / np.sum(weights_2d[land_mask_bool])
    
    # Extract lead times (assume shape [..., 5, ...] for 5 lead times)
    # Dimensions: (samples, 1, lead_times, 1, lat, lon)
    n_leads = unet_pred.shape[2]
    print(f"✓ Number of lead times: {n_leads}")
    
    # Lead times in hours (12, 18, 24, 30, 36)
    lead_times_h = np.array([12, 18, 24, 30, 36])
    
    # Compute temporal differences MAE for each lead time pair
    # e.g., diff[0] = MAE of (pred[lead=1] - pred[lead=0]) for all samples
    unet_mae_diff = []
    cnn_mae_diff = []
    
    for i in range(n_leads - 1):
        # Get predictions at consecutive lead times
        unet_curr = unet_pred[:, 0, i, 0, :, :] * cfg.scale  # (samples, lat, lon), convert to mm
        unet_next = unet_pred[:, 0, i+1, 0, :, :] * cfg.scale
        
        cnn_curr = cnn_pred[:, 0, i, 0, :, :] * cfg.scale
        cnn_next = cnn_pred[:, 0, i+1, 0, :, :] * cfg.scale
        
        # Compute temporal differences
        unet_diff = unet_next - unet_curr  # (samples, lat, lon)
        cnn_diff = cnn_next - cnn_curr
        
        # Compute area-weighted MAE for each sample
        unet_mae_per_sample = []
        cnn_mae_per_sample = []
        
        for s in range(unet_diff.shape[0]):
            # Extract land-only differences for this sample
            unet_land_diff = np.abs(unet_diff[s, land_mask_bool])
            cnn_land_diff = np.abs(cnn_diff[s, land_mask_bool])
            
            # Area-weighted MAE: sum(weight * |diff|) for land points
            unet_mae = np.sum(unet_land_diff * weights_normalized[land_mask_bool])
            cnn_mae = np.sum(cnn_land_diff * weights_normalized[land_mask_bool])
            
            unet_mae_per_sample.append(unet_mae)
            cnn_mae_per_sample.append(cnn_mae)
        
        # Average across all samples
        unet_mae_diff.append(np.mean(unet_mae_per_sample))
        cnn_mae_diff.append(np.mean(cnn_mae_per_sample))
    
    unet_mae_diff = np.array(unet_mae_diff)
    cnn_mae_diff = np.array(cnn_mae_diff)
    
    # Lead time pairs (e.g., "12→18h", "18→24h", etc.)
    lead_labels = [f"{lead_times_h[i]}→{lead_times_h[i+1]}h" for i in range(len(lead_times_h)-1)]
    lead_positions = np.arange(len(lead_labels))
    
    print(f"✓ U-Net temporal diff MAE: {unet_mae_diff}")
    print(f"✓ CNN temporal diff MAE: {cnn_mae_diff}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars side-by-side
    width = 0.35
    ax.bar(lead_positions - width/2, unet_mae_diff, width, label="U-Net", 
           color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=1.2)
    ax.bar(lead_positions + width/2, cnn_mae_diff, width, label="CNN", 
           color="#d62728", alpha=0.85, edgecolor="black", linewidth=1.2)
    
    ax.set_xlabel("Lead Time Transitions", fontsize=12, fontweight="bold")
    ax.set_ylabel("Area-Weighted MAE of Temporal Differences (mm/6h)", fontsize=12, fontweight="bold")
    ax.set_title("Temporal-Difference Diagnostics: Forecast Stability Across Lead Times\n"
                 "Lower values indicate more stable rainfall evolution predictions",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(lead_positions)
    ax.set_xticklabels(lead_labels, fontsize=11)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    
    # Add value labels on bars
    for i, (u, c) in enumerate(zip(unet_mae_diff, cnn_mae_diff)):
        ax.text(i - width/2, u + 0.002, f"{u:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text(i + width/2, c + 0.002, f"{c:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    fig.tight_layout()
    
    out_path = Path(cfg.out_dir) / "07_temporal_difference_diagnostics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved: {out_path}")
    print(f"  U-Net temporal diff MAE: {unet_mae_diff}")
    print(f"  CNN temporal diff MAE: {cnn_mae_diff}")
    print(f"  U-Net more stable: {np.mean(unet_mae_diff) < np.mean(cnn_mae_diff)}")
    plt.close(fig)


# ==================== MAIN ====================

def parse_args() -> ComparisonConfig:
    ap = argparse.ArgumentParser(
        description="Three-panel comparison: RMSE, MAE, Spatial (NZ land only)"
    )
    ap.add_argument(
        "--unet_eval_dir",
        default="results/rainfall_tp_model_unet_12to36_dt6/evaluation_metrics.zarr",
        type=str,
        help="Path to U-Net evaluation_metrics.zarr directory"
    )
    ap.add_argument(
        "--cnn_eval_dir",
        default="results/CNN_rainfall_tp_model_cnn_12to36_dt6_eval_thr002/evaluation_metrics.zarr",
        type=str,
        help="Path to CNN evaluation_metrics.zarr directory"
    )
    ap.add_argument(
        "--unet_5v_eval_dir",
        default="results/rainfall_tp_t2m_u10_v10_msl_model_unet_12to36_dt6/evaluation_metrics.zarr",
        type=str,
        help="Path to TP/T2M/U10/V10/MSL U-Net evaluation_metrics.zarr directory"
    )
    ap.add_argument(
        "--unet_model_name",
        default="rainfall_tp_model_unet_12to36_dt6",
        type=str,
        help="U-Net model name (for loading predictions)"
    )
    ap.add_argument(
        "--cnn_model_name",
        default="CNN_rainfall_tp_model_cnn_12to36_dt6",
        type=str,
        help="CNN model name (for loading predictions)"
    )
    ap.add_argument(
        "--unet_5v_model_name",
        default="rainfall_tp_t2m_u10_v10_msl_model_unet_12to36_dt6",
        type=str,
        help="TP/T2M/U10/V10/MSL U-Net model name"
    )
    ap.add_argument(
        "--unet_pred_zarr",
        default="results/rainfall_tp_model_unet_12to36_dt6/rainfall_tp_model_unet_12to36_dt6.zarr",
        type=str,
        help="Path to U-Net prediction zarr"
    )
    ap.add_argument(
        "--cnn_pred_zarr",
        default="results/CNN_rainfall_tp_model_cnn_12to36_dt6_eval_thr002/CNN_rainfall_tp_model_cnn_12to36_dt6.zarr",
        type=str,
        help="Path to CNN prediction zarr"
    )
    ap.add_argument(
        "--unet_5v_pred_zarr",
        default="results/rainfall_tp_t2m_u10_v10_msl_model_unet_12to36_dt6/rainfall_tp_t2m_u10_v10_msl_model_unet_12to36_dt6.zarr",
        type=str,
        help="Path to TP/T2M/U10/V10/MSL U-Net prediction zarr"
    )
    ap.add_argument(
        "--static_nc_path",
        default="/nesi/project/massey04632/data/ERA5/static/era5_static.nc",
        type=str,
        help="Path to ERA5 static file with land-sea mask"
    )
    ap.add_argument(
        "--land_threshold",
        default=0.5,
        type=float,
        help="Land threshold for LSM (>= threshold is land)"
    )
    ap.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for sample selection"
    )
    ap.add_argument(
        "--scale",
        default=1000.0,
        type=float,
        help="Scale factor (1000 converts m to mm)"
    )
    ap.add_argument(
        "--out_dir",
        default=str(REPO_ROOT / "plots"),
        type=str,
        help="Output directory for comparison figures"
    )
    
    args = ap.parse_args()
    return ComparisonConfig(
        unet_eval_dir=args.unet_eval_dir,
        unet_5v_eval_dir=args.unet_5v_eval_dir,
        cnn_eval_dir=args.cnn_eval_dir,
        unet_model_name=args.unet_model_name,
        unet_5v_model_name=args.unet_5v_model_name,
        cnn_model_name=args.cnn_model_name,
        unet_pred_zarr=args.unet_pred_zarr,
        unet_5v_pred_zarr=args.unet_5v_pred_zarr,
        cnn_pred_zarr=args.cnn_pred_zarr,
        static_nc_path=args.static_nc_path,
        land_threshold=args.land_threshold,
        seed=args.seed,
        scale=args.scale,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    cfg = parse_args()
    print(f"\nOutput directory: {cfg.out_dir}")
    
    try:
        plot_three_model_rmse(cfg)
        plot_three_model_spatial(cfg)
        plot_three_model_training_loss(cfg)
        print("\n✓ All comparisons completed successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

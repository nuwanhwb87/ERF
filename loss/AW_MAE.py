"""Compute area-weighted MAE from saved UNet/CNN predictions and run Wilcoxon tests.

This script does not retrain models. It uses saved prediction zarr files and
reconstructs truth directly from the test split in the ERA5 data file.

Outputs are saved separately inside each model result folder:
- MAE_per_sample.csv
- MAE_leadwise.csv
- MAE_per_sample.npy
- MAE_wilcoxon_vs_CNN.csv / .json (in UNet folder)
- MAE_wilcoxon_UNet_vs_CNN.csv / .json (in CNN folder)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_time_index(start_datetime: str, end_datetime: str, time_freq: str) -> pd.DatetimeIndex:
    start = pd.to_datetime(start_datetime)
    end = pd.to_datetime(end_datetime)
    try:
        return pd.date_range(start, end, freq=time_freq, inclusive="both")
    except TypeError:
        return pd.date_range(start, end, freq=time_freq)


def split_time_index(
    ti: pd.DatetimeIndex,
    split_mode: str,
    train_year_end=None,
    val_year_end=None,
    train_until=None,
    val_until=None,
):
    mode = str(split_mode).lower()
    if mode == "years":
        if train_year_end is None or val_year_end is None:
            raise ValueError("years split requires train_year_end and val_year_end")
        train_mask = ti.year <= int(train_year_end)
        val_mask = (ti.year > int(train_year_end)) & (ti.year <= int(val_year_end))
    elif mode == "dates":
        if train_until is None or val_until is None:
            raise ValueError("dates split requires train_until and val_until")
        tr_end = pd.to_datetime(train_until)
        va_end = pd.to_datetime(val_until)
        train_mask = ti <= tr_end
        val_mask = (ti > tr_end) & (ti <= va_end)
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    return len(ti), int(train_mask.sum()), int(val_mask.sum())


def resolve_dataset_file(data_directory: Path, variable_names: list[str], suffix: str) -> Path:
    prefix = "_".join(variable_names)
    candidates = [
        data_directory / f"{prefix}_{suffix}.npy",
        data_directory / f"{prefix}_{suffix}_5.625deg.npy",
    ]
    for p in candidates:
        if p.exists():
            return p

    matches = sorted(data_directory.glob(f"{prefix}_{suffix}*.npy"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No dataset file found for prefix '{prefix}_{suffix}' in {data_directory}")
    raise FileNotFoundError(f"Multiple dataset files found for prefix '{prefix}_{suffix}': {matches}")


def resolve_latlon_file(data_directory: Path, suffix: str) -> Path:
    candidates = [
        data_directory / f"latlon_{suffix}.npz",
        data_directory / f"latlon_{suffix}_5.625deg.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No lat/lon file found for suffix '{suffix}' in {data_directory}")


def compute_area_weights(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(lat)).reshape(-1, 1)
    w = np.repeat(w, lon.size, axis=1)
    w *= (lat.size * lon.size / np.sum(w))
    return w.astype(np.float32)


def load_land_mask(lsm_path: Path, h: int, w: int) -> np.ndarray:
    if not lsm_path.exists():
        raise FileNotFoundError(f"Land-sea mask not found: {lsm_path}")
    lsm = np.load(lsm_path).astype(np.float32)
    if lsm.ndim == 3 and lsm.shape[0] == 1:
        lsm = lsm[0]
    if lsm.ndim != 2:
        raise ValueError(f"Expected 2D lsm array, got shape {lsm.shape}")
    if lsm.shape != (h, w):
        raise ValueError(f"LSM shape {lsm.shape} does not match grid {(h, w)}")
    return (lsm > 0.5).astype(np.float32)


def build_test_indices(cfg: dict, max_horizon: int, spacing: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, int]:
    data_directory = Path(cfg["data_directory"])
    variable_names = cfg["variable_names"]
    num_variables = int(cfg["num_variables"])

    suffix = f"{cfg['start_datetime'][:4]}-{cfg['end_datetime'][:4]}"
    latlon = np.load(resolve_latlon_file(data_directory, suffix))
    lat = np.asarray(latlon["lat"], dtype=np.float64)
    lon = np.asarray(latlon["lon"], dtype=np.float64)

    ti = make_time_index(cfg["start_datetime"], cfg["end_datetime"], cfg["time_freq"])
    n_samples, n_train, n_val = split_time_index(
        ti,
        split_mode=cfg["split_mode"],
        train_year_end=cfg.get("train_year_end"),
        val_year_end=cfg.get("val_year_end"),
        train_until=cfg.get("train_until"),
        val_until=cfg.get("val_until"),
    )

    spinup = 24
    start = spinup + n_train + n_val
    stop = n_samples
    index_array = np.arange(start, stop - max_horizon)[::spacing]

    dataset_file = resolve_dataset_file(data_directory, variable_names, suffix)
    return index_array, lat, lon, dataset_file, num_variables


def compute_mae_per_sample(
    pred_path: Path,
    dataset_file: Path,
    num_samples_total: int,
    num_variables: int,
    h: int,
    w: int,
    sample_start_indices: np.ndarray,
    lead_times_h: np.ndarray,
    weights: np.ndarray,
    land_mask: np.ndarray | None,
) -> np.ndarray:
    pred = zarr.open_array(pred_path, mode="r")
    # expected pred shape: (n_test, n_ens, n_times, n_vars, H, W)
    if pred.shape[-2] != h or pred.shape[-1] != w:
        raise ValueError(f"Prediction grid mismatch for {pred_path}: {pred.shape[-2:]} vs {(h, w)}")

    mmap = np.memmap(
        str(dataset_file),
        dtype=np.float32,
        mode="r",
        shape=(num_samples_total, num_variables, h, w),
    )

    n_test = len(sample_start_indices)
    n_times = len(lead_times_h)
    out = np.zeros((n_test, n_times, num_variables), dtype=np.float32)

    for i, sidx in enumerate(sample_start_indices):
        y_index = sidx + lead_times_h
        truth = mmap[y_index, :, :, :]  # (T, V, H, W)

        ens_mean = np.asarray(pred[i], dtype=np.float32).mean(axis=0)  # (T, V, H, W)

        if land_mask is not None:
            truth = truth * land_mask[None, None, :, :]
            ens_mean = ens_mean * land_mask[None, None, :, :]

        abs_diff = np.abs(ens_mean - truth)  # (T, V, H, W)
        # Weighted mean over H,W; mimic training/eval behavior with full-grid mean.
        weighted = abs_diff * weights[None, None, :, :]
        out[i] = weighted.mean(axis=(-1, -2))

    return out


def save_mae_files(result_dir: Path, lead_times_h: np.ndarray, mae_per_sample: np.ndarray):
    result_dir.mkdir(parents=True, exist_ok=True)

    per_sample_csv = result_dir / "MAE_per_sample.csv"
    with per_sample_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "lead_time_h", "variable_index", "area_weighted_mae"])
        n_samples, n_times, n_vars = mae_per_sample.shape
        for s in range(n_samples):
            for t in range(n_times):
                for v in range(n_vars):
                    writer.writerow([s, int(lead_times_h[t]), v, float(mae_per_sample[s, t, v])])

    leadwise = mae_per_sample.mean(axis=(0, 2))
    leadwise_csv = result_dir / "MAE_leadwise.csv"
    with leadwise_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lead_time_h", "mean_area_weighted_mae"])
        for t, lead in enumerate(lead_times_h):
            writer.writerow([int(lead), float(leadwise[t])])

    np.save(result_dir / "MAE_per_sample.npy", mae_per_sample)
    return per_sample_csv, leadwise_csv


def run_wilcoxon(unet_mae: np.ndarray, cnn_mae: np.ndarray, lead_times_h: np.ndarray, alpha: float):
    rows = []
    for t in range(unet_mae.shape[1]):
        u = unet_mae[:, t, 0]
        c = cnn_mae[:, t, 0]
        stat, p = wilcoxon(u, c, alternative="two-sided")
        significant = bool(p < alpha)
        if significant and np.median(u) < np.median(c):
            result = "+"
        elif significant and np.median(u) > np.median(c):
            result = "-"
        else:
            result = "~"

        rows.append(
            {
                "lead_time_h": int(lead_times_h[t]),
                "comparison_model": "CNN Baseline",
                "result": result,
                "p_value": float(p),
                "significant": significant,
                "unet_median_mae": float(np.median(u)),
                "cnn_median_mae": float(np.median(c)),
                "improvement_percent": float(100.0 * (1.0 - (np.median(u) / np.median(c)))),
                "n_samples": int(u.shape[0]),
            }
        )

    return rows


def save_wilcoxon(result_dir: Path, rows: list[dict], file_stem: str):
    csv_path = result_dir / f"{file_stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lead_time_h",
                "comparison_model",
                "result",
                "p_value",
                "significant",
                "unet_median_mae",
                "cnn_median_mae",
                "improvement_percent",
                "n_samples",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = result_dir / f"{file_stem}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute area-weighted MAE and Wilcoxon test from saved UNet/CNN predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--unet-result-dir",
        type=Path,
        default=REPO_ROOT / "results" / "rainfall_tp_model_unet_12to36_dt6",
    )
    parser.add_argument(
        "--cnn-result-dir",
        type=Path,
        default=REPO_ROOT / "results" / "CNN_rainfall_tp_model_cnn_12to36_dt6_eval_thr002",
    )
    parser.add_argument(
        "--unet-model-config",
        type=Path,
        default=REPO_ROOT / "models" / "rainfall_tp_model_unet_12to36_dt6" / "config.json",
    )
    parser.add_argument("--t-min", type=int, default=12)
    parser.add_argument("--t-max", type=int, default=36)
    parser.add_argument("--t-direct", type=int, default=6)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()

    with args.unet_model_config.open("r", encoding="utf-8") as f:
        unet_cfg = json.load(f)

    lead_times_h = args.t_min + args.t_direct * np.arange(0, 1 + (args.t_max - args.t_min) // args.t_direct)

    sample_start_indices, lat, lon, dataset_file, num_variables = build_test_indices(
        cfg=unet_cfg,
        max_horizon=args.t_max,
        spacing=int(unet_cfg.get("spacing", 1)),
    )

    weights = compute_area_weights(lat, lon)

    land_mask = None
    if bool(unet_cfg.get("land_only", False)):
        land_mask = load_land_mask(Path(unet_cfg["lsm_path"]), len(lat), len(lon))

    # Use time-index length as total sample count for memmap shape.
    ti = make_time_index(unet_cfg["start_datetime"], unet_cfg["end_datetime"], unet_cfg["time_freq"])
    num_samples_total = len(ti)

    unet_pred_path = args.unet_result_dir / "rainfall_tp_model_unet_12to36_dt6.zarr"
    cnn_pred_path = args.cnn_result_dir / "CNN_rainfall_tp_model_cnn_12to36_dt6.zarr"

    if not unet_pred_path.exists():
        raise FileNotFoundError(f"UNet prediction zarr not found: {unet_pred_path}")
    if not cnn_pred_path.exists():
        raise FileNotFoundError(f"CNN prediction zarr not found: {cnn_pred_path}")

    unet_mae = compute_mae_per_sample(
        pred_path=unet_pred_path,
        dataset_file=dataset_file,
        num_samples_total=num_samples_total,
        num_variables=num_variables,
        h=len(lat),
        w=len(lon),
        sample_start_indices=sample_start_indices,
        lead_times_h=lead_times_h,
        weights=weights,
        land_mask=land_mask,
    )

    cnn_mae = compute_mae_per_sample(
        pred_path=cnn_pred_path,
        dataset_file=dataset_file,
        num_samples_total=num_samples_total,
        num_variables=num_variables,
        h=len(lat),
        w=len(lon),
        sample_start_indices=sample_start_indices,
        lead_times_h=lead_times_h,
        weights=weights,
        land_mask=land_mask,
    )

    if unet_mae.shape != cnn_mae.shape:
        raise ValueError(f"Shape mismatch: UNet {unet_mae.shape} vs CNN {cnn_mae.shape}")

    unet_saved = save_mae_files(args.unet_result_dir, lead_times_h, unet_mae)
    cnn_saved = save_mae_files(args.cnn_result_dir, lead_times_h, cnn_mae)

    rows = run_wilcoxon(unet_mae, cnn_mae, lead_times_h, args.alpha)
    unet_w = save_wilcoxon(args.unet_result_dir, rows, "MAE_wilcoxon_vs_CNN")
    cnn_w = save_wilcoxon(args.cnn_result_dir, rows, "MAE_wilcoxon_UNet_vs_CNN")

    print("Saved UNet MAE outputs:")
    for p in unet_saved + unet_w:
        print(f"  - {p}")

    print("Saved CNN MAE outputs:")
    for p in cnn_saved + cnn_w:
        print(f"  - {p}")

    print("Wilcoxon summary (UNet vs CNN, per-sample MAE):")
    for row in rows:
        print(
            f"  Lead {row['lead_time_h']} h: p={row['p_value']:.3e}, "
            f"result={row['result']}, significant={row['significant']}"
        )


if __name__ == "__main__":
    main()

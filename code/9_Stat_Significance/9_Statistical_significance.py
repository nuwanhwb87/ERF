"""Pairwise TP significance tests for CNN 1V, U-Net 1V, and U-Net 5V."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr
import zarr


REPO_ROOT = Path(__file__).resolve().parents[2]
LEAD_TIMES_H = np.array([12, 18, 24, 30, 36], dtype=int)


def load_land_mask(static_path: Path, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return an ascending-latitude land mask and its latitude coordinates."""
    with xr.open_dataset(static_path) as dataset:
        name = "lsm" if "lsm" in dataset else "land_sea_mask"
        if name not in dataset:
            raise ValueError("Static file has no land-sea mask.")
        mask = np.asarray(dataset[name].values).squeeze()
        latitude = np.asarray(dataset["latitude"].values, dtype=np.float64)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D land mask, got {mask.shape}.")
    if latitude[0] > latitude[-1]:
        latitude, mask = latitude[::-1], mask[::-1, :]
    return mask >= threshold, latitude


def load_truth_source(model_name: str, grid_shape: tuple[int, int]) -> tuple[np.memmap, np.ndarray]:
    """Load TP truth according to the saved U-Net 1V training/test split."""
    config_path = REPO_ROOT / "models" / model_name / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    time_index = pd.date_range(config["start_datetime"], config["end_datetime"], freq=config["time_freq"])
    train_until, val_until = pd.Timestamp(config["train_until"]), pd.Timestamp(config["val_until"])
    test_start = 24 + (time_index <= train_until).sum() + ((time_index > train_until) & (time_index <= val_until)).sum()
    starts = np.arange(test_start, len(time_index) - 36)[::int(config.get("spacing", 1))]
    variables = config["variable_names"]
    file_name = f"{'_'.join(variables)}_{config['start_datetime'][:4]}-{config['end_datetime'][:4]}.npy"
    data_path = Path(config["data_directory"]) / file_name
    data = np.memmap(data_path, dtype=np.float32, mode="r", shape=(len(time_index), config["num_variables"], *grid_shape))
    return data, starts


def compute_per_sample_errors(predictions: zarr.Array, truth: np.memmap, starts: np.ndarray, mask: np.ndarray, latitude: np.ndarray, scale: float) -> dict:
    """Compute TP area-weighted RMSE and MAE for each sample and forecast lead."""
    if predictions.ndim != 6 or predictions.shape[3] < 1:
        raise ValueError(f"Unexpected prediction shape: {predictions.shape}.")
    if predictions.shape[0] > len(starts) or tuple(predictions.shape[-2:]) != mask.shape:
        raise ValueError("Prediction samples or grid do not match the TP truth source.")
    weights = np.cos(np.deg2rad(latitude))[:, None] * mask
    weights /= weights.sum()
    errors = {}
    for lead_index, lead_hour in enumerate(LEAD_TIMES_H):
        forecast = np.asarray(predictions[:, 0, lead_index, 0], dtype=np.float64) * scale
        observed = np.asarray(truth[starts[:len(forecast)] + lead_hour, 0], dtype=np.float64) * scale
        difference = forecast - observed
        errors[lead_index] = {
            "rmse": np.sqrt(np.sum(difference ** 2 * weights, axis=(1, 2))),
            "mae": np.sum(np.abs(difference) * weights, axis=(1, 2)),
        }
    return errors


def run_wilcoxon_tests(errors_by_model: dict[str, dict]) -> pd.DataFrame:
    """Run paired, two-sided Wilcoxon tests for all model pairs."""
    rows = []
    for model_a, model_b in combinations(errors_by_model, 2):
        for lead_index, lead_hour in enumerate(LEAD_TIMES_H):
            for metric in ("rmse", "mae"):
                errors_a = errors_by_model[model_a][lead_index][metric]
                errors_b = errors_by_model[model_b][lead_index][metric]
                statistic, p_value = stats.wilcoxon(errors_a, errors_b, alternative="two-sided")
                median_a, median_b = np.median(errors_a), np.median(errors_b)
                rows.append({"model_a": model_a, "model_b": model_b, "lead_time_h": lead_hour, "metric": metric.upper(), "wilcoxon_statistic": statistic, "p_value": p_value, "significant_0.05": p_value < 0.05, "median_a": median_a, "median_b": median_b, "median_difference_a_minus_b": median_a - median_b})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired TP significance tests for CNN 1V, U-Net 1V, and U-Net 5V.")
    parser.add_argument("--unet_1v_pred_zarr", type=Path, default=REPO_ROOT / "results/rainfall_tp_model_unet_12to36_dt6/rainfall_tp_model_unet_12to36_dt6.zarr")
    parser.add_argument("--unet_5v_pred_zarr", type=Path, default=REPO_ROOT / "results/rainfall_tp_t2m_u10_v10_msl_model_unet_12to36_dt6/rainfall_tp_t2m_u10_v10_msl_model_unet_12to36_dt6.zarr")
    parser.add_argument("--cnn_1v_pred_zarr", type=Path, default=REPO_ROOT / "results/CNN_rainfall_tp_model_cnn_12to36_dt6_eval_thr002/CNN_rainfall_tp_model_cnn_12to36_dt6.zarr")
    parser.add_argument("--truth_model_name", default="rainfall_tp_model_unet_12to36_dt6")
    parser.add_argument("--static_nc_path", type=Path, default=Path("/nesi/project/massey04632/data/ERA5/static/era5_static.nc"))
    parser.add_argument("--out_csv", type=Path, default=REPO_ROOT / "plots/04_Comparison_statistical_significance_three_models.csv")
    parser.add_argument("--land_threshold", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=1000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    land_mask, latitude = load_land_mask(args.static_nc_path, args.land_threshold)
    truth, starts = load_truth_source(args.truth_model_name, land_mask.shape)
    paths = {"CNN 1V": args.cnn_1v_pred_zarr, "U-Net 1V": args.unet_1v_pred_zarr, "U-Net 5V": args.unet_5v_pred_zarr}
    errors = {name: compute_per_sample_errors(zarr.open_array(path, mode="r"), truth, starts, land_mask, latitude, args.scale) for name, path in paths.items()}
    results = run_wilcoxon_tests(errors)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_csv, index=False)
    print(results.to_string(index=False))
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
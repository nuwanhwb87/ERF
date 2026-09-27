#!/bin/bash -e

# JOB CONFIGURATION

#SBATCH --job-name=rainfall_train          # Job name
#SBATCH --account=massey04632            # Your project account
#SBATCH --partition=milan                # GPU partition with A100 support
#SBATCH --time=08:00:00                  # 8 hours for full 50-epoch training
#SBATCH --cpus-per-task=8                # CPU cores
#SBATCH --gpus-per-node=A100:1           # Request 1 typed A100 GPU
#SBATCH --mem=32GB                       # Memory

#SBATCH --output=ERROR_OUTPUT/%x_%j.out # Std output 
#SBATCH --error=ERROR_OUTPUT/%x_%j.err # Std error

########################
# ENVIRONMENT SETUP
########################

module --force purge                      # Clean environment (avoid conflicts)
module load NeSI              # load parent module
module load Miniconda3/4.12.0 # load Miniconda

# Enable conda
# source $(conda info --base)/etc/profile.d/conda.sh

# Activate your environment
source /opt/nesi/CS400_centos7_bdw/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate /nesi/project/massey04632/my-conda-env

echo "Conda env: $CONDA_PREFIX"
which python
python -V

echo "Running on node: $(hostname)"

########################
# RUN YOUR SCRIPTS
########################

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    # Fallback for manual local execution
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# If submitted from the slurm subfolder, move one level up to repository root.
if [[ ! -f "$PROJECT_ROOT/code/6_train_UNet/6_train_UNet_2V.py" && -f "$PROJECT_ROOT/../code/6_train_UNet/6_train_UNet_2V.py" ]]; then
    PROJECT_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
fi

cd "$PROJECT_ROOT"         # Move to project root for this repo checkout

if [[ ! -f "code/6_train_UNet/6_train_UNet_2V.py" ]]; then
    echo "ERROR: two-variable U-Net training script not found under $PROJECT_ROOT"
    exit 2
fi

# Training run (deterministic U-Net) for 12h-36h lead times, 6h steps.
METHOD=unet
RUN_NAME="rainfall_tp_t2m_model_${METHOD}_12to36_dt6"
RESULT_DIR="$PROJECT_ROOT/models/${RUN_NAME}"
SLURM_LOG_DIR="$PROJECT_ROOT/slurm/ERROR_OUTPUT"
TRAIN_SCRIPT="code/6_train_UNet/6_train_UNet_2V.py"
mkdir -p "$SLURM_LOG_DIR"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: ${TRAIN_SCRIPT} not found under $PROJECT_ROOT"
    exit 2
fi

# Live GPU monitor (1 sample/min): logs GPU utilization and memory usage while training runs.
GPU_MON_LOG="$SLURM_LOG_DIR/gpu_usage_${SLURM_JOB_ID}.csv"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "timestamp,name,utilization.gpu,memory.total(MiB),memory.used(MiB),memory.free(MiB)" > "$GPU_MON_LOG"
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits -l 60 >> "$GPU_MON_LOG" &
    GPU_MON_PID=$!
    echo "Started GPU monitor (pid=${GPU_MON_PID}), log=${GPU_MON_LOG}"
fi

if /nesi/project/massey04632/my-conda-env/bin/python "$TRAIN_SCRIPT" \
    --method $METHOD \
    --model deterministic \
    --epochs 50 \
    --spacing 1 \
    --batch_size 128 \
    --save_every 5 \
    --name "$RUN_NAME" \
    --t_min 12 \
    --t_max 36 \
    --delta_t 6 \
    --conditioning_times 0,-6 \
    --start_datetime 2015-01-01T00:00:00 \
    --end_datetime 2025-12-31T00:00:00 \
    --time_freq 1h \
    --split_mode dates \
    --train_until 2023-12-31T00:00:00 \
    --val_until 2024-12-31T00:00:00 \
    --land_only true \
    --lsm_path /nesi/project/massey04632/data/ERA5/static/lsm.npy \
    --variable_names total_precipitation,2m_temperature \
    --num_variables 2 \
    --data_directory /nesi/project/massey04632/data/ERA5/L2/2015_2025_2V \
    --result_directory "$PROJECT_ROOT/models"; then
    TRAIN_EXIT_CODE=0
else
    TRAIN_EXIT_CODE=$?
fi

# Stop GPU monitor after training.
if [[ -n "${GPU_MON_PID:-}" ]]; then
    kill "$GPU_MON_PID" >/dev/null 2>&1 || true
    wait "$GPU_MON_PID" 2>/dev/null || true
fi

echo "===== TRAINING SUMMARY ====="
echo "Run name: $RUN_NAME"
echo "Result directory: $RESULT_DIR"
echo "Land-only mode: enabled (ocean masked by lsm.npy in training script)"

if [[ -f "$RESULT_DIR/training_log.csv" ]]; then
    /nesi/project/massey04632/my-conda-env/bin/python - <<PY
import csv
from pathlib import Path

result_dir = Path("$RESULT_DIR")
train_log = result_dir / "training_log.csv"
grad_log = result_dir / "gradient_log.csv"
best_model = result_dir / "best_model.pth"

rows = []
with train_log.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            rows.append({
                "epoch": int(float(r["Epoch"])),
                "train_loss": float(r["Average Training Loss"]),
                "val_loss": float(r["Validation Loss"]),
            })
        except Exception:
            continue

if rows:
    best = min(rows, key=lambda x: x["val_loss"])
    last = rows[-1]
    print(f"Best epoch: {best['epoch']}")
    print(f"Best validation loss: {best['val_loss']:.6f}")
    print(f"Training loss at best epoch: {best['train_loss']:.6f}")
    print(f"Last epoch logged: {last['epoch']}")
    print(f"Last train/val loss: {last['train_loss']:.6f} / {last['val_loss']:.6f}")
else:
    print("No rows found in training_log.csv")

if grad_log.exists():
    with grad_log.open() as f:
        reader = csv.DictReader(f)
        first = next(reader, None)
        if first and "Total Params" in first:
            print(f"Model parameters (from gradient log): {first['Total Params']}")

if best_model.exists():
    size_mb = best_model.stat().st_size / (1024 * 1024)
    print(f"Best model file: {best_model}")
    print(f"Best model size: {size_mb:.2f} MiB")
else:
    print("best_model.pth not found")
PY
else
    echo "training_log.csv not found in $RESULT_DIR"
fi

if [[ -f "$GPU_MON_LOG" ]]; then
    echo "GPU usage log: $GPU_MON_LOG"
    echo "Last 5 GPU usage samples:"
    tail -n 5 "$GPU_MON_LOG" || true
fi

echo "Training exit code: $TRAIN_EXIT_CODE"
echo "============================"

if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
    exit $TRAIN_EXIT_CODE
fi

#echo "======================================"
#echo "JOB FINISHED"
#echo "======================================"
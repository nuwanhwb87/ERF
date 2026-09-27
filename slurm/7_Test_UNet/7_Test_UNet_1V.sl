#!/bin/bash -e

#SBATCH --job-name=rainfall_unet_test
#SBATCH --account=massey04632
#SBATCH --partition=milan
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
##SBATCH --gpus-per-node=A100:1
#SBATCH --mem=32GB
#SBATCH --output=ERROR_OUTPUT/%x_%j.out
#SBATCH --error=ERROR_OUTPUT/%x_%j.err

module --force purge
module load NeSI
module load GCC/12.3.0
module load Miniconda3/4.12.0

source /opt/nesi/CS400_centos7_bdw/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate /nesi/project/massey04632/my-conda-env

echo "Conda env: $CONDA_PREFIX"
which python
python -V
echo "Running on node: $(hostname)"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    # Fallback for manual local execution
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

if [[ ! -f "$PROJECT_ROOT/code/7_Test_UNet/7_Test_UNet_1V.py" && -f "$PROJECT_ROOT/../code/7_Test_UNet/7_Test_UNet_1V.py" ]]; then
    PROJECT_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
fi

cd "$PROJECT_ROOT"         # Move to project root for this repo checkout

TEST_SCRIPT="code/7_Test_UNet/7_Test_UNet_1V.py"
RUN_NAME="rainfall_tp_model_unet_12to36_dt6"
DATA_DIRECTORY="/nesi/project/massey04632/data/ERA5/L2/2015_2025_1V"
MODEL_DIRECTORY="$PROJECT_ROOT/models"
RESULT_DIRECTORY="$PROJECT_ROOT/results"

if [[ ! -f "$TEST_SCRIPT" ]]; then
    echo "ERROR: $TEST_SCRIPT not found under $PROJECT_ROOT"
    exit 2
fi

/nesi/project/massey04632/my-conda-env/bin/python "$TEST_SCRIPT" \
    --name "$RUN_NAME" \
    --model deterministic \
    --spacing 1 \
    --batch_size 1 \
    --t_min 12 \
    --t_max 36 \
    --t_iter 36 \
    --t_direct 6 \
    --n_ens 1 \
    --test_start 2024-12-31T01:00:00 \
    --test_end 2025-12-31T00:00:00 \
    --land_only true \
    --lsm_path /nesi/project/massey04632/data/ERA5/static/lsm.npy \
    --data_directory "$DATA_DIRECTORY" \
    --model_directory "$MODEL_DIRECTORY" \
    --result_directory "$RESULT_DIRECTORY"

echo "===== GPU INFORMATION ====="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not available on this node/partition"
fi
echo "==========================="

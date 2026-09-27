#!/bin/bash
set -euo pipefail

# JOB CONFIGURATION

#SBATCH --job-name=era5_prepare_residual   # Job name
#SBATCH --account=massey04632            # Your project account
#SBATCH --partition=milan                # GPU partition with A100 support
#SBATCH --time=08:00:00                  # 8 hours for full 50-epoch training
#SBATCH --cpus-per-task=8                # CPU cores
#SBATCH --gpus-per-node=A100:1           # Request 1 typed A100 GPU
#SBATCH --mem=32GB                       # Memory

#SBATCH --output=/home/nuwan39kyi4/00_nesi_projects/massey04632/ERF/slurm/ERROR_OUTPUT/%x_%j.out # Std output
#SBATCH --error=/home/nuwan39kyi4/00_nesi_projects/massey04632/ERF/slurm/ERROR_OUTPUT/%x_%j.err # Std error

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


########################
# RUN YOUR SCRIPTS
########################

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

if [[ ! -f "${PROJECT_ROOT}/code/4_prepare_residual.py" && -f "${PROJECT_ROOT}/../code/4_prepare_residual.py" ]]; then
    PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
fi

cd "${PROJECT_ROOT}"

if [[ ! -f "code/4_prepare_residual.py" ]]; then
    echo "ERROR: code/4_prepare_residual.py not found under ${PROJECT_ROOT}"
    exit 2
fi

# Run residual preparation
/nesi/project/massey04632/my-conda-env/bin/python code/4_prepare_residual.py \
    --file_directory /nesi/project/massey04632/data/ERA5/L1 \
    --save_directory /nesi/project/massey04632/data/ERA5/L2/2015_2025_2V \
    --chunk_size 50 \
    --start_datetime 2015-01-01T00:00:00 \
    --end_datetime 2025-12-31T00:00:00 \
    --time_freq 1d \
    --split_mode dates \
    --train_until 2023-12-31T00:00:00 \
    --val_until 2024-12-31T00:00:00

echo "Residual preparation completed!"


#!/bin/bash
#SBATCH --job-name=era5_prepare_dataset    # Job name
#SBATCH --account=massey04632            # Your project account
#SBATCH --partition=milan                # GPU partition with A100 support
#SBATCH --time=16:00:00                  # Time for full five-variable dataset preparation
#SBATCH --cpus-per-task=8                # CPU cores
#SBATCH --gpus-per-node=A100:1           # Request 1 typed A100 GPU
#SBATCH --mem=32GB                       # Memory
#SBATCH --output=slurm/ERROR_OUTPUT/%x_%j.out # Std output
#SBATCH --error=slurm/ERROR_OUTPUT/%x_%j.err # Std error

set -euo pipefail

# JOB CONFIGURATION

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

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/nesi/project/massey04632/ERF}"

cd "${PROJECT_ROOT}"

if [[ ! -f "code/3_Prepare_dataset_from_zarr/3_Prepare_dataset_from_zarr_5V.py" ]]; then
    echo "ERROR: five-variable dataset script not found under ${PROJECT_ROOT}"
    exit 2
fi

# Run dataset preparation
/nesi/project/massey04632/my-conda-env/bin/python code/3_Prepare_dataset_from_zarr/3_Prepare_dataset_from_zarr_5V.py \
    --file_directory /nesi/project/massey04632/data/ERA5/L1 \
    --save_directory /nesi/project/massey04632/data/ERA5/L2/2015_2025_5V \
    --folders "['total_precipitation','2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind','mean_sea_level_pressure']" \
    --long_names "['total_precipitation','2m_temperature','10m_u_component_of_wind','10m_v_component_of_wind','mean_sea_level_pressure']" \
    --short_names "['tp','t2m','u10','v10','msl']" \
    --num_variables 5 \
    --chunk_size 720 \
    --start_datetime 2015-01-01T00:00:00 \
    --end_datetime 2025-12-31T00:00:00 \
    --time_freq 1h \
    --split_mode dates \
    --train_until 2023-12-31T00:00:00 \
    --val_until 2024-12-31T00:00:00

echo "Dataset creation completed!"


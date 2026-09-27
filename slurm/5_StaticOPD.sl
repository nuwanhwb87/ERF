#!/bin/bash
set -euo pipefail

# JOB CONFIGURATION

#SBATCH --job-name=era5_static_opd         # Job name
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

if [[ ! -f "${PROJECT_ROOT}/Code/5_StaticOPD.py" && -f "${PROJECT_ROOT}/../Code/5_StaticOPD.py" ]]; then
    PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
fi

cd "${PROJECT_ROOT}"

if [[ ! -f "Code/5_StaticOPD.py" ]]; then
    echo "ERROR: Code/5_StaticOPD.py not found under ${PROJECT_ROOT}"
    exit 2
fi

# Run static variable download/extract
/nesi/project/massey04632/my-conda-env/bin/python Code/5_StaticOPD.py \
    --output_nc /nesi/project/massey04632/data/ERA5/static/era5_static.nc \
    --extract_dir /nesi/project/massey04632/data/ERA5/static \
    --year 2015 \
    --month 01 \
    --day 01 \
    --time 00:00 \
    --whole_region False

echo "Static variable preparation completed!"


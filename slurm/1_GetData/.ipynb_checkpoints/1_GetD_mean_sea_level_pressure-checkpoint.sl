#!/bin/bash
#SBATCH --job-name=era5_mean_sea_level_pressure
#SBATCH --account=massey04632
#SBATCH --partition=milan
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
##SBATCH --gpus-per-node=A100:1
#SBATCH --mem=32GB
#SBATCH --array=2015-2025%2
#SBATCH --output=/home/nuwan39kyi4/00_nesi_projects/massey04632/ERF/slurm/ERROR_OUTPUT/%x_%j.out
#SBATCH --error=/home/nuwan39kyi4/00_nesi_projects/massey04632/ERF/slurm/ERROR_OUTPUT/%x_%j.err

set -euo pipefail

module --force purge
module load NeSI
module load Miniconda3/4.12.0

source /opt/nesi/CS400_centos7_bdw/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate /nesi/project/massey04632/my-conda-env

echo "Node: $(hostname)"
echo "CWD before setup: $(pwd)"
echo "Conda env: ${CONDA_PREFIX}"
python -V

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

if [[ ! -f "${PROJECT_ROOT}/Code/1_get_L0_data_from_CDS/1_get_L0_2m_temperature.py" && -f "${PROJECT_ROOT}/../Code/1_get_L0_data_from_CDS/1_get_L0_2m_temperature.py" ]]; then
    PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
fi

cd "${PROJECT_ROOT}"
echo "Project root: ${PROJECT_ROOT}"

if [[ ! -f "Code/1_get_L0_data_from_CDS/1_get_L0_mean_sea_level_pressure.py" ]]; then
    echo "ERROR: mean_sea_level_pressure downloader not found under ${PROJECT_ROOT}"
    exit 2
fi

if [[ -n "${START_YM:-}" && -n "${END_YM:-}" ]]; then
    # Explicit override wins (useful for ad-hoc retries).
    :
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    TASK_YEAR="${SLURM_ARRAY_TASK_ID}"
    START_YM="${TASK_YEAR}01"
    END_YM="${TASK_YEAR}12"
else
    # Fallback when running without SLURM array.
    START_YM="201501"
    END_YM="202512"
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"
echo "Download window: START_YM=${START_YM}, END_YM=${END_YM}"

python Code/1_get_L0_data_from_CDS/1_get_L0_mean_sea_level_pressure.py \
    --var_name mean_sea_level_pressure \
    --start_ym "${START_YM}" \
    --end_ym "${END_YM}" \
    --L0_dir /nesi/project/massey04632/data/ERA5/L0_CDS \
    --whole_region false

echo "ERA5 L0 download job finished."
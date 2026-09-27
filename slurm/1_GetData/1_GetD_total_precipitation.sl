#!/bin/bash
set -euo pipefail

# SLURM resources and logging
#SBATCH --job-name=era5_l1_from_l0
#SBATCH --account=massey04632
#SBATCH --partition=milan
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=A100:1
#SBATCH --mem=32GB
#SBATCH --array=0-371%2
#SBATCH --output=/home/nuwan39kyi4/00_nesi_projects/massey04632/ERF/slurm/ERROR_OUTPUT/%x_%j.out
#SBATCH --error=/home/nuwan39kyi4/00_nesi_projects/massey04632/ERF/slurm/ERROR_OUTPUT/%x_%j.err

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

# If submitted from slurm/, move to repository root.
if [[ ! -f "${PROJECT_ROOT}/Code/1_get_L0_data_from_CDS.py" && -f "${PROJECT_ROOT}/../Code/1_get_L0_data_from_CDS.py" ]]; then
    PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
fi

cd "${PROJECT_ROOT}"
echo "Project root: ${PROJECT_ROOT}"

if [[ ! -f "Code/1_get_L0_data_from_CDS.py" ]]; then
    echo "ERROR: Code/1_get_L0_data_from_CDS.py not found under ${PROJECT_ROOT}"
    exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# NOTE: CDS download is network/IO bound.
# Default behavior uses a yearly SLURM array (one year per task),
# which avoids long single jobs hitting walltime.
#
# Array submit (default in this script):
#   sbatch slurm/1_GetData.sl
#
# Optional manual one-off range submit:
#   sbatch --array=2020-2020 --export=ALL,START_YM=202001,END_YM=202012 slurm/1_GetData.sl

if [[ -n "${START_YM:-}" && -n "${END_YM:-}" ]]; then
    # Explicit override wins (useful for ad-hoc retries).
    :
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    START_YM="$(python - "${SLURM_ARRAY_TASK_ID}" <<'PY'
import sys

offset = int(sys.argv[1])
year = 1995 + offset // 12
month = offset % 12 + 1
print(f"{year}{month:02d}")
PY
)"
    END_YM="${START_YM}"
else
    # Fallback when running without SLURM array.
    START_YM="201501"
    END_YM="202512"
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-NA}"
echo "Download window: START_YM=${START_YM}, END_YM=${END_YM}"

if ! python Code/1_get_L0_data_from_CDS/1_get_L0_total_precipitation.py \
    --var_name total_precipitation \
    --start_ym "${START_YM}" \
    --end_ym "${END_YM}" \
    --L0_dir /nesi/project/massey04632/data/ERA5/L0_CDS \
    --whole_region false; then
    echo "Download failed; resubmitting month ${START_YM}."
    sbatch --array="${SLURM_ARRAY_TASK_ID}" "${BASH_SOURCE[0]}"
    exit 1
fi

echo "ERA5 L0 download job finished."
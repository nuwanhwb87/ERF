#!/bin/bash

# SLURM resources and logging
#SBATCH --job-name=era5_l1_from_l0
#SBATCH --account=massey04632
#SBATCH --partition=milan
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
##SBATCH --gpus-per-node=A100:1
#SBATCH --mem=64GB
#SBATCH --output=ERROR_OUTPUT/%x_%j.out
#SBATCH --error=ERROR_OUTPUT/%x_%j.err

set -euo pipefail

# Environment modules and conda env
module --force purge
module load NeSI
module load Miniconda3/4.12.0

source /opt/nesi/CS400_centos7_bdw/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate /nesi/project/massey04632/my-conda-env

# Resolve repository root for both sbatch and manual runs
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	PROJECT_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

# If submitted from slurm/, move to repository root.
if [[ ! -f "${PROJECT_ROOT}/code/2_get_L1_data_from_L0/2_get_L1_data_from_L0.py" && -f "${PROJECT_ROOT}/../code/2_get_L1_data_from_L0/2_get_L1_data_from_L0.py" ]]; then
	PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
fi

# Run from project root so relative paths work
cd "${PROJECT_ROOT}"

# Fail fast if the target script is missing
if [[ ! -f "code/2_get_L1_data_from_L0/2_get_L1_data_from_L0.py" ]]; then
	echo "ERROR: L1 conversion script not found under ${PROJECT_ROOT}"
	exit 2
fi

# Print basic GPU info if available on node
if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

# Convert L0 monthly NetCDFs into/update L1 zarr
python code/2_get_L1_data_from_L0/2_get_L1_data_from_L0.py \
	--var_name "['total_precipitation', '2m_temperature','10m_u_component_of_wind', '10m_v_component_of_wind','mean_sea_level_pressure']" \
	--rebuild_vars "['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure']" \
	--L0_dir /nesi/project/massey04632/data/ERA5/L0_CDS \
	--L1_dir /nesi/project/massey04632/data/ERA5/L1

echo "L1 aggregation completed."


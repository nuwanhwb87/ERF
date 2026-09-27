#!/bin/bash -e
#SBATCH --job-name=rainfall_comparison_stats
#SBATCH --account=massey04632
#SBATCH --partition=genoa
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=L4:1
#SBATCH --mem=16G
#SBATCH --output=/nesi/project/massey04632/ERF/ERROR_OUTPUT/%x_%j.out
#SBATCH --error=/nesi/project/massey04632/ERF/ERROR_OUTPUT/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/nesi/project/massey04632/ERF
PYTHON_BIN=/nesi/project/massey04632/my-conda-env/bin/python

cd "$PROJECT_ROOT"

module --force purge
module load NeSI
module load Miniconda3/4.12.0

source /opt/nesi/CS400_centos7_bdw/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate /nesi/project/massey04632/my-conda-env

echo "========================================"
echo "Rainfall comparison and statistical-significance pipeline"
echo "Root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"
echo "Start: $(date)"
echo "========================================"

mkdir -p "$PROJECT_ROOT/ERROR_OUTPUT" "$PROJECT_ROOT/plots"

"$PYTHON_BIN" code/8_comparison/8_Comparison_All.py
"$PYTHON_BIN" code/9_Stat_Significance/9_Statistical_significance.py

echo "========================================"
echo "Comparison and significance tests completed successfully"
echo "Outputs under: $PROJECT_ROOT/plots"
echo "End: $(date)"
echo "========================================"

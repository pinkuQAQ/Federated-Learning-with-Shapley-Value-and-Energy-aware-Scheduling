#!/bin/bash
#SBATCH --job-name=FLSV_sig_stats
#SBATCH --partition=p3
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_sig_stats_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_sig_stats_%j.err

set -eo pipefail

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found."
    exit 1
fi

CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/home/zhaozhanshan/ENTER/envs/flsv}
if [ -d "${CONDA_ENV_PATH}" ]; then
    conda activate "${CONDA_ENV_PATH}"
else
    conda activate "${CONDA_ENV_NAME:-flsv}"
fi

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/zhaozhanshan/FLSV}
RUN_TAG=${RUN_TAG:?RUN_TAG must match the significance array run}
cd "${PROJECT_ROOT}"

python src/analyze_significance.py \
    --new-root "save/significance/${RUN_TAG}" \
    --existing "ours=save/beta025_confirmatory_20260712_235312/core/main/ours" \
    --existing "fedavg=save/sv_supp/20260617_120831/main" \
    --existing "fedprox=save/sv_supp/20260629_212435/main" \
    --existing "oort=save/sv_supp/20260629_212435/main" \
    --existing "gca=save/sv_supp/20260617_120831/main" \
    --existing "fedmsv=save/fedmsv_baseline_20260720/main" \
    --expected-pairs 10 \
    --output "save/significance/${RUN_TAG}/statistics"

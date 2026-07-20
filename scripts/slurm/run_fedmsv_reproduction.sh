#!/bin/bash
#SBATCH --job-name=FLSV_fedmsv_repro
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=120:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_fedmsv_repro_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_fedmsv_repro_%j.err

set -eo pipefail

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

CONDA_ENV_NAME=${CONDA_ENV_NAME:-flsv}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/home/zhaozhanshan/ENTER/envs/flsv}
if [ -d "${CONDA_ENV_PATH}" ]; then
    conda activate "${CONDA_ENV_PATH}"
else
    conda activate "${CONDA_ENV_NAME}"
fi

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/zhaozhanshan/FLSV}
cd "${PROJECT_ROOT}/src"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/save"

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
SEEDS=(${SEEDS:-42 123 2024})
LOW_QUALITY_TYPE=${LOW_QUALITY_TYPE:-label_flip}
LOW_QUALITY_FRACTION=${LOW_QUALITY_FRACTION:-0.3}
MAX_PERMUTATIONS=${MAX_PERMUTATIONS:-0}
GPU_ID=${GPU_ID:-0}

# The paper does not report optimizer settings. These explicit assumptions are
# isolated here so they can be swept without changing the Fed-MSV algorithm.
LR=${LR:-0.01}
MOMENTUM=${MOMENTUM:-0.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}

for SEED in "${SEEDS[@]}"; do
    OUT="fedmsv_reproduction/${RUN_TAG}/${LOW_QUALITY_TYPE}_p${LOW_QUALITY_FRACTION}/seed${SEED}"
    python federated_main.py \
        --dataset cifar --model lenet5 \
        --epochs 200 --num_users 60 --num_selected 12 \
        --local_ep 5 --local_bs 20 --optimizer sgd \
        --lr "${LR}" --momentum "${MOMENTUM}" --weight_decay "${WEIGHT_DECAY}" \
        --dirichlet_alpha 0.1 --test_size 10000 --gpu "${GPU_ID}" --seed "${SEED}" \
        --no_shapley --selection_method fedmsv \
        --fedmsv_guided_prefix 4 --fedmsv_epsilon_a 0.01 \
        --fedmsv_epsilon_b 0.01 --fedmsv_epsilon_c 0.1 \
        --fedmsv_max_permutations "${MAX_PERMUTATIONS}" \
        --fedmsv_utility_source test --fedmsv_utility_samples 0 \
        --fedmsv_low_quality_type "${LOW_QUALITY_TYPE}" \
        --fedmsv_low_quality_fraction "${LOW_QUALITY_FRACTION}" \
        --fedmsv_free_rider_std 0.01 --fedmsv_noise_variance 0.05 \
        --fedmsv_label_flip_fraction 0.5 \
        --output_folder "${OUT}"
done

echo "Fed-MSV reproduction results: ${PROJECT_ROOT}/save/fedmsv_reproduction/${RUN_TAG}"

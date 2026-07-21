#!/bin/bash
#SBATCH --job-name=FLSV_gca_paper
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_gca_paper_%A_%a.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_gca_paper_%A_%a.err

set -euo pipefail

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
conda activate flsv
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/zhaozhanshan/FLSV}
cd "${PROJECT_ROOT}/src"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/save"

DATASET=${DATASET:-cifar}
MODEL=${MODEL:-cnn}
EPOCHS=${EPOCHS:-100}
NUM_USERS=${NUM_USERS:-100}
NUM_SELECTED=${NUM_SELECTED:-5}
LOCAL_EP=${LOCAL_EP:-2}
LOCAL_BS=${LOCAL_BS:-32}
LR=${LR:-0.01}
DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-0.1}
TEST_SIZE=${TEST_SIZE:-10000}
GPU_ID=${GPU_ID:-0}
if [ -z "${RUN_TAG:-}" ]; then
    RUN_TAG="job${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
fi

# Ten paired seeds used by the main significance experiment.
SEEDS=(${SEEDS:-7 21 42 77 123 888 1001 2024 3407 31415})
# Values explicitly evaluated in the source GCA paper.
LAMBDA_VALUES=(${LAMBDA_VALUES:-0.3 0.4 0.5 0.6 0.7})

DP_CLIP_NORM=${DP_CLIP_NORM:-1.0}
CHANNEL_SIGMA=${CHANNEL_SIGMA:-0.1}
COMMON_DP_ARGS="--dp_advanced --dp_noise_schedule constant --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0"
DP_ARGS="--privacy_mode central --dp_clip_norm ${DP_CLIP_NORM} ${COMMON_DP_ARGS} --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier ${CHANNEL_SIGMA}"

BASE_ARGS="--dataset ${DATASET} --model ${MODEL} --epochs ${EPOCHS} --num_users ${NUM_USERS} --num_selected ${NUM_SELECTED} --local_ep ${LOCAL_EP} --local_bs ${LOCAL_BS} --lr ${LR} --dirichlet_alpha ${DIRICHLET_ALPHA} --test_size ${TEST_SIZE} --gpu ${GPU_ID}"
ENERGY_ARGS="--use_energy --sigma_squared 1.0 --initial_energy 500.0 --energy_threshold 50.0"
GCA_ARGS="--no_shapley --selection_method gca --gca_mode paper --gca_rho_dsi 0.5 --gca_rho_csi 0.5"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-4}
TOTAL_RUNS=$((${#SEEDS[@]} * ${#LAMBDA_VALUES[@]}))

echo "========================================"
echo "Job: ${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Start: $(date)"
echo "GCA source-faithful digital-FL reproduction"
echo "rho_DSI=0.5, rho_CSI=0.5"
echo "lambda_E values: ${LAMBDA_VALUES[*]}"
echo "seeds: ${SEEDS[*]}"
echo "This task handles indices congruent to ${TASK_ID} modulo ${TASK_COUNT}."
echo "Total runs across the array: ${TOTAL_RUNS}"
echo "Output tag: ${RUN_TAG}"
echo "========================================"

run_index=0
completed=0
for LAMBDA_E in "${LAMBDA_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        if ((run_index % TASK_COUNT != TASK_ID)); then
            run_index=$((run_index + 1))
            continue
        fi

        OUT="gca_paper/${RUN_TAG}/lambdaE${LAMBDA_E}/seed${SEED}"
        echo ""
        echo "----------------------------------------"
        echo "Run ${run_index}/${TOTAL_RUNS}: lambda_E=${LAMBDA_E}, seed=${SEED}"
        echo "Output: ${PROJECT_ROOT}/save/${OUT}"
        echo "Start: $(date)"
        echo "----------------------------------------"

        python federated_main.py \
            ${BASE_ARGS} --seed "${SEED}" \
            ${GCA_ARGS} --gca_lambda_energy "${LAMBDA_E}" \
            ${ENERGY_ARGS} ${DP_ARGS} \
            --output_folder "${OUT}"

        completed=$((completed + 1))
        run_index=$((run_index + 1))
    done
done

echo ""
echo "========================================"
echo "Task ${TASK_ID} finished ${completed} runs at $(date)."
echo "Results: ${PROJECT_ROOT}/save/gca_paper/${RUN_TAG}"
echo "For the predeclared main-table baseline, use lambda_E=0.5."
echo "Other lambda_E values are sensitivity results."
echo "========================================"

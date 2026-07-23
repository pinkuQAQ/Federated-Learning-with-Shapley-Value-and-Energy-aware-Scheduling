#!/bin/bash
#SBATCH --job-name=FLSV_oort_profile
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_oort_profile_%A_%a.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_oort_profile_%A_%a.err

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

CONDA_ENV_NAME=${CONDA_ENV_NAME:-flsv}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/home/zhaozhanshan/ENTER/envs/flsv}
if [ -d "${CONDA_ENV_PATH}" ]; then
    echo "Activating conda environment by path: ${CONDA_ENV_PATH}"
    conda activate "${CONDA_ENV_PATH}"
else
    echo "Activating conda environment by name: ${CONDA_ENV_NAME}"
    conda activate "${CONDA_ENV_NAME}"
fi

ITT_STUB=${ITT_STUB:-/data/home/zhaozhanshan/lib/libittnotify_stub.so}
if [ -f "${ITT_STUB}" ]; then
    export LD_PRELOAD="${ITT_STUB}"
fi

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
MOMENTUM=${MOMENTUM:-0.5}
WEIGHT_DECAY=${WEIGHT_DECAY:-5e-4}
DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-0.1}
TEST_SIZE=${TEST_SIZE:-10000}
GPU_ID=${GPU_ID:-0}
CHANNEL_SIGMA=${CHANNEL_SIGMA:-0.1}
SKIP_EXISTING=${SKIP_EXISTING:-1}

if [ -z "${RUN_TAG:-}" ]; then
    RUN_TAG="job${SLURM_ARRAY_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
fi

SEEDS=(${SEEDS:-7 21 42 77 123 888 1001 2024 3407 31415})

BASE_ARGS=(
    --dataset "${DATASET}" --model "${MODEL}" --epochs "${EPOCHS}"
    --num_users "${NUM_USERS}" --num_selected "${NUM_SELECTED}"
    --local_ep "${LOCAL_EP}" --local_bs "${LOCAL_BS}"
    --optimizer sgd --lr "${LR}" --momentum "${MOMENTUM}"
    --weight_decay "${WEIGHT_DECAY}" --dirichlet_alpha "${DIRICHLET_ALPHA}"
    --test_size "${TEST_SIZE}" --gpu "${GPU_ID}"
)

# Full ten-seed profile-based Oort configuration. These values are fixed before
# observing this run: source-style profile duration, scale-adapted 100-round
# Pacer, and a blacklist threshold suitable for an average exposure of 5.
OORT_ARGS=(
    --no_shapley --selection_method oort
    --oort_epsilon 0.9 --oort_epsilon_decay 0.98 --oort_epsilon_min 0.2
    --oort_duration_proxy profile --oort_reward_cap_samples 640
    --oort_profile_sigma 0.5 --oort_profile_compute_weight 0.5
    --oort_pacer_window 10 --oort_pacer_delta 10 --oort_round_threshold 30
    --oort_straggler_penalty 2.0 --oort_cutoff_util 0.95
    --oort_clip_percentile 95 --oort_blacklist_rounds 20
    --oort_blacklist_max_fraction 0.3 --oort_sample_window 5
)

ENERGY_ARGS=(
    --use_energy --sigma_squared 1.0 --channel_model rayleigh
    --initial_energy 500.0 --energy_threshold 50.0
)

COMMON_DP=(
    --privacy_mode central --dp_clip_norm 1.0
    --dp_advanced --dp_noise_schedule constant --dp_adaptive_clip
    --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8
    --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0
    --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0
    --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-4}

echo "========================================"
echo "Job: ${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Start: $(date)"
echo "Oort profile-based full ten-seed rerun"
echo "Seeds: ${SEEDS[*]}"
echo "Worker ${TASK_ID} handles seed indices modulo ${TASK_COUNT}"
echo "Output tag: ${RUN_TAG}"
echo "========================================"

completed=0
for index in "${!SEEDS[@]}"; do
    if ((index % TASK_COUNT != TASK_ID)); then
        continue
    fi

    SEED=${SEEDS[${index}]}
    OUT="oort_profile_full/${RUN_TAG}/seed${SEED}"
    if [ "${SKIP_EXISTING}" = "1" ] && compgen -G "${PROJECT_ROOT}/save/${OUT}/*.pkl" >/dev/null; then
        echo "[skip] Existing result: save/${OUT}"
        continue
    fi

    echo ""
    echo "----------------------------------------"
    echo "Seed ${SEED} (${index}/${#SEEDS[@]})"
    echo "Output: ${PROJECT_ROOT}/save/${OUT}"
    echo "Start: $(date)"
    echo "----------------------------------------"

    python federated_main.py \
        "${BASE_ARGS[@]}" --seed "${SEED}" \
        "${OORT_ARGS[@]}" \
        "${ENERGY_ARGS[@]}" "${COMMON_DP[@]}" \
        --output_folder "${OUT}"

    completed=$((completed + 1))
done

echo ""
echo "========================================"
echo "Worker ${TASK_ID} finished ${completed} runs at $(date)."
echo "Results: ${PROJECT_ROOT}/save/oort_profile_full/${RUN_TAG}"
echo "All four array workers together produce the ten paired seeds."
echo "========================================"

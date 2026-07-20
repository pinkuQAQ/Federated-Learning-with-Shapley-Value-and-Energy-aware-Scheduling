#!/bin/bash
#SBATCH --job-name=FLSV_fedmsv_rerun
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_fedmsv_rerun_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_fedmsv_rerun_%j.err

set -eo pipefail

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Start: $(date)"
echo "Task: Fed-MSV baseline supplementary rerun"
echo "========================================"

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
    conda activate "${CONDA_ENV_PATH}"
else
    conda activate "${CONDA_ENV_NAME}"
fi

ITT_STUB=${ITT_STUB:-/data/home/zhaozhanshan/lib/libittnotify_stub.so}
if [ -f "${ITT_STUB}" ]; then
    export LD_PRELOAD="${ITT_STUB}"
fi

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/zhaozhanshan/FLSV}
cd "${PROJECT_ROOT}/src"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/save"

# Examples:
#   sbatch scripts/slurm/run_fedmsv_rerun.sh
#   RUN_MAIN=1 RUN_CROSS=0 RUN_ALPHA=0 sbatch scripts/slurm/run_fedmsv_rerun.sh
#   RUN_MAIN=0 RUN_CROSS=0 RUN_ALPHA=1 sbatch scripts/slurm/run_fedmsv_rerun.sh
#   SEEDS="123 2024" RUN_TAG=<existing_tag> sbatch scripts/slurm/run_fedmsv_rerun.sh

RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
RUN_MAIN=${RUN_MAIN:-1}
RUN_CROSS=${RUN_CROSS:-1}
RUN_ALPHA=${RUN_ALPHA:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

DATASET=${DATASET:-cifar}
CROSS_DATASETS=(${CROSS_DATASETS:-fmnist mnist})
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
SEEDS=(${SEEDS:-42 123 2024})
ALPHAS=(${ALPHAS:-0.1 0.25 0.5 1.0})

FEDMSV_PREFIX=${FEDMSV_PREFIX:-4}
FEDMSV_EPS_A=${FEDMSV_EPS_A:-0.01}
FEDMSV_EPS_B=${FEDMSV_EPS_B:-0.01}
FEDMSV_EPS_C=${FEDMSV_EPS_C:-0.1}
FEDMSV_MAX_PERMUTATIONS=${FEDMSV_MAX_PERMUTATIONS:-0}
FEDMSV_UTILITY_SAMPLES=${FEDMSV_UTILITY_SAMPLES:-1000}

DP_CLIP_NORM=${DP_CLIP_NORM:-1.0}
CHANNEL_SIGMA=${CHANNEL_SIGMA:-0.1}

BASE_ARGS="--model ${MODEL} --epochs ${EPOCHS} --num_users ${NUM_USERS} --num_selected ${NUM_SELECTED} --local_ep ${LOCAL_EP} --local_bs ${LOCAL_BS} --optimizer sgd --lr ${LR} --momentum ${MOMENTUM} --weight_decay ${WEIGHT_DECAY} --test_size ${TEST_SIZE} --gpu ${GPU_ID}"
FEDMSV_ARGS="--no_shapley --selection_method fedmsv --fedmsv_guided_prefix ${FEDMSV_PREFIX} --fedmsv_epsilon_a ${FEDMSV_EPS_A} --fedmsv_epsilon_b ${FEDMSV_EPS_B} --fedmsv_epsilon_c ${FEDMSV_EPS_C} --fedmsv_max_permutations ${FEDMSV_MAX_PERMUTATIONS} --fedmsv_utility_source validation --fedmsv_utility_samples ${FEDMSV_UTILITY_SAMPLES} --fedmsv_low_quality_type none --fedmsv_low_quality_fraction 0.0"
COMMON_DP_ARGS="--dp_advanced --dp_noise_schedule constant --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0"
DP_ARGS="--privacy_mode central --dp_clip_norm ${DP_CLIP_NORM} ${COMMON_DP_ARGS} --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier ${CHANNEL_SIGMA}"

echo ""
echo "Configuration"
echo "  RUN_TAG=${RUN_TAG}"
echo "  SEEDS=${SEEDS[*]}"
echo "  EPOCHS=${EPOCHS}, N=${NUM_USERS}, K=${NUM_SELECTED}"
echo "  RUN_MAIN=${RUN_MAIN}, RUN_CROSS=${RUN_CROSS}, RUN_ALPHA=${RUN_ALPHA}"
echo "  Main alpha=${DIRICHLET_ALPHA}, alpha sweep=${ALPHAS[*]}"
echo "  Fed-MSV: m=${FEDMSV_PREFIX}, eps=(${FEDMSV_EPS_A},${FEDMSV_EPS_B},${FEDMSV_EPS_C})"
echo "  Fed-MSV permutations=${FEDMSV_MAX_PERMUTATIONS} (0 = K!/(K-m)!)"
echo "  Utility=validation, samples=${FEDMSV_UTILITY_SAMPLES}"
echo "  SKIP_EXISTING=${SKIP_EXISTING}"
echo "========================================"

run_cmd() {
    local label=$1
    local output=$2
    shift 2

    if [ "${SKIP_EXISTING}" = "1" ] && compgen -G "${PROJECT_ROOT}/save/${output}/*.pkl" >/dev/null; then
        echo "[skip] ${label}: existing result found under save/${output}"
        return
    fi

    echo ""
    echo "----------------------------------------"
    echo "${label}"
    echo "Output: ${output}"
    echo "Start: $(date)"
    echo "----------------------------------------"
    python federated_main.py "$@" --output_folder "${output}"
    echo "Done: $(date)"
}

run_fedmsv() {
    local dataset=$1
    local alpha=$2
    local seed=$3
    local output=$4
    run_cmd "Fed-MSV dataset=${dataset}, alpha=${alpha}, seed=${seed}" "${output}" \
        ${BASE_ARGS} --dataset "${dataset}" --dirichlet_alpha "${alpha}" --seed "${seed}" \
        ${FEDMSV_ARGS} ${DP_ARGS}
}

if [ "${RUN_MAIN}" = "1" ]; then
    echo ""
    echo "========================================"
    echo "[Group 1] CIFAR-10 main-table baseline"
    echo "========================================"
    for SEED in "${SEEDS[@]}"; do
        run_fedmsv "${DATASET}" "${DIRICHLET_ALPHA}" "${SEED}" \
            "sv_supp/${RUN_TAG}/main/seed${SEED}"
    done
fi

if [ "${RUN_CROSS}" = "1" ]; then
    echo ""
    echo "========================================"
    echo "[Group 2] Cross-dataset main-table baselines"
    echo "========================================"
    for CROSS_DATASET in "${CROSS_DATASETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_fedmsv "${CROSS_DATASET}" "${DIRICHLET_ALPHA}" "${SEED}" \
                "cross_dataset/${RUN_TAG}/${CROSS_DATASET}/seed${SEED}"
        done
    done
fi

if [ "${RUN_ALPHA}" = "1" ]; then
    echo ""
    echo "========================================"
    echo "[Group 3] CIFAR-10 Dirichlet-alpha sensitivity"
    echo "========================================"
    for ALPHA in "${ALPHAS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            run_fedmsv "cifar" "${ALPHA}" "${SEED}" \
                "sensitivity_multiseed/${RUN_TAG}/alpha/alpha${ALPHA}/seed${SEED}/fedmsv"
        done
    done
fi

cd "${PROJECT_ROOT}"

if [ "${RUN_MAIN}" = "1" ]; then
    python src/summarize_sv_supp_results.py --tag "${RUN_TAG}"
fi
if [ "${RUN_ALPHA}" = "1" ]; then
    python src/summarize_sensitivity_multiseed.py --tag "${RUN_TAG}"
fi

echo ""
echo "========================================"
echo "Fed-MSV rerun finished"
echo "Main: ${PROJECT_ROOT}/save/sv_supp/${RUN_TAG}/main"
echo "Cross: ${PROJECT_ROOT}/save/cross_dataset/${RUN_TAG}"
echo "Alpha: ${PROJECT_ROOT}/save/sensitivity_multiseed/${RUN_TAG}/alpha"
echo "End: $(date)"
echo "========================================"

#!/bin/bash
#SBATCH --job-name=FLSV_beta025
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_beta025_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_beta025_%j.err

set -eo pipefail

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Start: $(date)"
echo "Task: Confirmatory Ours rerun with beta=0.25 and legacy 1/K noise"
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

export LD_PRELOAD=${LD_PRELOAD:-/data/home/zhaozhanshan/lib/libittnotify_stub.so}

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/zhaozhanshan/FLSV}
cd "${PROJECT_ROOT}/src"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/save"

TASK_GROUP=${TASK_GROUP:-all}
RUN_TAG=${RUN_TAG:-beta025_confirmatory_$(date +%Y%m%d_%H%M%S)}
RUN_SUMMARY=${RUN_SUMMARY:-1}

EPOCHS=${EPOCHS:-100}
NUM_USERS=${NUM_USERS:-100}
NUM_SELECTED=${NUM_SELECTED:-5}
LOCAL_EP=${LOCAL_EP:-2}
LOCAL_BS=${LOCAL_BS:-32}
LR=${LR:-0.01}
MOMENTUM=${MOMENTUM:-0.5}
TEST_SIZE=${TEST_SIZE:-10000}
GPU_ID=${GPU_ID:-0}
SEEDS=(${SEEDS:-42 123 2024})

DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-0.1}
SELECTION_BETA=0.25
ENERGY_BUDGET=${ENERGY_BUDGET:-5.0}
ALPHAS=(${ALPHAS:-0.1 0.25 0.5 1.0})
SHAPLEY_BUDGETS=(${SHAPLEY_BUDGETS:-5 10 20 50})
CHANNEL_SIGMAS=(${CHANNEL_SIGMAS:-0.0 0.1 0.25 0.5})
CROSS_DATASETS=(${CROSS_DATASETS:-fmnist mnist})

RUN_MAIN=${RUN_MAIN:-1}
RUN_ABLATION=${RUN_ABLATION:-1}
RUN_ALPHA=${RUN_ALPHA:-1}
RUN_SHAPLEY_BUDGET=${RUN_SHAPLEY_BUDGET:-1}
RUN_CHANNEL=${RUN_CHANNEL:-1}
RUN_CROSS=${RUN_CROSS:-1}

DP_CLIP_NORM=${DP_CLIP_NORM:-1.0}
CHANNEL_SIGMA=${CHANNEL_SIGMA:-0.1}

ENERGY_ARGS=(
    --use_energy --sigma_squared 1.0 --channel_model rayleigh
    --initial_energy 500.0 --energy_threshold 50.0
)
SCHED_ARGS=(
    --selection_method hybrid --selection_beta "${SELECTION_BETA}"
    --use_lyapunov --lyapunov_V 10.0 --energy_budget "${ENERGY_BUDGET}"
    --sv_weight 0.7 --battery_weight 0.15 --channel_weight 0.15
)
SV_ARGS=(
    --shapley_estimator complementary --shapley_allocation neyman
    --shapley_pilot_samples 1 --shapley_max_iter 20
    --shapley_update_method mean --shapley_alpha 0.5
)
COMMON_DP=(
    --privacy_mode central --dp_clip_norm "${DP_CLIP_NORM}"
    --dp_advanced --dp_noise_schedule constant --dp_adaptive_clip
    --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8
    --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0
    --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0
    --dp_noise_multiplier 0.0
)

BASE_ARGS=()
set_base_args() {
    local dataset=$1
    local alpha=$2
    BASE_ARGS=(
        --dataset "${dataset}" --model cnn --epochs "${EPOCHS}"
        --num_users "${NUM_USERS}" --num_selected "${NUM_SELECTED}"
        --local_ep "${LOCAL_EP}" --local_bs "${LOCAL_BS}"
        --lr "${LR}" --momentum "${MOMENTUM}"
        --dirichlet_alpha "${alpha}" --test_size "${TEST_SIZE}"
        --gpu "${GPU_ID}"
    )
}

run_training() {
    local label=$1
    local output=$2
    shift 2
    echo ""
    echo "----------------------------------------"
    echo "${label}"
    echo "Output: ${output}"
    echo "Start: $(date)"
    echo "----------------------------------------"
    python federated_main.py "$@" --output_folder "${output}"
    echo "Done: $(date)"
}

run_ours() {
    local output=$1
    local dataset=$2
    local alpha=$3
    local seed=$4
    local shapley_budget=${5:-20}
    local channel_sigma=${6:-${CHANNEL_SIGMA}}
    set_base_args "${dataset}" "${alpha}"
    run_training "Ours beta=0.25: dataset=${dataset}, alpha=${alpha}, seed=${seed}" "${output}" \
        "${BASE_ARGS[@]}" --seed "${seed}" \
        "${SV_ARGS[@]}" --shapley_max_iter "${shapley_budget}" \
        "${ENERGY_ARGS[@]}" "${SCHED_ARGS[@]}" \
        "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${channel_sigma}"
}

run_core() {
    if [ "${RUN_MAIN}" = "1" ]; then
        for seed in "${SEEDS[@]}"; do
            run_ours "paper_alignment/${RUN_TAG}/core/main/ours/seed${seed}" cifar "${DIRICHLET_ALPHA}" "${seed}"
        done
    fi

    if [ "${RUN_ABLATION}" = "1" ]; then
        for seed in "${SEEDS[@]}"; do
            run_ours "paper_alignment/${RUN_TAG}/core/ablation/full/seed${seed}" cifar "${DIRICHLET_ALPHA}" "${seed}"
            set_base_args cifar "${DIRICHLET_ALPHA}"
            run_training "Ablation w/o SV beta=0.25, seed=${seed}" "paper_alignment/${RUN_TAG}/core/ablation/no_sv/seed${seed}" \
                "${BASE_ARGS[@]}" --seed "${seed}" --no_shapley --selection_method random \
                "${ENERGY_ARGS[@]}" --use_lyapunov --lyapunov_V 10.0 \
                --energy_budget "${ENERGY_BUDGET}" --selection_beta "${SELECTION_BETA}" \
                --sv_weight 0.7 --battery_weight 0.15 --channel_weight 0.15 \
                "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
            run_training "Ablation w/o Queue beta=0.25, seed=${seed}" "paper_alignment/${RUN_TAG}/core/ablation/no_queue/seed${seed}" \
                "${BASE_ARGS[@]}" --seed "${seed}" "${SV_ARGS[@]}" "${ENERGY_ARGS[@]}" \
                "${SCHED_ARGS[@]}" --disable_queue_penalty \
                "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
            run_training "Ablation w/o Energy beta=0.25, seed=${seed}" "paper_alignment/${RUN_TAG}/core/ablation/no_energy/seed${seed}" \
                "${BASE_ARGS[@]}" --seed "${seed}" "${SV_ARGS[@]}" \
                --selection_method hybrid --selection_beta "${SELECTION_BETA}" \
                "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
        done
    fi
}

run_sensitivity() {
    if [ "${RUN_ALPHA}" = "1" ]; then
        for alpha in "${ALPHAS[@]}"; do
            alpha_label=${alpha//./p}
            for seed in "${SEEDS[@]}"; do
                run_ours "paper_alignment/${RUN_TAG}/sensitivity/alpha/alpha${alpha_label}/ours/seed${seed}" cifar "${alpha}" "${seed}"
            done
        done
    fi

    if [ "${RUN_SHAPLEY_BUDGET}" = "1" ]; then
        for budget in "${SHAPLEY_BUDGETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_ours "paper_alignment/${RUN_TAG}/sensitivity/shapley_budget/M${budget}/seed${seed}" cifar "${DIRICHLET_ALPHA}" "${seed}" "${budget}"
            done
        done
    fi

    if [ "${RUN_CHANNEL}" = "1" ]; then
        for sigma in "${CHANNEL_SIGMAS[@]}"; do
            sigma_label=${sigma//./p}
            for seed in "${SEEDS[@]}"; do
                run_ours "paper_alignment/${RUN_TAG}/sensitivity/channel_noise/sigma${sigma_label}/seed${seed}" cifar "${DIRICHLET_ALPHA}" "${seed}" 20 "${sigma}"
            done
        done
    fi
}

run_cross() {
    if [ "${RUN_CROSS}" != "1" ]; then
        return
    fi
    for dataset in "${CROSS_DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_ours "paper_alignment/${RUN_TAG}/cross/${dataset}/ours/seed${seed}" "${dataset}" "${DIRICHLET_ALPHA}" "${seed}"
        done
    done
}

echo "Configuration"
echo "  TASK_GROUP=${TASK_GROUP}"
echo "  RUN_TAG=${RUN_TAG}"
echo "  SEEDS=${SEEDS[*]}"
echo "  beta=${SELECTION_BETA}, energy_budget=${ENERGY_BUDGET}, channel_sigma=${CHANNEL_SIGMA}"
echo "  Output root: ${PROJECT_ROOT}/save/paper_alignment/${RUN_TAG}"
echo "========================================"

case "${TASK_GROUP}" in
    core) run_core ;;
    sensitivity) run_sensitivity ;;
    cross) run_cross ;;
    all)
        run_core
        run_sensitivity
        run_cross
        ;;
    *)
        echo "ERROR: TASK_GROUP must be core, sensitivity, cross, or all."
        exit 1
        ;;
esac

if [ "${RUN_SUMMARY}" = "1" ]; then
    cd "${PROJECT_ROOT}"
    python src/summarize_alignment_sweeps.py --tag "${RUN_TAG}"
fi

echo ""
echo "========================================"
echo "Beta=0.25 confirmatory rerun finished"
echo "Results: ${PROJECT_ROOT}/save/paper_alignment/${RUN_TAG}"
echo "End: $(date)"
echo "========================================"

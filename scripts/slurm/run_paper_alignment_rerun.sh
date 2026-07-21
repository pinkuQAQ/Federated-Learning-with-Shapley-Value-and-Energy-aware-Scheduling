#!/bin/bash
#SBATCH --job-name=FLSV_hybrid_ours
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_hybrid_ours_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_hybrid_ours_%j.err

set -eo pipefail

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Start: $(date)"
echo "Task: Ours-only rerun with paper-aligned scheduler and legacy 1/K noise"
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

# TASK_GROUP: core, sensitivity, cross, or all. Direct sbatch runs all Ours groups.
TASK_GROUP=${TASK_GROUP:-all}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

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
SELECTION_BETA=${SELECTION_BETA:-1.0}
SELECTION_BETAS=(${SELECTION_BETAS:-0.1 0.25 0.5 1.0 2.0})
ENERGY_BUDGET=${ENERGY_BUDGET:-5.0}
ENERGY_BUDGETS=(${ENERGY_BUDGETS:-1.0 2.0 5.0 10.0})
ALPHAS=(${ALPHAS:-0.1 0.25 0.5 1.0})
SHAPLEY_BUDGETS=(${SHAPLEY_BUDGETS:-5 10 20 50})
CHANNEL_SIGMAS=(${CHANNEL_SIGMAS:-0.0 0.1 0.25 0.5})
CROSS_DATASETS=(${CROSS_DATASETS:-fmnist mnist})

RUN_MAIN=${RUN_MAIN:-1}
RUN_ABLATION=${RUN_ABLATION:-1}
RUN_BETA=${RUN_BETA:-1}
RUN_ENERGY_BUDGET=${RUN_ENERGY_BUDGET:-1}
RUN_ALPHA=${RUN_ALPHA:-1}
RUN_ESTIMATOR=${RUN_ESTIMATOR:-1}
RUN_SHAPLEY_BUDGET=${RUN_SHAPLEY_BUDGET:-1}
RUN_CHANNEL=${RUN_CHANNEL:-1}
RUN_CROSS=${RUN_CROSS:-1}
RUN_BASELINES=${RUN_BASELINES:-0}
RUN_SUMMARY=${RUN_SUMMARY:-1}

DP_CLIP_NORM=${DP_CLIP_NORM:-1.0}
CHANNEL_SIGMA=${CHANNEL_SIGMA:-0.1}
FEDPROX_MU=${FEDPROX_MU:-0.01}

ENERGY_ARGS=(
    --use_energy --sigma_squared 1.0 --channel_model rayleigh
    --initial_energy 500.0 --energy_threshold 50.0
)
SCHED_WEIGHTS=(--sv_weight 0.7 --battery_weight 0.15 --channel_weight 0.15)
SV_DEFAULT=(
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
    local beta=$5
    local budget=$6
    set_base_args "${dataset}" "${alpha}"
    run_training "Ours: dataset=${dataset}, alpha=${alpha}, beta=${beta}, budget=${budget}, seed=${seed}" "${output}" \
        "${BASE_ARGS[@]}" --seed "${seed}" \
        "${SV_DEFAULT[@]}" "${ENERGY_ARGS[@]}" \
        --use_lyapunov --lyapunov_V 10.0 --energy_budget "${budget}" --selection_beta "${beta}" \
        "${SCHED_WEIGHTS[@]}" "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
}

run_baseline() {
    local method=$1
    local output=$2
    local dataset=$3
    local alpha=$4
    local seed=$5
    set_base_args "${dataset}" "${alpha}"
    case "${method}" in
        fedavg)
            run_training "FedAvg: dataset=${dataset}, alpha=${alpha}, seed=${seed}" "${output}" \
                "${BASE_ARGS[@]}" --seed "${seed}" --no_shapley --selection_method random \
                "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
            ;;
        fedprox)
            run_training "FedProx: dataset=${dataset}, alpha=${alpha}, seed=${seed}" "${output}" \
                "${BASE_ARGS[@]}" --seed "${seed}" --no_shapley --selection_method random \
                --use_fedprox --fedprox_mu "${FEDPROX_MU}" \
                "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
            ;;
        oort)
            run_training "Oort: dataset=${dataset}, alpha=${alpha}, seed=${seed}" "${output}" \
                "${BASE_ARGS[@]}" --seed "${seed}" --no_shapley --selection_method oort \
                "${ENERGY_ARGS[@]}" "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
            ;;
        gca)
            run_training "GCA: dataset=${dataset}, alpha=${alpha}, seed=${seed}" "${output}" \
                "${BASE_ARGS[@]}" --seed "${seed}" --no_shapley --selection_method gca \
                --gca_mode paper --gca_rho_dsi 0.5 --gca_rho_csi 0.5 --gca_lambda_energy 0.5 \
                "${ENERGY_ARGS[@]}" "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
            ;;
        fedmsv)
            run_training "Fed-MSV: dataset=${dataset}, alpha=${alpha}, seed=${seed}" "${output}" \
                "${BASE_ARGS[@]}" --seed "${seed}" --no_shapley --selection_method fedmsv \
                --fedmsv_guided_prefix 4 --fedmsv_epsilon_a 0.01 \
                --fedmsv_epsilon_b 0.01 --fedmsv_epsilon_c 0.1 \
                --fedmsv_max_permutations 0 --fedmsv_utility_source validation \
                --fedmsv_utility_samples 1000 \
                "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
            ;;
        *)
            echo "ERROR: unknown baseline ${method}"
            exit 1
            ;;
    esac
}

run_core_group() {
    if [ "${RUN_MAIN}" = "1" ]; then
        for seed in "${SEEDS[@]}"; do
            run_ours "paper_alignment/${RUN_TAG}/core/main/ours/seed${seed}" cifar "${DIRICHLET_ALPHA}" "${seed}" "${SELECTION_BETA}" "${ENERGY_BUDGET}"
            if [ "${RUN_BASELINES}" = "1" ]; then
                for method in fedavg fedprox oort gca fedmsv; do
                    run_baseline "${method}" "paper_alignment/${RUN_TAG}/core/main/${method}/seed${seed}" cifar "${DIRICHLET_ALPHA}" "${seed}"
                done
            fi
        done
    fi

    if [ "${RUN_ABLATION}" = "1" ]; then
        for seed in "${SEEDS[@]}"; do
            run_ours "paper_alignment/${RUN_TAG}/core/ablation/full/seed${seed}" cifar "${DIRICHLET_ALPHA}" "${seed}" "${SELECTION_BETA}" "${ENERGY_BUDGET}"
            set_base_args cifar "${DIRICHLET_ALPHA}"
            run_training "Ablation w/o SV, seed=${seed}" "paper_alignment/${RUN_TAG}/core/ablation/no_sv/seed${seed}" \
                "${BASE_ARGS[@]}" --seed "${seed}" --no_shapley --selection_method random \
                "${ENERGY_ARGS[@]}" --use_lyapunov --lyapunov_V 10.0 \
                --energy_budget "${ENERGY_BUDGET}" --selection_beta "${SELECTION_BETA}" \
                "${SCHED_WEIGHTS[@]}" "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
            run_training "Ablation w/o Queue, seed=${seed}" "paper_alignment/${RUN_TAG}/core/ablation/no_queue/seed${seed}" \
                "${BASE_ARGS[@]}" --seed "${seed}" "${SV_DEFAULT[@]}" "${ENERGY_ARGS[@]}" \
                --use_lyapunov --disable_queue_penalty --lyapunov_V 10.0 \
                --energy_budget "${ENERGY_BUDGET}" --selection_beta "${SELECTION_BETA}" \
                "${SCHED_WEIGHTS[@]}" "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
            run_training "Ablation w/o Energy, seed=${seed}" "paper_alignment/${RUN_TAG}/core/ablation/no_energy/seed${seed}" \
                "${BASE_ARGS[@]}" --seed "${seed}" "${SV_DEFAULT[@]}" \
                "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
        done
    fi

    if [ "${RUN_BETA}" = "1" ]; then
        for beta in "${SELECTION_BETAS[@]}"; do
            beta_label=${beta//./p}
            for seed in "${SEEDS[@]}"; do
                run_ours "paper_alignment/${RUN_TAG}/core/beta/beta${beta_label}/seed${seed}" cifar "${DIRICHLET_ALPHA}" "${seed}" "${beta}" "${ENERGY_BUDGET}"
            done
        done
    fi

    if [ "${RUN_ENERGY_BUDGET}" = "1" ]; then
        for budget in "${ENERGY_BUDGETS[@]}"; do
            budget_label=${budget//./p}
            for seed in "${SEEDS[@]}"; do
                run_ours "paper_alignment/${RUN_TAG}/core/energy_budget/budget${budget_label}/seed${seed}" cifar "${DIRICHLET_ALPHA}" "${seed}" "${SELECTION_BETA}" "${budget}"
            done
        done
    fi
}

run_sensitivity_group() {
    if [ "${RUN_ALPHA}" = "1" ]; then
        for alpha in "${ALPHAS[@]}"; do
            alpha_label=${alpha//./p}
            for seed in "${SEEDS[@]}"; do
                run_ours "paper_alignment/${RUN_TAG}/sensitivity/alpha/alpha${alpha_label}/ours/seed${seed}" cifar "${alpha}" "${seed}" "${SELECTION_BETA}" "${ENERGY_BUDGET}"
                if [ "${RUN_BASELINES}" = "1" ]; then
                    for method in fedavg fedprox oort gca fedmsv; do
                        run_baseline "${method}" "paper_alignment/${RUN_TAG}/sensitivity/alpha/alpha${alpha_label}/${method}/seed${seed}" cifar "${alpha}" "${seed}"
                    done
                fi
            done
        done
    fi

    if [ "${RUN_ESTIMATOR}" = "1" ]; then
        for seed in "${SEEDS[@]}"; do
            set_base_args cifar "${DIRICHLET_ALPHA}"
            for spec in permutation complementary_uniform complementary_neyman; do
                case "${spec}" in
                    permutation) estimator_args=(--shapley_estimator permutation --shapley_max_iter 20) ;;
                    complementary_uniform) estimator_args=(--shapley_estimator complementary --shapley_allocation uniform --shapley_pilot_samples 1 --shapley_max_iter 20) ;;
                    complementary_neyman) estimator_args=(--shapley_estimator complementary --shapley_allocation neyman --shapley_pilot_samples 1 --shapley_max_iter 20) ;;
                esac
                run_training "Estimator=${spec}, seed=${seed}" "paper_alignment/${RUN_TAG}/sensitivity/estimator/${spec}/seed${seed}" \
                    "${BASE_ARGS[@]}" --seed "${seed}" "${estimator_args[@]}" --shapley_update_method mean \
                    "${ENERGY_ARGS[@]}" --use_lyapunov --lyapunov_V 10.0 \
                    --energy_budget "${ENERGY_BUDGET}" --selection_beta "${SELECTION_BETA}" \
                    "${SCHED_WEIGHTS[@]}" "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
            done
        done
    fi

    if [ "${RUN_SHAPLEY_BUDGET}" = "1" ]; then
        for budget in "${SHAPLEY_BUDGETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                set_base_args cifar "${DIRICHLET_ALPHA}"
                run_training "Shapley budget M=${budget}, seed=${seed}" "paper_alignment/${RUN_TAG}/sensitivity/shapley_budget/M${budget}/seed${seed}" \
                    "${BASE_ARGS[@]}" --seed "${seed}" --shapley_estimator complementary \
                    --shapley_allocation neyman --shapley_pilot_samples 1 --shapley_max_iter "${budget}" \
                    --shapley_update_method mean "${ENERGY_ARGS[@]}" \
                    --use_lyapunov --lyapunov_V 10.0 --energy_budget "${ENERGY_BUDGET}" \
                    --selection_beta "${SELECTION_BETA}" "${SCHED_WEIGHTS[@]}" \
                    "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
            done
        done
    fi

    if [ "${RUN_CHANNEL}" = "1" ]; then
        for sigma in "${CHANNEL_SIGMAS[@]}"; do
            sigma_label=${sigma//./p}
            for seed in "${SEEDS[@]}"; do
                set_base_args cifar "${DIRICHLET_ALPHA}"
                run_training "Channel sigma=${sigma}, seed=${seed}" "paper_alignment/${RUN_TAG}/sensitivity/channel_noise/sigma${sigma_label}/seed${seed}" \
                    "${BASE_ARGS[@]}" --seed "${seed}" "${SV_DEFAULT[@]}" "${ENERGY_ARGS[@]}" \
                    --use_lyapunov --lyapunov_V 10.0 --energy_budget "${ENERGY_BUDGET}" \
                    --selection_beta "${SELECTION_BETA}" "${SCHED_WEIGHTS[@]}" \
                    "${COMMON_DP[@]}" --dp_channel_noise_multiplier "${sigma}"
            done
        done
    fi
}

run_cross_group() {
    if [ "${RUN_CROSS}" != "1" ]; then
        return
    fi
    for dataset in "${CROSS_DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_ours "paper_alignment/${RUN_TAG}/cross/${dataset}/ours/seed${seed}" "${dataset}" "${DIRICHLET_ALPHA}" "${seed}" "${SELECTION_BETA}" "${ENERGY_BUDGET}"
            if [ "${RUN_BASELINES}" = "1" ]; then
                for method in fedavg fedprox oort gca fedmsv; do
                    run_baseline "${method}" "paper_alignment/${RUN_TAG}/cross/${dataset}/${method}/seed${seed}" "${dataset}" "${DIRICHLET_ALPHA}" "${seed}"
                done
            fi
        done
    done
}

echo "Configuration"
echo "  TASK_GROUP=${TASK_GROUP}"
echo "  RUN_TAG=${RUN_TAG}"
echo "  SEEDS=${SEEDS[*]}"
echo "  N=${NUM_USERS}, K=${NUM_SELECTED}, T=${EPOCHS}, alpha=${DIRICHLET_ALPHA}"
echo "  beta=${SELECTION_BETA}, energy_budget=${ENERGY_BUDGET}, channel_sigma=${CHANNEL_SIGMA}"
echo "  RUN_BASELINES=${RUN_BASELINES} (0 reuses legacy sv_supp baselines)"
echo "  RUN_SUMMARY=${RUN_SUMMARY}"
echo "  Output root: ${PROJECT_ROOT}/save/paper_alignment/${RUN_TAG}"
echo "========================================"

case "${TASK_GROUP}" in
    core) run_core_group ;;
    sensitivity) run_sensitivity_group ;;
    cross) run_cross_group ;;
    all)
        run_core_group
        run_sensitivity_group
        run_cross_group
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
echo "Hybrid Ours rerun finished"
echo "Results: ${PROJECT_ROOT}/save/paper_alignment/${RUN_TAG}"
echo "End: $(date)"
echo "========================================"

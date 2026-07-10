#!/bin/bash
#SBATCH --job-name=FLSV_ms_sens
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_ms_sens_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_ms_sens_%j.err

set -e

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Task: Multi-seed sensitivity experiments for heterogeneity and Shapley budget"
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
    echo "ERROR: conda.sh not found. Please check your Conda installation path."
    exit 1
fi
conda activate flsv
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

PROJECT_ROOT=/data/home/zhaozhanshan/FLSV
cd ${PROJECT_ROOT}/src
mkdir -p ${PROJECT_ROOT}/logs
mkdir -p ${PROJECT_ROOT}/save

DATASET=${DATASET:-cifar}
MODEL=${MODEL:-cnn}
EPOCHS=${EPOCHS:-100}
NUM_USERS=${NUM_USERS:-100}
NUM_SELECTED=${NUM_SELECTED:-5}
LOCAL_EP=${LOCAL_EP:-2}
LOCAL_BS=${LOCAL_BS:-32}
LR=${LR:-0.01}
TEST_SIZE=${TEST_SIZE:-10000}
GPU_ID=${GPU_ID:-0}
SEEDS=(${SEEDS:-42 123 2024})
ALPHAS=(${ALPHAS:-0.1 0.25 0.5 1.0})
BUDGETS=(${BUDGETS:-5 10 20 50})
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# Example:
#   RUN_ALPHA=1 RUN_BUDGET=0 sbatch scripts/slurm/run_multiseed_sensitivity_experiments.sh
RUN_ALPHA=${RUN_ALPHA:-1}
RUN_BUDGET=${RUN_BUDGET:-1}
RUN_ALPHA_BASELINES=${RUN_ALPHA_BASELINES:-1}

DP_CLIP_NORM=${DP_CLIP_NORM:-1.0}
CHANNEL_SIGMA=${CHANNEL_SIGMA:-0.1}
SELECTION_BETA=${SELECTION_BETA:-1.0}
SCHED_WEIGHTS="--sv_weight 0.7 --battery_weight 0.15 --channel_weight 0.15"
COMMON_DP_ARGS="--dp_advanced --dp_noise_schedule constant --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0"
DP_ARGS="--privacy_mode central --dp_clip_norm ${DP_CLIP_NORM} ${COMMON_DP_ARGS} --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier ${CHANNEL_SIGMA}"

BASE_ARGS="--dataset ${DATASET} --model ${MODEL} --epochs ${EPOCHS} --num_users ${NUM_USERS} --num_selected ${NUM_SELECTED} --local_ep ${LOCAL_EP} --local_bs ${LOCAL_BS} --lr ${LR} --test_size ${TEST_SIZE} --gpu ${GPU_ID}"
ENERGY_ARGS="--use_energy --sigma_squared 1.0 --initial_energy 500.0 --energy_threshold 50.0"
LYAP_ARGS="--use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 --selection_beta ${SELECTION_BETA}"
SV_UPDATE_ARGS="--shapley_update_method mean --shapley_alpha 0.5"
SV_CC_ARGS="--shapley_estimator complementary --shapley_allocation neyman --shapley_pilot_samples 1 --shapley_max_iter 20"
GCA_ARGS="--no_shapley --selection_method gca --gca_learning_weight 0.5 --gca_channel_weight 0.3 --gca_energy_weight 0.2"

echo ""
echo "Configuration"
echo "  RUN_TAG=${RUN_TAG}"
echo "  SEEDS=${SEEDS[*]}"
echo "  ALPHAS=${ALPHAS[*]}"
echo "  BUDGETS=${BUDGETS[*]}"
echo "  EPOCHS=${EPOCHS}, N=${NUM_USERS}, K=${NUM_SELECTED}"
echo "  RUN_ALPHA=${RUN_ALPHA}, RUN_ALPHA_BASELINES=${RUN_ALPHA_BASELINES}, RUN_BUDGET=${RUN_BUDGET}"
echo "  Output root: ${PROJECT_ROOT}/save/sensitivity_multiseed/${RUN_TAG}"
echo "========================================"

run_cmd() {
    echo ""
    echo "----------------------------------------"
    echo "$1"
    echo "Output folder: $2"
    echo "Start: $(date)"
    echo "----------------------------------------"
    shift 2
    python federated_main.py "$@"
    echo "Done: $(date)"
}

if [ "${RUN_ALPHA}" = "1" ]; then
    echo ""
    echo "========================================"
    echo "[Group 1] Multi-seed Dirichlet-alpha sensitivity"
    echo "========================================"
    for ALPHA in "${ALPHAS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            OUT_BASE="sensitivity_multiseed/${RUN_TAG}/alpha/alpha${ALPHA}/seed${SEED}"

            run_cmd "Ours alpha=${ALPHA}, seed=${SEED}" "${OUT_BASE}/ours" \
                ${BASE_ARGS} --dirichlet_alpha ${ALPHA} --seed ${SEED} \
                ${SV_CC_ARGS} ${SV_UPDATE_ARGS} \
                ${ENERGY_ARGS} ${LYAP_ARGS} \
                ${SCHED_WEIGHTS} ${DP_ARGS} \
                --output_folder "${OUT_BASE}/ours"

            if [ "${RUN_ALPHA_BASELINES}" = "1" ]; then
                run_cmd "FedAvg alpha=${ALPHA}, seed=${SEED}" "${OUT_BASE}/fedavg" \
                    ${BASE_ARGS} --dirichlet_alpha ${ALPHA} --seed ${SEED} \
                    --no_shapley --selection_method random \
                    ${DP_ARGS} \
                    --output_folder "${OUT_BASE}/fedavg"

                run_cmd "FedProx alpha=${ALPHA}, seed=${SEED}" "${OUT_BASE}/fedprox" \
                    ${BASE_ARGS} --dirichlet_alpha ${ALPHA} --seed ${SEED} \
                    --no_shapley --selection_method random --use_fedprox --fedprox_mu 0.01 \
                    ${DP_ARGS} \
                    --output_folder "${OUT_BASE}/fedprox"

                run_cmd "Oort alpha=${ALPHA}, seed=${SEED}" "${OUT_BASE}/oort" \
                    ${BASE_ARGS} --dirichlet_alpha ${ALPHA} --seed ${SEED} \
                    --no_shapley --selection_method oort \
                    ${ENERGY_ARGS} ${DP_ARGS} \
                    --output_folder "${OUT_BASE}/oort"

                run_cmd "GCA alpha=${ALPHA}, seed=${SEED}" "${OUT_BASE}/gca" \
                    ${BASE_ARGS} --dirichlet_alpha ${ALPHA} --seed ${SEED} \
                    ${GCA_ARGS} \
                    ${ENERGY_ARGS} ${DP_ARGS} \
                    --output_folder "${OUT_BASE}/gca"
            fi
        done
    done
fi

if [ "${RUN_BUDGET}" = "1" ]; then
    echo ""
    echo "========================================"
    echo "[Group 2] Multi-seed CC-Neyman Shapley sampling-budget sensitivity"
    echo "========================================"
    BUDGET_ALPHA=${BUDGET_ALPHA:-0.1}
    for M in "${BUDGETS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            OUT="sensitivity_multiseed/${RUN_TAG}/budget/M${M}/seed${SEED}"
            run_cmd "CC-Neyman budget M=${M}, alpha=${BUDGET_ALPHA}, seed=${SEED}" "${OUT}" \
                ${BASE_ARGS} --dirichlet_alpha ${BUDGET_ALPHA} --seed ${SEED} \
                --shapley_estimator complementary --shapley_allocation neyman --shapley_pilot_samples 1 --shapley_max_iter ${M} ${SV_UPDATE_ARGS} \
                ${ENERGY_ARGS} ${LYAP_ARGS} \
                ${SCHED_WEIGHTS} ${DP_ARGS} \
                --output_folder ${OUT}
        done
    done
fi

cd ${PROJECT_ROOT}

echo ""
echo "========================================"
echo "Multi-seed sensitivity experiments finished!"
echo "Results root: ${PROJECT_ROOT}/save/sensitivity_multiseed/${RUN_TAG}"
echo "End: $(date)"
echo "========================================"

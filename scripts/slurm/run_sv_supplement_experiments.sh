#!/bin/bash
#SBATCH --job-name=FLSV_sv_supp
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_sv_supp_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_sv_supp_%j.err

set -e

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Task: Supplementary experiments for complementary-contribution Shapley"
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

DATASET=cifar
MODEL=cnn
EPOCHS=${EPOCHS:-100}
NUM_USERS=${NUM_USERS:-100}
NUM_SELECTED=${NUM_SELECTED:-5}
LOCAL_EP=${LOCAL_EP:-2}
LOCAL_BS=${LOCAL_BS:-32}
LR=${LR:-0.01}
DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-0.1}
TEST_SIZE=${TEST_SIZE:-10000}
GPU_ID=${GPU_ID:-0}
SEEDS=(${SEEDS:-42 123 2024})
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# Experiment switches. Example:
#   RUN_MAIN=0 RUN_ABLATION=0 sbatch scripts/slurm/run_sv_supplement_experiments.sh
RUN_MAIN=${RUN_MAIN:-1}
RUN_ESTIMATORS=${RUN_ESTIMATORS:-1}
RUN_BUDGET=${RUN_BUDGET:-1}
RUN_ABLATION=${RUN_ABLATION:-1}

DP_CLIP_NORM=${DP_CLIP_NORM:-1.0}
CHANNEL_SIGMA=${CHANNEL_SIGMA:-0.1}
SELECTION_BETA=${SELECTION_BETA:-1.0}
SCHED_WEIGHTS="--sv_weight 0.7 --battery_weight 0.15 --channel_weight 0.15"
COMMON_DP_ARGS="--dp_advanced --dp_noise_schedule constant --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0"
DP_ARGS="--privacy_mode central --dp_clip_norm ${DP_CLIP_NORM} ${COMMON_DP_ARGS} --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier ${CHANNEL_SIGMA}"

BASE_ARGS="--dataset ${DATASET} --model ${MODEL} --epochs ${EPOCHS} --num_users ${NUM_USERS} --num_selected ${NUM_SELECTED} --local_ep ${LOCAL_EP} --local_bs ${LOCAL_BS} --lr ${LR} --dirichlet_alpha ${DIRICHLET_ALPHA} --test_size ${TEST_SIZE} --gpu ${GPU_ID}"
ENERGY_ARGS="--use_energy --sigma_squared 1.0 --initial_energy 500.0 --energy_threshold 50.0"
LYAP_ARGS="--use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 --selection_beta ${SELECTION_BETA}"
SV_UPDATE_ARGS="--shapley_update_method mean --shapley_alpha 0.5"
SV_CC_ARGS="--shapley_estimator complementary --shapley_allocation neyman --shapley_pilot_samples 1 --shapley_max_iter 20"

echo ""
echo "Configuration"
echo "  RUN_TAG=${RUN_TAG}"
echo "  SEEDS=${SEEDS[*]}"
echo "  EPOCHS=${EPOCHS}, N=${NUM_USERS}, K=${NUM_SELECTED}, alpha=${DIRICHLET_ALPHA}"
echo "  RUN_MAIN=${RUN_MAIN}, RUN_ESTIMATORS=${RUN_ESTIMATORS}, RUN_BUDGET=${RUN_BUDGET}, RUN_ABLATION=${RUN_ABLATION}"
echo "  Output root: ${PROJECT_ROOT}/save/sv_supp/${RUN_TAG}"
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

if [ "${RUN_MAIN}" = "1" ]; then
    echo ""
    echo "========================================"
    echo "[Group 1] Main comparison with complementary Shapley"
    echo "========================================"
    for SEED in "${SEEDS[@]}"; do
        OUT="sv_supp/${RUN_TAG}/main/seed${SEED}"

        run_cmd "Ours: complementary Shapley + Energy + Lyapunov, seed=${SEED}" "${OUT}" \
            ${BASE_ARGS} --seed ${SEED} \
            ${SV_CC_ARGS} ${SV_UPDATE_ARGS} \
            ${ENERGY_ARGS} ${LYAP_ARGS} \
            ${SCHED_WEIGHTS} ${DP_ARGS} \
            --output_folder ${OUT}

        run_cmd "FedAvg: random, seed=${SEED}" "${OUT}" \
            ${BASE_ARGS} --seed ${SEED} \
            --no_shapley --selection_method random \
            ${DP_ARGS} \
            --output_folder ${OUT}

        run_cmd "FedProx: random + proximal, seed=${SEED}" "${OUT}" \
            ${BASE_ARGS} --seed ${SEED} \
            --no_shapley --selection_method random --use_fedprox --fedprox_mu 0.01 \
            ${DP_ARGS} \
            --output_folder ${OUT}

        run_cmd "Oort: utility-aware selection, seed=${SEED}" "${OUT}" \
            ${BASE_ARGS} --seed ${SEED} \
            --no_shapley --selection_method oort \
            ${ENERGY_ARGS} ${DP_ARGS} \
            --output_folder ${OUT}

        run_cmd "GCA: gradient/channel/energy-aware selection, seed=${SEED}" "${OUT}" \
            ${BASE_ARGS} --seed ${SEED} \
            --no_shapley --selection_method gca --gca_mode paper --gca_rho_dsi 0.5 --gca_rho_csi 0.5 --gca_lambda_energy 0.5 \
            ${ENERGY_ARGS} ${DP_ARGS} \
            --output_folder ${OUT}

        run_cmd "Fed-MSV: modified-Shapley weighted sampling, seed=${SEED}" "${OUT}" \
            ${BASE_ARGS} --seed ${SEED} \
            --no_shapley --selection_method fedmsv \
            --fedmsv_guided_prefix 4 --fedmsv_epsilon_a 0.01 \
            --fedmsv_epsilon_b 0.01 --fedmsv_epsilon_c 0.1 \
            --fedmsv_max_permutations 0 --fedmsv_utility_source validation \
            --fedmsv_utility_samples 1000 \
            ${DP_ARGS} \
            --output_folder ${OUT}
    done
fi

if [ "${RUN_ESTIMATORS}" = "1" ]; then
    echo ""
    echo "========================================"
    echo "[Group 2] Shapley estimator comparison"
    echo "========================================"
    for SEED in "${SEEDS[@]}"; do
        OUT_BASE="sv_supp/${RUN_TAG}/estimator/seed${SEED}"

        run_cmd "Estimator=permutation, seed=${SEED}" "${OUT_BASE}/permutation_M20" \
            ${BASE_ARGS} --seed ${SEED} \
            --shapley_estimator permutation --shapley_max_iter 20 ${SV_UPDATE_ARGS} \
            ${ENERGY_ARGS} ${LYAP_ARGS} \
            ${SCHED_WEIGHTS} ${DP_ARGS} \
            --output_folder "${OUT_BASE}/permutation_M20"

        run_cmd "Estimator=complementary uniform, seed=${SEED}" "${OUT_BASE}/complementary_uniform_M20" \
            ${BASE_ARGS} --seed ${SEED} \
            --shapley_estimator complementary --shapley_allocation uniform --shapley_pilot_samples 1 --shapley_max_iter 20 ${SV_UPDATE_ARGS} \
            ${ENERGY_ARGS} ${LYAP_ARGS} \
            ${SCHED_WEIGHTS} ${DP_ARGS} \
            --output_folder "${OUT_BASE}/complementary_uniform_M20"

        run_cmd "Estimator=complementary neyman, seed=${SEED}" "${OUT_BASE}/complementary_neyman_M20" \
            ${BASE_ARGS} --seed ${SEED} \
            --shapley_estimator complementary --shapley_allocation neyman --shapley_pilot_samples 1 --shapley_max_iter 20 ${SV_UPDATE_ARGS} \
            ${ENERGY_ARGS} ${LYAP_ARGS} \
            ${SCHED_WEIGHTS} ${DP_ARGS} \
            --output_folder "${OUT_BASE}/complementary_neyman_M20"
    done
fi

if [ "${RUN_BUDGET}" = "1" ]; then
    echo ""
    echo "========================================"
    echo "[Group 3] Complementary Shapley sampling-budget sensitivity"
    echo "========================================"
    BUDGET_SEED=${BUDGET_SEED:-42}
    for M in 5 10 20 50; do
        OUT="sv_supp/${RUN_TAG}/budget/M${M}_seed${BUDGET_SEED}"
        run_cmd "Complementary neyman budget M=${M}, seed=${BUDGET_SEED}" "${OUT}" \
            ${BASE_ARGS} --seed ${BUDGET_SEED} \
            --shapley_estimator complementary --shapley_allocation neyman --shapley_pilot_samples 1 --shapley_max_iter ${M} ${SV_UPDATE_ARGS} \
            ${ENERGY_ARGS} ${LYAP_ARGS} \
            ${SCHED_WEIGHTS} ${DP_ARGS} \
            --output_folder ${OUT}
    done
fi

if [ "${RUN_ABLATION}" = "1" ]; then
    echo ""
    echo "========================================"
    echo "[Group 4] Ablation with complementary Shapley"
    echo "========================================"
    for SEED in "${SEEDS[@]}"; do
        OUT="sv_supp/${RUN_TAG}/ablation/seed${SEED}"

        run_cmd "Full: SV + Energy + Lyapunov, seed=${SEED}" "${OUT}" \
            ${BASE_ARGS} --seed ${SEED} \
            ${SV_CC_ARGS} ${SV_UPDATE_ARGS} \
            ${ENERGY_ARGS} ${LYAP_ARGS} \
            ${SCHED_WEIGHTS} ${DP_ARGS} \
            --output_folder ${OUT}

        run_cmd "w/o SV: random + Energy + Lyapunov, seed=${SEED}" "${OUT}" \
            ${BASE_ARGS} --seed ${SEED} \
            --no_shapley --selection_method random \
            ${ENERGY_ARGS} ${LYAP_ARGS} \
            ${SCHED_WEIGHTS} ${DP_ARGS} \
            --output_folder ${OUT}

        run_cmd "w/o Lyapunov: SV + Energy, seed=${SEED}" "${OUT}" \
            ${BASE_ARGS} --seed ${SEED} \
            ${SV_CC_ARGS} ${SV_UPDATE_ARGS} \
            ${ENERGY_ARGS} \
            ${DP_ARGS} \
            --output_folder ${OUT}

        run_cmd "w/o Energy: SV only, seed=${SEED}" "${OUT}" \
            ${BASE_ARGS} --seed ${SEED} \
            ${SV_CC_ARGS} ${SV_UPDATE_ARGS} \
            ${DP_ARGS} \
            --output_folder ${OUT}
    done
fi

cd ${PROJECT_ROOT}

echo ""
echo "========================================"
echo "Supplementary SV experiments finished!"
echo "Results root: ${PROJECT_ROOT}/save/sv_supp/${RUN_TAG}"
echo "End: $(date)"
echo "========================================"

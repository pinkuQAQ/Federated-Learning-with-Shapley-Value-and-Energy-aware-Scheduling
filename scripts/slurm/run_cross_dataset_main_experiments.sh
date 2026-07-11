#!/bin/bash
#SBATCH --job-name=FLSV_cross_ds
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_cross_dataset_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_cross_dataset_%j.err

set -e

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Task: Cross-dataset main comparison for FLSV"
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
if [ -d "/data/home/zhaozhanshan/ENTER/envs/flsv" ]; then
    conda activate /data/home/zhaozhanshan/ENTER/envs/flsv
else
    conda activate flsv
fi
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

PROJECT_ROOT=/data/home/zhaozhanshan/FLSV
cd ${PROJECT_ROOT}/src
mkdir -p ${PROJECT_ROOT}/logs
mkdir -p ${PROJECT_ROOT}/save

# Default run:
#   datasets = Fashion-MNIST + MNIST
#   methods  = Ours, FedAvg, FedProx, Oort, GCA
#   seeds    = 42, 123, 2024
#
# Example overrides:
#   DATASETS="fmnist" EPOCHS=20 sbatch scripts/slurm/run_cross_dataset_main_experiments.sh
#   SEEDS="42" RUN_TAG=test_cross_ds sbatch scripts/slurm/run_cross_dataset_main_experiments.sh

DATASETS=(${DATASETS:-fmnist mnist})
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
SEEDS=(${SEEDS:-42 123 2024})
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

RUN_OURS=${RUN_OURS:-1}
RUN_FEDAVG=${RUN_FEDAVG:-1}
RUN_FEDPROX=${RUN_FEDPROX:-1}
RUN_OORT=${RUN_OORT:-1}
RUN_GCA=${RUN_GCA:-1}

DP_CLIP_NORM=${DP_CLIP_NORM:-1.0}
CHANNEL_SIGMA=${CHANNEL_SIGMA:-0.1}
SELECTION_BETA=${SELECTION_BETA:-1.0}
SCHED_WEIGHTS="--sv_weight 0.7 --battery_weight 0.15 --channel_weight 0.15"
COMMON_DP_ARGS="--dp_advanced --dp_noise_schedule constant --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0"
DP_ARGS="--privacy_mode central --dp_clip_norm ${DP_CLIP_NORM} ${COMMON_DP_ARGS} --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier ${CHANNEL_SIGMA}"

ENERGY_ARGS="--use_energy --sigma_squared 1.0 --initial_energy 500.0 --energy_threshold 50.0"
LYAP_ARGS="--use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 --selection_beta ${SELECTION_BETA}"
SV_UPDATE_ARGS="--shapley_update_method mean --shapley_alpha 0.5"
SV_CC_ARGS="--shapley_estimator complementary --shapley_allocation neyman --shapley_pilot_samples 1 --shapley_max_iter 20"
FEDPROX_ARGS="--no_shapley --selection_method random --use_fedprox --fedprox_mu ${FEDPROX_MU:-0.01}"
OORT_ARGS="--no_shapley --selection_method oort"
GCA_ARGS="--no_shapley --selection_method gca --gca_learning_weight 0.5 --gca_channel_weight 0.3 --gca_energy_weight 0.2"

echo ""
echo "Configuration"
echo "  RUN_TAG=${RUN_TAG}"
echo "  DATASETS=${DATASETS[*]}"
echo "  SEEDS=${SEEDS[*]}"
echo "  EPOCHS=${EPOCHS}, N=${NUM_USERS}, K=${NUM_SELECTED}, alpha=${DIRICHLET_ALPHA}"
echo "  RUN_OURS=${RUN_OURS}, RUN_FEDAVG=${RUN_FEDAVG}, RUN_FEDPROX=${RUN_FEDPROX}, RUN_OORT=${RUN_OORT}, RUN_GCA=${RUN_GCA}"
echo "  Output root: ${PROJECT_ROOT}/save/cross_dataset/${RUN_TAG}"
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

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "[Dataset] ${DATASET}"
    echo "========================================"

    BASE_ARGS="--dataset ${DATASET} --model ${MODEL} --epochs ${EPOCHS} --num_users ${NUM_USERS} --num_selected ${NUM_SELECTED} --local_ep ${LOCAL_EP} --local_bs ${LOCAL_BS} --lr ${LR} --dirichlet_alpha ${DIRICHLET_ALPHA} --test_size ${TEST_SIZE} --gpu ${GPU_ID}"

    for SEED in "${SEEDS[@]}"; do
        OUT="cross_dataset/${RUN_TAG}/${DATASET}/seed${SEED}"

        if [ "${RUN_OURS}" = "1" ]; then
            run_cmd "Ours: ${DATASET}, seed=${SEED}" "${OUT}" \
                ${BASE_ARGS} --seed ${SEED} \
                ${SV_CC_ARGS} ${SV_UPDATE_ARGS} \
                ${ENERGY_ARGS} ${LYAP_ARGS} \
                ${SCHED_WEIGHTS} ${DP_ARGS} \
                --output_folder "${OUT}"
        fi

        if [ "${RUN_FEDAVG}" = "1" ]; then
            run_cmd "FedAvg: ${DATASET}, seed=${SEED}" "${OUT}" \
                ${BASE_ARGS} --seed ${SEED} \
                --no_shapley --selection_method random \
                ${DP_ARGS} \
                --output_folder "${OUT}"
        fi

        if [ "${RUN_FEDPROX}" = "1" ]; then
            run_cmd "FedProx: ${DATASET}, seed=${SEED}" "${OUT}" \
                ${BASE_ARGS} --seed ${SEED} \
                ${FEDPROX_ARGS} \
                ${DP_ARGS} \
                --output_folder "${OUT}"
        fi

        if [ "${RUN_OORT}" = "1" ]; then
            run_cmd "Oort: ${DATASET}, seed=${SEED}" "${OUT}" \
                ${BASE_ARGS} --seed ${SEED} \
                ${OORT_ARGS} \
                ${ENERGY_ARGS} ${DP_ARGS} \
                --output_folder "${OUT}"
        fi

        if [ "${RUN_GCA}" = "1" ]; then
            run_cmd "GCA: ${DATASET}, seed=${SEED}" "${OUT}" \
                ${BASE_ARGS} --seed ${SEED} \
                ${GCA_ARGS} \
                ${ENERGY_ARGS} ${DP_ARGS} \
                --output_folder "${OUT}"
        fi
    done
done

cd ${PROJECT_ROOT}

echo ""
echo "========================================"
echo "Cross-dataset main comparison finished!"
echo "Results root: ${PROJECT_ROOT}/save/cross_dataset/${RUN_TAG}"
echo "End: $(date)"
echo "========================================"

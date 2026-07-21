#!/bin/bash
#SBATCH --job-name=FLSV_gca
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_gca_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_gca_%j.err

set -e

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Task: GCA source-faithful indicator baseline for digital FL"
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

DP_CLIP_NORM=${DP_CLIP_NORM:-1.0}
CHANNEL_SIGMA=${CHANNEL_SIGMA:-0.1}
COMMON_DP_ARGS="--dp_advanced --dp_noise_schedule constant --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0"
DP_ARGS="--privacy_mode central --dp_clip_norm ${DP_CLIP_NORM} ${COMMON_DP_ARGS} --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier ${CHANNEL_SIGMA}"

BASE_ARGS="--dataset ${DATASET} --model ${MODEL} --epochs ${EPOCHS} --num_users ${NUM_USERS} --num_selected ${NUM_SELECTED} --local_ep ${LOCAL_EP} --local_bs ${LOCAL_BS} --lr ${LR} --dirichlet_alpha ${DIRICHLET_ALPHA} --test_size ${TEST_SIZE} --gpu ${GPU_ID}"
ENERGY_ARGS="--use_energy --sigma_squared 1.0 --initial_energy 500.0 --energy_threshold 50.0"
GCA_ARGS="--no_shapley --selection_method gca --gca_mode paper --gca_rho_dsi 0.5 --gca_rho_csi 0.5 --gca_lambda_energy 0.5"

echo ""
echo "Configuration"
echo "  RUN_TAG=${RUN_TAG}"
echo "  SEEDS=${SEEDS[*]}"
echo "  EPOCHS=${EPOCHS}, N=${NUM_USERS}, K=${NUM_SELECTED}, alpha=${DIRICHLET_ALPHA}"
echo "  Output root: ${PROJECT_ROOT}/save/sv_supp/${RUN_TAG}/main"
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

for SEED in "${SEEDS[@]}"; do
    OUT="sv_supp/${RUN_TAG}/main/seed${SEED}"
    run_cmd "GCA baseline, seed=${SEED}" "${OUT}" \
        ${BASE_ARGS} --seed ${SEED} \
        ${GCA_ARGS} ${ENERGY_ARGS} ${DP_ARGS} \
        --output_folder ${OUT}
done

cd ${PROJECT_ROOT}

echo ""
echo "========================================"
echo "GCA baseline finished!"
echo "Results root: ${PROJECT_ROOT}/save/sv_supp/${RUN_TAG}/main"
echo "Aggregate with:"
echo "  cd ${PROJECT_ROOT} && python src/summarize_sv_supp_results.py --tag ${RUN_TAG}"
echo "End: $(date)"
echo "========================================"

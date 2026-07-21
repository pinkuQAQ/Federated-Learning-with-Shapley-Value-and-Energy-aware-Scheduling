#!/bin/bash
#SBATCH --job-name=FLSV_sig
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_sig_%A_%a.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_sig_%A_%a.err

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
mkdir -p "${PROJECT_ROOT}/save" "${PROJECT_ROOT}/logs"

RUN_TAG=${RUN_TAG:-significance_$(date +%Y%m%d_%H%M%S)}
GPU_ID=${GPU_ID:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
WORKER_COUNT=${WORKER_COUNT:-4}

# Fixed before observing the new results. The five existing methods already
# have seeds 42, 123, and 2024. Fed-MSV currently has seed 42 only.
COMMON_NEW_SEEDS=(7 21 77 888 1001 3407 31415)
METHODS=(ours fedavg fedprox oort gca)
FEDMSV_NEW_SEEDS=(7 21 77 888 1001 3407 31415 123 2024)

TASKS=()
# Fed-MSV is substantially slower, so start its tasks first to reduce the
# array's overall wall-clock completion time.
for seed in "${FEDMSV_NEW_SEEDS[@]}"; do
    TASKS+=("fedmsv:${seed}")
done
for seed in "${COMMON_NEW_SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
        TASKS+=("${method}:${seed}")
    done
done

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
CHANNEL_SIGMA=${CHANNEL_SIGMA:-0.1}

COMMON_DP=(
    --privacy_mode central --dp_clip_norm 1.0
    --dp_advanced --dp_noise_schedule constant --dp_adaptive_clip
    --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8
    --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0
    --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0
    --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier "${CHANNEL_SIGMA}"
)

ENERGY_ARGS=(
    --use_energy --sigma_squared 1.0 --channel_model rayleigh
    --initial_energy 500.0 --energy_threshold 50.0
)

run_task() {
    local task_id=$1
    local method seed output
    local -a base_args method_args

    IFS=: read -r method seed <<< "${TASKS[${task_id}]}"
    output="significance/${RUN_TAG}/${method}/seed${seed}"

    if [ "${SKIP_EXISTING}" = "1" ] && compgen -G "${PROJECT_ROOT}/save/${output}/*.pkl" >/dev/null; then
        echo "[skip] Task ${task_id}: existing result save/${output}"
        return
    fi

    base_args=(
        --dataset cifar --model cnn --epochs "${EPOCHS}"
        --num_users "${NUM_USERS}" --num_selected "${NUM_SELECTED}"
        --local_ep "${LOCAL_EP}" --local_bs "${LOCAL_BS}"
        --optimizer sgd --lr "${LR}" --momentum "${MOMENTUM}"
        --weight_decay "${WEIGHT_DECAY}" --dirichlet_alpha "${DIRICHLET_ALPHA}"
        --test_size "${TEST_SIZE}" --gpu "${GPU_ID}" --seed "${seed}"
    )

    echo "========================================"
    echo "Worker: ${SLURM_ARRAY_TASK_ID:-0}/${WORKER_COUNT}, experiment task: ${task_id}"
    echo "Method: ${method}, seed: ${seed}"
    echo "Output: ${PROJECT_ROOT}/save/${output}"
    echo "Start: $(date)"
    echo "========================================"

    case "${method}" in
        ours)
            method_args=(
                --selection_method hybrid --selection_beta 0.25
                --shapley_estimator complementary --shapley_allocation neyman
                --shapley_pilot_samples 1 --shapley_max_iter 20
                --shapley_update_method mean --shapley_alpha 0.5
                --use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0
                --sv_weight 0.7 --battery_weight 0.15 --channel_weight 0.15
            )
            python federated_main.py "${base_args[@]}" "${method_args[@]}" \
                "${ENERGY_ARGS[@]}" "${COMMON_DP[@]}" --output_folder "${output}"
            ;;
        fedavg)
            python federated_main.py "${base_args[@]}" \
                --no_shapley --selection_method random \
                "${COMMON_DP[@]}" --output_folder "${output}"
            ;;
        fedprox)
            python federated_main.py "${base_args[@]}" \
                --no_shapley --selection_method random --use_fedprox --fedprox_mu 0.01 \
                "${COMMON_DP[@]}" --output_folder "${output}"
            ;;
        oort)
            python federated_main.py "${base_args[@]}" \
                --no_shapley --selection_method oort \
                "${ENERGY_ARGS[@]}" "${COMMON_DP[@]}" --output_folder "${output}"
            ;;
        gca)
            python federated_main.py "${base_args[@]}" \
                --no_shapley --selection_method gca \
                --gca_mode paper --gca_rho_dsi 0.5 --gca_rho_csi 0.5 --gca_lambda_energy 0.5 \
                "${ENERGY_ARGS[@]}" "${COMMON_DP[@]}" --output_folder "${output}"
            ;;
        fedmsv)
            python federated_main.py "${base_args[@]}" \
                --no_shapley --selection_method fedmsv \
                --fedmsv_guided_prefix 4 --fedmsv_epsilon_a 0.01 \
                --fedmsv_epsilon_b 0.01 --fedmsv_epsilon_c 0.1 \
                --fedmsv_max_permutations 0 --fedmsv_utility_source validation \
                --fedmsv_utility_samples 1000 --fedmsv_low_quality_type none \
                --fedmsv_low_quality_fraction 0.0 \
                "${COMMON_DP[@]}" --output_folder "${output}"
            ;;
        *)
            echo "ERROR: unsupported method ${method}"
            return 1
            ;;
    esac

    echo "Task ${task_id} done: $(date)"
}

if ! [[ "${WORKER_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: WORKER_COUNT must be a positive integer"
    exit 1
fi

WORKER_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ "${WORKER_ID}" -lt 0 ] || [ "${WORKER_ID}" -ge "${WORKER_COUNT}" ]; then
    echo "ERROR: worker ${WORKER_ID} is outside 0..$((WORKER_COUNT - 1))"
    exit 1
fi

echo "========================================"
echo "Job: ${SLURM_JOB_ID:-local}, worker: ${WORKER_ID}/${WORKER_COUNT}"
echo "Experiments assigned by task_id % ${WORKER_COUNT} = ${WORKER_ID}"
echo "Start: $(date)"
echo "========================================"

for ((task_id = WORKER_ID; task_id < ${#TASKS[@]}; task_id += WORKER_COUNT)); do
    run_task "${task_id}"
done

echo "Worker ${WORKER_ID} finished: $(date)"

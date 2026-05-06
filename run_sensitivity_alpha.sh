#!/bin/bash
#SBATCH --job-name=FLSV_sens_alpha
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_sens_alpha_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_sens_alpha_%j.err

# =============================================================================
# Task: Heterogeneity sensitivity sweep, single seed.
# Methods: Ours, FedAvg, PoC (3 methods x 3 alpha values = 9 runs)
# alpha: 0.25, 0.5, 1.0 (alpha=0.1 is already covered by the main runs)
# sigma_dp = 1.0, q=0.05, 100 epochs, seed=42, same operating point as the main table.
# Expect roughly 40 min/run x 9 runs = about 6 hours.
# =============================================================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Task: CIFAR-10 alpha sensitivity (single seed)"
echo "========================================"

source /data/home/zhaozhanshan/ENTER/etc/profile.d/conda.sh
conda activate flsv
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

cd /data/home/zhaozhanshan/FLSV/src
mkdir -p /data/home/zhaozhanshan/FLSV/logs
mkdir -p /data/home/zhaozhanshan/FLSV/save

DATASET=cifar
MODEL=cnn
EPOCHS=100
NUM_USERS=100
NUM_SELECTED=5
LOCAL_EP=2
LOCAL_BS=32
LR=0.01
SEED=42

ALPHAS=(0.25 0.5 1.0)

DP_CLIP_NORM=1.0
DP_ADAPTIVE_ARGS="--lightweight_dp --public_pretrain_epochs 3 --public_pretrain_samples 20000 --dp_advanced --dp_noise_schedule linear_increase --dp_noise_start_multiplier 0.7 --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0"
DP_NOISE_MULTIPLIER=1.0
DP_ARGS="--privacy_mode central --dp_clip_norm $DP_CLIP_NORM $DP_ADAPTIVE_ARGS --dp_noise_multiplier $DP_NOISE_MULTIPLIER"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

for ALPHA in "${ALPHAS[@]}"; do
    OUTPUT_FOLDER="sens_alpha${ALPHA}_seed${SEED}_${RUN_TAG}"

    echo ""
    echo "========================================"
    echo "Alpha=${ALPHA}  Seed=${SEED}"
    echo "Output: ${OUTPUT_FOLDER}"
    echo "========================================"

    echo "[1/3] Ours (SV + Energy + Lyapunov)"
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --shapley_update_method mean \
        --shapley_alpha 0.5 \
        --shapley_max_iter 20 \
        --use_energy \
        --sigma_squared 1.0 \
        --initial_energy 500.0 \
        --energy_threshold 50.0 \
        --use_lyapunov \
        --lyapunov_V 10.0 \
        --energy_budget 5.0 \
        $DP_ARGS \
        --output_folder $OUTPUT_FOLDER

    echo "[2/3] FedAvg (random)"
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --no_shapley \
        --selection_method random \
        $DP_ARGS \
        --output_folder $OUTPUT_FOLDER

    echo "[3/3] PoC"
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --no_shapley \
        --selection_method poc \
        $DP_ARGS \
        --output_folder $OUTPUT_FOLDER
done

echo ""
echo "========================================"
echo "Completed alpha sensitivity runs"
echo "End: $(date)"
echo "========================================"

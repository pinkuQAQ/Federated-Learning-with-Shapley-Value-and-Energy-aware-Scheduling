#!/bin/bash
#SBATCH --job-name=FLSV_cifar_ms
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_cifar_ms_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_cifar_ms_%j.err

# =============================================================================
# Task:  Main table ??CIFAR-10, ?=0.1, 3 seeds, 6 methods, 100 epochs
# Seeds: 42, 123, 2024
# Methods: Ours, FedAvg, PoC, UCB, FedProx, FedCS
# sigma_dp: 1.0 (default lightweight-update DP operating point, q=0.05)
# Expect ~40 min/run ? 3 ? 6 = ~12 h
# =============================================================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Task: CIFAR-10 main table (multi-seed)"
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
ALPHA=0.1
SEEDS=(42 123 2024)

# sigma_dp=1.5 is the default lightweight-update DP operating point; set to 0 for clipping-only.
DP_CLIP_NORM=1.0
DP_ADAPTIVE_ARGS="--lightweight_dp --public_pretrain_epochs 3 --public_pretrain_samples 20000 --dp_advanced --dp_noise_schedule linear_increase --dp_noise_start_multiplier 0.7 --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0"
DP_NOISE_MULTIPLIER=1.5
DP_ARGS="--privacy_mode central --dp_clip_norm $DP_CLIP_NORM $DP_ADAPTIVE_ARGS --dp_noise_multiplier $DP_NOISE_MULTIPLIER"

# FedCS deadline calibrated against ?_ch?=1.0 + default compute cost
FEDCS_DEADLINE=5.0

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

for SEED in "${SEEDS[@]}"; do
    OUTPUT_FOLDER="main_3seed_a${ALPHA}_sigma${DP_NOISE_MULTIPLIER}_seed${SEED}_${RUN_TAG}"

    echo ""
    echo "========================================"
    echo "Alpha=${ALPHA} Seed=${SEED}"
    echo "Output: ${OUTPUT_FOLDER}"
    echo "========================================"

    echo "[1/6] Ours (SV + Energy + Lyapunov)"
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

    echo "[2/6] FedAvg (random)"
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --no_shapley \
        --selection_method random \
        $DP_ARGS \
        --output_folder $OUTPUT_FOLDER

    echo "[3/6] PoC"
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --no_shapley \
        --selection_method poc \
        $DP_ARGS \
        --output_folder $OUTPUT_FOLDER

    echo "[4/6] UCB"
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --no_shapley \
        --selection_method ucb \
        --ucb_c 1.0 \
        $DP_ARGS \
        --output_folder $OUTPUT_FOLDER

    echo "[5/6] FedProx (mu=0.01)"
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --no_shapley \
        --selection_method random \
        --use_fedprox \
        --fedprox_mu 0.01 \
        $DP_ARGS \
        --output_folder $OUTPUT_FOLDER

    echo "[6/6] FedCS (deadline=${FEDCS_DEADLINE})"
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --no_shapley \
        --selection_method fedcs \
        --fedcs_deadline $FEDCS_DEADLINE \
        --use_energy \
        --sigma_squared 1.0 \
        --initial_energy 500.0 \
        --energy_threshold 50.0 \
        $DP_ARGS \
        --output_folder $OUTPUT_FOLDER
done

echo ""
echo "========================================"
echo "Completed CIFAR-10 multi-seed main runs"
echo "End: $(date)"
echo "========================================"

#!/bin/bash
#SBATCH --job-name=FLSV_stress
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_stress_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_stress_%j.err

# =============================================================================
# Task:  Energy stress — tight budget configuration so Lyapunov queue truly binds
# Setup: E_init=100 (vs default 500), E_threshold=10, same 6 methods, 1 seed,
#        α=0.1, 80 epochs. Purpose: show that at a tight budget some baselines
#        start depleting clients while Ours keeps participation balanced.
# Expect ~30 min/run × 6 methods ≈ 3 h
# =============================================================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Task: CIFAR-10 energy-stress test"
echo "========================================"

source /data/home/zhaozhanshan/ENTER/etc/profile.d/conda.sh
conda activate flsv
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

cd /data/home/zhaozhanshan/FLSV/src
mkdir -p /data/home/zhaozhanshan/FLSV/logs
mkdir -p /data/home/zhaozhanshan/FLSV/save

DATASET=cifar
MODEL=cnn
EPOCHS=80
NUM_USERS=100
NUM_SELECTED=10
LOCAL_EP=2
LOCAL_BS=32
LR=0.01
ALPHA=0.1
SEED=42

# Stress parameters — this is the one knob that makes queue actually bind
INITIAL_ENERGY=100.0
ENERGY_THRESHOLD=10.0
ENERGY_BUDGET=2.0

DP_CLIP_NORM=1.0
DP_NOISE_MULTIPLIER=0.01
DP_ARGS="--use_local_dp --dp_clip_norm $DP_CLIP_NORM --dp_noise_multiplier $DP_NOISE_MULTIPLIER"
FEDCS_DEADLINE=5.0
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
OUTPUT_FOLDER="stress_e${INITIAL_ENERGY}_a${ALPHA}_seed${SEED}_${RUN_TAG}"

echo ""
echo "========================================"
echo "Stress: E_init=${INITIAL_ENERGY}, E_th=${ENERGY_THRESHOLD}, Budget=${ENERGY_BUDGET}"
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
    --initial_energy $INITIAL_ENERGY \
    --energy_threshold $ENERGY_THRESHOLD \
    --use_lyapunov \
    --lyapunov_V 10.0 \
    --energy_budget $ENERGY_BUDGET \
    $DP_ARGS \
    --output_folder $OUTPUT_FOLDER

echo "[2/6] FedAvg"
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $ALPHA --seed $SEED \
    --no_shapley \
    --selection_method random \
    --use_energy \
    --sigma_squared 1.0 \
    --initial_energy $INITIAL_ENERGY \
    --energy_threshold $ENERGY_THRESHOLD \
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
    --use_energy \
    --sigma_squared 1.0 \
    --initial_energy $INITIAL_ENERGY \
    --energy_threshold $ENERGY_THRESHOLD \
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
    --use_energy \
    --sigma_squared 1.0 \
    --initial_energy $INITIAL_ENERGY \
    --energy_threshold $ENERGY_THRESHOLD \
    $DP_ARGS \
    --output_folder $OUTPUT_FOLDER

echo "[5/6] FedProx"
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $ALPHA --seed $SEED \
    --no_shapley \
    --selection_method random \
    --use_fedprox \
    --fedprox_mu 0.01 \
    --use_energy \
    --sigma_squared 1.0 \
    --initial_energy $INITIAL_ENERGY \
    --energy_threshold $ENERGY_THRESHOLD \
    $DP_ARGS \
    --output_folder $OUTPUT_FOLDER

echo "[6/6] FedCS"
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
    --initial_energy $INITIAL_ENERGY \
    --energy_threshold $ENERGY_THRESHOLD \
    $DP_ARGS \
    --output_folder $OUTPUT_FOLDER

echo ""
echo "========================================"
echo "Completed energy-stress runs"
echo "End: $(date)"
echo "========================================"

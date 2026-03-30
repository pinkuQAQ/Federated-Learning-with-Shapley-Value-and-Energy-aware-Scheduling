#!/bin/bash
#SBATCH --job-name=FLSV_fmnist_a025
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=96:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_fmnist_a025_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_fmnist_a025_%j.err

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "========================================"

source /data/home/zhaozhanshan/ENTER/etc/profile.d/conda.sh
conda activate flsv
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

cd /data/home/zhaozhanshan/FLSV/src
mkdir -p /data/home/zhaozhanshan/FLSV/logs
mkdir -p /data/home/zhaozhanshan/FLSV/save

DATASET=fmnist
MODEL=cnn
EPOCHS=100
NUM_USERS=100
NUM_SELECTED=10
LOCAL_EP=2
LOCAL_BS=32
LR=0.01
DIRICHLET_ALPHA=0.25
SEED=42
OUTPUT_FOLDER="${OUTPUT_FOLDER:-fmnist_alpha0.25_$(date +%Y%m%d)}"

echo "Task: FMNIST alpha=0.25"
echo "Output folder: $OUTPUT_FOLDER"

echo "[1/5] Running Ours..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --shapley_update_method mean \
    --shapley_alpha 0.5 \
    --shapley_max_iter 20 \
    --use_energy \
    --initial_energy 500.0 \
    --energy_threshold 50.0 \
    --use_lyapunov \
    --lyapunov_V 10.0 \
    --energy_budget 5.0 \
    --output_folder $OUTPUT_FOLDER

echo "[2/5] Running FedAvg..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --no_shapley \
    --selection_method random \
    --output_folder $OUTPUT_FOLDER

echo "[3/5] Running PoC..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --no_shapley \
    --selection_method poc \
    --output_folder $OUTPUT_FOLDER

echo "[4/5] Running UCB..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --no_shapley \
    --selection_method ucb \
    --ucb_c 1.0 \
    --output_folder $OUTPUT_FOLDER

echo "[5/5] Running FedProx..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --no_shapley \
    --selection_method random \
    --use_fedprox \
    --fedprox_mu 0.01 \
    --output_folder $OUTPUT_FOLDER

echo "Done. Results saved to: /data/home/zhaozhanshan/FLSV/save/$OUTPUT_FOLDER"

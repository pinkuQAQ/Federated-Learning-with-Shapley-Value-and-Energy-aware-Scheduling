#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
SRC_DIR="$REPO_DIR/src"
LOG_DIR="$REPO_DIR/logs"
SAVE_DIR="$REPO_DIR/save"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${CONDA_SH:-}" ]]; then
    source "$CONDA_SH"
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
    conda activate "$CONDA_ENV"
fi

if [[ -n "${LD_PRELOAD_PATH:-}" ]]; then
    export LD_PRELOAD="$LD_PRELOAD_PATH"
fi

mkdir -p "$LOG_DIR" "$SAVE_DIR"
cd "$SRC_DIR"

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

echo "========================================"
echo "Start: $(date)"
echo "Task: FMNIST alpha=0.25"
echo "Output folder: $OUTPUT_FOLDER"
echo "========================================"

echo "[1/5] Running Ours..."
"$PYTHON_BIN" federated_main.py \
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
"$PYTHON_BIN" federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --no_shapley \
    --selection_method random \
    --output_folder $OUTPUT_FOLDER

echo "[3/5] Running PoC..."
"$PYTHON_BIN" federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --no_shapley \
    --selection_method poc \
    --output_folder $OUTPUT_FOLDER

echo "[4/5] Running UCB..."
"$PYTHON_BIN" federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --no_shapley \
    --selection_method ucb \
    --ucb_c 1.0 \
    --output_folder $OUTPUT_FOLDER

echo "[5/5] Running FedProx..."
"$PYTHON_BIN" federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --no_shapley \
    --selection_method random \
    --use_fedprox \
    --fedprox_mu 0.01 \
    --output_folder $OUTPUT_FOLDER

echo "Done. Results saved to: $SAVE_DIR/$OUTPUT_FOLDER"

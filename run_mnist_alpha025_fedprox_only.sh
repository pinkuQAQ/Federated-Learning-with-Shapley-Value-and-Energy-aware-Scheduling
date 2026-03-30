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

DATASET=mnist
MODEL=cnn
EPOCHS=100
NUM_USERS=100
NUM_SELECTED=10
LOCAL_EP=2
LOCAL_BS=32
LR=0.01
DIRICHLET_ALPHA=0.25
SEED=42
OUTPUT_FOLDER="${OUTPUT_FOLDER:-mnist_alpha0.25_$(date +%Y%m%d)}"

echo "========================================"
echo "Start: $(date)"
echo "Task: MNIST alpha=0.25 FedProx only"
echo "Output folder: $OUTPUT_FOLDER"
echo "========================================"

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

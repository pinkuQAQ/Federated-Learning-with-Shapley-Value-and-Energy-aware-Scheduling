#!/bin/bash
#SBATCH --job-name=FLSV_ablation
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_ablation_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_ablation_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "========================================"

# 激活conda环境（自动探测路径）
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    for c in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/data/home/zhaozhanshan/miniconda3/etc/profile.d/conda.sh" \
        "/data/home/zhaozhanshan/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"
    do
        if [ -f "$c" ]; then
            source "$c"
            break
        fi
    done
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda command not found on this node."
    exit 1
fi

conda activate flsv || {
    echo "ERROR: conda env 'flsv' not found."
    conda info --envs || true
    exit 1
}

if [ -f /data/home/zhaozhanshan/lib/libittnotify_stub.so ]; then
    export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so
fi

cd /data/home/zhaozhanshan/FLSV/src
mkdir -p /data/home/zhaozhanshan/FLSV/logs
mkdir -p /data/home/zhaozhanshan/FLSV/save

DATASET=cifar
MODEL=cnn
EPOCHS=100
NUM_USERS=100
NUM_SELECTED=10
LOCAL_EP=2
LOCAL_BS=32
LR=0.01
DIRICHLET_ALPHA=0.1
SEED=42
OUTPUT_FOLDER="${OUTPUT_FOLDER:-ablation_$(date +%Y%m%d_%H%M%S)}"

echo "Task: ablation rerun"
echo "Output folder: $OUTPUT_FOLDER"

echo "[1/4] Running Full..."
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
    --use_crypto \
    --output_folder $OUTPUT_FOLDER

echo "[2/4] Running w/o Crypto..."
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

echo "[3/4] Running w/o Lyapunov..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --selection_method greedy \
    --shapley_update_method mean \
    --shapley_alpha 0.5 \
    --shapley_max_iter 20 \
    --use_crypto \
    --output_folder $OUTPUT_FOLDER

echo "[4/4] Running w/o SV..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --no_shapley \
    --selection_method random \
    --use_energy \
    --initial_energy 500.0 \
    --energy_threshold 50.0 \
    --use_lyapunov \
    --lyapunov_V 10.0 \
    --energy_budget 5.0 \
    --use_crypto \
    --output_folder $OUTPUT_FOLDER

echo "Done. Results saved to: /data/home/zhaozhanshan/FLSV/save/$OUTPUT_FOLDER"

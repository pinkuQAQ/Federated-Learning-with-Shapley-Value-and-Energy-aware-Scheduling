#!/bin/bash
#SBATCH --job-name=FLSV_baseline
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "========================================"

# 激活conda环境
source /data/home/zhaozhanshan/ENTER/etc/profile.d/conda.sh
conda activate flsv
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

# 切换到src目录
cd /data/home/zhaozhanshan/FLSV/src

# 创建日志目录
mkdir -p /data/home/zhaozhanshan/FLSV/logs
mkdir -p /data/home/zhaozhanshan/FLSV/save

# 共享参数
DATASET=cifar
MODEL=cnn
EPOCHS=100
NUM_USERS=100
NUM_SELECTED=10
LOCAL_EP=5
LOCAL_BS=32
LR=0.01
DIRICHLET_ALPHA=0.1
SEED=42
OUTPUT_FOLDER="baseline_cmp_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "[1/5] Running Ours (SV + Energy + Lyapunov)..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --shapley_update_method mean \
    --shapley_alpha 0.5 \
    --use_energy \
    --sigma_squared 1.0 \
    --initial_energy 500.0 \
    --energy_threshold 50.0 \
    --use_lyapunov \
    --lyapunov_V 10.0 \
    --energy_budget 5.0 \
    --output_folder $OUTPUT_FOLDER
echo "[1/5] Done!"

echo ""
echo "[2/5] Running FedAvg (Random)..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --no_shapley \
    --selection_method random \
    --output_folder $OUTPUT_FOLDER
echo "[2/5] Done!"

echo ""
echo "[3/5] Running PoC (Power of Choice)..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --no_shapley \
    --selection_method poc \
    --output_folder $OUTPUT_FOLDER
echo "[3/5] Done!"

echo ""
echo "[4/5] Running UCB1..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --no_shapley \
    --selection_method ucb \
    --ucb_c 1.0 \
    --output_folder $OUTPUT_FOLDER
echo "[4/5] Done!"

echo ""
echo "[5/5] Running FedProx (mu=0.01)..."
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
echo "[5/5] Done!"

echo ""
echo "========================================"
echo "All experiments finished!"
echo "Results saved to: ../save/$OUTPUT_FOLDER"
echo "End: $(date)"
echo "========================================"

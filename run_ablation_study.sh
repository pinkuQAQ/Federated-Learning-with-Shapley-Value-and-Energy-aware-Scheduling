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

# 激活conda环境
source /data/home/zhaozhanshan/ENTER/etc/profile.d/conda.sh
conda activate flsv
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

# 切换到src目录
cd /data/home/zhaozhanshan/FLSV/src

# 创建目录
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
OUTPUT_FOLDER="ablation_$(date +%Y%m%d_%H%M%S)"

# ============================================
# [1/4] Ours 完整方法（基准）
# 指标：准确率 / 能耗 / Q(t) / 加密开销
# ============================================
echo ""
echo "[1/4] Running Ours (Full: SV + Energy + Lyapunov + Crypto)..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --shapley_update_method mean \
    --shapley_alpha 0.5 \
    --use_energy \
    --initial_energy 500.0 \
    --energy_threshold 50.0 \
    --use_lyapunov \
    --lyapunov_V 10.0 \
    --energy_budget 5.0 \
    --use_crypto \
    --output_folder $OUTPUT_FOLDER
echo "[1/4] Done!"

# ============================================
# [2/4] w/o Crypto
# 指标：准确率应与[1/4]相同，体现加密零开销
# ============================================
echo ""
echo "[2/4] Running w/o Crypto (SV + Energy + Lyapunov)..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --shapley_update_method mean \
    --shapley_alpha 0.5 \
    --use_energy \
    --initial_energy 500.0 \
    --energy_threshold 50.0 \
    --use_lyapunov \
    --lyapunov_V 10.0 \
    --energy_budget 5.0 \
    --output_folder $OUTPUT_FOLDER
echo "[2/4] Done!"

# ============================================
# [3/4] w/o Lyapunov
# 指标：Q(t)不稳定，能量约束无法保证
# ============================================
echo ""
echo "[3/4] Running w/o Lyapunov (SV + Energy + Crypto)..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --shapley_update_method mean \
    --shapley_alpha 0.5 \
    --use_energy \
    --initial_energy 500.0 \
    --energy_threshold 50.0 \
    --use_crypto \
    --output_folder $OUTPUT_FOLDER
echo "[3/4] Done!"

# ============================================
# [4/4] w/o SV（随机选择）
# 指标：准确率低于Ours，体现SV选择的贡献
# ============================================
echo ""
echo "[4/4] Running w/o SV (Random + Energy + Lyapunov + Crypto)..."
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
echo "[4/4] Done!"

echo ""
echo "========================================"
echo "All ablation experiments finished!"
echo "Results saved to: /data/home/zhaozhanshan/FLSV/save/$OUTPUT_FOLDER"
echo "End: $(date)"
echo "========================================"

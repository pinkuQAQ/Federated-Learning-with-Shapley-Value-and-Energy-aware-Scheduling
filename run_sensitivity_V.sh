#!/bin/bash
#SBATCH --job-name=FLSV_sens_V
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_sens_V_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_sens_V_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Sensitivity Analysis: Lyapunov V"
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

# 共享参数（与消融实验保持一致）
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
OUTPUT_FOLDER="sens_V_$(date +%Y%m%d_%H%M%S)"

echo "Output folder: $OUTPUT_FOLDER"
echo ""

# ============================================
# V=1  (强能量约束，弱 Shapley 贡献权重)
# ============================================
echo "[1/5] Running V=1 ..."
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
    --lyapunov_V 1.0 \
    --energy_budget 5.0 \
    --use_crypto \
    --output_folder $OUTPUT_FOLDER
echo "[1/5] Done!"

# ============================================
# V=5
# ============================================
echo ""
echo "[2/5] Running V=5 ..."
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
    --lyapunov_V 5.0 \
    --energy_budget 5.0 \
    --use_crypto \
    --output_folder $OUTPUT_FOLDER
echo "[2/5] Done!"

# ============================================
# V=10  (默认值，与消融实验一致)
# ============================================
echo ""
echo "[3/5] Running V=10 (default) ..."
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
echo "[3/5] Done!"

# ============================================
# V=20
# ============================================
echo ""
echo "[4/5] Running V=20 ..."
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
    --lyapunov_V 20.0 \
    --energy_budget 5.0 \
    --use_crypto \
    --output_folder $OUTPUT_FOLDER
echo "[4/5] Done!"

# ============================================
# V=50  (弱能量约束，强 Shapley 贡献权重)
# ============================================
echo ""
echo "[5/5] Running V=50 ..."
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
    --lyapunov_V 50.0 \
    --energy_budget 5.0 \
    --use_crypto \
    --output_folder $OUTPUT_FOLDER
echo "[5/5] Done!"

echo ""
echo "========================================"
echo "Sensitivity analysis for V finished!"
echo "Results saved to: /data/home/zhaozhanshan/FLSV/save/$OUTPUT_FOLDER"
echo "End: $(date)"
echo "========================================"

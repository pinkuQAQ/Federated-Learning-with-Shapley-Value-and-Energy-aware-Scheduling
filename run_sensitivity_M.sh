#!/bin/bash
#SBATCH --job-name=FLSV_sens_M
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_sens_M_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_sens_M_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Sensitivity Analysis: MC-Shapley iterations M"
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
DP_CLIP_NORM=1.0
DP_NOISE_MULTIPLIER=0.05
DP_ARGS="--use_local_dp --dp_clip_norm $DP_CLIP_NORM --dp_noise_multiplier $DP_NOISE_MULTIPLIER"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# ============================================
# M=5  (极少迭代，Shapley 估计最不准确)
# ============================================
echo "[1/5] Running M=5 ..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --shapley_update_method mean \
    --shapley_alpha 0.5 \
    --shapley_max_iter 5 \
    --use_energy \
    --initial_energy 500.0 \
    --energy_threshold 50.0 \
    --use_lyapunov \
    --lyapunov_V 10.0 \
    --energy_budget 5.0 \
    $DP_ARGS \
    --output_folder sens_M5_$RUN_TAG
echo "[1/5] Done!"

# ============================================
# M=10
# ============================================
echo ""
echo "[2/5] Running M=10 ..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --shapley_update_method mean \
    --shapley_alpha 0.5 \
    --shapley_max_iter 10 \
    --use_energy \
    --initial_energy 500.0 \
    --energy_threshold 50.0 \
    --use_lyapunov \
    --lyapunov_V 10.0 \
    --energy_budget 5.0 \
    $DP_ARGS \
    --output_folder sens_M10_$RUN_TAG
echo "[2/5] Done!"

# ============================================
# M=20  (默认值，与消融实验一致)
# ============================================
echo ""
echo "[3/5] Running M=20 (default) ..."
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
    $DP_ARGS \
    --output_folder sens_M20_$RUN_TAG
echo "[3/5] Done!"

# ============================================
# M=50
# ============================================
echo ""
echo "[4/5] Running M=50 ..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --shapley_update_method mean \
    --shapley_alpha 0.5 \
    --shapley_max_iter 50 \
    --use_energy \
    --initial_energy 500.0 \
    --energy_threshold 50.0 \
    --use_lyapunov \
    --lyapunov_V 10.0 \
    --energy_budget 5.0 \
    $DP_ARGS \
    --output_folder sens_M50_$RUN_TAG
echo "[4/5] Done!"

# ============================================
# M=100  (高精度，开销最大，用于验证 M=20 已饱和)
# ============================================
echo ""
echo "[5/5] Running M=100 ..."
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected $NUM_SELECTED \
    --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
    --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
    --shapley_update_method mean \
    --shapley_alpha 0.5 \
    --shapley_max_iter 100 \
    --use_energy \
    --initial_energy 500.0 \
    --energy_threshold 50.0 \
    --use_lyapunov \
    --lyapunov_V 10.0 \
    --energy_budget 5.0 \
    $DP_ARGS \
    --output_folder sens_M100_$RUN_TAG
echo "[5/5] Done!"

echo ""
echo "========================================"
echo "Sensitivity analysis for M finished!"
echo "Results saved to:"
echo "/data/home/zhaozhanshan/FLSV/save/sens_M5_$RUN_TAG"
echo "/data/home/zhaozhanshan/FLSV/save/sens_M10_$RUN_TAG"
echo "/data/home/zhaozhanshan/FLSV/save/sens_M20_$RUN_TAG"
echo "/data/home/zhaozhanshan/FLSV/save/sens_M50_$RUN_TAG"
echo "/data/home/zhaozhanshan/FLSV/save/sens_M100_$RUN_TAG"
echo "End: $(date)"
echo "========================================"

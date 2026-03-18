#!/bin/bash
#SBATCH --job-name=FLSV_noniid
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=96:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_noniid_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_noniid_%j.err

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

DATASET=cifar
MODEL=cnn
EPOCHS=100
NUM_USERS=100
NUM_SELECTED=10
LOCAL_EP=2
LOCAL_BS=32
LR=0.01
SEED=42

# 4个不同的 Dirichlet α 值
ALPHAS=(0.1 0.25 0.5 1.0)

TOTAL=$((${#ALPHAS[@]} * 5))
COUNT=0

for ALPHA in "${ALPHAS[@]}"; do
    OUTPUT_FOLDER="noniid_cmp_alpha${ALPHA}_$(date +%Y%m%d)"

    echo ""
    echo "========================================"
    echo "Alpha = ${ALPHA}"
    echo "Output: ${OUTPUT_FOLDER}"
    echo "========================================"

    COUNT=$((COUNT + 1))
    echo "[${COUNT}/${TOTAL}] Ours (alpha=${ALPHA})..."
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --shapley_update_method mean --shapley_alpha 0.5 --shapley_max_iter 20 \
        --use_energy --sigma_squared 1.0 --initial_energy 500.0 --energy_threshold 50.0 \
        --use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 \
        --output_folder $OUTPUT_FOLDER

    COUNT=$((COUNT + 1))
    echo "[${COUNT}/${TOTAL}] FedAvg (alpha=${ALPHA})..."
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --no_shapley --selection_method random \
        --output_folder $OUTPUT_FOLDER

    COUNT=$((COUNT + 1))
    echo "[${COUNT}/${TOTAL}] PoC (alpha=${ALPHA})..."
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --no_shapley --selection_method poc \
        --output_folder $OUTPUT_FOLDER

    COUNT=$((COUNT + 1))
    echo "[${COUNT}/${TOTAL}] UCB1 (alpha=${ALPHA})..."
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --no_shapley --selection_method ucb --ucb_c 1.0 \
        --output_folder $OUTPUT_FOLDER

    COUNT=$((COUNT + 1))
    echo "[${COUNT}/${TOTAL}] FedProx (alpha=${ALPHA})..."
    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $ALPHA --seed $SEED \
        --no_shapley --selection_method random --use_fedprox --fedprox_mu 0.01 \
        --output_folder $OUTPUT_FOLDER

    echo "Alpha=${ALPHA} done! Results: /data/home/zhaozhanshan/FLSV/save/${OUTPUT_FOLDER}"
done

echo ""
echo "========================================"
echo "All non-IID experiments finished!"
echo "End: $(date)"
echo "========================================"

#!/bin/bash
#SBATCH --job-name=FLSV_cifar_ms
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_cifar_ms_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_cifar_ms_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Task: CIFAR-10 multi-seed main results"
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
ALPHAS=(0.25 0.5)
SEEDS=(42 52 62 72 82)
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"

for ALPHA in "${ALPHAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUTPUT_FOLDER="cifar_ms_alpha${ALPHA}_seed${SEED}_${RUN_TAG}"

        echo ""
        echo "========================================"
        echo "Alpha=${ALPHA} Seed=${SEED}"
        echo "Output: ${OUTPUT_FOLDER}"
        echo "========================================"

        echo "[1/5] Ours"
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
            --initial_energy 500.0 \
            --energy_threshold 50.0 \
            --use_lyapunov \
            --lyapunov_V 10.0 \
            --energy_budget 5.0 \
            --output_folder $OUTPUT_FOLDER

        echo "[2/5] FedAvg"
        python federated_main.py \
            --dataset $DATASET --model $MODEL --epochs $EPOCHS \
            --num_users $NUM_USERS --num_selected $NUM_SELECTED \
            --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
            --dirichlet_alpha $ALPHA --seed $SEED \
            --no_shapley \
            --selection_method random \
            --output_folder $OUTPUT_FOLDER

        echo "[3/5] PoC"
        python federated_main.py \
            --dataset $DATASET --model $MODEL --epochs $EPOCHS \
            --num_users $NUM_USERS --num_selected $NUM_SELECTED \
            --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
            --dirichlet_alpha $ALPHA --seed $SEED \
            --no_shapley \
            --selection_method poc \
            --output_folder $OUTPUT_FOLDER

        echo "[4/5] UCB"
        python federated_main.py \
            --dataset $DATASET --model $MODEL --epochs $EPOCHS \
            --num_users $NUM_USERS --num_selected $NUM_SELECTED \
            --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
            --dirichlet_alpha $ALPHA --seed $SEED \
            --no_shapley \
            --selection_method ucb \
            --ucb_c 1.0 \
            --output_folder $OUTPUT_FOLDER

        echo "[5/5] FedProx"
        python federated_main.py \
            --dataset $DATASET --model $MODEL --epochs $EPOCHS \
            --num_users $NUM_USERS --num_selected $NUM_SELECTED \
            --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
            --dirichlet_alpha $ALPHA --seed $SEED \
            --no_shapley \
            --selection_method random \
            --use_fedprox \
            --fedprox_mu 0.01 \
            --output_folder $OUTPUT_FOLDER
    done
done

echo ""
echo "========================================"
echo "Completed CIFAR-10 multi-seed main runs"
echo "End: $(date)"
echo "========================================"

#!/bin/bash
#SBATCH --job-name=FLSV_multidataset
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=96:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_multidataset_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_multidataset_%j.err

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

MODEL=cnn
EPOCHS=100
NUM_USERS=100
NUM_SELECTED=10
LOCAL_EP=2
LOCAL_BS=32
LR=0.01
SEED=42

# 三个 non-IID 程度：强/中/弱
ALPHAS=(0.1 0.5 1.0)
DATASETS=(mnist fmnist)

TOTAL=$((${#DATASETS[@]} * ${#ALPHAS[@]} * 5))
COUNT=0

for DATASET in "${DATASETS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        OUTPUT_FOLDER="multidataset_${DATASET}_alpha${ALPHA}_$(date +%Y%m%d)"

        echo ""
        echo "========================================"
        echo "Dataset=${DATASET}  Alpha=${ALPHA}"
        echo "Output: ${OUTPUT_FOLDER}"
        echo "========================================"

        COUNT=$((COUNT + 1))
        echo "[${COUNT}/${TOTAL}] Ours (${DATASET}, alpha=${ALPHA})..."
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
        echo "[${COUNT}/${TOTAL}] FedAvg (${DATASET}, alpha=${ALPHA})..."
        python federated_main.py \
            --dataset $DATASET --model $MODEL --epochs $EPOCHS \
            --num_users $NUM_USERS --num_selected $NUM_SELECTED \
            --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
            --dirichlet_alpha $ALPHA --seed $SEED \
            --no_shapley --selection_method random \
            --output_folder $OUTPUT_FOLDER

        COUNT=$((COUNT + 1))
        echo "[${COUNT}/${TOTAL}] PoC (${DATASET}, alpha=${ALPHA})..."
        python federated_main.py \
            --dataset $DATASET --model $MODEL --epochs $EPOCHS \
            --num_users $NUM_USERS --num_selected $NUM_SELECTED \
            --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
            --dirichlet_alpha $ALPHA --seed $SEED \
            --no_shapley --selection_method poc \
            --output_folder $OUTPUT_FOLDER

        COUNT=$((COUNT + 1))
        echo "[${COUNT}/${TOTAL}] UCB (${DATASET}, alpha=${ALPHA})...")
        python federated_main.py \
            --dataset $DATASET --model $MODEL --epochs $EPOCHS \
            --num_users $NUM_USERS --num_selected $NUM_SELECTED \
            --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
            --dirichlet_alpha $ALPHA --seed $SEED \
            --no_shapley --selection_method ucb --ucb_c 1.0 \
            --output_folder $OUTPUT_FOLDER

        COUNT=$((COUNT + 1))
        echo "[${COUNT}/${TOTAL}] FedProx (${DATASET}, alpha=${ALPHA})..."
        python federated_main.py \
            --dataset $DATASET --model $MODEL --epochs $EPOCHS \
            --num_users $NUM_USERS --num_selected $NUM_SELECTED \
            --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
            --dirichlet_alpha $ALPHA --seed $SEED \
            --no_shapley --selection_method random --use_fedprox --fedprox_mu 0.01 \
            --output_folder $OUTPUT_FOLDER

        echo "Done: Dataset=${DATASET} Alpha=${ALPHA}"
    done
done

echo ""
echo "========================================"
echo "All multi-dataset experiments finished!"
echo "End: $(date)"
echo "========================================"

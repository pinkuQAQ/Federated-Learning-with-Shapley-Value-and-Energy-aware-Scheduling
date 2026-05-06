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
NUM_SELECTED=5
LOCAL_EP=2
LOCAL_BS=32
LR=0.01
DIRICHLET_ALPHA=0.1
SEED=42
DP_CLIP_NORM=1.0
DP_ADAPTIVE_ARGS="--lightweight_dp --public_pretrain_epochs 3 --public_pretrain_samples 20000 --dp_advanced --dp_noise_schedule linear_increase --dp_noise_start_multiplier 0.7 --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0"
DP_NOISE_MULTIPLIER=1.0
DP_ARGS="--privacy_mode central --dp_clip_norm $DP_CLIP_NORM $DP_ADAPTIVE_ARGS --dp_noise_multiplier $DP_NOISE_MULTIPLIER --dp_shapley_alpha 0.9"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

for M in 5 10 20 50; do
    OUTPUT_FOLDER="sens_M${M}_${RUN_TAG}"

    echo ""
    echo "========================================"
    echo "Running MC-Shapley M=${M}"
    echo "Output: ${OUTPUT_FOLDER}"
    echo "========================================"

    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $NUM_SELECTED \
        --local_ep $LOCAL_EP --local_bs $LOCAL_BS --lr $LR \
        --dirichlet_alpha $DIRICHLET_ALPHA --seed $SEED \
        --shapley_update_method mean \
        --shapley_alpha 0.5 \
        --shapley_max_iter $M \
        --use_energy \
        --initial_energy 500.0 \
        --energy_threshold 50.0 \
        --use_lyapunov \
        --lyapunov_V 10.0 \
        --energy_budget 5.0 \
        $DP_ARGS \
        --output_folder $OUTPUT_FOLDER

    echo "M=${M} done"
done

echo ""
echo "========================================"
echo "Sensitivity analysis for MC-Shapley M finished!"
echo "Run tag: $RUN_TAG"
echo "End: $(date)"
echo "========================================"

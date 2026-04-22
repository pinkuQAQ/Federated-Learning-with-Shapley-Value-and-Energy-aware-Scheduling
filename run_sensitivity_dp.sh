#!/bin/bash
#SBATCH --job-name=FLSV_sens_dp
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_sens_dp_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_sens_dp_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Task: Noise-multiplier sweep (σ_dp) for the optional upload-perturbation module"
echo "Note: the sweep spans σ_dp ∈ {0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0}."
echo "      σ_dp ≤ 0.2 is the utility-preserving regime (no formal DP)."
echo "      σ_dp ∈ {0.5, 1.0} is included to mark the regime where (ε,δ)"
echo "      starts to bind meaningfully at the cost of accuracy."
echo "========================================"

source /data/home/zhaozhanshan/ENTER/etc/profile.d/conda.sh
conda activate flsv
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

cd /data/home/zhaozhanshan/FLSV/src
mkdir -p /data/home/zhaozhanshan/FLSV/logs
mkdir -p /data/home/zhaozhanshan/FLSV/save

DATASET=cifar
MODEL=cnn
EPOCHS=80
NUM_USERS=100
NUM_SELECTED=10
LOCAL_EP=2
LOCAL_BS=32
LR=0.01
DIRICHLET_ALPHA=0.1
SEED=42
DP_CLIP_NORM=1.0
NOISE_MULTIPLIERS=(0.0 0.01 0.05 0.1 0.2 0.5 1.0)
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

for SIGMA_DP in "${NOISE_MULTIPLIERS[@]}"; do
    OUTPUT_FOLDER="sens_dp_sigma${SIGMA_DP}_${RUN_TAG}"

    echo ""
    echo "========================================"
    echo "sigma_dp = ${SIGMA_DP}"
    echo "Output: ${OUTPUT_FOLDER}"
    echo "========================================"

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
        --use_local_dp \
        --dp_clip_norm $DP_CLIP_NORM \
        --dp_noise_multiplier $SIGMA_DP \
        --output_folder $OUTPUT_FOLDER
done

echo ""
echo "========================================"
echo "Sensitivity analysis for local DP finished!"
echo "End: $(date)"
echo "========================================"

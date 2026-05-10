#!/bin/bash
#SBATCH --job-name=FLSV_channel_dp
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_channel_dp_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_channel_dp_%j.err

set -e

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Task: Lightweight channel-noise-assisted privacy sweep"
echo "========================================"

source ~/.bashrc
conda activate flsv
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

cd /data/home/zhaozhanshan/FLSV/src
mkdir -p /data/home/zhaozhanshan/FLSV/logs
mkdir -p /data/home/zhaozhanshan/FLSV/save

DATASET=cifar
MODEL=cnn
EPOCHS=100
NUM_USERS=100
LOCAL_EP=2
LOCAL_BS=32
LR=0.01
DIRICHLET_ALPHA=0.1
SEED=42
DP_CLIP_NORM=1.0
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

COMMON_DP_ARGS="--dp_advanced --dp_noise_schedule linear_increase --dp_noise_start_multiplier 0.7 --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0"

# Format: "K unused_algorithmic_sigma base_channel_sigma"
# Channel-only route: privacy perturbation comes only from equivalent channel noise.
CONFIGS=(
  "10 0.0 0.5"
  "10 0.0 0.75"
  "10 0.0 1.0"
)

echo ""
echo "[baseline] K=10, clipping only, no aggregate privacy noise"
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected 10 \
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
    --privacy_mode central \
    --dp_clip_norm $DP_CLIP_NORM \
    $COMMON_DP_ARGS \
    --dp_noise_multiplier 0.0 \
    --dp_channel_noise_multiplier 0.0 \
    --output_folder channel_dp_channelonly_K10_full_ch0_${RUN_TAG}

for CFG in "${CONFIGS[@]}"; do
    read -r K SIGMA_EFF CH_SIGMA <<< "$CFG"
    OUTPUT_FOLDER="channel_dp_channelonly_K${K}_full_ch${CH_SIGMA}_${RUN_TAG}"

    echo ""
    echo "========================================"
    echo "K=${K}, algorithmic sigma=${SIGMA_EFF}, base channel sigma=${CH_SIGMA}"
    echo "Output: ${OUTPUT_FOLDER}"
    echo "========================================"

    python federated_main.py \
        --dataset $DATASET --model $MODEL --epochs $EPOCHS \
        --num_users $NUM_USERS --num_selected $K \
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
        --privacy_mode central \
        --dp_clip_norm $DP_CLIP_NORM \
        $COMMON_DP_ARGS \
        --dp_noise_multiplier $SIGMA_EFF \
        --dp_channel_noise_multiplier $CH_SIGMA \
        --output_folder $OUTPUT_FOLDER
done

cd /data/home/zhaozhanshan/FLSV
python src/summarize_dp_results.py --pattern "channel_dp_*_${RUN_TAG}" > "channel_dp_summary_${RUN_TAG}.txt"
cat "channel_dp_summary_${RUN_TAG}.txt"

echo ""
echo "========================================"
echo "Lightweight channel-assisted privacy sweep finished!"
echo "Summary: /data/home/zhaozhanshan/FLSV/channel_dp_summary_${RUN_TAG}.txt"
echo "End: $(date)"
echo "========================================"

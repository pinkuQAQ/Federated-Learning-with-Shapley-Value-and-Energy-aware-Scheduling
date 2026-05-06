#!/bin/bash
#SBATCH --job-name=FLSV_chdp_rescue
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_chdp_rescue_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_chdp_rescue_%j.err

set -e

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "Task: Channel-noise-assisted lightweight DP rescue sweep"
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

COMMON_DP_ARGS="--lightweight_dp --public_pretrain_epochs 3 --public_pretrain_samples 20000 --dp_advanced --dp_noise_schedule linear_increase --dp_noise_start_multiplier 0.7 --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode topup --dp_channel_gain_cap 2.0"

# Format: "K target_sigma_eff base_channel_sigma"
# These choices target epsilon_h roughly in the 3--5 range while reducing per-round aggregate noise through larger K.
CONFIGS=(
  "10 2.0 0.8"
  "15 2.5 1.0"
  "20 3.0 1.2"
)

echo ""
echo "[baseline] K=15, clipping only, no aggregate DP noise"
python federated_main.py \
    --dataset $DATASET --model $MODEL --epochs $EPOCHS \
    --num_users $NUM_USERS --num_selected 15 \
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
    --output_folder chdp_rescue_K15_sigma0_ch0_${RUN_TAG}

for CFG in "${CONFIGS[@]}"; do
    read -r K SIGMA_EFF CH_SIGMA <<< "$CFG"
    OUTPUT_FOLDER="chdp_rescue_K${K}_sigma${SIGMA_EFF}_ch${CH_SIGMA}_${RUN_TAG}"

    echo ""
    echo "========================================"
    echo "K=${K}, target sigma_eff=${SIGMA_EFF}, base channel sigma=${CH_SIGMA}"
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
python src/summarize_cdp_local.py --pattern "chdp_rescue_*_${RUN_TAG}" > "chdp_rescue_summary_${RUN_TAG}.txt"
cat "chdp_rescue_summary_${RUN_TAG}.txt"

echo ""
echo "========================================"
echo "Channel-assisted DP rescue sweep finished!"
echo "Summary: /data/home/zhaozhanshan/FLSV/chdp_rescue_summary_${RUN_TAG}.txt"
echo "End: $(date)"
echo "========================================"

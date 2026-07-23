#!/bin/bash
#SBATCH --job-name=FLSV_sched_ablation
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_sched_ablation_%A_%a.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_sched_ablation_%A_%a.err

set -eo pipefail

source ~/.bashrc
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /data/home/zhaozhanshan/ENTER/envs/flsv
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

PROJECT_ROOT=/data/home/zhaozhanshan/FLSV
cd /data/home/zhaozhanshan/FLSV/src
mkdir -p /data/home/zhaozhanshan/FLSV/logs /data/home/zhaozhanshan/FLSV/save

SEEDS=(7 21 42 77 123 888 1001 2024 3407 31415)
TASK_ID=$SLURM_ARRAY_TASK_ID
RUN_TAG=job$SLURM_ARRAY_JOB_ID
FILTER_DEFAULT=0.30
case $TASK_ID in
    0) VARIANT=legacy; COLD_START=round_robin; FILTER_Q=0.0 ;;
    1) VARIANT=online; COLD_START=online; FILTER_Q=0.0 ;;
    2) VARIANT=channel_filter; COLD_START=round_robin; FILTER_Q=$FILTER_DEFAULT ;;
    3) VARIANT=online_channel; COLD_START=online; FILTER_Q=$FILTER_DEFAULT ;;
    *) echo Invalid-array-task-$TASK_ID; exit 2 ;;
esac

echo ========================================
echo Job-$SLURM_ARRAY_JOB_ID-$TASK_ID
echo Variant-$VARIANT
echo Cold-start-$COLD_START
echo Channel-filter-quantile-$FILTER_Q
echo Seeds-${SEEDS[@]}
echo ========================================

for SEED in ${SEEDS[@]}; do
    OUT=ours_scheduler_ablation/$RUN_TAG/$VARIANT/seed$SEED
    if compgen -G /data/home/zhaozhanshan/FLSV/save/$OUT/'*.pkl' >/dev/null; then
        echo Skip-existing-$OUT
        continue
    fi
    echo Running-$VARIANT-seed-$SEED
    echo Output-$OUT

    python federated_main.py \
        --dataset cifar --model cnn --epochs 100 \
        --num_users 100 --num_selected 5 \
        --local_ep 2 --local_bs 32 \
        --optimizer sgd --lr 0.01 --momentum 0.5 --weight_decay 5e-4 \
        --dirichlet_alpha 0.1 --test_size 10000 --gpu 0 --seed $SEED \
        --selection_method hybrid --selection_beta 0.25 \
        --use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 \
        --sv_weight 0.7 --battery_weight 0.15 --channel_weight 0.15 \
        --shapley_estimator complementary --shapley_allocation neyman \
        --shapley_pilot_samples 1 --shapley_max_iter 20 \
        --shapley_update_method mean --shapley_alpha 0.5 \
        --shapley_cold_start $COLD_START --shapley_exploration_slots 1 \
        --shapley_ucb_c 0.25 --channel_filter_quantile $FILTER_Q \
        --channel_min_gain 0.0 \
        --use_energy --sigma_squared 1.0 --channel_model rayleigh \
        --initial_energy 500.0 --energy_threshold 50.0 \
        --privacy_mode central --dp_clip_norm 1.0 \
        --dp_advanced --dp_noise_schedule constant --dp_adaptive_clip \
        --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 \
        --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 \
        --dp_channel_assisted --dp_channel_mode channel_only \
        --dp_channel_gain_cap 2.0 --dp_noise_multiplier 0.0 \
        --dp_channel_noise_multiplier 0.1 --output_folder $OUT
done

echo Finished-$VARIANT

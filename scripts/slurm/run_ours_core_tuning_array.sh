#!/bin/bash
#SBATCH --job-name=FLSV_core_tune
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_core_tune_%A_%a.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_core_tune_%A_%a.err

set -eo pipefail

source ~/.bashrc
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate /data/home/zhaozhanshan/ENTER/envs/flsv
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

PROJECT_ROOT=/data/home/zhaozhanshan/FLSV
cd /data/home/zhaozhanshan/FLSV/src
mkdir -p /data/home/zhaozhanshan/FLSV/logs /data/home/zhaozhanshan/FLSV/save

# New development seeds: do not reuse the ten seeds already inspected.
TUNING_SEEDS=(11 29 101)
FILTER_QUANTILES=(0.30 0.40 0.50 0.60)
SHAPLEY_BUDGETS=(10 20)
UPDATE_SPECS=(mean exp0p2 exp0p5 exp0p8)
WEIGHT_SPECS=(base ch20 ch25)

TASK_ID=$SLURM_ARRAY_TASK_ID
TASK_COUNT=4
RUN_TAG=job$SLURM_ARRAY_JOB_ID
SKIP_EXISTING=1
TOTAL_CONFIGS=$((${#FILTER_QUANTILES[@]} * ${#SHAPLEY_BUDGETS[@]} * ${#UPDATE_SPECS[@]} * ${#WEIGHT_SPECS[@]}))
TOTAL_RUNS=$((TOTAL_CONFIGS * ${#TUNING_SEEDS[@]}))

set_update_args() {
    case $1 in
        mean) UPDATE_METHOD=mean; UPDATE_ALPHA=0.5 ;;
        exp0p2) UPDATE_METHOD=exponential; UPDATE_ALPHA=0.2 ;;
        exp0p5) UPDATE_METHOD=exponential; UPDATE_ALPHA=0.5 ;;
        exp0p8) UPDATE_METHOD=exponential; UPDATE_ALPHA=0.8 ;;
        *) echo ERROR-unknown-update-spec-$1; exit 2 ;;
    esac
}

set_weight_args() {
    case $1 in
        base) SV_WEIGHT=0.70; BATTERY_WEIGHT=0.15; CHANNEL_WEIGHT=0.15 ;;
        ch20) SV_WEIGHT=0.65; BATTERY_WEIGHT=0.15; CHANNEL_WEIGHT=0.20 ;;
        ch25) SV_WEIGHT=0.60; BATTERY_WEIGHT=0.15; CHANNEL_WEIGHT=0.25 ;;
        *) echo ERROR-unknown-weight-spec-$1; exit 2 ;;
    esac
}

echo ========================================
echo Job-$SLURM_ARRAY_JOB_ID-$TASK_ID
echo Core-tuning-grid-$TOTAL_CONFIGS-configurations-$TOTAL_RUNS-runs
echo Worker-$TASK_ID-of-$TASK_COUNT
echo Tuning-seeds-${TUNING_SEEDS[@]}
echo Output-tag-$RUN_TAG
echo ========================================

run_index=0
completed=0
for FILTER_Q in ${FILTER_QUANTILES[@]}; do
  for SHAPLEY_M in ${SHAPLEY_BUDGETS[@]}; do
    for UPDATE_SPEC in ${UPDATE_SPECS[@]}; do
      set_update_args $UPDATE_SPEC
      for WEIGHT_SPEC in ${WEIGHT_SPECS[@]}; do
        set_weight_args $WEIGHT_SPEC
        Q_LABEL=${FILTER_Q/./p}
        CONFIG=q$Q_LABEL-M$SHAPLEY_M-$UPDATE_SPEC-$WEIGHT_SPEC
        for SEED in ${TUNING_SEEDS[@]}; do
          if ((run_index % TASK_COUNT != TASK_ID)); then
            run_index=$((run_index + 1))
            continue
          fi
          OUT=ours_core_tuning/$RUN_TAG/$CONFIG/seed$SEED
          if [ $SKIP_EXISTING = 1 ] && compgen -G /data/home/zhaozhanshan/FLSV/save/$OUT/'*.pkl' >/dev/null; then
            echo Skip-existing-$OUT
            run_index=$((run_index + 1))
            continue
          fi
          echo Running-index-$run_index-config-$CONFIG-seed-$SEED

          python federated_main.py \
            --dataset cifar --model cnn --epochs 100 \
            --num_users 100 --num_selected 5 \
            --local_ep 2 --local_bs 32 \
            --optimizer sgd --lr 0.01 --momentum 0.5 --weight_decay 5e-4 \
            --dirichlet_alpha 0.1 --test_size 10000 --gpu 0 --seed $SEED \
            --selection_method hybrid --selection_beta 0.25 \
            --shapley_cold_start round_robin \
            --shapley_estimator complementary --shapley_allocation neyman \
            --shapley_pilot_samples 1 --shapley_max_iter $SHAPLEY_M \
            --shapley_update_method $UPDATE_METHOD --shapley_alpha $UPDATE_ALPHA \
            --use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 \
            --sv_weight $SV_WEIGHT --battery_weight $BATTERY_WEIGHT \
            --channel_weight $CHANNEL_WEIGHT \
            --channel_filter_quantile $FILTER_Q --channel_min_gain 0.0 \
            --use_energy --sigma_squared 1.0 --channel_model rayleigh \
            --initial_energy 500.0 --energy_threshold 50.0 \
            --privacy_mode central --dp_clip_norm 1.0 \
            --dp_advanced --dp_noise_schedule constant --dp_adaptive_clip \
            --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 \
            --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 \
            --dp_channel_assisted --dp_channel_mode channel_only \
            --dp_channel_gain_cap 2.0 --dp_noise_multiplier 0.0 \
            --dp_channel_noise_multiplier 0.1 --output_folder $OUT

          completed=$((completed + 1))
          run_index=$((run_index + 1))
        done
      done
    done
  done
done

echo Worker-$TASK_ID-finished-$completed-runs
echo Results-/data/home/zhaozhanshan/FLSV/save/ours_core_tuning/$RUN_TAG

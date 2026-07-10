#!/bin/bash
#SBATCH --job-name=FLSV_ch_eps
#SBATCH --partition=p3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=120:00:00
#SBATCH --output=/data/home/zhaozhanshan/FLSV/logs/slurm_ch_eps_%j.out
#SBATCH --error=/data/home/zhaozhanshan/FLSV/logs/slurm_ch_eps_%j.err

set -e

echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "Task: Main-setting channel-noise epsilon_ref sweep"
echo "========================================"

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base)
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found. Please check your Conda installation path."
    exit 1
fi
# The cluster may list the project environment by absolute path instead of
# the short name "flsv". Prefer the known path, but keep env vars overrideable.
CONDA_ENV_NAME=${CONDA_ENV_NAME:-flsv}
CONDA_ENV_PATH=${CONDA_ENV_PATH:-/data/home/zhaozhanshan/ENTER/envs/flsv}
if [ -d "${CONDA_ENV_PATH}" ]; then
    echo "Activating conda env by path: ${CONDA_ENV_PATH}"
    conda activate "${CONDA_ENV_PATH}"
else
    echo "Activating conda env by name: ${CONDA_ENV_NAME}"
    conda activate "${CONDA_ENV_NAME}"
fi
export LD_PRELOAD=/data/home/zhaozhanshan/lib/libittnotify_stub.so

PROJECT_ROOT=/data/home/zhaozhanshan/FLSV
cd "${PROJECT_ROOT}/src"
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/save"

# Match the main-table operating point in the current manuscript.
DATASET=${DATASET:-cifar}
MODEL=${MODEL:-cnn}
EPOCHS=${EPOCHS:-100}
NUM_USERS=${NUM_USERS:-100}
NUM_SELECTED=${NUM_SELECTED:-5}
LOCAL_EP=${LOCAL_EP:-2}
LOCAL_BS=${LOCAL_BS:-32}
LR=${LR:-0.01}
DIRICHLET_ALPHA=${DIRICHLET_ALPHA:-0.1}
TEST_SIZE=${TEST_SIZE:-10000}
GPU_ID=${GPU_ID:-0}
SEEDS=(${SEEDS:-42 123 2024})
CHANNEL_SIGMAS=(${CHANNEL_SIGMAS:-0.0 0.1 0.25 0.5})
SELECTION_BETA=${SELECTION_BETA:-1.0}
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

DP_CLIP_NORM=${DP_CLIP_NORM:-1.0}
SCHED_WEIGHTS="--sv_weight 0.7 --battery_weight 0.15 --channel_weight 0.15"
COMMON_DP_ARGS="--dp_advanced --dp_noise_schedule constant --dp_adaptive_clip --dp_clip_scope layer --dp_clip_percentile 80 --dp_clip_ema 0.8 --dp_clip_growth 1.2 --dp_min_clip_norm 0.05 --dp_max_clip_norm 1.0 --dp_channel_assisted --dp_channel_mode channel_only --dp_channel_gain_cap 2.0"

BASE_ARGS="--dataset ${DATASET} --model ${MODEL} --epochs ${EPOCHS} --num_users ${NUM_USERS} --num_selected ${NUM_SELECTED} --local_ep ${LOCAL_EP} --local_bs ${LOCAL_BS} --lr ${LR} --dirichlet_alpha ${DIRICHLET_ALPHA} --test_size ${TEST_SIZE} --gpu ${GPU_ID}"
ENERGY_ARGS="--use_energy --sigma_squared 1.0 --initial_energy 500.0 --energy_threshold 50.0"
LYAP_ARGS="--use_lyapunov --lyapunov_V 10.0 --energy_budget 5.0 --selection_beta ${SELECTION_BETA}"
SV_UPDATE_ARGS="--shapley_update_method mean --shapley_alpha 0.5"
SV_CC_ARGS="--shapley_estimator complementary --shapley_allocation neyman --shapley_pilot_samples 1 --shapley_max_iter 20"

echo ""
echo "Configuration"
echo "  RUN_TAG=${RUN_TAG}"
echo "  SEEDS=${SEEDS[*]}"
echo "  CHANNEL_SIGMAS=${CHANNEL_SIGMAS[*]}"
echo "  EPOCHS=${EPOCHS}, N=${NUM_USERS}, K=${NUM_SELECTED}, alpha=${DIRICHLET_ALPHA}"
echo "  local_ep=${LOCAL_EP}, batch=${LOCAL_BS}, lr=${LR}"
echo "  Output pattern: ${PROJECT_ROOT}/save/channel_eps_main_${RUN_TAG}_ch*_seed*"
echo "========================================"

run_cmd() {
    echo ""
    echo "----------------------------------------"
    echo "$1"
    echo "Output folder: $2"
    echo "Start: $(date)"
    echo "----------------------------------------"
    shift 2
    python federated_main.py "$@"
    echo "Done: $(date)"
}

for CH_SIGMA in "${CHANNEL_SIGMAS[@]}"; do
    CH_LABEL=${CH_SIGMA//./p}
    DP_ARGS="--privacy_mode central --dp_clip_norm ${DP_CLIP_NORM} ${COMMON_DP_ARGS} --dp_noise_multiplier 0.0 --dp_channel_noise_multiplier ${CH_SIGMA}"

    for SEED in "${SEEDS[@]}"; do
        OUT="channel_eps_main_${RUN_TAG}_ch${CH_LABEL}_seed${SEED}"
        run_cmd "Ours channel multiplier=${CH_SIGMA}, seed=${SEED}" "${OUT}" \
            ${BASE_ARGS} --seed ${SEED} \
            ${SV_CC_ARGS} ${SV_UPDATE_ARGS} \
            ${ENERGY_ARGS} ${LYAP_ARGS} \
            ${SCHED_WEIGHTS} ${DP_ARGS} \
            --output_folder "${OUT}"
    done
done

cd "${PROJECT_ROOT}"

SUMMARY_TXT="channel_eps_main_summary_${RUN_TAG}.txt"
python src/summarize_dp_results.py --pattern "channel_eps_main_${RUN_TAG}_*" > "${SUMMARY_TXT}"
cat "${SUMMARY_TXT}"

python - "${RUN_TAG}" <<'PY'
import csv
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np

run_tag = sys.argv[1]
root = Path("save")
folders = sorted(root.glob(f"channel_eps_main_{run_tag}_ch*_seed*"))
rows = []
for folder in folders:
    pkls = sorted(folder.glob("*.pkl"))
    if not pkls:
        continue
    match = re.search(r"_ch([^_]+)_seed(\d+)$", folder.name)
    if not match:
        continue
    ch = float(match.group(1).replace("p", "."))
    seed = int(match.group(2))
    with pkls[0].open("rb") as f:
        data = pickle.load(f)
    acc = np.asarray(data.get("test_accuracy", []), dtype=np.float64)
    if acc.size and np.nanmax(acc) <= 1.5:
        acc = acc * 100.0
    dp_stats = data.get("dp_statistics", {}) or {}
    eps = float(dp_stats.get("update_epsilon", math.inf))
    rows.append({
        "channel_multiplier": ch,
        "seed": seed,
        "epsilon_ref": eps,
        "last5_acc": float(acc[-5:].mean()) if acc.size >= 5 else math.nan,
        "final_acc": float(acc[-1]) if acc.size else math.nan,
        "best_acc": float(np.nanmax(acc)) if acc.size else math.nan,
        "folder": folder.name,
    })

summary_dir = root / "channel_eps_main" / run_tag / "summary_tables"
summary_dir.mkdir(parents=True, exist_ok=True)

run_csv = summary_dir / "channel_epsilon_runs.csv"
with run_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["channel_multiplier", "seed", "epsilon_ref", "last5_acc", "final_acc", "best_acc", "folder"])
    writer.writeheader()
    writer.writerows(rows)

groups = {}
for row in rows:
    groups.setdefault(row["channel_multiplier"], []).append(row)

def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std

agg_rows = []
for ch in sorted(groups):
    group = groups[ch]
    eps_values = [r["epsilon_ref"] for r in group]
    finite_eps = [v for v in eps_values if math.isfinite(v)]
    eps_mean = math.inf if not finite_eps else float(np.mean(finite_eps))
    last5_mean, last5_std = mean_std([r["last5_acc"] for r in group])
    final_mean, final_std = mean_std([r["final_acc"] for r in group])
    best_mean, best_std = mean_std([r["best_acc"] for r in group])
    agg_rows.append({
        "channel_multiplier": ch,
        "n": len(group),
        "seeds": "[" + ",".join(str(r["seed"]) for r in sorted(group, key=lambda x: x["seed"])) + "]",
        "epsilon_ref_mean": eps_mean,
        "last5_acc_mean": last5_mean,
        "last5_acc_std": last5_std,
        "final_acc_mean": final_mean,
        "final_acc_std": final_std,
        "best_acc_mean": best_mean,
        "best_acc_std": best_std,
    })

summary_csv = summary_dir / "channel_epsilon_summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()) if agg_rows else ["channel_multiplier"])
    writer.writeheader()
    writer.writerows(agg_rows)

def fmt_eps(value):
    if math.isinf(value):
        return "inf"
    if abs(value) >= 1000:
        return f"{value:.2e}"
    return f"{value:.2f}"

table_tex = summary_dir / "channel_epsilon_table.tex"
with table_tex.open("w", encoding="utf-8") as f:
    f.write("% Generated by run_channel_epsilon_main_sweep.sh\n")
    f.write("% Columns: channel multiplier, epsilon_ref, last-5 accuracy, final accuracy\n")
    for row in agg_rows:
        f.write(
            f"{row['channel_multiplier']:.2f} & "
            f"{fmt_eps(row['epsilon_ref_mean'])} & "
            f"{row['last5_acc_mean']:.2f} $\\pm$ {row['last5_acc_std']:.2f} & "
            f"{row['final_acc_mean']:.2f} $\\pm$ {row['final_acc_std']:.2f} \\\\\n"
        )

print(f"Wrote {run_csv}")
print(f"Wrote {summary_csv}")
print(f"Wrote {table_tex}")
PY

echo ""
echo "========================================"
echo "Main-setting channel epsilon sweep finished!"
echo "Raw row summary: ${PROJECT_ROOT}/${SUMMARY_TXT}"
echo "Aggregated tables: ${PROJECT_ROOT}/save/channel_eps_main/${RUN_TAG}/summary_tables"
echo "End: $(date)"
echo "========================================"

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize supplementary SV experiment results.

The script scans save/sv_supp/<run_tag>/ and writes compact CSV, JSON, and
LaTeX tables under save/sv_supp/<run_tag>/summary_tables/.
"""

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SAVE = ROOT / "save"

MAIN_ORDER = [
    ("hybrid_SV_Energy_Lyapunov_CDP", "Ours"),
    ("random_CDP", "FedAvg"),
    ("random_FedProx_CDP", "FedProx"),
    ("oort_Energy_CDP", "Oort"),
    ("gca_Energy_CDP", "GCA"),
]

ABLATION_ORDER = [
    ("hybrid_SV_Energy_Lyapunov_CDP", "Full"),
    ("random_Energy_Lyapunov_CDP", "w/o SV"),
    ("hybrid_SV_Energy_NoQueue_CDP", "w/o Queue"),
    ("hybrid_SV_CDP", "w/o Energy"),
]

ESTIMATOR_ORDER = [
    ("permutation_M20", "Permutation"),
    ("complementary_uniform_M20", "CC-uniform"),
    ("complementary_neyman_M20", "CC-Neyman"),
]


def normalize_accuracy(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size and np.nanmax(arr) <= 1.5:
        arr = arr * 100.0
    return arr


def method_from_filename(path):
    stem = Path(path).stem
    marker = "_B[32]_"
    idx = stem.find(marker)
    return stem[idx + len(marker):] if idx >= 0 else stem


def as_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return float(default)


def load_record(path, group, label=None):
    with Path(path).open("rb") as f:
        data = pickle.load(f)

    args = data.get("args", {}) or {}
    acc = normalize_accuracy(data.get("test_accuracy", []))
    loss = np.asarray(data.get("train_loss", []), dtype=np.float64)
    train_acc = normalize_accuracy(data.get("train_accuracy", []))

    final_energy = data.get("final_client_energy", None)
    if final_energy is None:
        energy_stats = data.get("energy_statistics", {}) or {}
        final_energy = energy_stats.get("current_energy", [])
    final_energy = np.asarray(final_energy, dtype=np.float64)

    energy_stats = data.get("energy_statistics", {}) or {}
    consumption = energy_stats.get("consumption_history", []) or []
    if consumption:
        cons_values = np.concatenate([np.asarray(x, dtype=np.float64).ravel() for x in consumption])
    else:
        cons_values = np.asarray([], dtype=np.float64)

    lyap_stats = data.get("lyapunov_statistics", {}) or {}
    shapley_time = data.get("shapley_time_history", []) or []
    shapley_times = np.asarray(
        [as_float(item.get("time_s")) for item in shapley_time if isinstance(item, dict) and "time_s" in item],
        dtype=np.float64,
    )

    dp_stats = data.get("dp_statistics", {}) or {}
    participation = np.asarray(data.get("client_participation_counts", []), dtype=np.float64)

    return {
        "group": group,
        "label": label or method_from_filename(path),
        "method_tag": method_from_filename(path),
        "path": str(Path(path).relative_to(ROOT)),
        "seed": int(args.get("seed", -1)),
        "epochs": int(args.get("epochs", len(acc))),
        "num_users": int(args.get("num_users", 0)),
        "num_selected": int(args.get("num_selected", 0)),
        "dirichlet_alpha": as_float(args.get("dirichlet_alpha")),
        "shapley_estimator": args.get("shapley_estimator", ""),
        "shapley_allocation": args.get("shapley_allocation", ""),
        "shapley_max_iter": int(args.get("shapley_max_iter", 0) or 0),
        "final_acc": as_float(acc[-1]) if acc.size else np.nan,
        "last5_acc": as_float(acc[-min(5, len(acc)):].mean()) if acc.size else np.nan,
        "best_acc": as_float(acc.max()) if acc.size else np.nan,
        "final_train_acc": as_float(train_acc[-1]) if train_acc.size else np.nan,
        "final_train_loss": as_float(loss[-1]) if loss.size else np.nan,
        "avg_shapley_time_s": as_float(np.nanmean(shapley_times)) if shapley_times.size else np.nan,
        "total_shapley_time_s": as_float(np.nansum(shapley_times)) if shapley_times.size else 0.0,
        "final_energy_mean": as_float(np.nanmean(final_energy)) if final_energy.size else np.nan,
        "final_energy_min": as_float(np.nanmin(final_energy)) if final_energy.size else np.nan,
        "depleted_clients": int(np.sum(final_energy < as_float(args.get("energy_threshold", 50.0)))) if final_energy.size else 0,
        "avg_round_energy": as_float(np.nanmean(cons_values)) if cons_values.size else np.nan,
        "queue_mean": as_float(lyap_stats.get("queue_mean")),
        "queue_max": as_float(lyap_stats.get("queue_max")),
        "lyapunov_value": as_float(lyap_stats.get("lyapunov_value")),
        "privacy_epsilon": as_float(dp_stats.get("update_epsilon")),
        "channel_sigma": as_float(dp_stats.get("channel_noise_multiplier", args.get("dp_channel_noise_multiplier", np.nan))),
        "participation_std": as_float(np.nanstd(participation)) if participation.size else np.nan,
        "participation_max": as_float(np.nanmax(participation)) if participation.size else np.nan,
    }


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return float(arr.mean()), float(arr.std(ddof=0))


def summarize(records, metric_keys):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["label"]].append(row)

    summaries = {}
    for label, rows in grouped.items():
        item = {"label": label, "n": len(rows)}
        for key in metric_keys:
            m, s = mean_std([r.get(key, np.nan) for r in rows])
            item[f"{key}_mean"] = m
            item[f"{key}_std"] = s
        item["seeds"] = sorted({r["seed"] for r in rows if r["seed"] >= 0})
        summaries[label] = item
    return summaries


def fmt_pm(mean, std, digits=2):
    if not np.isfinite(mean):
        return "N/A"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path, summaries, order, columns):
    best_label = None
    if summaries:
        best_label = max(
            summaries,
            key=lambda label: summaries[label].get("last5_acc_mean", float("-inf")),
        )
    lines = ["% Generated by summarize_sv_supp_results.py"]
    for _, label in order:
        stats = summaries.get(label)
        if stats is None:
            row = [label] + ["N/A"] * len(columns)
        else:
            row = [label]
            for metric, digits in columns:
                cell = fmt_pm(stats.get(f"{metric}_mean", np.nan), stats.get(f"{metric}_std", np.nan), digits)
                if metric == "last5_acc" and label == best_label and cell != "N/A":
                    cell = "\\textbf{" + cell + "}"
                row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_run(root):
    records = []

    for method_tag, label in MAIN_ORDER:
        for path in sorted((root / "main").rglob(f"*{method_tag}.pkl")):
            records.append(load_record(path, "main", label))

    for method_tag, label in ABLATION_ORDER:
        for path in sorted((root / "ablation").rglob(f"*{method_tag}.pkl")):
            records.append(load_record(path, "ablation", label))

    for folder_tag, label in ESTIMATOR_ORDER:
        for path in sorted((root / "estimator").glob(f"seed*/{folder_tag}/*.pkl")):
            records.append(load_record(path, "estimator", label))

    for path in sorted((root / "budget").glob("M*_seed*/*.pkl")):
        max_iter = 0
        try:
            max_iter = int(path.parent.name.split("_")[0].lstrip("M"))
        except Exception:
            pass
        records.append(load_record(path, "budget", f"M={max_iter}"))

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="Run tag under save/sv_supp. Defaults to latest.")
    args = parser.parse_args()

    base = SAVE / "sv_supp"
    if args.tag:
        run_root = base / args.tag
    else:
        candidates = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise FileNotFoundError("No run directories found under save/sv_supp")
        run_root = candidates[-1]

    records = collect_run(run_root)
    out = run_root / "summary_tables"
    out.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in records for key in row.keys()})
    write_csv(out / "all_runs.csv", records, fieldnames)

    metric_keys = [
        "last5_acc",
        "final_acc",
        "best_acc",
        "avg_shapley_time_s",
        "total_shapley_time_s",
        "final_energy_mean",
        "final_energy_min",
        "avg_round_energy",
        "queue_mean",
        "queue_max",
        "participation_std",
    ]

    all_summary = {}
    for group in ["main", "estimator", "budget", "ablation"]:
        group_records = [r for r in records if r["group"] == group]
        summaries = summarize(group_records, metric_keys)
        all_summary[group] = summaries
        rows = list(summaries.values())
        write_csv(out / f"{group}_summary.csv", rows, sorted({k for row in rows for k in row.keys()}) if rows else ["label"])

    write_latex(
        out / "main_table.tex",
        all_summary["main"],
        MAIN_ORDER,
        [("last5_acc", 2), ("final_acc", 2), ("final_energy_mean", 2), ("queue_mean", 2)],
    )
    write_latex(
        out / "estimator_table.tex",
        all_summary["estimator"],
        ESTIMATOR_ORDER,
        [("last5_acc", 2), ("avg_shapley_time_s", 2), ("total_shapley_time_s", 1), ("final_energy_mean", 2)],
    )
    budget_order = [(f"M={m}", f"M={m}") for m in [5, 10, 20, 50]]
    write_latex(
        out / "budget_table.tex",
        all_summary["budget"],
        budget_order,
        [("last5_acc", 2), ("final_acc", 2), ("avg_shapley_time_s", 2), ("total_shapley_time_s", 1)],
    )
    write_latex(
        out / "ablation_table.tex",
        all_summary["ablation"],
        ABLATION_ORDER,
        [("last5_acc", 2), ("final_acc", 2), ("final_energy_mean", 2), ("queue_mean", 2)],
    )

    (out / "summary.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    print(f"Run root: {run_root.relative_to(ROOT)}")
    print(f"Records: {len(records)}")
    print(f"Written: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

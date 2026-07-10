#!/usr/bin/env python
"""Summarize paper-alignment reruns across all experiment groups."""

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SAVE = ROOT / "save"


def as_float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_accuracy(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size and np.nanmax(arr) <= 1.5:
        arr = arr * 100.0
    return arr


def flatten_consumption(history):
    arrays = [np.asarray(item, dtype=np.float64).ravel() for item in (history or [])]
    return np.concatenate(arrays) if arrays else np.asarray([], dtype=np.float64)


def load_record(path, run_root):
    with path.open("rb") as handle:
        data = pickle.load(handle)

    args = data.get("args", {}) or {}
    acc = normalize_accuracy(data.get("test_accuracy", []))
    energy_stats = data.get("energy_statistics", {}) or {}
    constraint = data.get("energy_constraint_statistics", {}) or {}
    lyapunov = data.get("lyapunov_statistics", {}) or {}
    final_energy = np.asarray(
        data.get("final_client_energy", energy_stats.get("current_energy", [])),
        dtype=np.float64,
    )
    consumption = flatten_consumption(energy_stats.get("consumption_history", []))
    observations = np.asarray(data.get("shapley_observation_counts", []), dtype=np.float64)
    participation = np.asarray(data.get("client_participation_counts", []), dtype=np.float64)
    aggregate_records = [
        item for item in (data.get("dp_round_history", []) or [])
        if isinstance(item, dict) and "aggregation_max_weight" in item
    ]
    aggregation_weights = np.asarray(
        [as_float(item.get("aggregation_max_weight")) for item in aggregate_records],
        dtype=np.float64,
    )

    relative_parent = path.parent.relative_to(run_root)
    parts = relative_parent.parts
    series_parts = parts[:-1] if parts and parts[-1].startswith("seed") else parts
    series = "/".join(series_parts) or "root"

    return {
        "series": series,
        "seed": int(args.get("seed", -1)),
        "path": str(path.relative_to(ROOT)),
        "dataset": args.get("dataset", ""),
        "selection_method": args.get("selection_method", ""),
        "selection_beta": as_float(args.get("selection_beta")),
        "energy_budget": as_float(args.get("energy_budget")),
        "dirichlet_alpha": as_float(args.get("dirichlet_alpha")),
        "shapley_estimator": args.get("shapley_estimator", ""),
        "shapley_allocation": args.get("shapley_allocation", ""),
        "shapley_max_iter": int(args.get("shapley_max_iter", 0) or 0),
        "initial_rounds": int(args.get("initial_rounds", 0) or 0),
        "last5_acc": as_float(acc[-min(5, acc.size):].mean()) if acc.size else np.nan,
        "final_acc": as_float(acc[-1]) if acc.size else np.nan,
        "best_acc": as_float(np.nanmax(acc)) if acc.size else np.nan,
        "avg_energy_per_use": as_float(np.nanmean(consumption)) if consumption.size else np.nan,
        "final_energy_min": as_float(np.nanmin(final_energy)) if final_energy.size else np.nan,
        "queue_mean": as_float(lyapunov.get("queue_mean")),
        "queue_max": as_float(lyapunov.get("queue_max")),
        "max_time_average_energy": as_float(constraint.get("max_time_average_energy")),
        "max_budget_violation": as_float(constraint.get("max_budget_violation")),
        "mean_budget_violation": as_float(constraint.get("mean_budget_violation")),
        "constraint_satisfied_fraction": as_float(constraint.get("constraint_satisfied_fraction")),
        "queue_over_horizon_max": as_float(np.nanmax(constraint.get("queue_over_horizon", [])))
        if np.asarray(constraint.get("queue_over_horizon", [])).size else np.nan,
        "mean_aggregation_max_weight": as_float(np.nanmean(aggregation_weights))
        if aggregation_weights.size else np.nan,
        "shapley_observed_clients": int(np.sum(observations > 0)) if observations.size else 0,
        "shapley_observation_mean": as_float(np.nanmean(observations)) if observations.size else np.nan,
        "participation_std": as_float(np.nanstd(participation)) if participation.size else np.nan,
    }


def mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return np.nan, np.nan
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def summarize(records):
    metric_keys = [
        "last5_acc", "final_acc", "best_acc", "avg_energy_per_use",
        "final_energy_min", "queue_mean", "queue_max",
        "max_time_average_energy", "max_budget_violation",
        "mean_budget_violation", "constraint_satisfied_fraction",
        "queue_over_horizon_max", "mean_aggregation_max_weight",
        "shapley_observed_clients", "shapley_observation_mean",
        "participation_std",
    ]
    grouped = defaultdict(list)
    for row in records:
        grouped[row["series"]].append(row)

    rows = []
    for series, items in sorted(grouped.items()):
        summary = {
            "series": series,
            "n": len(items),
            "seeds": "[" + ",".join(str(seed) for seed in sorted({x["seed"] for x in items})) + "]",
        }
        for key in metric_keys:
            mean, std = mean_std([item.get(key, np.nan) for item in items])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
        rows.append(summary)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["series"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="Run tag under save/paper_alignment")
    args = parser.parse_args()

    run_root = SAVE / "paper_alignment" / args.tag
    if not run_root.exists():
        raise FileNotFoundError(f"Run directory not found: {run_root}")

    records = [load_record(path, run_root) for path in sorted(run_root.rglob("*.pkl"))]
    summaries = summarize(records)
    output = run_root / "summary_tables"
    write_csv(output / "all_runs.csv", records)
    write_csv(output / "group_summary.csv", summaries)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_tag": args.tag,
        "records": [{key: json_value(value) for key, value in row.items()} for row in records],
        "summary": [{key: json_value(value) for key, value in row.items()} for row in summaries],
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(f"Run root: {run_root.relative_to(ROOT)}")
    print(f"Records: {len(records)}")
    print(f"Written: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

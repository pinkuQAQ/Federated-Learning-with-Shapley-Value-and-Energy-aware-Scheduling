#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize lightweight local CDP smoke-test runs."""

import argparse
import math
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SAVE = ROOT / "save"


def compute_epsilon(sigma, q, rounds, delta=1e-5):
    if sigma <= 0:
        return float("inf")
    orders = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]
    best = float("inf")
    for alpha in orders:
        terms = []
        for k in range(alpha + 1):
            log_binom = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
            terms.append(
                log_binom
                + k * math.log(max(q, 1e-300))
                + (alpha - k) * math.log(max(1.0 - q, 1e-300))
                + (k * k - k) / (2.0 * sigma * sigma)
            )
        m = max(terms)
        rdp = (m + math.log(sum(math.exp(v - m) for v in terms))) / (alpha - 1)
        best = min(best, rounds * rdp + math.log(1.0 / delta) / (alpha - 1))
    return best


def sigma_for_round(base_sigma, round_idx, rounds, schedule="constant", start_fraction=0.7):
    if schedule == "constant" or rounds <= 1:
        return float(base_sigma)
    progress = min(max(round_idx / float(max(rounds - 1, 1)), 0.0), 1.0)
    start_fraction = min(max(float(start_fraction), 1e-6), 1.0)
    if schedule == "linear_increase":
        factor = start_fraction + (1.0 - start_fraction) * progress
    elif schedule == "cosine_increase":
        factor = start_fraction + (1.0 - start_fraction) * (1.0 - math.cos(math.pi * progress)) / 2.0
    else:
        factor = 1.0
    return float(base_sigma) * factor


def compute_epsilon_schedule(sigmas, q, delta=1e-5):
    orders = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]
    total = {alpha: 0.0 for alpha in orders}
    for sigma in sigmas:
        if sigma <= 0:
            return float("inf")
        for alpha in orders:
            terms = []
            for k in range(alpha + 1):
                log_binom = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
                terms.append(
                    log_binom
                    + k * math.log(max(q, 1e-300))
                    + (alpha - k) * math.log(max(1.0 - q, 1e-300))
                    + (k * k - k) / (2.0 * sigma * sigma)
                )
            m = max(terms)
            total[alpha] += (m + math.log(sum(math.exp(v - m) for v in terms))) / (alpha - 1)

    best = float("inf")
    for alpha, rdp in total.items():
        best = min(best, rdp + math.log(1.0 / delta) / (alpha - 1))
    return best


def fmt_eps(eps):
    if math.isinf(eps):
        return "inf"
    if eps >= 1000:
        return f"{eps:.3e}"
    return f"{eps:.3f}"


def normalize_acc(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr * 100.0 if arr.size and np.nanmax(arr) <= 1.5 else arr


def read_args_value(args_obj, name, default=None):
    if isinstance(args_obj, dict):
        return args_obj.get(name, default)
    return getattr(args_obj, name, default)


def summarize_folder(folder):
    pkls = sorted(folder.glob("*.pkl"))
    if not pkls:
        return None

    with pkls[0].open("rb") as f:
        data = pickle.load(f)

    args = data.get("args", {})
    acc = normalize_acc(data.get("test_accuracy", []))
    if acc.size == 0:
        return None

    sigma = read_args_value(args, "dp_noise_multiplier", 0.0)
    privacy_mode = read_args_value(args, "privacy_mode", "none")
    if privacy_mode is None:
        privacy_mode = "central"

    rounds = int(read_args_value(args, "epochs", len(acc)))
    users = int(read_args_value(args, "num_users", 1))
    selected = int(read_args_value(args, "num_selected", 1))
    delta = float(read_args_value(args, "dp_delta", 1e-5))
    sample_rate = selected / float(users)
    if privacy_mode == "none":
        eps = float("inf")
    elif read_args_value(args, "dp_advanced", False) and read_args_value(args, "dp_noise_schedule", "constant") != "constant":
        schedule = read_args_value(args, "dp_noise_schedule", "constant")
        start = float(read_args_value(args, "dp_noise_start_multiplier", 0.7))
        sigmas = [sigma_for_round(float(sigma), t, rounds, schedule, start) for t in range(rounds)]
        eps = compute_epsilon_schedule(sigmas, sample_rate, delta)
    else:
        eps = compute_epsilon(float(sigma), sample_rate, rounds, delta)

    dp_stats = data.get("dp_statistics", {}) or {}
    score_dp = bool(read_args_value(args, "dp_score_dp", False) or dp_stats.get("score_dp", False))
    score_sigma = float(read_args_value(args, "dp_score_noise_multiplier", dp_stats.get("score_noise_multiplier", 0.0)) or 0.0)
    score_clip = float(read_args_value(args, "dp_score_clip_norm", dp_stats.get("score_clip_norm", 0.0)) or 0.0)
    if not score_dp:
        score_eps = float("inf")
    elif "score_epsilon" in dp_stats:
        score_eps = float(dp_stats["score_epsilon"])
    else:
        score_eps = compute_epsilon(score_sigma, sample_rate, rounds, delta)
    update_eps = float(dp_stats.get("update_epsilon", eps))
    scope = str(read_args_value(args, "trainable_scope", dp_stats.get("trainable_scope", "full")))

    history = data.get("dp_round_history", [])
    client_items = [x for x in history if int(x.get("client", -1)) >= 0]
    aggregate_items = [x for x in history if int(x.get("client", -1)) < 0]
    noise_values = [float(x.get("noise_std", 0.0)) for x in aggregate_items if "noise_std" in x]
    noise_mult_values = [float(x.get("noise_multiplier", sigma)) for x in aggregate_items if "noise_multiplier" in x]
    clip_values = [float(x.get("clip_factor", 1.0)) for x in client_items if "clip_factor" in x]
    clip_norm_values = [float(x.get("clip_norm", 0.0)) for x in client_items if "clip_norm" in x]

    return {
        "folder": folder.name,
        "mode": privacy_mode,
        "sigma": float(sigma),
        "score_epsilon": score_eps,
        "update_epsilon": update_eps,
        "score_sigma": score_sigma,
        "score_clip": score_clip,
        "scope": scope,
        "final": float(acc[-1]),
        "last3": float(acc[-min(3, len(acc)):].mean()),
        "noise": float(np.mean(noise_values)) if noise_values else 0.0,
        "sigma_avg": float(np.mean(noise_mult_values)) if noise_mult_values else float(sigma),
        "clip": float(np.mean(clip_values)) if clip_values else 1.0,
        "clip_norm": float(np.mean(clip_norm_values)) if clip_norm_values else float(read_args_value(args, "dp_clip_norm", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="Run tag printed by local CDP bat scripts")
    parser.add_argument("--pattern", default="local_cdp_*", help="Folder glob under save/, e.g. local_cdp_sweep_*")
    args = parser.parse_args()

    pattern = args.pattern
    if args.tag:
        pattern = pattern.rstrip("*") + f"*_{args.tag}"
    folders = sorted([p for p in SAVE.glob(pattern) if p.is_dir()])
    rows = [summarize_folder(folder) for folder in folders]
    rows = [row for row in rows if row is not None]

    if not rows:
        print("No local CDP result folders found.")
        return

    rows.sort(key=lambda r: (r["mode"] != "none", r["sigma"]))

    print(f"{'folder':42s} {'mode':>8s} {'scope':>7s} {'sigma':>8s} {'sig_avg':>8s} {'upd_eps':>12s} {'score_eps':>12s} {'final':>9s} {'last3':>9s} {'agg_noise':>9s} {'C_avg':>8s} {'clip':>8s}")
    print("-" * 153)
    for row in rows:
        print(
            f"{row['folder'][:42]:42s} "
            f"{row['mode']:>8s} "
            f"{row['scope'][:7]:>7s} "
            f"{row['sigma']:8.3f} "
            f"{row['sigma_avg']:8.3f} "
            f"{fmt_eps(row['update_epsilon']):>12s} "
            f"{fmt_eps(row['score_epsilon']):>12s} "
            f"{row['final']:9.2f} "
            f"{row['last3']:9.2f} "
            f"{row['noise']:9.4f} "
            f"{row['clip_norm']:8.4f} "
            f"{row['clip']:8.4f}"
        )


if __name__ == "__main__":
    main()

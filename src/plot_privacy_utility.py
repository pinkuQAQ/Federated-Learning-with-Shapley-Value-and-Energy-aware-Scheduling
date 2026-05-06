#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot the utility effect of lightweight-update DP noise strength."""

import pickle
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent.parent
SAVE = ROOT / "save"
FIG_OUT = ROOT / "latex" / "figures" / "privacy_utility.pdf"
PNG_OUT = SAVE / "privacy_utility.png"


def normalize(a):
    a = np.asarray(a, dtype=np.float64)
    return a * 100.0 if a.max() <= 1.5 else a


def compute_epsilon(sigma, q=0.05, rounds=100, delta=1e-5):
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


def compute_epsilon_schedule(sigmas, q=0.05, delta=1e-5):
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


def read_arg(args_obj, name, default=None):
    if isinstance(args_obj, dict):
        return args_obj.get(name, default)
    return getattr(args_obj, name, default)


def epsilon_from_run(data, fallback_sigma):
    args = data.get("args", {})
    sigma = float(read_arg(args, "dp_noise_multiplier", fallback_sigma))
    rounds = int(read_arg(args, "epochs", 100))
    num_users = int(read_arg(args, "num_users", 100))
    num_selected = int(read_arg(args, "num_selected", 10))
    delta = float(read_arg(args, "dp_delta", 1e-5))
    q = num_selected / float(num_users)
    if read_arg(args, "dp_advanced", False) and read_arg(args, "dp_noise_schedule", "constant") != "constant":
        schedule = read_arg(args, "dp_noise_schedule", "constant")
        start = float(read_arg(args, "dp_noise_start_multiplier", 0.7))
        sigmas = [sigma_for_round(sigma, t, rounds, schedule, start) for t in range(rounds)]
        return compute_epsilon_schedule(sigmas, q=q, delta=delta)
    return compute_epsilon(sigma, q=q, rounds=rounds, delta=delta)


def eps_label(eps):
    if math.isinf(eps):
        return r"$\infty$"
    if eps >= 1000:
        return f"{eps:.1e}"
    return f"{eps:.1f}"


def main():
    sigmas = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0]
    epsilons = [compute_epsilon(s) for s in sigmas]
    last5 = []

    for idx, sigma in enumerate(sigmas):
        folders = sorted(SAVE.glob(f"sens_dp_sigma{sigma}_*"))
        if not folders:
            last5.append(np.nan)
            continue

        pkls = list(folders[-1].glob("*.pkl"))
        if not pkls:
            last5.append(np.nan)
            continue

        with open(pkls[0], "rb") as f:
            data = pickle.load(f)
        epsilons[idx] = epsilon_from_run(data, sigma)
        acc = normalize(data["test_accuracy"])
        last5.append(acc[-5:].mean())

    last5 = np.array(last5)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(sigmas))

    ax.plot(
        x,
        last5,
        marker="o",
        color="#2563eb",
        linewidth=1.9,
        label="Test acc. (last 5)",
    )
    ax.set_xlabel(r"Head-update noise multiplier $\sigma_{\mathrm{dp}}$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in sigmas])
    ax.set_ylim(0, 55)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=False)

    fig.suptitle(
        "Lightweight-update DP utility tradeoff on CIFAR-10, "
        r"$\alpha=0.1$, $q=0.05$, $V=10$, $T=100$"
    )
    fig.tight_layout()

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    SAVE.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=140, bbox_inches="tight")

    print(f"wrote {FIG_OUT}\nwrote {PNG_OUT}")
    print("\n sigma_dp | last-5 acc (%)")
    print("-" * 30)
    for sigma, eps, acc in zip(sigmas, epsilons, last5):
        print(f" {sigma:>8.2f} | {acc:5.2f}")


if __name__ == "__main__":
    main()

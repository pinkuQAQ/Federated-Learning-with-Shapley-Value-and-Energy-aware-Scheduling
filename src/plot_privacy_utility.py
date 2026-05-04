#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot the utility effect of optional upload perturbation."""

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


def compute_epsilon(sigma, q=0.1, rounds=100, delta=1e-5):
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


def eps_label(eps):
    if math.isinf(eps):
        return r"$\infty$"
    if eps >= 1000:
        return f"{eps:.1e}"
    return f"{eps:.1f}"


def main():
    sigmas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    epsilons = [compute_epsilon(s) for s in sigmas]
    last5 = []

    for sigma in sigmas:
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
    ax.set_xlabel(r"Noise multiplier $\sigma_{\mathrm{dp}}$ / privacy budget $\varepsilon_{\mathrm{priv}}$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{eps_label(e)}" for s, e in zip(sigmas, epsilons)])
    ax.set_ylim(0, 55)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=False)

    fig.suptitle(
        "Privacy-utility tradeoff on CIFAR-10, "
        r"$\alpha=0.1$, $V=10$, $T=100$, $\delta=10^{-5}$"
    )
    fig.tight_layout()

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    SAVE.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=140, bbox_inches="tight")

    print(f"wrote {FIG_OUT}\nwrote {PNG_OUT}")
    print("\n sigma | epsilon_priv | last-5 acc (%)")
    print("-" * 42)
    for sigma, eps, acc in zip(sigmas, epsilons, last5):
        print(f" {sigma:>5.2f} | {eps_label(eps):>12s} | {acc:5.2f}")


if __name__ == "__main__":
    main()

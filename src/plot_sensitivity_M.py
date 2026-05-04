#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot sensitivity to the MC-Shapley iteration budget M."""

import pickle
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent.parent
SAVE = ROOT / "save"
FIG_OUT = ROOT / "latex" / "figures" / "sensitivity_M.pdf"
PNG_OUT = SAVE / "sensitivity_M.png"


def normalize(a):
    a = np.asarray(a, dtype=np.float64)
    return a * 100.0 if a.max() <= 1.5 else a


def smooth_ema(arr, alpha=0.1):
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def load_latest_curve(m):
    folders = sorted(SAVE.glob(f"sens_M{m}_*"))
    if not folders:
        return None
    pkls = list(folders[-1].glob("*.pkl"))
    if not pkls:
        return None
    with open(pkls[0], "rb") as f:
        data = pickle.load(f)
    return normalize(data["test_accuracy"])


def main():
    ms = [5, 10, 20, 50]
    curves = {m: load_latest_curve(m) for m in ms}
    curves = {m: c for m, c in curves.items() if c is not None}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [2.2, 1]}
    )

    cmap = plt.get_cmap("viridis")
    for i, m in enumerate(ms):
        if m not in curves:
            continue
        acc = smooth_ema(curves[m], 0.1)
        x = np.arange(1, len(acc) + 1)
        ax1.plot(
            x,
            acc,
            label=f"$M={m}$",
            color=cmap(i / max(1, len(ms) - 1)),
            linewidth=1.6,
        )
    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_title(r"Convergence trajectories for varying $M$")
    ax1.grid(alpha=0.3, linestyle=":")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_xlim(1, 100)
    ax1.set_ylim(8, 55)

    ms_present = [m for m in ms if m in curves]
    last5_vals = [curves[m][-5:].mean() for m in ms_present]
    colors = [cmap(i / max(1, len(ms_present) - 1)) for i in range(len(ms_present))]
    bars = ax2.bar(
        [str(m) for m in ms_present],
        last5_vals,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
    )
    for bar, val in zip(bars, last5_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.4, f"{val:.1f}",
                 ha="center", fontsize=9)
    ax2.set_xlabel(r"MC iterations $M$")
    ax2.set_ylabel("Last-5 test accuracy (%)")
    ax2.set_title(r"Last-5 acc. vs. $M$")
    ax2.set_ylim(0, 55)
    ax2.grid(alpha=0.3, axis="y", linestyle=":")

    fig.suptitle(
        r"Sensitivity to MC-Shapley iterations $M$ -- CIFAR-10, "
        r"$\alpha=0.1$, $\sigma_{\mathrm{dp}}=0.01$, single seed=42",
        fontsize=11,
    )
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {FIG_OUT}\nwrote {PNG_OUT}")

    if ms_present:
        print("\n M | last-5 acc (%)")
        print("-" * 22)
        for m, val in zip(ms_present, last5_vals):
            print(f"{m:2d} | {val:6.2f}")


if __name__ == "__main__":
    main()

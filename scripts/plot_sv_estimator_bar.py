#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot Shapley estimator accuracy with evaluation-count annotations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "latex" / "figures"
PDF_OUT = FIG_DIR / "sv_estimator_bar.pdf"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", action="store_true", help="Also write a PNG copy.")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["Permutation", "CC-uniform", "CC-Neyman"]
    means = np.array([32.70, 34.07, 33.81])
    stds = np.array([3.46, 1.91, 2.12])
    eval_counts = [r"$\leq 82$", r"$\leq 42$", r"$\leq 42$"]

    colors = ["#9aa6b2", "#3b82c4", "#2f9e8f"]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        width=0.58,
        color=colors,
        edgecolor="#222222",
        linewidth=0.6,
        capsize=3.5,
        error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#333333"},
        zorder=3,
    )

    ax.set_ylabel("Last-5 accuracy (%)", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.tick_params(axis="y", labelsize=7.8)
    ax.set_ylim(26, 37.6)
    ax.set_yticks([26, 28, 30, 32, 34, 36])
    ax.grid(axis="y", linestyle="--", linewidth=0.45, color="#d0d7de", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, eval_label in zip(bars, eval_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            27.05,
            f"Eval./round\n{eval_label}",
            ha="center",
            va="center",
            fontsize=6.8,
            linespacing=0.95,
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.18,rounding_size=0.05",
                "facecolor": "white",
                "edgecolor": "#d0d7de",
                "linewidth": 0.35,
                "alpha": 0.92,
            },
        )

    ax.text(
        0.02,
        0.98,
        r"$M=20,\ K=5,\ T=100$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#4b5563",
    )

    fig.tight_layout(pad=0.35)
    fig.savefig(PDF_OUT, bbox_inches="tight")
    print(f"wrote {PDF_OUT}")
    if args.png:
        png_out = FIG_DIR / "sv_estimator_bar.png"
        fig.savefig(png_out, dpi=320, bbox_inches="tight")
        print(f"wrote {png_out}")


if __name__ == "__main__":
    main()

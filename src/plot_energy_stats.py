#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot minimum residual energy for the energy-aware ablation."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent.parent
FIG_OUT = ROOT / "latex" / "figures" / "energy_stats.pdf"
PNG_OUT = ROOT / "save" / "energy_stats.png"


LABELS = ["Full", "w/o SV", "w/o Lyap", "FedCS"]
MIN_E = np.array([254.82, 415.64, 327.52, 496.72])
MIN_E_STD = np.array([28.80, 17.26, 46.53, 0.14])


def main():
    x = np.arange(len(LABELS))

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    colors = ["#2563eb", "#0f766e", "#f97316", "#64748b"]
    bars = ax.bar(x, MIN_E, color=colors, width=0.58)

    ax.axhline(50, color="#dc2626", linestyle="--", linewidth=1.2, label="Eligibility threshold")
    ax.set_ylabel("Minimum residual energy (J)")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 540)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    for bar, value in zip(bars, MIN_E):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 18,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(r"Minimum residual energy on CIFAR-10, $\alpha=0.1$, $T=100$")
    fig.tight_layout()

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches="tight")
    fig.savefig(PNG_OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {FIG_OUT}")
    print(f"wrote {PNG_OUT}")


if __name__ == "__main__":
    main()

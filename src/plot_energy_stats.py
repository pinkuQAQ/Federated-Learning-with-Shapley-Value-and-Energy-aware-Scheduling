#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot minimum residual energy from current main and ablation runs."""

import argparse
import pickle
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent.parent
SAVE = ROOT / "save"
FIG_OUT = ROOT / "latex" / "figures" / "energy_stats.pdf"
PNG_OUT = SAVE / "energy_stats.png"

CONFIGS = [
    ("Full", "main", "hybrid_SV_Energy_Lyapunov_CDP"),
    ("w/o SV", "ablation", "random_Energy_Lyapunov_CDP"),
    ("w/o Lyap", "ablation", "hybrid_SV_Energy_CDP"),
    ("FedCS", "main", "fedcs_Energy_CDP"),
]


def find_tag(path):
    stem = Path(path).stem
    marker = "_B[32]_"
    idx = stem.find(marker)
    return stem[idx + len(marker):] if idx >= 0 else None


def latest_run(group, tag):
    root = SAVE / group
    if tag:
        return root / tag
    runs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run directories found under {root}")
    return runs[-1]


def collect_min_energy(run_dir, method_tag):
    mins = []
    for seed_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        for pkl in seed_dir.glob("*.pkl"):
            if find_tag(pkl) != method_tag:
                continue
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            stats = data.get("energy_statistics") or {}
            energies = np.asarray(stats.get("current_energy", data.get("final_client_energy", [])), dtype=float)
            if energies.size:
                mins.append(float(energies.min()))
    return np.asarray(mins, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-tag", default=None)
    parser.add_argument("--ablation-tag", default=None)
    args = parser.parse_args()

    roots = {
        "main": latest_run("main", args.main_tag),
        "ablation": latest_run("ablation", args.ablation_tag),
    }
    print(f"using main={roots['main']}")
    print(f"using ablation={roots['ablation']}")

    labels, means, stds = [], [], []
    for label, group, method_tag in CONFIGS:
        values = collect_min_energy(roots[group], method_tag)
        if values.size == 0:
            continue
        labels.append(label)
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=0)))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    colors = ["#2563eb", "#0f766e", "#f97316", "#64748b"][: len(labels)]
    bars = ax.bar(x, means, color=colors, width=0.58)

    ax.axhline(50, color="#dc2626", linestyle="--", linewidth=1.2, label="Eligibility threshold")
    ax.set_ylabel("Minimum residual energy (J)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 540)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 18, f"{value:.0f}",
                ha="center", va="bottom", fontsize=9)

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

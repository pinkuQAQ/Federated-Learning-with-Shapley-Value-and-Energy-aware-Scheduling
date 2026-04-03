#!/usr/bin/env python
"""Generate the baseline comparison figure in the same visual style as Fig. 2."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "save" / "noniid_cmp_alpha0.25_20260320"
OUTPUT_DIR = ROOT / "paper_latex" / "figures"
EMA_ALPHA = 0.1

METHOD_STYLES = {
    "Ours": {"color": "#000000", "linewidth": 1.1},
    "FedAvg": {"color": "#1f77b4", "linewidth": 0.95},
    "PoC": {"color": "#ff7f0e", "linewidth": 0.95},
    "UCB": {"color": "#2ca02c", "linewidth": 0.95},
    "FedProx": {"color": "#d62728", "linewidth": 0.95},
}

METHOD_ORDER = ["Ours", "FedAvg", "PoC", "UCB", "FedProx"]


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def exponential_moving_average(values: list[float], alpha: float = EMA_ALPHA) -> list[float]:
    if not values:
        return []

    ema = [values[0]]
    for value in values[1:]:
        ema.append(alpha * value + (1.0 - alpha) * ema[-1])
    return ema


def identify_method(path: Path) -> str:
    name = path.name.lower()
    if "hybrid_sv_energy" in name:
        return "Ours"
    if "random_fedprox" in name:
        return "FedProx"
    if "random" in name:
        return "FedAvg"
    if "poc" in name:
        return "PoC"
    if "ucb" in name:
        return "UCB"
    raise ValueError(f"Unrecognized baseline result file: {path.name}")


def load_curves() -> dict[str, list[float]]:
    curves: dict[str, list[float]] = {}
    for path in sorted(BASELINE_DIR.glob("*.pkl")):
        method = identify_method(path)
        result = load_pickle(path)
        curves[method] = [value * 100.0 for value in result["test_accuracy"]]
    return curves


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlim(1, 100)
    ax.set_ylim(10, 55)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.grid(True, linestyle="-", linewidth=0.35, alpha=0.18)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    ax.tick_params(width=0.65, labelsize=8.0)


def generate_figure() -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "figure.dpi": 300,
            "axes.linewidth": 0.65,
        }
    )

    curves = load_curves()
    rounds = list(range(1, 101))

    fig, ax = plt.subplots(1, 1, figsize=(4.25, 3.35), constrained_layout=True)
    for method in METHOD_ORDER:
        ax.plot(
            rounds,
            exponential_moving_average(curves[method]),
            label=method,
            linestyle="-",
            **METHOD_STYLES[method],
        )

    style_axis(ax)
    ax.legend(
        loc="lower right",
        fontsize=6.7,
        framealpha=0.86,
        ncol=2,
        handlelength=1.5,
        borderpad=0.30,
        labelspacing=0.25,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "baseline_convergence.pdf"
    png_path = OUTPUT_DIR / "baseline_convergence.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


if __name__ == "__main__":
    pdf_file, png_file = generate_figure()
    print(f"Saved PDF: {pdf_file}")
    print(f"Saved PNG: {png_file}")

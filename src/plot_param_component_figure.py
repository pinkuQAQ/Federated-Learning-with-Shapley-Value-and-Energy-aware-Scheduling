#!/usr/bin/env python
"""Generate Fig. style EMA accuracy curves for ablation and sensitivity studies.

The script reads only completed experiment files under ``save/`` and ignores
``save/objects``. It produces a single-column figure with three stacked panels:

1. Ablation curves
2. Sensitivity to the Lyapunov control parameter V
3. Sensitivity to the MC-Shapley iteration count M
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "save"
OUTPUT_DIR = ROOT / "paper_latex" / "figures"

ABLATION_DIR = SAVE_DIR / "ablation_20260330_181547"
V_DIRS = {
    1: SAVE_DIR / "sens_V1_20260330_182151",
    5: SAVE_DIR / "sens_V5_20260330_182151",
    10: SAVE_DIR / "sens_V10_20260330_182151",
    20: SAVE_DIR / "sens_V20_20260330_182151",
    50: SAVE_DIR / "sens_V50_20260330_182151",
}
M_DIRS = {
    5: SAVE_DIR / "sens_M5_20260330_190405",
    10: SAVE_DIR / "sens_M10_20260330_190405",
    20: SAVE_DIR / "sens_M20_20260330_190405",
    50: SAVE_DIR / "sens_M50_20260330_190405",
    100: SAVE_DIR / "sens_M100_20260330_190405",
}

EMA_ALPHA = 0.1

ABLATION_STYLES = {
    "Full": {"color": "#000000", "linestyle": "-", "linewidth": 1.1},
    "w/o Lyapunov": {"color": "#ff7f0e", "linestyle": "-", "linewidth": 0.95},
    "w/o SV": {"color": "#d62728", "linestyle": "-", "linewidth": 0.95},
}

PARAM_STYLES = [
    {"color": "#000000", "linestyle": "-", "linewidth": 1.0},
    {"color": "#1f77b4", "linestyle": "-", "linewidth": 0.92},
    {"color": "#ff7f0e", "linestyle": "-", "linewidth": 0.92},
    {"color": "#2ca02c", "linestyle": "-", "linewidth": 0.92},
    {"color": "#d62728", "linestyle": "-", "linewidth": 0.92},
]


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def first_pickle(folder: Path) -> Path:
    files = sorted(folder.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No pkl files found in {folder}")
    return files[0]


def exponential_moving_average(values: list[float], alpha: float = EMA_ALPHA) -> list[float]:
    if not values:
        return []

    ema = [values[0]]
    for value in values[1:]:
        ema.append(alpha * value + (1.0 - alpha) * ema[-1])
    return ema


def identify_ablation_variant(result: dict) -> str:
    args = result.get("args", {})
    use_sv = bool(args.get("use_shapley"))
    use_lyapunov = bool(args.get("use_lyapunov"))
    use_energy = bool(args.get("use_energy"))

    if use_sv and use_lyapunov and use_energy:
        return "Full"
    if use_sv and not use_lyapunov:
        return "w/o Lyapunov"
    if not use_sv and use_lyapunov and use_energy:
        return "w/o SV"
    raise ValueError(f"Unrecognized ablation configuration: {args}")


def load_ablation_curves() -> dict[str, list[float]]:
    curves: dict[str, list[float]] = {}
    for path in sorted(ABLATION_DIR.glob("*.pkl")):
        result = load_pickle(path)
        variant = identify_ablation_variant(result)
        curves[variant] = [value * 100.0 for value in result["test_accuracy"]]
    return curves


def load_parameter_curves(folders: dict[int, Path]) -> dict[int, list[float]]:
    curves: dict[int, list[float]] = {}
    for key in sorted(folders):
        result = load_pickle(first_pickle(folders[key]))
        curves[key] = [value * 100.0 for value in result["test_accuracy"]]
    return curves


def style_axis(ax: plt.Axes, title: str, show_xlabel: bool) -> None:
    ax.set_xlim(1, 100)
    ax.set_ylim(10, 55)
    ax.set_ylabel("Test Accuracy (%)")
    if show_xlabel:
        ax.set_xlabel("Communication Round")
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4.0)
    ax.grid(True, linestyle="-", linewidth=0.35, alpha=0.18)
    ax.set_xticks([20, 40, 60, 80, 100])
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    ax.tick_params(width=0.65, labelsize=8.0)


def plot_ablation_panel(ax: plt.Axes, curves: dict[str, list[float]]) -> None:
    order = ["Full", "w/o Lyapunov", "w/o SV"]
    rounds = list(range(1, 101))
    for name in order:
        ax.plot(rounds, exponential_moving_average(curves[name]), label=name, **ABLATION_STYLES[name])
    style_axis(ax, "(a) Ablation Accuracy Comparison", show_xlabel=False)
    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.86, ncol=1, handlelength=1.5, borderpad=0.30, labelspacing=0.25)


def plot_parameter_panel(ax: plt.Axes, curves: dict[int, list[float]], prefix: str, title: str, show_xlabel: bool) -> None:
    rounds = list(range(1, 101))
    for style, key in zip(PARAM_STYLES, sorted(curves)):
        ax.plot(rounds, exponential_moving_average(curves[key]), label=f"{prefix}={key}", **style)
    style_axis(ax, title, show_xlabel=show_xlabel)
    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.86, ncol=2, handlelength=1.5, borderpad=0.30, labelspacing=0.25)


def generate_figure() -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "figure.dpi": 300,
            "axes.linewidth": 0.65,
        }
    )

    ablation = load_ablation_curves()
    sens_v = load_parameter_curves(V_DIRS)
    sens_m = load_parameter_curves(M_DIRS)

    # Use a taller canvas so that after LaTeX rescales to one column,
    # each panel still keeps enough vertical breathing room.
    fig, axes = plt.subplots(3, 1, figsize=(4.25, 9.1), constrained_layout=True)

    plot_ablation_panel(axes[0], ablation)
    plot_parameter_panel(axes[1], sens_v, prefix="V", title="(b) Sensitivity to V", show_xlabel=False)
    plot_parameter_panel(axes[2], sens_m, prefix="M", title="(c) Sensitivity to M", show_xlabel=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "param_component_curves.pdf"
    png_path = OUTPUT_DIR / "param_component_curves.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


if __name__ == "__main__":
    pdf_file, png_file = generate_figure()
    print(f"Saved PDF: {pdf_file}")
    print(f"Saved PNG: {png_file}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_sensitivity_V.py — Lyapunov V 敏感性单 seed 曲线 + 末段汇总。
输出 paper_latex/figures/sensitivity_V.pdf
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
SAVE = ROOT / 'save'
FIG_OUT = ROOT / 'latex' / 'figures' / 'sensitivity_V.pdf'
PNG_OUT = SAVE / 'sensitivity_V.png'


def normalize(a):
    a = np.asarray(a, dtype=np.float64)
    return a * 100.0 if a.max() <= 1.5 else a


def smooth_ema(arr, alpha=0.1):
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def main():
    Vs = [1, 5, 10, 20, 50]
    curves = {}
    for v in Vs:
        folders = sorted(SAVE.glob(f'sens_V{v}_*'))
        if not folders:
            continue
        pkls = list(folders[-1].glob('*.pkl'))
        if pkls:
            with open(pkls[0], 'rb') as f:
                d = pickle.load(f)
            curves[v] = normalize(d['test_accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                    gridspec_kw={'width_ratios': [2.2, 1]})

    # left: smoothed curves
    cmap = plt.get_cmap('viridis')
    for i, v in enumerate(Vs):
        if v not in curves:
            continue
        acc = smooth_ema(curves[v], 0.1)
        x = np.arange(1, len(acc) + 1)
        ax1.plot(x, acc, label=f'$V={v}$', color=cmap(i / max(1, len(Vs) - 1)),
                 linewidth=1.6)
    ax1.set_xlabel('Communication round')
    ax1.set_ylabel('Test accuracy (%)')
    ax1.set_title(r'Convergence trajectories for varying $V$')
    ax1.grid(alpha=0.3, linestyle=':')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(1, 100)
    ax1.set_ylim(8, 50)

    # right: bar of last5 by V
    Vs_present = [v for v in Vs if v in curves]
    last5_vals = [curves[v][-5:].mean() for v in Vs_present]
    bars = ax2.bar([str(v) for v in Vs_present], last5_vals,
                    color=[cmap(i / max(1, len(Vs_present) - 1)) for i in range(len(Vs_present))],
                    edgecolor='black', linewidth=0.7)
    for bar, val in zip(bars, last5_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
                  f'{val:.1f}', ha='center', fontsize=9)
    ax2.set_xlabel(r'$V$')
    ax2.set_ylabel('Last-5 test accuracy (%)')
    ax2.set_title(r'Last-5 acc.\ vs.\ $V$')
    ax2.set_ylim(0, 55)
    ax2.grid(alpha=0.3, axis='y', linestyle=':')

    fig.suptitle(r'Sensitivity to Lyapunov $V$ — CIFAR-10, $\alpha=0.1$, '
                  r'$\sigma_{\mathrm{dp}}=0.01$, single seed=42', fontsize=11)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches='tight')
    fig.savefig(PNG_OUT, dpi=140, bbox_inches='tight')
    print(f'wrote {FIG_OUT}\nwrote {PNG_OUT}')


if __name__ == '__main__':
    main()

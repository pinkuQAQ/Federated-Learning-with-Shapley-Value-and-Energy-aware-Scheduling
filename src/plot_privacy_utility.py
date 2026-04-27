#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_privacy_utility.py — σ_dp ↔ accuracy ↔ (ε, δ) 三栏图。
左轴：last-5 accuracy；右轴：log10(ε) 在 δ=1e-5 下的 RDP→DP 转换。
输出 paper_latex/figures/privacy_utility.pdf
"""
import math
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
SAVE = ROOT / 'save'
FIG_OUT = ROOT / 'paper_latex' / 'figures' / 'privacy_utility.pdf'
PNG_OUT = SAVE / 'privacy_utility.png'

import sys
sys.path.insert(0, str(ROOT / 'src'))
from compute_privacy import compute_total_epsilon


def normalize(a):
    a = np.asarray(a, dtype=np.float64)
    return a * 100.0 if a.max() <= 1.5 else a


def main():
    sigmas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    last5 = []
    for s in sigmas:
        folders = sorted(SAVE.glob(f'sens_dp_sigma{s}_*'))
        if not folders:
            last5.append(np.nan); continue
        pkls = list(folders[-1].glob('*.pkl'))
        if not pkls:
            last5.append(np.nan); continue
        with open(pkls[0], 'rb') as f:
            d = pickle.load(f)
        a = normalize(d['test_accuracy'])
        last5.append(a[-5:].mean())
    last5 = np.array(last5)

    # ε(σ) at q=0.1, T=100, δ=1e-5
    epsilons = []
    for s in sigmas:
        if s <= 0:
            epsilons.append(float('inf'))
        else:
            epsilons.append(compute_total_epsilon(sigma=s, q=0.1, T=100, delta=1e-5))
    epsilons = np.array(epsilons)

    fig, ax1 = plt.subplots(figsize=(7.0, 4.4))
    color1 = 'tab:blue'
    ax1.plot(sigmas, last5, marker='o', color=color1, linewidth=1.8, label='Test acc. (last 5)')
    ax1.set_xlabel(r'Noise multiplier $\sigma_{\mathrm{dp}}$')
    ax1.set_ylabel('Test accuracy (%)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(sigmas)
    ax1.set_xticklabels([str(s) for s in sigmas])
    ax1.grid(alpha=0.3, linestyle=':')
    ax1.set_ylim(0, 55)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    finite_eps = [e if not math.isinf(e) else None for e in epsilons]
    plot_eps = [e if e is not None else np.nan for e in finite_eps]
    ax2.semilogy([s for s in sigmas if s > 0], [e for e in plot_eps[1:]],
                 marker='s', color=color2, linewidth=1.5, linestyle='--',
                 label=r'$\varepsilon$ (Gaussian, $\delta=10^{-5}$)')
    ax2.set_ylabel(r'$\varepsilon$ at $\delta=10^{-5}$ (log scale)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(10, color=color2, linestyle=':', alpha=0.5)
    ax2.text(0.97, 11, r'$\varepsilon=10$ (loose)', color=color2, fontsize=8, ha='right')

    fig.suptitle('Noise vs. utility on CIFAR-10, '
                 r'$\alpha=0.1$, $V=10$, $T=100$, $q=K/N=0.1$, single seed=42')
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches='tight')
    fig.savefig(PNG_OUT, dpi=140, bbox_inches='tight')

    print(f'wrote {FIG_OUT}\nwrote {PNG_OUT}')
    print('\n  σ_dp |  acc%  |   ε       ')
    print('-' * 32)
    for s, a, e in zip(sigmas, last5, epsilons):
        es = 'inf' if math.isinf(e) else f'{e:8.2f}'
        print(f'  {s:>5.2f} | {a:5.2f} | {es}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_ablation_curves.py — 消融多 seed 曲线 + ±1σ 阴影。
输出 paper_latex/figures/ablation_curves.pdf
"""
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
SAVE = ROOT / 'save'
FIG_OUT = ROOT / 'latex' / 'figures' / 'ablation_curves.pdf'
PNG_OUT = SAVE / 'ablation_curves.png'

VARIANTS = [
    ('hybrid_SV_Energy_Lyapunov_LDP', 'Full (SV + Lyap + Energy)', 'C0', '-'),
    ('random_Energy_Lyapunov_LDP',     'w/o SV',                    'C1', '--'),
    ('hybrid_SV_Energy_LDP',           'w/o Lyapunov',              'C2', '-.'),
    ('hybrid_SV_LDP',                  'w/o Energy (SV-only)',      'C3', ':'),
]


def find_tag(name):
    s = Path(name).stem
    m = '_B[32]_'
    i = s.find(m)
    return s[i + len(m):] if i >= 0 else None


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
    data = defaultdict(list)
    for folder in sorted(SAVE.glob('ablation_3seed_a0.1_seed*_180923')):
        for pkl in folder.glob('*.pkl'):
            t = find_tag(pkl.name)
            if t is None:
                continue
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            data[t].append(normalize(d['test_accuracy']))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for tag, label, color, ls in VARIANTS:
        if tag not in data or not data[tag]:
            continue
        L = min(len(a) for a in data[tag])
        runs = np.stack([a[:L] for a in data[tag]], axis=0)
        smooth = np.stack([smooth_ema(r, alpha=0.1) for r in runs], axis=0)
        m = smooth.mean(axis=0)
        x = np.arange(1, L + 1)
        ax.plot(x, m, color=color, linestyle=ls, label=label, linewidth=1.6)

    ax.set_xlabel('Communication round')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(r'Ablation on CIFAR-10, $\alpha=0.1$, '
                 'mean over 3 seeds')
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.85)
    ax.set_xlim(1, 100)
    ax.set_ylim(8, 50)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches='tight')
    fig.savefig(PNG_OUT, dpi=140, bbox_inches='tight')
    print(f'wrote {FIG_OUT}\nwrote {PNG_OUT}')


if __name__ == '__main__':
    main()

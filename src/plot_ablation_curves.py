#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot ablation curves from save/main and save/ablation run folders."""
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
    ('main', 'hybrid_SV_Energy_Lyapunov_CDP', 'Full', 'C0', '-'),
    ('ablation', 'random_Energy_Lyapunov_CDP', 'w/o SV', 'C1', '--'),
    ('ablation', 'hybrid_SV_Energy_CDP', 'w/o Lyapunov', 'C2', '-.'),
    ('ablation', 'hybrid_SV_CDP', 'w/o Energy', 'C3', ':'),
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-tag', default=None, help='Run tag under save/main. Defaults to latest.')
    parser.add_argument('--ablation-tag', default=None, help='Run tag under save/ablation. Defaults to latest.')
    args = parser.parse_args()

    def latest_run(group, tag):
        root = SAVE / group
        if tag:
            return root / tag
        runs = sorted([p for p in root.iterdir() if p.is_dir()])
        if not runs:
            raise FileNotFoundError(f'No run directories found under {root}')
        return runs[-1]

    roots = {
        'main': latest_run('main', args.main_tag),
        'ablation': latest_run('ablation', args.ablation_tag),
    }
    print(f"using main={roots['main']}")
    print(f"using ablation={roots['ablation']}")

    data = defaultdict(list)
    for group, tag, *_ in VARIANTS:
        for folder in sorted([p for p in roots[group].iterdir() if p.is_dir()]):
            for pkl in folder.glob('*.pkl'):
                if find_tag(pkl.name) != tag:
                    continue
                with open(pkl, 'rb') as f:
                    d = pickle.load(f)
                data[tag].append(normalize(d['test_accuracy']))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for _, tag, label, color, ls in VARIANTS:
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
    ax.set_title(r'Ablation on CIFAR-10, $\alpha=0.1$, channel-only privacy')
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

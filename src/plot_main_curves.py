#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot main-comparison curves from saved main-comparison runs."""
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
SAVE = ROOT / 'save'
FIG_OUT = ROOT / 'latex' / 'figures' / 'baseline_convergence.pdf'
PNG_OUT = SAVE / 'baseline_convergence.png'

METHODS = [
    ('hybrid_SV_Energy_Lyapunov_CDP', 'Ours',     'C0', '-'),
    ('random_CDP',                     'FedAvg',   'C1', '--'),
    ('random_FedProx_CDP',             'FedProx',  'C2', '-.'),
    ('poc_CDP',                        'PoC',      'C3', ':'),
    ('oort_Energy_CDP',                'Oort',     'C5', '-.'),
    ('gca_Energy_CDP',                 'GCA',      'C4', (0, (3, 1, 1, 1))),
]


def find_tag(name: str):
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
    parser.add_argument('--tag', default=None, help='Run tag under save/main/. Defaults to latest.')
    parser.add_argument('--root', default=None,
                        help='Optional main-result root, e.g. save/sv_supp/<tag>/main.')
    args = parser.parse_args()

    data = defaultdict(list)
    if args.root:
        run_dir = Path(args.root)
    elif args.tag and (SAVE / 'sv_supp' / args.tag / 'main').exists():
        run_dir = SAVE / 'sv_supp' / args.tag / 'main'
    else:
        main_root = SAVE / 'main'
        if args.tag:
            run_dir = main_root / args.tag
        else:
            run_dirs = sorted([p for p in main_root.iterdir() if p.is_dir()])
            if not run_dirs:
                raise FileNotFoundError('No run directories found under save/main')
            run_dir = run_dirs[-1]
    if not run_dir.exists():
        raise FileNotFoundError(f'Run directory not found: {run_dir}')
    folders = sorted([p for p in run_dir.iterdir() if p.is_dir()])
    print(f'using {run_dir}')
    for folder in folders:
        for pkl in folder.glob('*.pkl'):
            t = find_tag(pkl.name)
            if t is None:
                continue
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            data[t].append(normalize(d['test_accuracy']))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for tag, label, color, ls in METHODS:
        if tag not in data or not data[tag]:
            continue
        # Pad to common length
        L = min(len(a) for a in data[tag])
        runs = np.stack([a[:L] for a in data[tag]], axis=0)
        runs_smooth = np.stack([smooth_ema(r, alpha=0.1) for r in runs], axis=0)
        mean = runs_smooth.mean(axis=0)
        x = np.arange(1, L + 1)
        ax.plot(x, mean, color=color, linestyle=ls, label=label, linewidth=1.6)

    ax.set_xlabel('Communication round')
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title(r'CIFAR-10, Dirichlet $\alpha=0.1$, channel-only privacy, '
                 'mean over 3 seeds')
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.85)
    ax.set_xlim(1, max(L for tag in data if data[tag] for L in [min(len(a) for a in data[tag])]))
    ax.set_ylim(8, 50)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches='tight')
    fig.savefig(PNG_OUT, dpi=140, bbox_inches='tight')
    print(f'wrote {FIG_OUT}\nwrote {PNG_OUT}')


if __name__ == '__main__':
    main()

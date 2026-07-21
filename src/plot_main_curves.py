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

METHODS = [
    ('hybrid_SV_Energy_Lyapunov_CDP', 'Ours',     'C0', '-'),
    ('random_CDP',                     'FedAvg',   'C1', '--'),
    ('random_FedProx_CDP',             'FedProx',  'C2', '-.'),
    ('oort_Energy_CDP',                'Oort',     'C5', '-.'),
    ('gca_Energy_CDP',                 'GCA',      'C4', (0, (3, 1, 1, 1))),
    ('fedmsv_CDP',                     'Fed-MSV',  'C6', (0, (5, 1))),
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
    parser.add_argument('--merge-root', action='append', default=[],
                        help='Additional result root to merge by method tag and seed.')
    parser.add_argument('--replace-root', default=None,
                        help='Optional root whose runs replace matching method tags from the base root.')
    parser.add_argument('--replace-tags', default='',
                        help='Comma-separated method tags to replace from --replace-root.')
    parser.add_argument('--png', action='store_true',
                        help='Also write a PNG copy under save/.')
    args = parser.parse_args()

    data = defaultdict(dict)
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
    if args.root:
        pkls = sorted(run_dir.rglob('*.pkl'))
    else:
        folders = sorted([p for p in run_dir.iterdir() if p.is_dir()])
        pkls = [pkl for folder in folders for pkl in folder.glob('*.pkl')]
    print(f'using {run_dir}')
    for pkl in pkls:
        t = find_tag(pkl.name)
        if t is None:
            continue
        with open(pkl, 'rb') as f:
            d = pickle.load(f)
        seed = int((d.get('args') or {}).get('seed', len(data[t])))
        data[t][seed] = normalize(d['test_accuracy'])

    for merge_root in args.merge_root:
        merge_dir = Path(merge_root)
        if not merge_dir.exists():
            raise FileNotFoundError(f'Merge directory not found: {merge_dir}')
        for pkl in sorted(merge_dir.rglob('*.pkl')):
            t = find_tag(pkl.name)
            if t is None:
                continue
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            seed = int((d.get('args') or {}).get('seed', len(data[t])))
            data[t][seed] = normalize(d['test_accuracy'])
        print(f'merged {merge_dir}')

    replace_tags = {tag.strip() for tag in args.replace_tags.split(',') if tag.strip()}
    if args.replace_root:
        if not replace_tags:
            raise ValueError('--replace-tags is required when --replace-root is used')
        replace_dir = Path(args.replace_root)
        if not replace_dir.exists():
            raise FileNotFoundError(f'Replacement directory not found: {replace_dir}')
        for tag in replace_tags:
            data[tag] = {}
        for pkl in sorted(replace_dir.rglob('*.pkl')):
            t = find_tag(pkl.name)
            if t not in replace_tags:
                continue
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            seed = int((d.get('args') or {}).get('seed', len(data[t])))
            data[t][seed] = normalize(d['test_accuracy'])
        print(f'replaced {sorted(replace_tags)} from {replace_dir}')

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    lengths = []
    for tag, label, color, ls in METHODS:
        print(f'{label}: {len(data[tag])} runs, seeds={sorted(data[tag])}')
        if tag not in data or not data[tag]:
            continue
        # Pad to common length
        method_runs = list(data[tag].values())
        L = min(len(a) for a in method_runs)
        lengths.append(L)
        runs = np.stack([a[:L] for a in method_runs], axis=0)
        runs_smooth = np.stack([smooth_ema(r, alpha=0.1) for r in runs], axis=0)
        mean = runs_smooth.mean(axis=0)
        x = np.arange(1, L + 1)
        ax.plot(x, mean, color=color, linestyle=ls, label=label, linewidth=1.6)

    ax.set_xlabel('Communication round')
    ax.set_ylabel('Test accuracy (%)')
    run_counts = [len(data[tag]) for tag, _, _, _ in METHODS if data[tag]]
    count_label = str(min(run_counts)) if run_counts and len(set(run_counts)) == 1 else 'matched'
    ax.set_title(r'CIFAR-10, Dirichlet $\alpha=0.1$, channel-noise diagnostic, '
                 f'mean over {count_label} seeds')
    ax.grid(alpha=0.3, linestyle=':')
    ax.legend(loc='lower right', fontsize=9, ncol=2, framealpha=0.85)
    if not lengths:
        raise ValueError(f'No plottable method runs found under {run_dir}')
    ax.set_xlim(1, max(lengths))
    ax.set_ylim(8, 50)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, bbox_inches='tight')
    print(f'wrote {FIG_OUT}')
    if args.png:
        png_out = SAVE / 'baseline_convergence.png'
        fig.savefig(png_out, dpi=140, bbox_inches='tight')
        print(f'wrote {png_out}')


if __name__ == '__main__':
    main()

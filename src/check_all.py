#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_all.py — 一次性把 main / ablation / stress / sens_V / sens_dp 都过一遍，
打印每个实验的 last-5 mean (±std), final acc, last20 slope。仅供我们检查趋势。
"""
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).parent.parent
SAVE = ROOT / 'save'


def load_pkl(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    a = np.asarray(d['test_accuracy'], dtype=np.float64)
    return a * 100.0 if a.max() <= 1.5 else a


def stats(acc):
    return dict(
        final=acc[-1],
        last5=acc[-5:].mean(),
        slope=np.polyfit(np.arange(20), acc[-20:], 1)[0],
        std20=acc[-20:].std(),
    )


def find_tag(name):
    s = Path(name).stem
    m = '_B[32]_'
    i = s.find(m)
    return s[i + len(m):] if i >= 0 else None


# ---------- main ----------
print('\n[ MAIN — α=0.1 σ=0.01 100ep × 3 seeds ]')
data = defaultdict(list)
for folder in sorted(SAVE.glob('main_3seed_a0.1_sigma0.01_seed*_180923')):
    seed = int(folder.name.split('_seed')[1].split('_')[0])
    for pkl in folder.glob('*.pkl'):
        tag = find_tag(pkl.name)
        if tag is None:
            continue
        acc = load_pkl(pkl)
        data[tag].append((seed, stats(acc)))
order = ['hybrid_SV_Energy_Lyapunov_LDP', 'random_LDP', 'random_FedProx_LDP',
         'poc_LDP', 'ucb_LDP', 'fedcs_Energy_LDP']
labels = ['Ours', 'FedAvg', 'FedProx', 'PoC', 'UCB', 'FedCS']
for tag, lab in zip(order, labels):
    if tag not in data:
        print(f'  {lab:10s}: missing'); continue
    f = np.array([s['final'] for _, s in data[tag]])
    l5 = np.array([s['last5'] for _, s in data[tag]])
    sl = np.array([s['slope'] for _, s in data[tag]])
    print(f'  {lab:10s}: final {f.mean():5.2f}±{f.std():4.2f}  '
          f'last5 {l5.mean():5.2f}±{l5.std():4.2f}  slope {sl.mean():+.3f}')

# ---------- ablation ----------
print('\n[ ABLATION — α=0.1 100ep × 3 seeds ]')
abl = defaultdict(list)
for folder in sorted(SAVE.glob('ablation_3seed_a0.1_seed*_180923')):
    seed = int(folder.name.split('_seed')[1].split('_')[0])
    for pkl in folder.glob('*.pkl'):
        tag = find_tag(pkl.name)
        abl[tag].append((seed, stats(load_pkl(pkl))))
abl_order = [
    ('hybrid_SV_Energy_Lyapunov_LDP', 'Full'),
    ('random_Energy_Lyapunov_LDP',     'w/o SV'),
    ('hybrid_SV_Energy_LDP',           'w/o Lyap'),
    ('hybrid_SV_LDP',                  'w/o Energy'),
]
for tag, lab in abl_order:
    if tag not in abl:
        print(f'  {lab:12s}: missing'); continue
    l5 = np.array([s['last5'] for _, s in abl[tag]])
    sl = np.array([s['slope'] for _, s in abl[tag]])
    print(f'  {lab:12s}: last5 {l5.mean():5.2f}±{l5.std():4.2f}  slope {sl.mean():+.3f}')

# ---------- stress ----------
print('\n[ STRESS — E_init=100, α=0.1, 1 seed ]')
stress_dirs = sorted(SAVE.glob('stress_e100.0_a0.1_seed42_*'))
if stress_dirs:
    for pkl in sorted(stress_dirs[-1].glob('*.pkl')):
        tag = find_tag(pkl.name)
        with open(pkl, 'rb') as f:
            d = pickle.load(f)
        acc = load_pkl(pkl)
        st = stats(acc)
        # energy/queue stats if present
        es = d.get('energy_statistics', {})
        ls = d.get('lyapunov_statistics', {})
        avg_e = np.mean(es.get('current_energy', [np.nan]))
        min_e = np.min(es.get('current_energy', [np.nan]))
        depleted = int(np.sum(np.asarray(es.get('current_energy', [])) < 10.0))
        max_q = ls.get('queue_max', np.nan) if isinstance(ls, dict) else np.nan
        print(f'  {tag[:38]:38s}: final {st["final"]:5.2f}%  last5 {st["last5"]:5.2f}%  '
              f'avgE {avg_e:6.2f}  minE {min_e:6.2f}  depl {depleted:2d}  maxQ {max_q}')

# ---------- V sensitivity ----------
print('\n[ V SENSITIVITY ]')
for v in [1, 5, 10, 20, 50]:
    folders = sorted(SAVE.glob(f'sens_V{v}_*'))
    if not folders:
        print(f'  V={v:>2d}: missing'); continue
    pkl = next(folders[-1].glob('*.pkl'))
    st = stats(load_pkl(pkl))
    print(f'  V={v:>2d}: final {st["final"]:5.2f}%  last5 {st["last5"]:5.2f}%  slope {st["slope"]:+.3f}')

# ---------- DP sigma sensitivity ----------
print('\n[ σ_dp SENSITIVITY ]')
for s in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    folders = sorted(SAVE.glob(f'sens_dp_sigma{s}_*'))
    if not folders:
        print(f'  σ={s:>5.2f}: missing'); continue
    pkl = next(folders[-1].glob('*.pkl'))
    st = stats(load_pkl(pkl))
    print(f'  σ={s:>5.2f}: final {st["final"]:5.2f}%  last5 {st["last5"]:5.2f}%')

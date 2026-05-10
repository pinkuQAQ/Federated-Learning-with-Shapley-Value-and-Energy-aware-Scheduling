#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compute_privacy.py — 客户端 Gaussian 机制的 (ε, δ) 离线核算

复用现有 sens_dp_sigma* / baseline / ablation 实验的 .pkl 中保存的
args.dp_noise_multiplier、args.dp_clip_norm、args.epochs、args.num_users、
args.num_selected，按标准 RDP 复合给出 (ε, δ)。

我们建模的是：每一轮被选中的客户端各自独立执行一次 ℓ2 裁剪到 C 后加 N(0, σ_dp² C² I)
噪声的 Gaussian mechanism。客户端层面看，"是否被选中" 是一个 q = K/N 的子采样事件
（按 average per-round 近似），合在一起就是一个 sub-sampled Gaussian mechanism，
其 RDP 通过经典上界给出，最后 RDP→(ε, δ) 转换。

参考：
  Mironov, "Rényi Differential Privacy", CSF 2017.
  Wang, Balle, Kasiviswanathan, "Subsampled Rényi Differential Privacy", AISTATS 2019.
  Abadi et al., "Deep Learning with Differential Privacy" (moments accountant) 2016.

用法：
  python compute_privacy.py --sigma 0.01 --T 100 --q 0.1 --delta 1e-5
  python compute_privacy.py --pkl ../save/main_channel_dp_3seed_a0.1_K15_sigma2.5_ch1.0_seed42_<tag>/<file>.pkl
  python compute_privacy.py --table  # 打印所有典型 σ 的 (ε, δ) 对照表
"""
import argparse
import math
import os
import pickle
import sys
from typing import List, Optional, Sequence


# ----------- 核心 RDP 公式 -----------

def gaussian_rdp(alpha: float, sigma: float) -> float:
    """无子采样 Gaussian mechanism 在 order α 下的 RDP：α / (2 σ²)."""
    if sigma <= 0:
        return float('inf')
    return alpha / (2.0 * sigma * sigma)


def subsampled_gaussian_rdp(alpha: float, sigma: float, q: float) -> float:
    """
    Sub-sampled Gaussian mechanism 的 RDP 上界（Wang et al. 2019 Thm 9 简化形式）。
    对整数 α ≥ 2：
        ε(α) ≤ (1/(α-1)) * log( sum_{k=0..α} C(α,k) (1-q)^{α-k} q^k * exp((k²-k)/(2σ²)) )
    本实现仅支持整数 α ≥ 2；选 order 时请只传整数。
    """
    if not (0 < q <= 1):
        raise ValueError("q must be in (0, 1].")
    if abs(alpha - round(alpha)) > 1e-9 or int(round(alpha)) < 2:
        raise ValueError("alpha must be an integer >= 2 for this RDP bound.")
    a = int(round(alpha))
    log_terms = []
    for k in range(a + 1):
        log_binom = math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1)
        log_q_term = k * math.log(max(q, 1e-300))
        log_1mq_term = (a - k) * math.log(max(1.0 - q, 1e-300))
        log_exp_term = (k * k - k) / (2.0 * sigma * sigma)
        log_terms.append(log_binom + log_q_term + log_1mq_term + log_exp_term)
    max_lt = max(log_terms)
    log_sum = max_lt + math.log(sum(math.exp(lt - max_lt) for lt in log_terms))
    return log_sum / (a - 1)


def rdp_to_dp(rdp_at_orders: Sequence[float], orders: Sequence[float], delta: float) -> float:
    """
    将一组 (α, RDP(α)) 转成 (ε, δ)，取最优 α 给出最小 ε。
    使用 Mironov 2017 的转换：ε = RDP(α) + log(1/δ) / (α - 1)
    """
    best_eps = float('inf')
    for alpha, rdp in zip(orders, rdp_at_orders):
        if alpha <= 1:
            continue
        eps = rdp + math.log(1.0 / delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps = eps
    return best_eps


def compute_total_epsilon(sigma: float, q: float, T: int, delta: float = 1e-5,
                          orders: Optional[Sequence[float]] = None) -> float:
    """
    给定每轮 sub-sampled Gaussian 机制（噪声乘子 σ、采样率 q）、复合 T 轮后转 (ε, δ)。
    """
    if sigma <= 0:
        return float('inf')
    if orders is None:
        orders = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128]
    rdp_per_round = [subsampled_gaussian_rdp(a, sigma, q) for a in orders]
    rdp_total = [r * T for r in rdp_per_round]
    return rdp_to_dp(rdp_total, orders, delta)


# ----------- I/O 入口 -----------

def from_args(sigma: float, q: float, T: int, delta: float) -> None:
    eps = compute_total_epsilon(sigma=sigma, q=q, T=T, delta=delta)
    print(f"σ_dp={sigma}, q={q:.4f}, T={T}, δ={delta:.0e}  =>  ε ≈ {eps:.3f}")


def from_pkl(pkl_path: str, delta: float) -> None:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    args = data.get('args', None)
    if args is None:
        print(f"[skip] {pkl_path}: no args in pkl")
        return
    privacy_mode = getattr(args, 'privacy_mode', None)
    use_dp = privacy_mode == 'central'
    sigma = getattr(args, 'dp_noise_multiplier', 0.0)
    if not use_dp or sigma <= 0:
        print(f"{os.path.basename(pkl_path)}: DP off (σ_dp={sigma}) -> epsilon = inf (no formal guarantee)")
        return
    K = getattr(args, 'num_selected', 10)
    N = getattr(args, 'num_users', 100)
    T = getattr(args, 'epochs', 100)
    q = K / float(N)
    eps = compute_total_epsilon(sigma=sigma, q=q, T=T, delta=delta)
    print(f"{os.path.basename(pkl_path)}: σ_dp={sigma}, K/N={K}/{N}={q:.3f}, T={T} "
          f"=> (ε={eps:.3f}, δ={delta:.0e})")


def print_table(q: float, T: int, delta: float, sigmas: Sequence[float]) -> None:
    print(f"\nGaussian-mechanism (ε, δ) accounting at q={q:.3f}, T={T}, δ={delta:.0e}\n")
    print(f"{'sigma_dp':>10s} {'epsilon':>12s}")
    print('-' * 24)
    for s in sigmas:
        if s <= 0:
            print(f"{s:>10.4f} {'inf':>12s}  (no formal DP)")
            continue
        eps = compute_total_epsilon(sigma=s, q=q, T=T, delta=delta)
        print(f"{s:>10.4f} {eps:>12.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', type=float, default=None,
                        help='noise multiplier σ_dp; required unless --pkl or --table')
    parser.add_argument('--q', type=float, default=0.1,
                        help='per-round sampling rate K/N (default 0.1)')
    parser.add_argument('--T', type=int, default=100,
                        help='number of communication rounds (default 100)')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='target δ (default 1e-5)')
    parser.add_argument('--pkl', type=str, default=None,
                        help='read sigma/q/T from a saved experiment pkl')
    parser.add_argument('--table', action='store_true',
                        help='print a comparison table over standard sigmas')
    parser.add_argument('--sigmas', type=float, nargs='*',
                        default=[0.0, 0.01, 0.05, 0.1, 0.2],
                        help='sigma list for --table')
    args = parser.parse_args()

    if args.pkl is not None:
        from_pkl(args.pkl, delta=args.delta)
        return

    if args.table:
        print_table(q=args.q, T=args.T, delta=args.delta, sigmas=args.sigmas)
        return

    if args.sigma is None:
        print('Error: provide --sigma or --pkl or --table.', file=sys.stderr)
        sys.exit(2)

    from_args(sigma=args.sigma, q=args.q, T=args.T, delta=args.delta)


if __name__ == '__main__':
    main()

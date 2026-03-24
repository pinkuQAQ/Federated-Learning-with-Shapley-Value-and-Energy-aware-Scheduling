#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
隐私保护验证实验
对比加密与不加密实验的准确率和开销，验证AES-256-GCM零精度影响
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def information_leakage_analysis(save_dir='../save/crypto_15202603_1708'):
    """
    分析加密前后的准确率差异和加密开销
    从实验结果文件中加载真实数据进行对比
    """
    print("=" * 60)
    print("隐私保护分析：加密开销与准确率影响")
    print("=" * 60)

    pkl_files = list(Path(save_dir).glob('*.pkl'))

    results = {
        'no_crypto': None,
        'with_crypto': None
    }

    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            args = data.get('args', {})
            filename = pkl_file.name

            if args.get('use_crypto'):
                results['with_crypto'] = data
                print(f"加载加密实验: {filename}")
            elif 'random' in filename.lower() or args.get('selection_method') == 'random':
                results['no_crypto'] = data
                print(f"加载无加密实验: {filename}")
        except Exception as e:
            print(f"加载失败: {e}")
            continue

    # 计算准确率差异
    if results['no_crypto'] and results['with_crypto']:
        acc_no_crypto = results['no_crypto']['test_accuracy'][-1]
        acc_with_crypto = results['with_crypto']['test_accuracy'][-1]

        print(f"\n准确率对比:")
        print(f"  无加密: {acc_no_crypto:.4f}")
        print(f"  有加密: {acc_with_crypto:.4f}")
        print(f"  准确率差异: {abs(acc_no_crypto - acc_with_crypto)*100:.4f}%")

        # 加密开销
        if 'crypto_statistics' in results['with_crypto']:
            stats = results['with_crypto']['crypto_statistics']
            print(f"\n加密开销:")
            print(f"  算法: {stats.get('algorithm', 'AES-256-GCM')}")
            print(f"  密钥长度: {stats.get('key_size_bits', 256)} bits")
            print(f"  总加密操作: {stats.get('total_encrypt_ops', 0)}")
            print(f"  明文数据量: {stats.get('total_plaintext_KB', 0):.2f} KB")
            print(f"  密文数据量: {stats.get('total_ciphertext_KB', 0):.2f} KB")
            print(f"  元数据膨胀: {stats.get('total_encrypt_ops', 0) * 28} bytes "
                  f"(nonce + tag)")
    else:
        print("\n未找到可对比的实验结果文件")

    print("=" * 60)


if __name__ == '__main__':
    information_leakage_analysis()

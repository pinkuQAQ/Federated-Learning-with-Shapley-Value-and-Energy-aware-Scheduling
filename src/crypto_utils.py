#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
方案2：AES-256-GCM 对称加密 + 即时明文销毁
威胁模型：诚实但好奇（honest-but-curious）服务器
隐私声明：客户端梯度在传输和存储中均以密文形式存在，
          服务器仅在计算期间短暂持有明文，计算完成后立即销毁。
"""

import pickle
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


class ClientCrypto:
    """单个客户端的加密会话，持有该客户端的 AES-256 会话密钥"""

    def __init__(self, client_id):
        self.client_id = client_id
        self.session_key = get_random_bytes(32)   # AES-256：32字节密钥
        self.encrypt_count = 0
        self.decrypt_count = 0
        self.total_plaintext_bytes = 0
        self.total_ciphertext_bytes = 0

    def encrypt_weights(self, weights_dict):
        """
        序列化并加密模型权重字典。
        使用 AES-256-GCM（认证加密），同时提供保密性和完整性验证。

        Returns:
            dict: 包含 ciphertext / nonce / tag / client_id 的加密包
        """
        plaintext = pickle.dumps(weights_dict)
        self.total_plaintext_bytes += len(plaintext)

        cipher = AES.new(self.session_key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)

        self.total_ciphertext_bytes += len(ciphertext)
        self.encrypt_count += 1

        # 明文序列化后立即销毁
        del plaintext

        return {
            'ciphertext': ciphertext,
            'nonce': cipher.nonce,   # 12字节随机 nonce，每次加密唯一
            'tag': tag,              # 16字节认证标签，防止篡改
            'client_id': self.client_id
        }

    def decrypt_weights(self, encrypted_package):
        """
        解密并反序列化模型权重。
        调用方有责任在使用后立即 del 返回值，确保明文不持久化。

        Returns:
            dict: 模型权重字典（调用方必须在用完后 del）
        """
        cipher = AES.new(
            self.session_key,
            AES.MODE_GCM,
            nonce=encrypted_package['nonce']
        )
        # decrypt_and_verify 同时解密并验证认证标签
        plaintext = cipher.decrypt_and_verify(
            encrypted_package['ciphertext'],
            encrypted_package['tag']
        )
        weights = pickle.loads(plaintext)
        del plaintext   # 立即销毁反序列化前的字节串

        self.decrypt_count += 1
        return weights


class CryptoManager:
    """
    管理所有客户端的加密会话。

    协议流程：
      1. 初始化时为每个客户端生成独立的 AES-256 会话密钥
      2. 客户端训练完成后调用 encrypt(client_id, weights) 加密上传
      3. 服务器调用 decrypt_and_destroy(pkg) 获取明文
      4. 服务器使用明文后立即 del，只持久化密文
    """

    def __init__(self, num_clients):
        self.clients = {i: ClientCrypto(i) for i in range(num_clients)}
        self.rounds_processed = 0

    def encrypt(self, client_id, weights_dict):
        """客户端侧：加密本地模型权重"""
        return self.clients[client_id].encrypt_weights(weights_dict)

    def decrypt_and_destroy(self, encrypted_package):
        """
        服务器侧：解密权重包。
        注意：调用方须在使用完返回值后立即执行 del，
        确保明文生命周期仅限于单次计算。
        """
        client_id = encrypted_package['client_id']
        return self.clients[client_id].decrypt_weights(encrypted_package)

    def get_statistics(self):
        total_plain = sum(c.total_plaintext_bytes for c in self.clients.values())
        total_cipher = sum(c.total_ciphertext_bytes for c in self.clients.values())
        total_enc = sum(c.encrypt_count for c in self.clients.values())

        return {
            'algorithm': 'AES-256-GCM',
            'key_size_bits': 256,
            'num_clients': len(self.clients),
            'total_encrypt_ops': total_enc,
            'total_plaintext_KB': total_plain / 1024,
            'total_ciphertext_KB': total_cipher / 1024,
            'overhead_ratio': total_cipher / total_plain if total_plain > 0 else 0,
        }

    def print_statistics(self, epoch=None):
        stats = self.get_statistics()
        prefix = f"  [加密-Round {epoch}]" if epoch is not None else "  [加密统计]"
        print(f"{prefix} 算法: {stats['algorithm']}, 密钥长度: {stats['key_size_bits']}位")
        print(f"{prefix} 累计加密操作: {stats['total_encrypt_ops']}次")
        print(f"{prefix} 传输数据量: {stats['total_plaintext_KB']:.1f} KB (明文) → "
              f"{stats['total_ciphertext_KB']:.1f} KB (密文), "
              f"膨胀率: {stats['overhead_ratio']:.3f}x")

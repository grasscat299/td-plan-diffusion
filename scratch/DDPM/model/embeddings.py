"""
t/angleなど、スカラーのバッチをsinusoidal embeddingに変換する共通ユーティリティ。
model_transformer.py / model_unet.py の両方から使う。
"""
from __future__ import annotations

import math

import torch


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x(shape (b,))をTransformerの位置エンコーディングと同じ式でdim次元のベクトルに変換する。
    """
    half_dim = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half_dim, device=x.device) / half_dim)
    args = x.float().unsqueeze(1) * freq.unsqueeze(0)  # (b, half_dim)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # (b, dim)

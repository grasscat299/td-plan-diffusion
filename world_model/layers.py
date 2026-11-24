"""モデルの部品。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(x: torch.Tensor, dim: int) -> torch.Tensor:
    """x (n,) を Transformer の位置エンコーディングと同じ式で (n, dim) にする。"""
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device) / half)
    args = x.float().unsqueeze(1) * freq.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def _modulate(x: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """adaLN の scale-shift: x * (1 + g) + b。x (Bn, N, d)、g・b (Bn, d)。"""
    return x * (1.0 + g.unsqueeze(1)) + b.unsqueeze(1)


class TowerBlock(nn.Module):
    """adaLN-Zero のブロック。self-attention（sa）・相手のタワーへの cross-attention（ca）・MLP（ff）を別々に呼べる。
    3 つのタワーを並列に結合するため、各層で sa → ca → ff の順に、タワーをまたいで呼ぶ。"""

    def __init__(self, d: int, num_heads: int, ffn_mult: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.SiLU(), nn.Linear(d * ffn_mult, d))
        self.adaLN = nn.Linear(d, 9 * d)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def params(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.adaLN(c).chunk(9, dim=-1)

    def sa(self, h, p, key_padding_mask=None):
        g, b, a = p[0], p[1], p[2]
        x = _modulate(self.norm1(h), g, b)
        o, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return h + a.unsqueeze(1) * self.drop(o)

    def ca(self, h, p, kv, key_padding_mask=None):
        g, b, a = p[3], p[4], p[5]
        q = _modulate(self.norm2(h), g, b)
        o, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        return h + a.unsqueeze(1) * self.drop(o)

    def ff(self, h, p):
        g, b, a = p[6], p[7], p[8]
        return h + a.unsqueeze(1) * self.drop(self.mlp(_modulate(self.norm3(h), g, b)))


class BoardCA(nn.Module):
    """配置タワーが盤面のトークンだけを読む cross-attention（敵・資源を読む CA とは別の重みとゲート）。"""

    def __init__(self, d: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.adaLN = nn.Linear(d, 3 * d)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)

    def forward(self, h, c, kv):
        g, b, a = self.adaLN(c).chunk(3, dim=-1)
        o, _ = self.attn(_modulate(self.norm(h), g, b), kv, kv, need_weights=False)
        return h + a.unsqueeze(1) * self.drop(o)


class TimeCompress(nn.Module):
    """時系列のトークン (Bn, T, d) を、時間方向の線形 1 枚で (Bn, t_c, d) にする。層ごとに別の重み。"""

    def __init__(self, T: int, t_c: int):
        super().__init__()
        self.lin = nn.Linear(T, t_c)

    def forward(self, h):
        return self.lin(h.transpose(1, 2)).transpose(1, 2)


class FinalLayer(nn.Module):
    """最後の adaLN（ゲートなし）と線形。どちらも 0 で初期化する。"""

    def __init__(self, d: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.adaLN = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, out_dim)
        for m in (self.adaLN, self.proj):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, h, c):
        g, b = self.adaLN(c).chunk(2, dim=-1)
        return self.proj(_modulate(self.norm(h), g, b))


class _RouteBlock(nn.Module):
    def __init__(self, d: int, num_heads: int, ffn_mult: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * ffn_mult), nn.SiLU(), nn.Linear(d * ffn_mult, d))

    def forward(self, h, key_padding_mask):
        x = self.norm1(h)
        o, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        h = h + o
        return h + self.mlp(self.norm2(h))


class RouteKVEncoder(nn.Module):
    """敵ごとの通り道 (Bn, N_r, route_dim) を、敵タワーの cross-attention の K/V (Bn, N_r, d) にする。"""

    def __init__(self, d: int, max_route_points: int, num_layers: int, num_heads: int, ffn_mult: int, in_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, d)
        self.lin2 = nn.Linear(d, d)
        self.blocks = nn.ModuleList([_RouteBlock(d, num_heads, ffn_mult) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d)
        self.register_buffer("pe_route", sinusoidal_embedding(torch.arange(max_route_points), d))

    def forward(self, route, padding_mask):
        h = self.lin2(F.silu(self.lin1(route))) + self.pe_route.unsqueeze(0)
        for blk in self.blocks:
            h = blk(h, padding_mask)
        return self.norm(h)


class MapEncoder(nn.Module):
    """盤面 (B, R, C) の地形と升ごとの特徴を (B, R*C, d) のトークンにする。行・列の学習する位置埋め込みと、小さな self-attention。
    トークンの並びは行優先（idx = row * C + col）。"""

    def __init__(self, d: int, n_rows: int, n_cols: int, num_heads: int, ffn_mult: int,
                 n_tile_types: int, tile_feat_dim: int, num_layers: int = 1):
        super().__init__()
        self.n_rows, self.n_cols = n_rows, n_cols
        self.tile_emb = nn.Embedding(n_tile_types, d)
        self.tile_feat_lin = nn.Linear(tile_feat_dim, d)
        self.pos_row = nn.Parameter(torch.randn(n_rows, d) * 0.02)
        self.pos_col = nn.Parameter(torch.randn(n_cols, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, num_heads, d * ffn_mult, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, tile_map, tile_feat):
        B, R, C = tile_map.shape
        x = self.tile_emb(tile_map.reshape(B, R * C).long())
        x = x + (self.pos_row[:, None, :] + self.pos_col[None, :, :]).reshape(R * C, -1).unsqueeze(0)
        x = x + self.tile_feat_lin(tile_feat.reshape(B, R * C, -1).to(x.dtype))
        return self.norm(self.encoder(x))


# 射程の形の格子（右向きが基準の相対位置 (dc, dr)）
RANGE_DC0, RANGE_DC1, RANGE_DR0, RANGE_DR1 = -3, 7, -3, 3
RANGE_GRID_N = (RANGE_DC1 - RANGE_DC0 + 1) * (RANGE_DR1 - RANGE_DR0 + 1)   # 77


def range_grid(ope_range: torch.Tensor) -> torch.Tensor:
    """ope_range (B, N_op, K, 2) -> 射程の形の 0/1 の格子 (B, N_op, 77)。格子の外の点と詰め物は捨てる。"""
    dc, dr = ope_range[..., 0].long(), ope_range[..., 1].long()
    ok = (dc >= RANGE_DC0) & (dc <= RANGE_DC1) & (dr >= RANGE_DR0) & (dr <= RANGE_DR1)
    idx = torch.where(ok, (dc - RANGE_DC0) * (RANGE_DR1 - RANGE_DR0 + 1) + (dr - RANGE_DR0),
                      torch.full_like(dc, RANGE_GRID_N))
    g = torch.zeros(*idx.shape[:-1], RANGE_GRID_N + 1, dtype=torch.float32, device=idx.device)
    g.scatter_(-1, idx, 1.0)
    return g[..., :RANGE_GRID_N]


def range_cov(tile_pass: torch.Tensor, ope_range: torch.Tensor, n_rows: int, n_cols: int) -> torch.Tensor:
    """升 p に枠を置いて向き d にした時の、射程の中の「敵が通る延べ時間」の和。
    tile_pass (B, R*C)・ope_range (B, N_op, K, 2) -> (B, N_op, 4, R*C)。
    向きの回転: 0 = (dc, dr)、1 = (-dr, dc)、2 = (-dc, -dr)、3 = (dr, -dc)（行は下が増える）。盤面の外の点は数えない。"""
    B, S, K, _ = ope_range.shape
    dc, dr = ope_range[..., 0].long(), ope_range[..., 1].long()
    rot_c = torch.stack([dc, -dr, -dc, dr], dim=2)
    rot_r = torch.stack([dr, dc, -dr, -dc], dim=2)
    P = n_rows * n_cols
    rows = torch.arange(P, device=tile_pass.device) // n_cols
    cols = torch.arange(P, device=tile_pass.device) % n_cols
    c2 = cols.view(1, 1, 1, P, 1) + rot_c.unsqueeze(3)
    r2 = rows.view(1, 1, 1, P, 1) + rot_r.unsqueeze(3)
    ok = (c2 >= 0) & (c2 < n_cols) & (r2 >= 0) & (r2 < n_rows)
    idx = (r2.clamp(0, n_rows - 1) * n_cols + c2.clamp(0, n_cols - 1)).reshape(B, -1)
    vals = torch.gather(tile_pass.to(torch.float32), 1, idx).reshape(B, S, 4, P, K)
    return (vals * ok.to(vals.dtype)).sum(dim=-1)

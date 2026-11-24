"""
scratch/DDPM/model/model_dit.py をMNIST用に持ってきたコピー。

design.mdの「InputData.angle(b,)についても同じ構造（sinusoidal embedding->linear->silu->linear）
でangle_emb(b,c*16)を計算する」という部分を、数字ラベル(0〜9)向けに学習可能な離散埋め込み
(nn.Embedding(10, c*16))に置き換えている。angle専用のlinear(angle_linear_1/2)とsinusoidal
embeddingは使わず、label_embeddingで直接ラベルからベクトルを引く。y = t_emb + label_emb。

sinusoidal embeddingは「値が近い→埋め込みが近い」という連続値向けの構造を課してしまい、
6と9のように見た目が近い数字が埋め込み上は遠くなる等、離散クラスの条件付けとしては
本質的に不向きという仮説の検証用。t_emb側の計算(sinusoidal embedding->linear->silu->linear)は
元のまま変更していない。

それ以外（DiTBlock/FinalLayer/patchify/unpatchify等）はscratch/DDPM/model/model_dit.pyと
完全に同じ。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scratch.DDPM.data import InputData
from scratch.DDPM.model.embeddings import sinusoidal_embedding

_HIDDEN_CHANNEL_MULTIPLIER = 32  # linearで c -> c*32 に拡大する倍率（model_transformer.pyと揃える）
_PATCH_SIZE = 2                  # design.mdの「p=4など」の例に従い決め打ち。h,wはこれで割り切れる前提
_NUM_DIT_BLOCKS = 4               # design.mdに具体的な数の指定がないため決め打ち（model_transformer.pyの_NUM_TRANSFORMER_LAYERSと揃える）
_NUM_ATTENTION_HEADS = 4          # 同上
_MLP_HIDDEN_MULTIPLIER = 2        # design.mdの「linear(c*16->c*32)->silu->linear(c*32->c*16)」に対応する倍率
_NUM_LABEL_CLASSES = 10           # MNISTの数字ラベル0〜9


def _modulate(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """adaLNのscale-shift: x*(1+γ)+β。x: (b,N,hidden_dim), gamma/beta: (b,hidden_dim)"""
    return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class DiTBlock(nn.Module):
    """design.mdのDiT Block（adaLN-Zero）: LN->scale-shift->MHSA->scale->残差 を2回（Attention, MLP）"""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * _MLP_HIDDEN_MULTIPLIER),
            nn.SiLU(),
            nn.Linear(hidden_dim * _MLP_HIDDEN_MULTIPLIER, hidden_dim),
        )
        self.adaLN_modulation = nn.Linear(hidden_dim, hidden_dim * 6)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN_modulation(y).chunk(6, dim=-1)

        h = _modulate(self.norm1(x), gamma1, beta1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + alpha1.unsqueeze(1) * attn_out

        h = _modulate(self.norm2(x), gamma2, beta2)
        x = x + alpha2.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """design.mdの最終adaLN（γ,βのみ、ゲートαはなし）-> linear(c*16 -> p*p*c)"""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.adaLN_modulation = nn.Linear(hidden_dim, hidden_dim * 2)
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.adaLN_modulation(y).chunk(2, dim=-1)
        x = _modulate(self.norm(x), gamma, beta)
        return self.linear(x)


class DiTModel(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        super().__init__()
        if h % _PATCH_SIZE != 0 or w % _PATCH_SIZE != 0:
            raise ValueError(f"h={h}, w={w} は patch_size={_PATCH_SIZE} で割り切れる必要がある")

        self.c = c
        self.h = h
        self.w = w
        self.patch_size = _PATCH_SIZE
        self.grid_h = h // _PATCH_SIZE
        self.grid_w = w // _PATCH_SIZE

        hidden_dim = c * _HIDDEN_CHANNEL_MULTIPLIER
        self.hidden_dim = hidden_dim

        self.time_linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.time_linear_2 = nn.Linear(hidden_dim, hidden_dim)

        self.label_embedding = nn.Embedding(_NUM_LABEL_CLASSES, hidden_dim)

        patch_dim = c * _PATCH_SIZE * _PATCH_SIZE
        self.patch_linear = nn.Linear(patch_dim, hidden_dim)

        # flat(reshape)がrow-major(index = row*grid_w + col)であることに合わせてrow/col indexを組む
        # （model_transformer.pyの2次元positional encodingと同じ方針）
        row_ids = torch.arange(self.grid_h).repeat_interleave(self.grid_w)  # (grid_h*grid_w,)
        col_ids = torch.arange(self.grid_w).repeat(self.grid_h)              # (grid_h*grid_w,)
        half_dim = hidden_dim // 2
        row_emb = sinusoidal_embedding(row_ids, half_dim)
        col_emb = sinusoidal_embedding(col_ids, half_dim)
        pos_emb = torch.cat([row_emb, col_emb], dim=1)  # (grid_h*grid_w, hidden_dim)
        self.register_buffer("pos_emb", pos_emb)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, _NUM_ATTENTION_HEADS) for _ in range(_NUM_DIT_BLOCKS)]
        )
        self.final_layer = FinalLayer(hidden_dim, patch_dim)

    def _patchify(self, xt: torch.Tensor) -> torch.Tensor:
        """(b,c,h,w) -> (b, grid_h*grid_w, c*p*p)"""
        b, c, h, w = xt.shape
        p = self.patch_size
        x = xt.reshape(b, c, self.grid_h, p, self.grid_w, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (b, grid_h, grid_w, c, p, p)
        return x.reshape(b, self.grid_h * self.grid_w, c * p * p)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """(b, grid_h*grid_w, c*p*p) -> (b,c,h,w)"""
        b = patches.shape[0]
        p = self.patch_size
        x = patches.reshape(b, self.grid_h, self.grid_w, self.c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (b, c, grid_h, p, grid_w, p)
        return x.reshape(b, self.c, self.h, self.w)

    def forward(self, data: InputData) -> torch.Tensor:
        t_emb = sinusoidal_embedding(data.t, self.hidden_dim)
        t_emb = self.time_linear_1(t_emb)
        t_emb = F.silu(t_emb)
        t_emb = self.time_linear_2(t_emb)  # (b, hidden_dim)

        label_emb = self.label_embedding(data.angle.long())  # (b, hidden_dim)

        y = t_emb + label_emb  # (b, hidden_dim) 条件ベクトル

        patches = self._patchify(data.xt)          # (b, N, c*p*p)
        hidden = self.patch_linear(patches)         # (b, N, hidden_dim)
        hidden = hidden + self.pos_emb.unsqueeze(0)  # 2次元sin-cos位置埋め込みを加算

        for block in self.blocks:
            hidden = block(hidden, y)

        out_patches = self.final_layer(hidden, y)   # (b, N, c*p*p)
        return self._unpatchify(out_patches)         # (b, c, h, w)

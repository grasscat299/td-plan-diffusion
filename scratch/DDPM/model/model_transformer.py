"""
design.md の ### Model セクション（transformer）をそのまま実装した denoiseネットワーク。

InputData(xt: (b,c,h,w))
  -> flat        (b, c, h*w)
  -> permute     (b, h*w, c)
  -> linear      (b, h*w, c*16)
  -> t_embを加算 (b, h*w, c*16) + (b, 1, c*16)
  -> angle_embを加算 (b, h*w, c*16) + (b, 1, c*16)
  -> 2次元(row/col)positional encodingを加算 (b, h*w, c*16) + (1, h*w, c*16)
  -> transformer * N   (b, h*w, c*16)
  -> linear(c*16->c*8) -> relu -> linear(c*8->c)  (b, h*w, c)
  -> permute + reshape して (b, c, h, w) に戻す
     （design.mdのClassセクションでforwardの戻り値がTensor(shape(batch,C,H,W))と
       決まっているため、最後にこの形に戻す処理を補っている）

InputData.t(b,)
  -> sinusoidal embedding  (b, c*16)
  -> linear -> silu -> linear(c*16->c*16)  (b, c*16)
  （transformerに入れる前のhiddenに加算するため、hidden_dim(=c*16)に揃えている）

InputData.angle(b,)
  -> sinusoidal embedding  (b, c*16)
  -> linear -> silu -> linear(c*16->c*16)  (b, c*16)
  （t_embと同じ構造・同じ式だが、angle専用の重み(angle_linear_1/2)を持つ）

Nやattention head数はdesign.mdのModel__init__(c, h, w)というシグネチャ上
コンストラクタ引数として渡せないため、モジュール定数として決め打ちにしている。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import InputData
from .embeddings import sinusoidal_embedding

_HIDDEN_CHANNEL_MULTIPLIER = 16  # linearで c -> c*16 に拡大する倍率
_NUM_TRANSFORMER_LAYERS = 4      # design.mdに具体的な数の指定がないため決め打ち
_NUM_ATTENTION_HEADS = 4         # 同上
_FEEDFORWARD_DIM_MULTIPLIER = 4  # TransformerEncoderLayerのdim_feedforward = hidden_dim * この倍率（Transformer論文の慣例に合わせる。指定しないとPyTorchのデフォルト2048が使われ、hidden_dimに対して不釣り合いに大きくなるため）


class TransformerModel(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        super().__init__()
        self.c = c
        self.h = h
        self.w = w

        hidden_dim = c * _HIDDEN_CHANNEL_MULTIPLIER
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(c, hidden_dim)

        self.time_linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.time_linear_2 = nn.Linear(hidden_dim, hidden_dim)

        self.angle_linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.angle_linear_2 = nn.Linear(hidden_dim, hidden_dim)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=_NUM_ATTENTION_HEADS,
            dim_feedforward=hidden_dim * _FEEDFORWARD_DIM_MULTIPLIER,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=_NUM_TRANSFORMER_LAYERS)

        # flat(reshape)がrow-major(index = row*w + col)であることに合わせてrow/col indexを組む
        row_ids = torch.arange(h).repeat_interleave(w)  # (h*w,)
        col_ids = torch.arange(w).repeat(h)              # (h*w,)
        half_dim = hidden_dim // 2
        row_emb = sinusoidal_embedding(row_ids, half_dim)  # (h*w, half_dim)
        col_emb = sinusoidal_embedding(col_ids, half_dim)  # (h*w, half_dim)
        pos_emb = torch.cat([row_emb, col_emb], dim=1)            # (h*w, hidden_dim)
        self.register_buffer("pos_emb", pos_emb)

        self.output_linear_1 = nn.Linear(hidden_dim, c * 8)
        self.output_linear_2 = nn.Linear(c * 8, c)

    def forward(self, data: InputData) -> torch.Tensor:
        batch = data.xt.shape[0]

        t_emb = sinusoidal_embedding(data.t, self.hidden_dim)  # (b, c*16)
        t_emb = self.time_linear_1(t_emb)
        t_emb = F.silu(t_emb)
        t_emb = self.time_linear_2(t_emb)                            # (b, c*16)

        angle_emb = sinusoidal_embedding(data.angle, self.hidden_dim)  # (b, c*16)
        angle_emb = self.angle_linear_1(angle_emb)
        angle_emb = F.silu(angle_emb)
        angle_emb = self.angle_linear_2(angle_emb)                    # (b, c*16)

        flattened = data.xt.reshape(batch, self.c, self.h * self.w)  # (b, c, h*w)
        tokens = flattened.permute(0, 2, 1)                          # (b, h*w, c)

        hidden = self.input_linear(tokens)                           # (b, h*w, c*16)
        hidden = hidden + t_emb.unsqueeze(1)                         # t_embをbroadcast加算
        hidden = hidden + angle_emb.unsqueeze(1)                     # angle_embをbroadcast加算
        hidden = hidden + self.pos_emb.unsqueeze(0)                  # 2次元(row/col)positional encodingをbroadcast加算
        hidden = self.transformer(hidden)                            # (b, h*w, c*16)

        hidden = self.output_linear_1(hidden)                        # (b, h*w, c*8)
        hidden = torch.relu(hidden)
        output_tokens = self.output_linear_2(hidden)                 # (b, h*w, c)

        output_flattened = output_tokens.permute(0, 2, 1)            # (b, c, h*w)
        noise_pred = output_flattened.reshape(batch, self.c, self.h, self.w)  # (b, c, h, w)
        return noise_pred

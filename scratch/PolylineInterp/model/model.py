"""
design.md の「モデル（model/model.py）」に対応するPolylineTransformer。
N個のcheckpointをTransformer Encoderでエンコードし、L_max個の学習可能クエリ（DETR風）を
Transformer Decoderでcross-attentionさせて、等間隔点列を予測する。

出力ヘッドはhead_typeで切り替える。
- "linear": decoder出力からLinear(d_model->2)で座標を直接回帰する（従来通り）。
- "interp": decoder出力からN個のcheckpointへのattention重み(softmax)を予測し、
  出力座標をcheckpoints座標の加重和（凸結合）として明示的に計算する
  （Pointer Networks的な発想）。重みが1つのcheckpointに集中すれば、その座標を
  厳密に再現できるため、頂点(checkpoint)が丸まる問題への対策として導入した。
"""
from __future__ import annotations

import torch
import torch.nn as nn

_HEAD_TYPES = ("linear", "interp")


class PolylineTransformer(nn.Module):
    def __init__(
        self,
        n_checkpoints: int,
        l_max: int,
        d_model: int,
        n_heads: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        dim_feedforward: int,
        head_type: str,
    ) -> None:
        super().__init__()
        if head_type not in _HEAD_TYPES:
            raise ValueError(f"unknown head_type: {head_type!r} (choices: {_HEAD_TYPES})")

        self.n_checkpoints = n_checkpoints
        self.l_max = l_max
        self.head_type = head_type

        self.input_proj = nn.Linear(2, d_model)
        self.input_pos_emb = nn.Parameter(torch.randn(n_checkpoints, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        self.query_emb = nn.Parameter(torch.randn(l_max, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        if head_type == "linear":
            self.output_head = nn.Linear(d_model, 2)
        else:  # "interp"
            self.output_head = nn.Linear(d_model, n_checkpoints)

    def forward(self, checkpoints: torch.Tensor) -> torch.Tensor:
        """checkpoints: (batch, n_checkpoints, 2) -> pred: (batch, l_max, 2)"""
        batch = checkpoints.shape[0]

        x = self.input_proj(checkpoints) + self.input_pos_emb.unsqueeze(0)
        memory = self.encoder(x)  # (batch, n_checkpoints, d_model). Nは固定長なのでpadding maskは不要

        q = self.query_emb.unsqueeze(0).expand(batch, -1, -1)  # (batch, l_max, d_model)
        h = self.decoder(tgt=q, memory=memory)  # (batch, l_max, d_model)

        if self.head_type == "linear":
            return self.output_head(h)  # (batch, l_max, 2)

        attn_logits = self.output_head(h)  # (batch, l_max, n_checkpoints)
        weights = attn_logits.softmax(dim=-1)  # (batch, l_max, n_checkpoints). 各行の和が1
        return torch.bmm(weights, checkpoints)  # (batch, l_max, 2)

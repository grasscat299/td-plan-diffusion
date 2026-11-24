"""
design.md の ### Model セクション（Unet）をそのまま実装した denoiseネットワーク。

design.mdの表記との対応:
- c_img: 画像のchannel数（__init__の引数 c）
- c: Unet内部の基準channel数（design.mdのUnetセクションで使われる"c"はこちら。
  __init__の引数cとは別物で、design.mdに具体的な値の指定がないため
  _BASE_CHANNEL_MULTIPLIER倍として決め打ちにしている）

emb(t), emb(angle) (shape (b, c*4)):
  PE(t) -> linear(c*4->c*8) -> swish -> linear(c*8->c*4)  : emb(t)
  PE(angle)についても同じ構造・別重みでemb(angle)を計算する。
  Unet全体で1回だけ計算し、全resnet系ブロックに共通で渡す。

Resnet/encode_Resnet/decode_Resnetの各ブロックは、1つ目のconv後・2つ目のGroupNorm+Reluの前で、
emb(t)・emb(angle)をそれぞれそのブロックのchannel数へlinearで射影してから加算する
（射影用のlinearはブロックごとに個別の重みを持つ）。
encode_conv（ダウンサンプリング）・引き伸ばし（アップサンプリング）のjoint部分には
embは注入しない。

全てのconvの直前にGroupNorm->Relu(pre-norm)を挟む。残差結合の"+"自体にはnormをかけず、
入力をそのまま加算する（恒等写像の勾配経路を保つため）。GroupNormが無いと、残差加算を
繰り返すごとに活性値が指数的に増大し、学習中に勾配爆発を起こす（実際にnorm無しで
学習させたところ数epochで発散した）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import InputData
from .embeddings import sinusoidal_embedding

_BASE_CHANNEL_MULTIPLIER = 32  # design.mdに具体的な値の指定がないため決め打ち（内部基準channel = c_img * この倍率）
_EMB_HIDDEN_MULTIPLIER = 2     # emb MLPの中間層はemb_dim(=base*4)のこの倍率まで拡張する


class _EmbeddingMLP(nn.Module):
    """emb(t) = W2*swish(W1*PE(t)+b1)+b2 の2層MLP。emb(angle)にも同じクラスを使う（重みは別インスタンス）。"""

    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(emb_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear_1(x)
        h = F.silu(h)
        return self.linear_2(h)


class ResnetBlock(nn.Module):
    """design.mdのResnet: (b,h,w,c)->GroupNorm->Relu->conv->+emb->GroupNorm->Relu->conv->+残差"""

    def __init__(self, channels: int, emb_dim: int, num_groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.linear_t = nn.Linear(emb_dim, channels)
        self.linear_angle = nn.Linear(emb_dim, channels)

    def forward(self, x: torch.Tensor, emb_t: torch.Tensor, emb_angle: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = torch.relu(h)
        h = self.conv1(h)
        h = h + self.linear_t(emb_t)[:, :, None, None] + self.linear_angle(emb_angle)[:, :, None, None]
        h = self.norm2(h)
        h = torch.relu(h)
        h = self.conv2(h)
        return h + x


class ResampleResnetBlock(nn.Module):
    """
    design.mdのencode_Resnet(c->2c)・decode_Resnet(3c->c)に共通の構造:
    (b,h,w,in)->GroupNorm->Relu->conv(in->out)->+emb->GroupNorm->Relu->conv(out->out)
    ->+残差(conv(in->out,k=1)で調整、normは通さない)
    """

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, num_groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.linear_t = nn.Linear(emb_dim, out_channels)
        self.linear_angle = nn.Linear(emb_dim, out_channels)

    def forward(self, x: torch.Tensor, emb_t: torch.Tensor, emb_angle: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = torch.relu(h)
        h = self.conv1(h)
        h = h + self.linear_t(emb_t)[:, :, None, None] + self.linear_angle(emb_angle)[:, :, None, None]
        h = self.norm2(h)
        h = torch.relu(h)
        h = self.conv2(h)
        return h + self.residual_conv(x)


class UnetModel(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        super().__init__()
        self.c_img = c
        self.h = h
        self.w = w

        base = c * _BASE_CHANNEL_MULTIPLIER
        self.base = base
        emb_dim = base * 4
        self.emb_dim = emb_dim

        self.time_mlp = _EmbeddingMLP(emb_dim, emb_dim * _EMB_HIDDEN_MULTIPLIER)
        self.angle_mlp = _EmbeddingMLP(emb_dim, emb_dim * _EMB_HIDDEN_MULTIPLIER)

        self.in_conv = nn.Conv2d(c, base, kernel_size=3, padding=1)

        # GroupNormのgroup数は基準channel数base(=design.mdの"c")に揃える。
        # 各ブロックのchannel数は必ずbaseの倍数(c,2c,3c,4c,6c)になるため、これで常に割り切れる。
        num_groups = base

        # layer1
        self.layer1_resnet1 = ResnetBlock(base, emb_dim, num_groups)
        self.layer1_resnet2 = ResnetBlock(base, emb_dim, num_groups)
        self.down1 = nn.Conv2d(base, base, kernel_size=3, stride=2, padding=1)  # encode_conv

        # layer2
        self.layer2_encode_resnet = ResampleResnetBlock(base, base * 2, emb_dim, num_groups)
        self.layer2_resnet = ResnetBlock(base * 2, emb_dim, num_groups)
        self.down2 = nn.Conv2d(base * 2, base * 2, kernel_size=3, stride=2, padding=1)  # encode_conv

        # layer3
        self.layer3_encode_resnet = ResampleResnetBlock(base * 2, base * 4, emb_dim, num_groups)
        self.layer3_resnet = ResnetBlock(base * 4, emb_dim, num_groups)

        # layer2 (decode)
        self.up1_conv = nn.Conv2d(base * 4, base * 4, kernel_size=3, padding=1)
        self.layer2_decode_resnet = ResampleResnetBlock(base * 6, base * 2, emb_dim, num_groups)
        self.layer2_decode_resnet2 = ResnetBlock(base * 2, emb_dim, num_groups)

        # layer1 (decode)
        self.up2_conv = nn.Conv2d(base * 2, base * 2, kernel_size=3, padding=1)
        self.layer1_decode_resnet = ResampleResnetBlock(base * 3, base, emb_dim, num_groups)
        self.layer1_decode_resnet2 = ResnetBlock(base, emb_dim, num_groups)

        self.out_conv = nn.Conv2d(base, c, kernel_size=3, padding=1)

    def forward(self, data: InputData) -> torch.Tensor:
        emb_t = self.time_mlp(sinusoidal_embedding(data.t, self.emb_dim))
        emb_angle = self.angle_mlp(sinusoidal_embedding(data.angle, self.emb_dim))

        h = self.in_conv(data.xt)

        h = self.layer1_resnet1(h, emb_t, emb_angle)
        skip1 = self.layer1_resnet2(h, emb_t, emb_angle)
        h = self.down1(skip1)

        h = self.layer2_encode_resnet(h, emb_t, emb_angle)
        skip2 = self.layer2_resnet(h, emb_t, emb_angle)
        h = self.down2(skip2)

        h = self.layer3_encode_resnet(h, emb_t, emb_angle)
        h = self.layer3_resnet(h, emb_t, emb_angle)

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up1_conv(h)
        h = torch.cat([h, skip2], dim=1)
        h = self.layer2_decode_resnet(h, emb_t, emb_angle)
        h = self.layer2_decode_resnet2(h, emb_t, emb_angle)

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up2_conv(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.layer1_decode_resnet(h, emb_t, emb_angle)
        h = self.layer1_decode_resnet2(h, emb_t, emb_angle)

        return self.out_conv(h)

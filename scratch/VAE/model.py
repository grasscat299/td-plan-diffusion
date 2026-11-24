"""
design.md の class セクション（EncoderOutput/VAEOutput/Encoder/Decoder/VAE）、
function セクションの Reparameterize、Model セクション（KL-VAE系アーキテクチャ）の実装。

design.mdの表記との対応:
- base: Encoder/Decoder内部の基準channel数（design.mdの"base = c * 16"）
- AttentionBlockは design.md の決定事項に従い、構造だけ実装してEncoder/Decoderの
  forwardからは呼ばない（コメントアウトして無効化する）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

_BASE_CHANNEL_MULTIPLIER = 16  # design.mdに明記の値（base = c * 16）


@dataclass
class EncoderOutput:
    mean: torch.Tensor    # (batch, c, h, w)
    logvar: torch.Tensor  # (batch, c, h, w) — variance そのものではなく log(variance)


@dataclass
class VAEOutput:
    x_hat: torch.Tensor   # (batch, c, 4h, 4w)
    mean: torch.Tensor    # (batch, c, h, w)
    logvar: torch.Tensor  # (batch, c, h, w)


def Reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mean)
    return mean + std * eps


class ResBlock(nn.Module):
    """
    design.mdのResBlock: GroupNorm->SiLU->conv->GroupNorm->SiLU->conv->+residual。
    DDPM UnetのResnetBlockと異なりemb injectionは無い（t/angleの条件付けが無いため）。
    channel数が変わる場合は残差側にconv(in->out, k=1, s=1)を挟む（normは通さない）。
    """

    def __init__(self, in_channels: int, out_channels: int, num_groups: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """
    design.mdのAttentionBlock: 最小解像度(h,w)に1つだけ挿入する想定の1ヘッドSelf-Attention。
    design.mdの決定事項により現時点ではEncoder/Decoderのforwardから呼ばれておらず無効化されている
    （将来有効化する場合に備えて構造だけ残す）。
    """

    def __init__(self, channels: int, num_groups: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x)
        h_ = h_.flatten(2).permute(0, 2, 1)  # (b, h*w, c)
        q, k, v = self.to_q(h_), self.to_k(h_), self.to_v(h_)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        h_ = attn @ v
        h_ = h_.permute(0, 2, 1).reshape(b, c, h, w)
        return x + h_


class Encoder(nn.Module):
    """design.mdの Model > Encoder をそのまま実装した圧縮ネットワーク。"""

    def __init__(self, c: int, h: int, w: int):
        super().__init__()
        self.c, self.h, self.w = c, h, w

        base = c * _BASE_CHANNEL_MULTIPLIER
        num_groups = base

        self.conv_in = nn.Conv2d(c, base, kernel_size=3, padding=1)

        # level1 (4h,4w)
        self.level1_resnet1 = ResBlock(base, base, num_groups)
        self.level1_resnet2 = ResBlock(base, base, num_groups)
        self.down1 = nn.Conv2d(base, base, kernel_size=3, stride=2, padding=1)  # downsample

        # level2 (2h,2w)
        self.level2_resnet1 = ResBlock(base, base * 2, num_groups)
        self.level2_resnet2 = ResBlock(base * 2, base * 2, num_groups)
        self.down2 = nn.Conv2d(base * 2, base * 2, kernel_size=3, stride=2, padding=1)  # downsample

        # level3 (h,w) 最小解像度
        self.level3_resnet1 = ResBlock(base * 2, base * 4, num_groups)
        self.level3_resnet2 = ResBlock(base * 4, base * 4, num_groups)
        # self.attention = AttentionBlock(base * 4, num_groups)  # design.mdの決定により無効化
        self.level3_resnet3 = ResBlock(base * 4, base * 4, num_groups)

        self.norm_out = nn.GroupNorm(num_groups, base * 4)
        self.conv_out = nn.Conv2d(base * 4, c * 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        h = self.conv_in(x)

        h = self.level1_resnet1(h)
        h = self.level1_resnet2(h)
        h = self.down1(h)

        h = self.level2_resnet1(h)
        h = self.level2_resnet2(h)
        h = self.down2(h)

        h = self.level3_resnet1(h)
        h = self.level3_resnet2(h)
        # h = self.attention(h)  # design.mdの決定により無効化
        h = self.level3_resnet3(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        mean, logvar = h.chunk(2, dim=1)
        return EncoderOutput(mean=mean, logvar=logvar)


class Decoder(nn.Module):
    """design.mdの Model > Decoder をそのまま実装した復元ネットワーク（Encoderの対称形）。"""

    def __init__(self, c: int, h: int, w: int):
        super().__init__()
        self.c, self.h, self.w = c, h, w

        base = c * _BASE_CHANNEL_MULTIPLIER
        num_groups = base

        self.conv_in = nn.Conv2d(c, base * 4, kernel_size=3, padding=1)

        # level3 (h,w) 最小解像度
        self.level3_resnet1 = ResBlock(base * 4, base * 4, num_groups)
        # self.attention = AttentionBlock(base * 4, num_groups)  # design.mdの決定により無効化
        self.level3_resnet2 = ResBlock(base * 4, base * 4, num_groups)
        self.up1_conv = nn.Conv2d(base * 4, base * 4, kernel_size=3, padding=1)  # upsample後のconv

        # level2 (2h,2w)
        self.level2_resnet1 = ResBlock(base * 4, base * 2, num_groups)
        self.level2_resnet2 = ResBlock(base * 2, base * 2, num_groups)
        self.up2_conv = nn.Conv2d(base * 2, base * 2, kernel_size=3, padding=1)  # upsample後のconv

        # level1 (4h,4w)
        self.level1_resnet1 = ResBlock(base * 2, base, num_groups)
        self.level1_resnet2 = ResBlock(base, base, num_groups)

        self.norm_out = nn.GroupNorm(num_groups, base)
        self.conv_out = nn.Conv2d(base, c, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)

        h = self.level3_resnet1(h)
        # h = self.attention(h)  # design.mdの決定により無効化
        h = self.level3_resnet2(h)

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up1_conv(h)

        h = self.level2_resnet1(h)
        h = self.level2_resnet2(h)

        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.up2_conv(h)

        h = self.level1_resnet1(h)
        h = self.level1_resnet2(h)

        h = self.norm_out(h)
        h = F.silu(h)
        return self.conv_out(h)


class VAE(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        super().__init__()
        self.encoder = Encoder(c, h, w)
        self.decoder = Decoder(c, h, w)

    def forward(self, x: torch.Tensor) -> VAEOutput:
        encoder_output = self.encoder(x)
        z = Reparameterize(encoder_output.mean, encoder_output.logvar)
        x_hat = self.decoder(z)
        return VAEOutput(x_hat=x_hat, mean=encoder_output.mean, logvar=encoder_output.logvar)

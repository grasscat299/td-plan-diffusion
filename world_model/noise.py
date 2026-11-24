"""拡散のノイズのスケジュールと、置ける升の扱い。拡散するのは計画（配置タワー）だけ。"""
from __future__ import annotations

import torch

from .config import DiffusionConfig

RANGED_IDX = 7                  # キャラクターの属性のうち、高台にしか置けない種類なら 1 の列
TILE_GROUND, TILE_HIGH = 1, 2   # 地形の値: 地上・高台


def zero_terminal_a_bar(a_bar: torch.Tensor) -> torch.Tensor:
    """ᾱ の最後を 0 にする（Lin et al. 2023, "Common Diffusion Noise Schedules and Sample Steps are Flawed"）。
    √ᾱ を、最初の値は変えずに、最後が 0 になるように平行移動して拡大する。最大のノイズの学習の入力が純粋なノイズになり、推論の出発点と一致する。"""
    sq = a_bar.double().sqrt()
    s0, sT = sq[0].clone(), sq[-1].clone()
    sq = (sq - sT) * (s0 / (s0 - sT))
    return (sq ** 2).to(a_bar.dtype)


class NoiseScheduler:
    def __init__(self, cfg: DiffusionConfig):
        self.num_timesteps = cfg.num_timesteps
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_timesteps)
        a_bar = torch.cumprod(1.0 - betas, dim=0)
        self.a_bar = zero_terminal_a_bar(a_bar) if cfg.zero_terminal_snr else a_bar

    def to(self, device) -> "NoiseScheduler":
        self.a_bar = self.a_bar.to(device)
        return self

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """x0 (B, N_op, c_op)・t (B,)（1 始まり）-> x_t。"""
        ab = self.a_bar[t - 1].view(-1, 1, 1)
        noise = torch.randn_like(x0) if noise is None else noise
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise


def legal_tile_mask(tile_map: torch.Tensor, ope_attr: torch.Tensor) -> torch.Tensor:
    """tile_map (B, R, C)・ope_attr (B, N_op, A)（正規化前）-> 置ける升 (B, N_op, R*C) bool。
    高台専用の枠は高台、それ以外は地上だけ。置ける升が 1 つも無い枠は、全部の升を置ける扱いにする（softmax が NaN にならないため）。"""
    B = tile_map.shape[0]
    tm = tile_map.reshape(B, 1, -1)
    ranged = (ope_attr[..., RANGED_IDX] > 0.5).unsqueeze(-1)
    legal = torch.where(ranged, tm == TILE_HIGH, tm == TILE_GROUND)
    return legal | ~legal.any(dim=-1, keepdim=True)


def mask_logits(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """置けない升の logit を最小値にする（softmax・argmax が置ける升の中だけになる）。"""
    return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)


def pin_illegal(x_tile: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """拡散の入力の升の部分（one-hot を ±1 にした値）で、置けない升を −1 に固定する（ノイズを足さない）。"""
    return torch.where(legal, x_tile, torch.full_like(x_tile, -1.0))

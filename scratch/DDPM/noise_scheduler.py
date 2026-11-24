"""
design.md の NoiseScheduler。βスケジュールから ᾱ_t を事前計算し、
closed-formで forward diffusion (ノイズ化) を1発で行う。
"""
from __future__ import annotations

import torch

from .data import InputData, RandomPickedData


class NoiseScheduler:
    def __init__(self, num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        self.a_t_bar = torch.cumprod(alphas, dim=0)  # (T,)

    def to(self, device: torch.device) -> "NoiseScheduler":
        self.a_t_bar = self.a_t_bar.to(device)
        return self

    def add_noise(self, data: RandomPickedData) -> InputData:
        a_bar_t = self.a_t_bar[data.t - 1].view(-1, 1, 1, 1)  # tは1-indexedなので-1
        eps = torch.randn_like(data.x0)
        xt = a_bar_t.sqrt() * data.x0 + (1.0 - a_bar_t).sqrt() * eps
        return InputData(xt=xt, t=data.t, eps=eps, angle=data.angle)

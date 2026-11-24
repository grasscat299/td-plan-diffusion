"""
config.tomlの読み込みと、checkpoint同梱configの復元を同じ経路（config_from_dict）で行う。
model-dev-convention準拠。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class VaeConfig:
    load_path: str
    image_channels: int
    latent_height: int  # model_vae_128.ptを学習した時点の値と一致させる
    latent_width: int


@dataclass
class ModelConfig:
    type: str  # build_model()に渡すモデル名。"transformer" or "unet" or "dit"


@dataclass
class DiffusionConfig:
    num_timesteps: int


@dataclass
class TrainConfig:
    lr: float
    epoch: int
    steps_per_epoch: int
    batch: int


@dataclass
class WandbConfig:
    project: str
    group: str


@dataclass
class Config:
    vae: VaeConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    train: TrainConfig
    wandb: WandbConfig


def config_from_dict(raw: dict) -> Config:
    return Config(
        vae=VaeConfig(**raw["vae"]),
        model=ModelConfig(**raw["model"]),
        diffusion=DiffusionConfig(**raw["diffusion"]),
        train=TrainConfig(**raw["train"]),
        wandb=WandbConfig(**raw["wandb"]),
    )


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return config_from_dict(raw)

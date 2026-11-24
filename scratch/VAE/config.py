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
class ModelConfig:
    image_channels: int
    latent_height: int  # 入力画像は (4*latent_height, 4*latent_width)
    latent_width: int


@dataclass
class TrainConfig:
    lr: float
    epoch: int
    steps_per_epoch: int
    batch: int
    kl_weight: float


@dataclass
class WandbConfig:
    project: str
    group: str


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig
    wandb: WandbConfig


def config_from_dict(raw: dict) -> Config:
    return Config(
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        wandb=WandbConfig(**raw["wandb"]),
    )


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return config_from_dict(raw)

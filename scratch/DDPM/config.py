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
    type: str  # denoiseネットワークの種類。"transformer" / "unet" / "dit"（build_model()に渡す）
    image_channels: int
    image_height: int
    image_width: int


@dataclass
class DiffusionConfig:
    num_timesteps: int  # 拡散過程のタイムステップ数T


@dataclass
class TrainConfig:
    lr: float
    epoch: int
    steps_per_epoch: int  # 1epochあたりBuildSwissRollで生成するバッチ数
    batch: int


@dataclass
class WandbConfig:
    project: str
    group: str


@dataclass
class Config:
    model: ModelConfig
    diffusion: DiffusionConfig
    train: TrainConfig
    wandb: WandbConfig


def config_from_dict(raw: dict) -> Config:
    return Config(
        model=ModelConfig(**raw["model"]),
        diffusion=DiffusionConfig(**raw["diffusion"]),
        train=TrainConfig(**raw["train"]),
        wandb=WandbConfig(**raw["wandb"]),
    )


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return config_from_dict(raw)

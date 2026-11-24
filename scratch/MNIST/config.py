"""
config.tomlの読み込みと、checkpoint同梱configの復元を同じ経路（config_from_dict）で行う。
model-dev-convention準拠。

[experiment] kind で3系統の実験（latent_dit / pixel_transformer / vae）を切り替える
（scratch/MNIST/design.md参照）。
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
class ExperimentConfig:
    kind: str  # "latent_dit" | "pixel_transformer" | "vae"


@dataclass
class VaeConfig:
    image_channels: int
    latent_height: int  # kind="vae"では入力画像は(4*latent_height, 4*latent_width)
    latent_width: int
    kl_weight: float     # kind="vae"の学習でのみ使う
    load_path: str        # kind="latent_dit"で読み込む凍結VAEのcheckpoint


@dataclass
class LatentDitConfig:
    dit_variant: str  # "local_dit" | "model_dit"
    loss_type: str    # "mse" | "l1"


@dataclass
class PixelTransformerConfig:
    model_type: str
    image_channels: int
    image_height: int
    image_width: int


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
    experiment: ExperimentConfig
    vae: VaeConfig
    latent_dit: LatentDitConfig
    pixel_transformer: PixelTransformerConfig
    diffusion: DiffusionConfig
    train: TrainConfig
    wandb: WandbConfig


def config_from_dict(raw: dict) -> Config:
    return Config(
        experiment=ExperimentConfig(**raw["experiment"]),
        vae=VaeConfig(**raw["vae"]),
        latent_dit=LatentDitConfig(**raw["latent_dit"]),
        pixel_transformer=PixelTransformerConfig(**raw["pixel_transformer"]),
        diffusion=DiffusionConfig(**raw["diffusion"]),
        train=TrainConfig(**raw["train"]),
        wandb=WandbConfig(**raw["wandb"]),
    )


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return config_from_dict(raw)

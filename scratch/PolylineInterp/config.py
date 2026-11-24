"""
config.tomlの読み込みと、checkpoint同梱configの復元を同じ経路（config_from_dict）で行う。
model-dev-convention準拠。

l_maxはBuildDataset実行後にしか確定しない値のため、config.tomlには書かず、
main.pyでデータ生成後にModelConfig.l_maxへセットしてから使う（詳細はdesign.mdの
「l_maxの循環依存」に関するやり取りを参照）。
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
class DataConfig:
    n_checkpoints: int
    closed: bool
    coord_min: float
    coord_max: float
    step: float
    min_triangle_area: float
    num_train_samples: int
    num_val_samples: int
    seed: int


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    dim_feedforward: int
    head_type: str  # "linear"（座標を直接回帰） or "interp"（checkpoint座標への補間）
    l_max: int | None = None  # config.tomlには書かない。BuildDataset実行後に確定する。


@dataclass
class TrainConfig:
    lr: float
    epoch: int
    batch: int


@dataclass
class WandbConfig:
    project: str
    group: str


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    wandb: WandbConfig


def config_from_dict(raw: dict) -> Config:
    return Config(
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        wandb=WandbConfig(**raw["wandb"]),
    )


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return config_from_dict(raw)

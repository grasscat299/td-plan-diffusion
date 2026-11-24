"""
複数のdenoiseネットワーク実装（TransformerModel / UnetModel / DiTModel）を名前で切り替えるためのfactory。
TrainSchedule/OneStep側はforward(data: InputData) -> Tensorのインターフェースしか
使わないため、build_model()が返すモジュールを差し替えるだけでアーキテクチャを切り替えられる。
"""
from __future__ import annotations

import torch.nn as nn

from .model_dit import DiTModel
from .model_transformer import TransformerModel
from .model_unet import UnetModel

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "transformer": TransformerModel,
    "unet": UnetModel,
    "dit": DiTModel,
}


def build_model(name: str, c: int, h: int, w: int) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model name: {name!r} (choices: {sorted(MODEL_REGISTRY)})")
    return MODEL_REGISTRY[name](c=c, h=h, w=w)

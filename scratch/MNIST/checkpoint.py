"""
model-dev-conventionのcheckpoint規約: 学習済み重みと、その学習に使ったconfigスナップショットを
同じファイルに同梱する。推論側は--ckptを差し替えるだけでモデル構造を復元できる。

WebUI対応前に保存した checkpoint（torch.save(net.state_dict()) の生の state_dict）も
そのまま読めるようにしている（その場合 config=None を返す）。
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from .config import Config, config_from_dict

_FORMAT = "MNIST/ckpt-v1"


def save_checkpoint(path: str | Path, net: torch.nn.Module, cfg: Config) -> None:
    torch.save(
        {"format": _FORMAT, "state_dict": net.state_dict(), "config": dataclasses.asdict(cfg)},
        path,
    )


def load_checkpoint(path: str | Path, map_location=None) -> tuple[dict, Config | None]:
    """
    同梱configが無い旧形式（生のstate_dict）はconfig=Noneで返す（後方互換）。
    """
    payload = torch.load(path, map_location=map_location)
    if "format" not in payload:
        return payload, None
    raw_config = payload.get("config")
    cfg = config_from_dict(raw_config) if raw_config is not None else None
    return payload["state_dict"], cfg

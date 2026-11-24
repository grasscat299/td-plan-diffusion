"""
model-dev-conventionのcheckpoint規約: 学習済み重みと、その学習に使ったconfigスナップショットを
同じファイルに同梱する。推論側は--ckptを差し替えるだけでモデル構造(l_max含む)を復元できる。
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from .config import Config, config_from_dict

_FORMAT = "PolylineInterp/ckpt-v1"


def save_checkpoint(path: str | Path, net: torch.nn.Module, cfg: Config) -> None:
    torch.save(
        {"format": _FORMAT, "state_dict": net.state_dict(), "config": dataclasses.asdict(cfg)},
        path,
    )


def load_checkpoint(path: str | Path, map_location=None) -> tuple[dict, Config | None]:
    """
    同梱configが無い/壊れている旧形式はconfig=Noneで返す（後方互換）。
    """
    payload = torch.load(path, map_location=map_location)
    state_dict = payload["state_dict"]
    raw_config = payload.get("config")
    cfg = config_from_dict(raw_config) if raw_config is not None else None
    return state_dict, cfg

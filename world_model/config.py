"""モデルの設定。`config.example.toml` を読み、型付きの値にする。"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

# 配置の枠のトークンの並び: [出す時刻, 退く時刻, 枠を使うか, 向き 4, 升 n_rows*n_cols, スキル]
PLAN_TD, PLAN_TR, PLAN_USE, PLAN_DIR0, PLAN_TILE0 = 0, 1, 2, 3, 7


@dataclass(frozen=True)
class ShapeConfig:
    T: int                 # 時系列の長さ（秒）
    n_en: int              # 敵の数の上限
    n_op: int              # 配置の枠の数
    en_dim: int            # 敵の時系列の 1 秒あたりの次元（存在・x・y・HP）
    c_op: int              # 配置の枠のトークンの次元
    n_rows: int            # 盤面の行数
    n_cols: int            # 盤面の列数
    n_tile_types: int      # 地形の種類の数
    max_route_points: int  # 通り道の点の数の上限
    route_dim: int         # 通り道の 1 点の次元（x・y・その点での待機時間）
    enemy_attr_dim: int    # 敵の属性の次元
    ope_attr_dim: int      # キャラクターの属性の次元
    tile_feat_dim: int     # 升ごとの特徴の次元
    cov_channel: int       # 升ごとの特徴のうち、敵が通る延べ時間（log1p）の列

    @property
    def n_tiles(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def n_skill_cols(self) -> int:
        return self.c_op - PLAN_TILE0 - self.n_tiles

    @property
    def n_resource(self) -> int:
        """資源タワーの出力の次元: 配置コスト 1 + 枠ごとの HP・SP・スキルの発動回数。"""
        return 1 + 3 * self.n_op


@dataclass(frozen=True)
class ModelConfig:
    d: int
    num_layers: int
    num_heads: int
    ffn_mult: int
    cond_mlp_mult: int
    route_num_layers: int
    route_num_heads: int
    t_c: int               # 敵・資源の時系列を配置タワーに渡す時の、時間方向の長さ
    op_dropout: float = 0.0


@dataclass(frozen=True)
class DiffusionConfig:
    num_timesteps: int
    beta_start: float
    beta_end: float
    zero_terminal_snr: bool = True


@dataclass(frozen=True)
class Config:
    shape: ShapeConfig
    model: ModelConfig
    diffusion: DiffusionConfig


def _make(cls, raw: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"{cls.__name__} に無い項目: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Config(
        shape=_make(ShapeConfig, raw["shape"]),
        model=_make(ModelConfig, raw["model"]),
        diffusion=_make(DiffusionConfig, raw["diffusion"]),
    )

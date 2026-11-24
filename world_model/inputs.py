"""モデルの入力と出力の型。B = バッチ、N_en = 敵の数、N_op = 配置の枠の数、T = 秒、R・C = 盤面の行・列。"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelInput:
    plan_xt: torch.Tensor            # (B, N_op, c_op)  ノイズの入った計画
    t: torch.Tensor                  # (B,) long        拡散のステップ（1..num_timesteps）
    route: torch.Tensor              # (B, N_en, max_route_points, route_dim)  敵ごとの通り道（前詰め・残りは 0）
    route_padding_mask: torch.Tensor # (B, N_en, max_route_points) bool       True = 通り道の詰め物
    known_en_x0: torch.Tensor        # (B, N_en, T, en_dim)      出現の時点で分かっている値（位置・HP）
    known_en_mask: torch.Tensor      # (B, N_en, T) bool         True = known_en_x0 を使う秒
    enemy_attr: torch.Tensor         # (B, N_en, enemy_attr_dim) 敵の属性。全部 0 の行 = 敵がいない行
    tile_map: torch.Tensor           # (B, R, C) long            地形
    tile_feat: torch.Tensor          # (B, R*C, tile_feat_dim)   升ごとの特徴
    avail: torch.Tensor              # (B, N_op) float           枠を使えるか（0/1）
    ope_attr: torch.Tensor           # (B, N_op, ope_attr_dim)   枠のキャラクターの属性（正規化前の値）
    ope_range: torch.Tensor          # (B, N_op, K, 2) long      射程の相対位置 (dc, dr)。右向きが基準。詰め物は盤面の外の大きな値
    ope_valid: torch.Tensor          # (B, N_op) bool            編成にある枠


@dataclass
class ModelOutput:
    plan: torch.Tensor       # (B, N_op, c_op)          計画の x0 の予測（升の部分は logits）
    enemy: torch.Tensor      # (B, N_en, T, en_dim)     敵の時系列の予測（存在は logit）
    resource: torch.Tensor   # (B, T, 1 + 3*N_op)       配置コストと、枠ごとの HP・SP・スキルの発動回数の予測
    success: torch.Tensor    # (B,)                     全部の敵を倒せるかの logit

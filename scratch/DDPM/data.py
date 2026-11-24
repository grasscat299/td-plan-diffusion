"""
design.md の「ノイズ化部分」に対応するデータクラス群と、
トイデータセット（2次元swiss roll）の生成関数。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# swiss roll の theta = 1.5π*(1+2r) (r∈[0,1]) は最大でも 4.5π 程度なので、
# |theta*cos(theta)|, |theta*sin(theta)| を [-1, 1] に収める安全なスケール。
_SWISS_ROLL_SCALE = 15.0

# 1枚の画像の中に、渦の形が見えるだけの点数を描画する。
_NUMBER_OF_SWISS_ROLL_POINTS_PER_IMAGE = 2000


@dataclass
class RandomPickedData:
    """ノイズ化前のデータ。x0からtをサンプリングして保持する。"""
    x0: torch.Tensor            # (batch, C, H, W)
    num_timesteps: int          # T
    angle: torch.Tensor         # (batch,) — BuildSwissRollが生成した回転角度をそのまま渡す
    t: torch.Tensor = field(init=False)  # (batch,) — 1 〜 num_timesteps の一様分布からサンプリング

    def __post_init__(self) -> None:
        batch = self.x0.shape[0]
        self.t = torch.randint(1, self.num_timesteps + 1, (batch,), device=self.x0.device)


@dataclass
class InputData:
    """ノイズ化後のデータ。NoiseScheduler.add_noise() の出力。"""
    xt: torch.Tensor    # (batch, C, H, W)
    t: torch.Tensor     # (batch,)
    eps: torch.Tensor   # (batch, C, H, W)
    angle: torch.Tensor  # (batch,)


def _sample_swiss_roll_points(n: int, angle: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    2次元swiss roll上の点をn個サンプリングする。戻り値は(points, angle_used)。
    points: shape (n, 2)。
    開始点の角度（渦全体の向き）はangleで指定でき、指定しない場合は呼び出しごとに
    [0, 2π)から1回だけランダムに決めて、全n点に同じ回転をかける
    （点ごとに別々の角度にすると渦の形が崩れるため）。
    """
    r = torch.rand(n)
    theta = 1.5 * math.pi * (1.0 + 2.0 * r)
    x = theta * torch.cos(theta)
    y = theta * torch.sin(theta)

    angle_used = angle if angle is not None else torch.rand(()) * 2.0 * math.pi
    cos_offset = torch.cos(angle_used)
    sin_offset = torch.sin(angle_used)
    rotated_x = x * cos_offset - y * sin_offset
    rotated_y = x * sin_offset + y * cos_offset

    points = torch.stack([rotated_x, rotated_y], dim=1)
    return points, angle_used


def _rasterize_points_onto_image(points: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    複数の点をまとめて1枚の二値画像に描画する。戻り値 shape (h, w)。
    拡散モデルの慣習に合わせて、値の範囲は[-1, 1]にする（背景=-1, 点=1）。
    """
    normalized = points / _SWISS_ROLL_SCALE  # [-1, 1] へ正規化
    rows = ((1.0 - (normalized[:, 1] + 1.0) / 2.0) * (h - 1)).round().long().clamp(0, h - 1)
    cols = (((normalized[:, 0] + 1.0) / 2.0) * (w - 1)).round().long().clamp(0, w - 1)
    image = torch.full((h, w), -1.0)
    image[rows, cols] = 1.0
    return image


def BuildSwissRoll(batch: int, c: int, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    swiss roll全体の渦が見える画像をbatch枚生成する。
    画像ごとに点を独立にサンプリングし直すので、渦の向き（開始角度）も画像ごとに変わる。
    戻り値は(x0, angle)。x0のshapeは(batch, c, h, w)、angleのshapeは(batch,)。
    """
    images = torch.zeros(batch, h, w)
    angles = torch.zeros(batch)
    for i in range(batch):
        points, angle = _sample_swiss_roll_points(_NUMBER_OF_SWISS_ROLL_POINTS_PER_IMAGE)
        images[i] = _rasterize_points_onto_image(points, h, w)
        angles[i] = angle
    return images.unsqueeze(1).expand(batch, c, h, w).clone(), angles


def BuildSwissRollAtAngles(batch: int, c: int, h: int, w: int, angles: torch.Tensor) -> torch.Tensor:
    """
    指定したangles(shape (batch,))ごとに、その回転角度のswiss roll画像を1枚ずつ生成する。
    主にinferでの正解画像（ground truth）表示など、可視化用途に使う。
    戻り値のshapeは(batch, c, h, w)。
    """
    images = torch.zeros(batch, h, w)
    for i in range(batch):
        points, _ = _sample_swiss_roll_points(_NUMBER_OF_SWISS_ROLL_POINTS_PER_IMAGE, angle=angles[i])
        images[i] = _rasterize_points_onto_image(points, h, w)
    return images.unsqueeze(1).expand(batch, c, h, w).clone()

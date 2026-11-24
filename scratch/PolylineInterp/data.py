"""
design.md の「データ生成（data.py）」に対応するデータクラス群と、
三角形（N=3）toyデータの生成関数。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Sample:
    checkpoints: torch.Tensor  # (N, 2)
    targets: torch.Tensor      # (L_max, 2)
    mask: torch.Tensor         # (L_max,) bool


def _triangle_area(checkpoints: torch.Tensor) -> float:
    """3点(shape(3,2))が作る三角形の面積を返す（shoelace公式）。"""
    p0, p1, p2 = checkpoints[0], checkpoints[1], checkpoints[2]
    cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
    return abs(cross.item()) / 2.0


def GenerateTriangleCheckpoints(coord_min: float, coord_max: float, min_triangle_area: float) -> torch.Tensor:
    """
    [coord_min, coord_max]^2 の一様分布から3点をランダムに生成する。
    面積がmin_triangle_area未満の退化三角形（ほぼ一直線）は再サンプリングする。
    戻り値のshapeは(3, 2)。
    """
    while True:
        checkpoints = torch.rand(3, 2) * (coord_max - coord_min) + coord_min
        if _triangle_area(checkpoints) >= min_triangle_area:
            return checkpoints


def _segments(checkpoints: torch.Tensor, closed: bool) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """checkpointsを順に結んだ経路のセグメント(始点,終点)列。closed=Trueなら最後に始点へ戻るセグメントを含む。"""
    n = checkpoints.shape[0]
    pairs = [(checkpoints[i], checkpoints[i + 1]) for i in range(n - 1)]
    if closed:
        pairs.append((checkpoints[n - 1], checkpoints[0]))
    return pairs


def PolylineArcLength(checkpoints: torch.Tensor, closed: bool) -> float:
    """checkpointsを順に結んだ経路（closed=Trueなら閉路）の全長（弧長）を返す。"""
    return sum((b - a).norm().item() for a, b in _segments(checkpoints, closed))


def SampleEquidistantPoints(checkpoints: torch.Tensor, step: float, closed: bool) -> torch.Tensor:
    """
    セグメント（checkpoint間の各辺）ごとに独立して等間隔サンプリングする。各セグメントは
    `num_intervals = max(1, round(seg_len/step))` 個の区間に分割し、その区間数分だけ終点側の
    点（区間の右端）を出力列に加える。区間の右端は必ずセグメントの終点（=checkpoint）と一致するため、
    checkpointsは（stepの倍数と一致するかに関わらず）必ず出力列に含まれる。実際の間隔は
    セグメントごとの `seg_len / num_intervals` になり、指定したstepの近似値になる
    （辺がstepよりかなり短い場合はnum_intervals=1、すなわち区間内の中間点は生成されない）。
    1点目は必ずcheckpoints[0]。戻り値のshapeは(L_actual, 2)。
    """
    segments = _segments(checkpoints, closed)

    points = [checkpoints[0]]
    for a, b in segments:
        seg_len = (b - a).norm().item()
        num_intervals = max(1, round(seg_len / step))
        for j in range(1, num_intervals + 1):
            points.append(a + (b - a) * (j / num_intervals))
    return torch.stack(points)


def BuildDataset(
    num_samples: int,
    n_checkpoints: int,
    coord_min: float,
    coord_max: float,
    step: float,
    closed: bool,
    min_triangle_area: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    num_samples個の三角形toyサンプルを生成する。
    戻り値は(checkpoints_all, targets, mask, l_max)。
    checkpoints_all: (num_samples, n_checkpoints, 2)
    targets: (num_samples, l_max, 2)  余った分は0padding
    mask: (num_samples, l_max) bool  有効な出力位置がTrue
    """
    assert n_checkpoints == 3, "PolylineInterpは現状N=3の三角形のみ対応"

    checkpoints_list = []
    points_list = []
    for _ in range(num_samples):
        checkpoints = GenerateTriangleCheckpoints(coord_min, coord_max, min_triangle_area)
        points = SampleEquidistantPoints(checkpoints, step, closed)
        checkpoints_list.append(checkpoints)
        points_list.append(points)

    l_max = max(points.shape[0] for points in points_list)
    checkpoints_all = torch.stack(checkpoints_list, dim=0)
    targets = torch.zeros(num_samples, l_max, 2)
    mask = torch.zeros(num_samples, l_max, dtype=torch.bool)
    for i, points in enumerate(points_list):
        l_actual = points.shape[0]
        targets[i, :l_actual] = points
        mask[i, :l_actual] = True

    return checkpoints_all, targets, mask, l_max

"""
design.md の BuildMNISTBatch。

torchvision.datasets.MNISTでMNISTデータをダウンロード・キャッシュし、
バッチをランダムサンプリングしてswiss roll実験と同じ値域・shapeに変換する。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision

_DATA_ROOT = "scratch/MNIST/data"  # .gitignoreのdata/パターンで除外される

_mnist_dataset: torchvision.datasets.MNIST | None = None


def _get_mnist_dataset() -> torchvision.datasets.MNIST:
    global _mnist_dataset
    if _mnist_dataset is None:
        _mnist_dataset = torchvision.datasets.MNIST(root=_DATA_ROOT, train=True, download=True)
    return _mnist_dataset


def BuildMNISTBatch(batch: int, c: int, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    MNIST(28x28グレースケール)からbatch枚をランダムサンプリングし、(h,w)に引き伸ばして返す。
    c=1（グレースケール）を前提とする。
    """
    dataset = _get_mnist_dataset()
    indices = torch.randint(0, len(dataset), (batch,))

    images = dataset.data[indices].unsqueeze(1).float() / 255.0  # (batch,1,28,28), [0,1]
    images = F.interpolate(images, size=(h, w), mode="bilinear", align_corners=False)
    images = images * 2.0 - 1.0  # [0,1] -> [-1,1]（swiss rollの値域に合わせる）

    label = dataset.targets[indices].float()  # (batch,)
    return images, label


def BuildMNISTImagesByLabel(labels: torch.Tensor, c: int, h: int, w: int) -> torch.Tensor:
    """
    指定したlabels(shape (batch,))それぞれについて、その数字の実際のMNIST画像を1枚ずつ取得する。
    主にinferでの正解画像(ground truth)表示など、可視化用途に使う。
    """
    dataset = _get_mnist_dataset()
    indices = []
    for label in labels.tolist():
        candidates = (dataset.targets == int(label)).nonzero(as_tuple=True)[0]
        indices.append(candidates[torch.randint(0, len(candidates), (1,))].item())

    images = dataset.data[indices].unsqueeze(1).float() / 255.0  # (batch,1,28,28), [0,1]
    images = F.interpolate(images, size=(h, w), mode="bilinear", align_corners=False)
    return images * 2.0 - 1.0  # [0,1] -> [-1,1]

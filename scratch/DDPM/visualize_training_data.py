"""
main.pyの学習で実際に使われている設定(c, h, w)でBuildSwissRollを呼び出し、
生成される画像のサンプルを並べて保存するデモスクリプト。
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from .config import load_config
from .data import BuildSwissRoll
from .main import CONFIG_PATH

NUM_SAMPLES_TO_SHOW = 16
OUTPUT_IMAGE_PATH = "scratch/DDPM/training_data_sample.png"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    c, h, w = cfg.model.image_channels, cfg.model.image_height, cfg.model.image_width
    x0, _angle = BuildSwissRoll(batch=NUM_SAMPLES_TO_SHOW, c=c, h=h, w=w)

    figure, axes = plt.subplots(2, NUM_SAMPLES_TO_SHOW // 2, figsize=(2 * NUM_SAMPLES_TO_SHOW // 2, 4))
    for axis, image in zip(axes.flat, x0):
        axis.imshow(image[0].numpy(), cmap="gray")
        axis.axis("off")

    figure.suptitle(f"BuildSwissRoll samples ({h}x{w}, as used in training)")
    figure.tight_layout()
    figure.savefig(OUTPUT_IMAGE_PATH, dpi=120)
    print(f"saved: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()

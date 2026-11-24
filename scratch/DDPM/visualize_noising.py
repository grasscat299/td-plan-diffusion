"""
Swiss roll全体を表す画像を1枚作り、t=5, 50, 100でノイズをかけた結果を
元画像と並べて4枚表示するデモスクリプト。

design.md の RandomPickedData はtをランダムに選ぶ仕様だが、
このスクリプトでは可視化のために意図的にtを固定値(5, 50, 100)に上書きして使う。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from .data import BuildSwissRoll, RandomPickedData
from .noise_scheduler import NoiseScheduler

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUMBER_OF_DIFFUSION_TIMESTEPS = 100
TARGET_TIMESTEPS_TO_VISUALIZE = [5, 50, 100]
OUTPUT_IMAGE_PATH = "scratch/DDPM/swiss_roll_noising.png"


def add_noise_at_specific_timestep(
    x0: torch.Tensor,
    angle: torch.Tensor,
    target_timestep: int,
    noise_scheduler: NoiseScheduler,
) -> torch.Tensor:
    """
    x0に対して、ランダムに選ばれたtではなく指定したtでノイズをかける。
    RandomPickedDataは生成時にtをランダムサンプリングしてしまうため、
    生成した直後にtを指定の値へ上書きしてからadd_noise()を呼ぶ。
    """
    random_picked_data = RandomPickedData(x0, noise_scheduler.num_timesteps, angle)
    random_picked_data.t = torch.tensor([target_timestep])
    input_data = noise_scheduler.add_noise(random_picked_data)
    return input_data.xt


def main() -> None:
    x0, angle = BuildSwissRoll(batch=1, c=1, h=IMAGE_HEIGHT, w=IMAGE_WIDTH)
    noise_scheduler = NoiseScheduler(NUMBER_OF_DIFFUSION_TIMESTEPS)

    noised_images = []
    for target_timestep in TARGET_TIMESTEPS_TO_VISUALIZE:
        noised_image = add_noise_at_specific_timestep(x0, angle, target_timestep, noise_scheduler)
        noised_images.append(noised_image)

    images_to_show = [x0] + noised_images
    titles_to_show = ["t=0 (original)"] + [f"t={timestep}" for timestep in TARGET_TIMESTEPS_TO_VISUALIZE]

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, image, title in zip(axes, images_to_show, titles_to_show):
        single_image_as_numpy_array = image[0, 0].detach().numpy()
        axis.imshow(single_image_as_numpy_array, cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(OUTPUT_IMAGE_PATH)
    print(f"saved: {OUTPUT_IMAGE_PATH}")
    plt.show()


if __name__ == "__main__":
    main()

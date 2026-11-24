"""
transformer_train.pyで学習した重みを読み込み、28x28のpixel空間で直接
（VAEを経由せず）、数字ラベル0〜9それぞれを指定してSAMPLING_STEPS回の逆拡散で生成し、
正解画像と並べて保存し、WebUIに送るデモスクリプト（experiment.kind="pixel_transformer"）。

逆拡散のロジック(respaced timesteps上のancestral sampling)はscratch/DDPM/infer.pyのsample()を
そのまま再利用する（angleとlabelを区別しない汎用実装のため。コピーしない）。

モデル構造はcheckpointに同梱されたconfigから復元する。WebUI対応前の旧形式checkpoint（configなし）の
場合だけ、main.pyから渡された（config.tomlの）cfgを使う。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from scratch.DDPM.infer import sample
from scratch.DDPM.model import build_model
from scratch.DDPM.noise_scheduler import NoiseScheduler
from train_webui import client as webui

from .checkpoint import load_checkpoint
from .config import Config
from .data import BuildMNISTImagesByLabel

SAMPLING_STEPS = 100
NUM_SAMPLES_TO_GENERATE = 10  # 数字ラベル0〜9を1つずつ生成する

NUM_PROGRESSION_STEPS_TO_SHOW = 10  # 逆拡散の何段階分をグリッド画像に並べるか
PROGRESSION_IMAGE_PATH = "scratch/MNIST/transformer_sampling_progression.png"


def run(load_path: str, cfg: Config, device: torch.device) -> None:
    state_dict, ckpt_cfg = load_checkpoint(load_path, map_location=device)
    cfg = ckpt_cfg or cfg
    c = cfg.pixel_transformer.image_channels
    h = cfg.pixel_transformer.image_height
    w = cfg.pixel_transformer.image_width

    net = build_model(cfg.pixel_transformer.model_type, c=c, h=h, w=w)
    net.load_state_dict(state_dict)
    net.to(device)

    noise_scheduler = NoiseScheduler(cfg.diffusion.num_timesteps)
    noise_scheduler.to(device)

    # 0〜9のラベルを1つずつ指定して生成する（angle引数にlabelをそのまま渡す）
    label = torch.arange(NUM_SAMPLES_TO_GENERATE, dtype=torch.float32, device=device)

    _generated_images, intermediates = sample(
        net,
        noise_scheduler,
        batch=NUM_SAMPLES_TO_GENERATE,
        c=c,
        h=h,
        w=w,
        sampling_steps=SAMPLING_STEPS,
        device=device,
        angle=label,
    )

    reference_images = BuildMNISTImagesByLabel(label.detach().cpu(), c, h, w)

    num_shown = min(NUM_PROGRESSION_STEPS_TO_SHOW, len(intermediates))
    shown_indices = torch.linspace(0, len(intermediates) - 1, num_shown).round().long().unique().tolist()
    num_cols = len(shown_indices) + 1  # +1 は正解画像列

    prog_figure, prog_axes = plt.subplots(
        NUM_SAMPLES_TO_GENERATE, num_cols,
        figsize=(2 * num_cols, 2 * NUM_SAMPLES_TO_GENERATE),
    )
    for row in range(NUM_SAMPLES_TO_GENERATE):
        for col, idx in enumerate(shown_indices):
            axis = prog_axes[row, col]
            # xtはtanh等の最終活性化が無く[-1,1]に収まらないため、vmin/vmaxを固定して
            # imshowの自動コントラスト調整で品質が見えなくなるのを防ぐ(scratch/VAE/infer.py参照)
            axis.imshow(intermediates[idx][row, 0].detach().cpu().numpy(), cmap="gray", vmin=-1, vmax=1)
            axis.axis("off")
            if row == 0:
                axis.set_title("x_T" if idx == 0 else f"step {idx}", fontsize=8)

        gt_axis = prog_axes[row, num_cols - 1]
        gt_axis.imshow(reference_images[row, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        gt_axis.axis("off")
        if row == 0:
            gt_axis.set_title("ground truth", fontsize=8)

    prog_figure.suptitle("Reverse diffusion in pixel space (28x28, transformer, label-conditioned) vs. ground truth")
    prog_figure.tight_layout()
    prog_figure.savefig(PROGRESSION_IMAGE_PATH, dpi=120)
    print(f"saved: {PROGRESSION_IMAGE_PATH}")
    webui.log_image("infer/progression", prog_figure)

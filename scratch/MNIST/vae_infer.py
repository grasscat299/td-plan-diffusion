"""
vae_train.pyで学習した重みを読み込み、MNIST画像をVAEに通してdecoderの入力(z)・出力(x_hat)・
正解データ(x)をNUM_SAMPLES_TO_GENERATE行のグリッド画像として保存し、WebUIに送るデモスクリプト
（experiment.kind="vae"）。scratch/VAE/infer.pyと同じ構成
（BuildSwissRollの代わりにBuildMNISTBatchを使う点だけが違う）。

モデル構造はcheckpointに同梱されたconfigから復元する。WebUI対応前の旧形式checkpoint（configなし）の
場合だけ、main.pyから渡された（config.tomlの）cfgを使う。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from scratch.VAE.model import Reparameterize, VAE
from train_webui import client as webui

from .checkpoint import load_checkpoint
from .config import Config
from .data import BuildMNISTBatch

NUM_SAMPLES_TO_GENERATE = 4
OUTPUT_IMAGE_PATH = "scratch/MNIST/vae_reconstruction.png"


def run(load_path: str, cfg: Config, device: torch.device) -> None:
    state_dict, ckpt_cfg = load_checkpoint(load_path, map_location=device)
    cfg = ckpt_cfg or cfg
    c, h, w = cfg.vae.image_channels, cfg.vae.latent_height, cfg.vae.latent_width

    net = VAE(c=c, h=h, w=w)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    x, _label = BuildMNISTBatch(NUM_SAMPLES_TO_GENERATE, c, 4 * h, 4 * w)
    x = x.to(device)

    with torch.no_grad():
        encoder_output = net.encoder(x)
        z = Reparameterize(encoder_output.mean, encoder_output.logvar)
        x_hat = net.decoder(z)

    # zはlatent表現で値域がx/x_hatとは異なるためNoneのまま(パネルごとの自動スケール)にし、
    # x_hat/xは同じ[-1,1]スケールに固定して見比べる（固定しないとx_hatの粗さが
    # imshowの自動コントラスト調整で見えなくなる）
    columns = [
        ("decoder input (z)", z, None, None),
        ("decoder output (x_hat)", x_hat, -1, 1),
        ("ground truth (x)", x, -1, 1),
    ]

    figure, axes = plt.subplots(
        NUM_SAMPLES_TO_GENERATE, len(columns),
        figsize=(3 * len(columns), 3 * NUM_SAMPLES_TO_GENERATE),
    )
    for row in range(NUM_SAMPLES_TO_GENERATE):
        for col, (title, images, vmin, vmax) in enumerate(columns):
            axis = axes[row, col]
            axis.imshow(images[row, 0].detach().cpu().numpy(), cmap="gray", vmin=vmin, vmax=vmax)
            axis.axis("off")
            if row == 0:
                axis.set_title(title)

    figure.tight_layout()
    figure.savefig(OUTPUT_IMAGE_PATH)
    print(f"saved: {OUTPUT_IMAGE_PATH}")
    webui.log_image("infer/reconstruction", figure)

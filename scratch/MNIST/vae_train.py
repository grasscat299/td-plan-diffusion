"""
design.md の VaeTrainSchedule（experiment.kind="vae"）。

scratch/VAE/train.py の TrainSchedule と全く同じ構造で、x0の生成元だけを
scratch.MNIST.data.BuildMNISTBatch に差し替えたもの。KLLoss/ReconLoss/Loss/OneStepは
scratch.VAE.train のものをそのまま再利用する（コピーしない）。
"""
from __future__ import annotations

import torch
from torch.optim import Adam

from scratch.VAE.train import OneStep
from train_webui import client as webui

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config
from .data import BuildMNISTBatch


def VaeTrainSchedule(
    net: torch.nn.Module,
    cfg: Config,
    save_path: str,
    load_path: str | None,
    device: torch.device,
) -> None:
    if load_path is not None:
        state_dict, _ = load_checkpoint(load_path, map_location=device)
        net.load_state_dict(state_dict)

    c, h, w = cfg.vae.image_channels, cfg.vae.latent_height, cfg.vae.latent_width
    net.to(device)
    optimizer = Adam(net.parameters(), lr=cfg.train.lr)
    # webui.init 前（テスト等）なら define_plot/add は warning を出して無視される
    step_loss_plot = webui.define_plot(
        "train/loss", kind="line", title="batchごとのloss", xlabel="step", ylabel="loss", yscale="log",
    )
    epoch_loss_plot = webui.define_plot(
        "epoch/loss", kind="line", title="epochごとのloss", xlabel="epoch", ylabel="loss", yscale="log",
    )

    step = 0
    for ep in range(cfg.train.epoch):
        epoch_loss = 0.0
        for _ in range(cfg.train.steps_per_epoch):
            x0, _label = BuildMNISTBatch(cfg.train.batch, c, 4 * h, 4 * w)
            x0 = x0.to(device)
            loss = OneStep(net, optimizer, x0, cfg.vae.kl_weight)
            loss_value = loss.item()
            epoch_loss += loss_value
            step_loss_plot.add(step, loss_value, step=step)
            step += 1

        avg_loss = epoch_loss / cfg.train.steps_per_epoch
        print(f"epoch {ep + 1}/{cfg.train.epoch}  loss={avg_loss:.4f}")
        epoch_loss_plot.add(ep + 1, avg_loss, series="train", step=step)

    save_checkpoint(save_path, net, cfg)

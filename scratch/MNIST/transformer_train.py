"""
VAEのlatent空間を経由せず、28x28のpixel空間で直接DDPM(TransformerModel)を
数字ラベル条件付きで学習するTrainSchedule（experiment.kind="pixel_transformer"）。

scratch/DDPM/train.pyのTrainScheduleと同じ構造だが、あちら(BuildSwissRoll固定)とは
データ元がBuildMNISTBatchである点、条件付けの値がangle(連続値)ではなくlabel(0.0〜9.0)
である点が違う。RandomPickedData/NoiseScheduler/OneStepはscratch.DDPMのものを
そのまま再利用する（コピーしない）。
"""
from __future__ import annotations

import torch
from torch.optim import Adam

from scratch.DDPM.data import RandomPickedData
from scratch.DDPM.noise_scheduler import NoiseScheduler
from scratch.DDPM.train import OneStep
from train_webui import client as webui

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config
from .data import BuildMNISTBatch


def TrainSchedule(
    net: torch.nn.Module,
    noise_scheduler: NoiseScheduler,
    cfg: Config,
    save_path: str,
    load_path: str | None,
    device: torch.device,
) -> None:
    if load_path is not None:
        state_dict, _ = load_checkpoint(load_path, map_location=device)
        net.load_state_dict(state_dict)

    c = cfg.pixel_transformer.image_channels
    h = cfg.pixel_transformer.image_height
    w = cfg.pixel_transformer.image_width
    net.to(device)
    noise_scheduler.to(device)
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
            x0, label = BuildMNISTBatch(cfg.train.batch, c, h, w)
            x0, label = x0.to(device), label.to(device)
            random_picked_data = RandomPickedData(x0, noise_scheduler.num_timesteps, angle=label)
            loss = OneStep(net, optimizer, noise_scheduler, random_picked_data)
            loss_value = loss.item()
            epoch_loss += loss_value
            step_loss_plot.add(step, loss_value, step=step)
            step += 1

        avg_loss = epoch_loss / cfg.train.steps_per_epoch
        print(f"epoch {ep + 1}/{cfg.train.epoch}  loss={avg_loss:.4f}")
        epoch_loss_plot.add(ep + 1, avg_loss, series="train", step=step)

    save_checkpoint(save_path, net, cfg)

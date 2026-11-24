"""
design.md の Loss / OneStep / TrainSchedule。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim import Adam

from train_webui import client as webui

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config
from .data import BuildSwissRoll, RandomPickedData
from .noise_scheduler import NoiseScheduler


def Loss(noise_pred: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(noise_pred, eps)


def OneStep(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    noise_scheduler: NoiseScheduler,
    random_picked_data: RandomPickedData,
) -> torch.Tensor:
    optimizer.zero_grad()
    input_data = noise_scheduler.add_noise(random_picked_data)
    noise_pred = net(input_data)
    loss = Loss(noise_pred, input_data.eps)
    loss.backward()
    optimizer.step()
    return loss


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

    c, h, w = cfg.model.image_channels, cfg.model.image_height, cfg.model.image_width
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
            x0, angle = BuildSwissRoll(cfg.train.batch, c, h, w)
            x0, angle = x0.to(device), angle.to(device)
            random_picked_data = RandomPickedData(x0, noise_scheduler.num_timesteps, angle)
            loss = OneStep(net, optimizer, noise_scheduler, random_picked_data)
            loss_value = loss.item()
            epoch_loss += loss_value
            step_loss_plot.add(step, loss_value, step=step)
            step += 1

        avg_loss = epoch_loss / cfg.train.steps_per_epoch
        print(f"epoch {ep + 1}/{cfg.train.epoch}  loss={avg_loss:.4f}")
        epoch_loss_plot.add(ep + 1, avg_loss, series="train", step=step)

    save_checkpoint(save_path, net, cfg)

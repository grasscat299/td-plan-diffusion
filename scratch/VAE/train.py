"""
design.md の KLLoss / ReconLoss / Loss / OneStep / TrainSchedule。

x0の生成はscratch/DDPM/data.pyのBuildSwissRollを再利用する（angleはVAEでは使わないため無視する）。
TrainScheduleは現行のscratch/DDPM/train.pyと同じく、num_batchesで事前にx0_listを作るのではなく、
steps_per_epochごとに毎回1バッチ生成する方式にする。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim import Adam

from scratch.DDPM.data import BuildSwissRoll
from train_webui import client as webui

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config
from .model import VAEOutput


def KLLoss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp())).mean()


def ReconLoss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(x_hat, x)


def Loss(output: VAEOutput, x: torch.Tensor, kl_weight: float) -> torch.Tensor:
    return ReconLoss(output.x_hat, x) + kl_weight * KLLoss(output.mean, output.logvar)


def OneStep(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    kl_weight: float,
) -> torch.Tensor:
    optimizer.zero_grad()
    output = net(x)
    loss = Loss(output, x, kl_weight)
    loss.backward()
    optimizer.step()
    return loss


def TrainSchedule(
    net: torch.nn.Module,
    cfg: Config,
    save_path: str,
    load_path: str | None,
    device: torch.device,
) -> None:
    if load_path is not None:
        state_dict, _ = load_checkpoint(load_path, map_location=device)
        net.load_state_dict(state_dict)

    c, h, w = cfg.model.image_channels, cfg.model.latent_height, cfg.model.latent_width
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
            x0, _angle = BuildSwissRoll(cfg.train.batch, c, 4 * h, 4 * w)
            x0 = x0.to(device)
            loss = OneStep(net, optimizer, x0, cfg.train.kl_weight)
            loss_value = loss.item()
            epoch_loss += loss_value
            step_loss_plot.add(step, loss_value, step=step)
            step += 1

        avg_loss = epoch_loss / cfg.train.steps_per_epoch
        print(f"epoch {ep + 1}/{cfg.train.epoch}  loss={avg_loss:.4f}")
        epoch_loss_plot.add(ep + 1, avg_loss, series="train", step=step)

    save_checkpoint(save_path, net, cfg)

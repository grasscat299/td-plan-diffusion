"""
design.md の BuildLatentBatch / TrainSchedule。

RandomPickedData/InputData/NoiseScheduler/Model/Loss/OneStepは(batch,C,H,W)の
汎用テンソルとして書かれており、pixel空間かlatent空間かを区別しないため、
scratch.DDPM.data / scratch.DDPM.train のものをそのまま再利用する（コピーしない）。
"""
from __future__ import annotations

import torch
from torch.optim import Adam

from scratch.DDPM.data import BuildSwissRoll, RandomPickedData
from scratch.DDPM.noise_scheduler import NoiseScheduler
from scratch.DDPM.train import OneStep
from scratch.VAE.model import Reparameterize, VAE
from train_webui import client as webui

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config


@torch.no_grad()
def BuildLatentBatch(
    vae: VAE,
    batch: int,
    c_img: int,
    h: int,
    w: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0_pixel, angle = BuildSwissRoll(batch, c_img, 4 * h, 4 * w)
    x0_pixel = x0_pixel.to(device)
    encoder_output = vae.encoder(x0_pixel)
    z = Reparameterize(encoder_output.mean, encoder_output.logvar)
    return z, angle.to(device)


def TrainSchedule(
    net: torch.nn.Module,
    vae: VAE,
    noise_scheduler: NoiseScheduler,
    cfg: Config,
    save_path: str,
    load_path: str | None,
    device: torch.device,
) -> None:
    if load_path is not None:
        state_dict, _ = load_checkpoint(load_path, map_location=device)
        net.load_state_dict(state_dict)

    c_img, h, w = cfg.vae.image_channels, cfg.vae.latent_height, cfg.vae.latent_width

    net.to(device)
    noise_scheduler.to(device)
    vae.to(device)
    vae.eval()  # VAEは凍結（勾配を流さない・重みも更新しない）

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
            z, angle = BuildLatentBatch(vae, cfg.train.batch, c_img, h, w, device)
            random_picked_data = RandomPickedData(z, noise_scheduler.num_timesteps, angle)
            loss = OneStep(net, optimizer, noise_scheduler, random_picked_data)
            loss_value = loss.item()
            epoch_loss += loss_value
            step_loss_plot.add(step, loss_value, step=step)
            step += 1

        avg_loss = epoch_loss / cfg.train.steps_per_epoch
        print(f"epoch {ep + 1}/{cfg.train.epoch}  loss={avg_loss:.4f}")
        epoch_loss_plot.add(ep + 1, avg_loss, series="train", step=step)

    save_checkpoint(save_path, net, cfg)

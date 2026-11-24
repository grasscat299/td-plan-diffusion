"""
design.md の BuildLatentMNISTBatch / TrainSchedule（experiment.kind="latent_dit"）。

scratch/LatentDDPM/train.py の BuildLatentBatch/TrainSchedule と全く同じ構造で、
x0の生成元をBuildMNISTBatchに、条件付けの値をangle(連続値の回転角度)から
label(0.0〜9.0の数字ラベル)に差し替えたもの。ただしRandomPickedDataのフィールド名は
angleのままなので、labelをそのままangle引数に渡す（scratch.DDPM側のモデルコードは
一切変更しない）。

Loss/OneStepはscratch.DDPM.train(MSE固定)をそのまま使わず、cfg.latent_dit.loss_typeで
MSE/L1を切り替えられるようにこのファイルで独自に定義している。

denoiseネットワークはcfg.latent_dit.dit_variantで2種類を切り替えられる（build_net）:
- "local_dit": scratch/MNIST/model_dit.pyのDiTModel（数字ラベルをnn.Embeddingで条件付け）
- "model_dit": scratch.DDPM.model.build_model("dit", ...)（angleスロットにsinusoidal
  embeddingでラベルを転用、scratch/DDPM/model/model_dit.py本体をそのまま使う）

build_net/load_frozen_vaeはmain.py（学習エントリー）とinfer.py（推論エントリー）の
両方から呼ばれるためここに置く（infer.pyがmain.pyをimportせずに済むように）。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from scratch.DDPM.data import RandomPickedData
from scratch.DDPM.model import build_model
from scratch.DDPM.noise_scheduler import NoiseScheduler
from scratch.VAE.model import Reparameterize, VAE
from train_webui import client as webui

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config
from .data import BuildMNISTBatch
from .model_dit import DiTModel


def build_net(cfg: Config) -> nn.Module:
    """cfg.latent_dit.dit_variantに従ってdenoiseネットワークを構築する。"""
    c, h, w = cfg.vae.image_channels, cfg.vae.latent_height, cfg.vae.latent_width
    if cfg.latent_dit.dit_variant == "local_dit":
        return DiTModel(c=c, h=h, w=w)
    if cfg.latent_dit.dit_variant == "model_dit":
        return build_model("dit", c=c, h=h, w=w)
    raise ValueError(
        f"unknown dit_variant: {cfg.latent_dit.dit_variant!r} (choices: 'local_dit', 'model_dit')"
    )


def load_frozen_vae(cfg: Config, device: torch.device) -> VAE:
    """
    kind="latent_dit"の学習・推論で使う、凍結済みVAEをcfg.vae.load_pathから読み込む。
    このckptはkind="vae"でMNIST自身が保存した新形式（config同梱）か、それ以前の
    生state_dict(旧形式)のどちらかなので、load_checkpointでどちらの形式も吸収する。
    """
    load_path = cfg.vae.load_path
    if not Path(load_path).exists():
        raise FileNotFoundError(
            f"VAE checkpoint がありません: {load_path}（先に experiment.kind=\"vae\" で --mode train）"
        )
    state_dict, _ckpt_cfg = load_checkpoint(load_path, map_location=device)
    vae = VAE(c=cfg.vae.image_channels, h=cfg.vae.latent_height, w=cfg.vae.latent_width)
    vae.load_state_dict(state_dict)
    print(f"loaded vae: {load_path}")
    return vae


def Loss(noise_pred: torch.Tensor, eps: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(noise_pred, eps)
    if loss_type == "l1":
        return F.l1_loss(noise_pred, eps)
    raise ValueError(f"unknown loss_type: {loss_type!r} (choices: 'mse', 'l1')")


def OneStep(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    noise_scheduler: NoiseScheduler,
    random_picked_data: RandomPickedData,
    loss_type: str,
) -> torch.Tensor:
    optimizer.zero_grad()
    input_data = noise_scheduler.add_noise(random_picked_data)
    noise_pred = net(input_data)
    loss = Loss(noise_pred, input_data.eps, loss_type)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def BuildLatentMNISTBatch(
    vae: VAE,
    batch: int,
    c_img: int,
    h: int,
    w: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0_pixel, label = BuildMNISTBatch(batch, c_img, 4 * h, 4 * w)
    x0_pixel = x0_pixel.to(device)
    encoder_output = vae.encoder(x0_pixel)
    z = Reparameterize(encoder_output.mean, encoder_output.logvar)
    return z, label.to(device)


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
            z, label = BuildLatentMNISTBatch(vae, cfg.train.batch, c_img, h, w, device)
            random_picked_data = RandomPickedData(z, noise_scheduler.num_timesteps, angle=label)
            loss = OneStep(net, optimizer, noise_scheduler, random_picked_data, cfg.latent_dit.loss_type)
            loss_value = loss.item()
            epoch_loss += loss_value
            step_loss_plot.add(step, loss_value, step=step)
            step += 1

        avg_loss = epoch_loss / cfg.train.steps_per_epoch
        print(f"epoch {ep + 1}/{cfg.train.epoch}  loss={avg_loss:.4f}")
        epoch_loss_plot.add(ep + 1, avg_loss, series="train", step=step)

    save_checkpoint(save_path, net, cfg)

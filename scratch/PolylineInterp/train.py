"""
design.md の MaskedMSELoss / OneStep / TrainSchedule。

design.mdのTrainScheduleは内部でBuildDatasetを呼ぶ想定だったが、netの構築に必要な
l_max（Config.model.l_max）はBuildDatasetを実行してみないと確定しないため、
「netを先に作ってTrainScheduleへ渡す」設計と循環依存になる。ユーザー確認の上、
main.py側で先にBuildDatasetを実行してl_maxを確定・netを構築し、生成済みのデータ
（checkpoints/targets/mask）をTrainScheduleへ渡す形に変更している。
"""
from __future__ import annotations

import torch
from torch.optim import Adam

from train_webui import client as webui

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config


def MaskedMSELoss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff2 = (pred - target) ** 2  # (batch, l_max, 2)
    mask_f = mask.unsqueeze(-1).float()  # (batch, l_max, 1)
    return (diff2 * mask_f).sum() / (mask.sum() * pred.shape[-1])


def OneStep(
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoints: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad()
    pred = net(checkpoints)
    loss = MaskedMSELoss(pred, targets, mask)
    loss.backward()
    optimizer.step()
    return loss


def TrainSchedule(
    net: torch.nn.Module,
    cfg: Config,
    checkpoints_all: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    val_checkpoints: torch.Tensor,
    val_targets: torch.Tensor,
    val_mask: torch.Tensor,
    save_path: str,
    load_path: str | None,
    device: torch.device,
) -> None:
    if load_path is not None:
        state_dict, _ = load_checkpoint(load_path, map_location=device)
        net.load_state_dict(state_dict)

    net.to(device)
    checkpoints_all, targets, mask = checkpoints_all.to(device), targets.to(device), mask.to(device)
    val_checkpoints = val_checkpoints.to(device)
    val_targets, val_mask = val_targets.to(device), val_mask.to(device)

    optimizer = Adam(net.parameters(), lr=cfg.train.lr)
    # webui.init 前（テスト等）なら define_plot/add は warning を出して無視される
    step_loss_plot = webui.define_plot(
        "train/loss", kind="line", title="batchごとのloss", xlabel="step", ylabel="loss", yscale="log",
    )
    epoch_loss_plot = webui.define_plot(
        "epoch/loss", kind="line", title="epochごとのloss", xlabel="epoch", ylabel="loss", yscale="log",
    )

    num_train = checkpoints_all.shape[0]
    step = 0
    for ep in range(cfg.train.epoch):
        perm = torch.randperm(num_train, device=device)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, num_train, cfg.train.batch):
            idx = perm[start:start + cfg.train.batch]
            loss = OneStep(net, optimizer, checkpoints_all[idx], targets[idx], mask[idx])
            loss_value = loss.item()
            epoch_loss += loss_value
            num_batches += 1
            step_loss_plot.add(step, loss_value, step=step)
            step += 1

        avg_loss = epoch_loss / num_batches
        net.eval()
        with torch.no_grad():
            val_pred = net(val_checkpoints)
            val_loss = MaskedMSELoss(val_pred, val_targets, val_mask).item()
        net.train()

        print(f"epoch {ep + 1}/{cfg.train.epoch}  loss={avg_loss:.4f}  val_loss={val_loss:.4f}")
        epoch_loss_plot.add(ep + 1, avg_loss, series="train", step=step)
        epoch_loss_plot.add(ep + 1, val_loss, series="val", step=step)

    save_checkpoint(save_path, net, cfg)

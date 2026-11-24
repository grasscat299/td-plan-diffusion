"""
main.pyの--mode inferから呼ばれる推論・可視化ロジック。学習した重みを読み込み、latent空間の純粋な
ノイズからSAMPLING_STEPS回の逆拡散でswiss rollのlatentを生成し、VAEのDecoderでpixel空間に戻して
保存・WebUIへの送信を行う。

逆拡散のロジック(respaced timesteps上のancestral sampling)はscratch/DDPM/infer.pyのsample()を
そのまま再利用する（latent空間かpixel空間かを区別しない汎用実装のため。コピーしない）。

モデル構造はcheckpointに同梱されたconfigから復元する。WebUI対応前の旧形式checkpoint（configなし）の
場合だけ、main.pyから渡された（config.tomlの）cfgを使う。
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from scratch.DDPM.data import BuildSwissRollAtAngles
from scratch.DDPM.infer import sample
from scratch.DDPM.model import build_model
from scratch.DDPM.noise_scheduler import NoiseScheduler
from scratch.VAE.checkpoint import load_checkpoint as load_vae_checkpoint
from scratch.VAE.model import VAE
from train_webui import client as webui

from .checkpoint import load_checkpoint
from .config import Config

SAMPLING_STEPS = 100
NUM_SAMPLES_TO_GENERATE = 4

NUM_PROGRESSION_STEPS_TO_SHOW = 10  # 逆拡散の何段階分をグリッド画像に並べるか
PROGRESSION_IMAGE_PATH = "scratch/LatentDDPM/sampling_progression.png"
LATENT_PROGRESSION_IMAGE_PATH = "scratch/LatentDDPM/sampling_progression_latent.png"


def _load_vae(cfg: Config, device: torch.device) -> VAE:
    """[vae]セクションのload_pathからVAEの重みを読み込む（main.py._load_vaeと同じ規則）。

    WebUIは学習開始・inferの度にこのプロジェクトのコードのみを学習用PCへrsyncし、
    checkpoints/ は同期しないため、VAEのcheckpointは学習用PC上のscratch/VAE/checkpointsに
    あらかじめ置いておく必要がある。
    """
    vae_path = cfg.vae.load_path
    if not Path(vae_path).exists():
        raise FileNotFoundError(
            f"VAE checkpoint がありません: {vae_path}\n"
            "WebUI経由の実行はこのプロジェクトのコードのみを学習用PCへ同期し、checkpoints/ は"
            "同期されません。学習用PC上の scratch/VAE/checkpoints に、[vae] load_path の"
            "VAE checkpoint を事前に置いてください。"
        )
    state_dict, _vae_cfg = load_vae_checkpoint(vae_path, map_location=device)
    vae = VAE(c=cfg.vae.image_channels, h=cfg.vae.latent_height, w=cfg.vae.latent_width)
    vae.load_state_dict(state_dict)
    return vae


def run(load_path: str, cfg: Config, device: torch.device) -> None:
    state_dict, ckpt_cfg = load_checkpoint(load_path, map_location=device)
    cfg = ckpt_cfg or cfg
    c_img, h, w = cfg.vae.image_channels, cfg.vae.latent_height, cfg.vae.latent_width

    net = build_model(cfg.model.type, c=c_img, h=h, w=w)
    net.load_state_dict(state_dict)
    net.to(device)

    vae = _load_vae(cfg, device)
    vae.to(device)
    vae.eval()

    noise_scheduler = NoiseScheduler(cfg.diffusion.num_timesteps)
    noise_scheduler.to(device)

    # 生成する渦の回転角度を[0, 2π)で等間隔に用意し、angle_embによる条件付けを確認できるようにする
    angle = torch.linspace(0, 2.0 * math.pi, NUM_SAMPLES_TO_GENERATE + 1, device=device)[:-1]

    _generated_latents, intermediates = sample(
        net,
        noise_scheduler,
        batch=NUM_SAMPLES_TO_GENERATE,
        c=c_img,
        h=h,
        w=w,
        sampling_steps=SAMPLING_STEPS,
        device=device,
        angle=angle,
    )

    # ── 逆拡散の各段階(latentをVAE Decoderで都度pixel化) + 同じangleでの正解画像 ─────
    reference_images = BuildSwissRollAtAngles(
        NUM_SAMPLES_TO_GENERATE, c_img, 4 * h, 4 * w,
        angle.detach().cpu(),
    )

    num_shown = min(NUM_PROGRESSION_STEPS_TO_SHOW, len(intermediates))
    shown_indices = torch.linspace(0, len(intermediates) - 1, num_shown).round().long().unique().tolist()
    num_cols = len(shown_indices) + 1  # +1 は正解画像列

    with torch.no_grad():
        decoded_steps = [vae.decoder(intermediates[idx]) for idx in shown_indices]

    prog_figure, prog_axes = plt.subplots(
        NUM_SAMPLES_TO_GENERATE, num_cols,
        figsize=(2 * num_cols, 2 * NUM_SAMPLES_TO_GENERATE),
    )
    for row in range(NUM_SAMPLES_TO_GENERATE):
        for col, idx in enumerate(shown_indices):
            axis = prog_axes[row, col]
            axis.imshow(decoded_steps[col][row, 0].detach().cpu().numpy(), cmap="gray", vmin=-1, vmax=1)
            axis.axis("off")
            if row == 0:
                axis.set_title("x_T (latent)" if idx == 0 else f"step {idx}", fontsize=8)

        gt_axis = prog_axes[row, num_cols - 1]
        gt_axis.imshow(reference_images[row, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
        gt_axis.axis("off")
        if row == 0:
            gt_axis.set_title("ground truth", fontsize=8)

    prog_figure.suptitle("Reverse diffusion in latent space -> VAE decode, vs. ground truth")
    prog_figure.tight_layout()
    prog_figure.savefig(PROGRESSION_IMAGE_PATH, dpi=120)
    print(f"saved: {PROGRESSION_IMAGE_PATH}")
    webui.log_image("infer/progression", prog_figure)

    # ── 逆拡散の各段階を、VAE Decoderを経由せずlatentのまま表示（DDPMの生の出力のみ） ──
    shown_latents = [intermediates[idx] for idx in shown_indices]
    latent_vmin = min(t.min().item() for t in shown_latents)
    latent_vmax = max(t.max().item() for t in shown_latents)

    latent_figure, latent_axes = plt.subplots(
        NUM_SAMPLES_TO_GENERATE, len(shown_indices),
        figsize=(2 * len(shown_indices), 2 * NUM_SAMPLES_TO_GENERATE),
    )
    for row in range(NUM_SAMPLES_TO_GENERATE):
        for col, idx in enumerate(shown_indices):
            axis = latent_axes[row, col]
            # 全パネルで同じvmin/vmax(shown_latents全体のmin/max)を使い、逆拡散が進むにつれて
            # latentの値がどう変化するかをコントラストの差として比較できるようにする
            axis.imshow(
                intermediates[idx][row, 0].detach().cpu().numpy(),
                cmap="gray", vmin=latent_vmin, vmax=latent_vmax,
            )
            axis.axis("off")
            if row == 0:
                axis.set_title("x_T (latent)" if idx == 0 else f"step {idx}", fontsize=8)

    latent_figure.suptitle("Reverse diffusion in latent space (DDPM raw output, no VAE decode)")
    latent_figure.tight_layout()
    latent_figure.savefig(LATENT_PROGRESSION_IMAGE_PATH, dpi=120)
    print(f"saved: {LATENT_PROGRESSION_IMAGE_PATH}")
    webui.log_image("infer/progression_latent", latent_figure)

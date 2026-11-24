"""
main.pyの--mode inferから呼ばれる推論・可視化ロジック。学習した重みを読み込み、純粋なノイズ画像から
SAMPLING_STEPS回の逆拡散でswiss roll画像を生成し、WebUIに送る。

design.mdにはサンプリング(逆拡散)のクラス設計が定義されていないため、
Ho et al. (2020)のancestral samplingを、学習時のタイムステップ数(T)から
SAMPLING_STEPS個だけ等間隔に間引いた"respaced"タイムステップ列に適用する形で実装する
（間引かず100ステップ全部やると重いため）。
InputData.epsはModel.forwardで使われないため、型を満たすためのダミー値を渡している。

モデル構造はcheckpointに同梱されたconfigから復元する。WebUI対応前の旧形式checkpoint（configなし）の
場合だけ、main.pyから渡された（config.tomlの）cfgを使う。
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch

from train_webui import client as webui

from .checkpoint import load_checkpoint
from .config import Config
from .data import BuildSwissRollAtAngles, InputData
from .model import build_model
from .noise_scheduler import NoiseScheduler

SAMPLING_STEPS = 100
NUM_SAMPLES_TO_GENERATE = 4
OUTPUT_IMAGE_PATH = "scratch/DDPM/sampled.png"

NUM_PROGRESSION_STEPS_TO_SHOW = 10  # 逆拡散の何段階分をグリッド画像に並べるか
PROGRESSION_IMAGE_PATH = "scratch/DDPM/sampling_progression.png"


def _respaced_timesteps(num_timesteps: int, sampling_steps: int) -> list[int]:
    """
    1〜num_timestepsからsampling_steps個を等間隔に選び、逆拡散の順序(降順)で返す。
    """
    indices = torch.linspace(1, num_timesteps, sampling_steps).round().long().tolist()
    return sorted(set(indices), reverse=True)


@torch.no_grad()
def sample(
    net: torch.nn.Module,
    noise_scheduler: NoiseScheduler,
    batch: int,
    c: int,
    h: int,
    w: int,
    sampling_steps: int,
    device: torch.device,
    angle: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    x_T 〜 N(0,I) から出発し、間引いたタイムステップ列に沿って逆拡散し、x0(生成画像)を返す。
    angle(shape (batch,))は生成したい渦の回転角度で、全ステップを通してモデルへの条件付けに使う。
    戻り値の2要素目は、x_T(開始時点)を含む各ステップ後の画像のスナップショット一覧。
    """
    net.eval()
    timesteps = _respaced_timesteps(noise_scheduler.num_timesteps, sampling_steps)
    a_t_bar = noise_scheduler.a_t_bar

    x = torch.randn(batch, c, h, w, device=device)
    intermediates = [x.clone()]

    for i, t_cur in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0

        a_bar_cur = a_t_bar[t_cur - 1]
        a_bar_prev = a_t_bar[t_prev - 1] if t_prev > 0 else torch.tensor(1.0, device=device)
        alpha_cur = a_bar_cur / a_bar_prev  # respaceした区間分の実効alpha
        beta_cur = 1.0 - alpha_cur

        t_batch = torch.full((batch,), t_cur, dtype=torch.long, device=device)
        input_data = InputData(xt=x, t=t_batch, eps=torch.zeros_like(x), angle=angle)
        eps_pred = net(input_data)

        mean = (x - beta_cur / (1.0 - a_bar_cur).sqrt() * eps_pred) / alpha_cur.sqrt()
        noise = torch.randn_like(x) if t_prev > 0 else torch.zeros_like(x)
        x = mean + beta_cur.sqrt() * noise
        intermediates.append(x.clone())

    return x, intermediates


def run(load_path: str, cfg: Config, device: torch.device) -> None:
    state_dict, ckpt_cfg = load_checkpoint(load_path, map_location=device)
    cfg = ckpt_cfg or cfg
    c, h, w = cfg.model.image_channels, cfg.model.image_height, cfg.model.image_width

    net = build_model(cfg.model.type, c=c, h=h, w=w)
    net.load_state_dict(state_dict)
    net.to(device)
    noise_scheduler = NoiseScheduler(cfg.diffusion.num_timesteps)
    noise_scheduler.to(device)

    # 生成する渦の回転角度を[0, 2π)で等間隔に用意し、angle_embによる条件付けを確認できるようにする
    angle = torch.linspace(0, 2.0 * math.pi, NUM_SAMPLES_TO_GENERATE + 1, device=device)[:-1]

    generated_images, intermediates = sample(
        net,
        noise_scheduler,
        batch=NUM_SAMPLES_TO_GENERATE,
        c=c,
        h=h,
        w=w,
        sampling_steps=SAMPLING_STEPS,
        device=device,
        angle=angle,
    )

    figure, axes = plt.subplots(1, NUM_SAMPLES_TO_GENERATE, figsize=(4 * NUM_SAMPLES_TO_GENERATE, 4))
    for axis, image, a in zip(axes, generated_images, angle):
        axis.imshow(image[0].detach().cpu().numpy(), cmap="gray")
        axis.set_title(f"angle={a.item():.2f}")
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(OUTPUT_IMAGE_PATH)
    print(f"saved: {OUTPUT_IMAGE_PATH}")
    webui.log_image("infer/sampled", figure)

    # ── 逆拡散の各段階 + 同じangleでの正解画像をグリッド画像に保存 ─────────────
    reference_images = BuildSwissRollAtAngles(
        NUM_SAMPLES_TO_GENERATE, c, h, w, angle.detach().cpu(),
    )

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
            axis.imshow(intermediates[idx][row, 0].detach().cpu().numpy(), cmap="gray")
            axis.axis("off")
            if row == 0:
                axis.set_title("x_T" if idx == 0 else f"step {idx}", fontsize=8)

        gt_axis = prog_axes[row, num_cols - 1]
        gt_axis.imshow(reference_images[row, 0].numpy(), cmap="gray")
        gt_axis.axis("off")
        if row == 0:
            gt_axis.set_title("ground truth", fontsize=8)

    prog_figure.suptitle("Reverse diffusion progression (noise -> generated) vs. ground truth")
    prog_figure.tight_layout()
    prog_figure.savefig(PROGRESSION_IMAGE_PATH, dpi=120)
    print(f"saved: {PROGRESSION_IMAGE_PATH}")
    webui.log_image("infer/progression", prog_figure)

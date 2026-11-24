"""エントリーポイント。--mode {train,infer} で学習/推論を呼び分ける（scratch/PolylineInterpと同じ流儀）。

    venv/bin/python -m scratch.DDPM.main --mode train
    venv/bin/python -m scratch.DDPM.main --mode infer  [--ckpt path]

checkpointパスは[wandb] groupから解決する（checkpoints/model_{group}.pt）。--ckpt省略時、
inferはこの解決規則で読みに行く。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_webui import client as webui

from . import infer as infer_mod
from .config import Config, load_config
from .model import build_model
from .noise_scheduler import NoiseScheduler
from .train import TrainSchedule

CONFIG_PATH = "scratch/DDPM/config.toml"
_CKPT_DIR = Path("scratch/DDPM/checkpoints")


def _save_path(cfg: Config) -> str:
    return str(_CKPT_DIR / f"model_{cfg.wandb.group}.pt")


def run_train(cfg: Config, device: torch.device) -> None:
    save_path = _save_path(cfg)
    load_path = save_path if Path(save_path).exists() else None
    if load_path is not None:
        print(f"loaded: {load_path}")

    net = build_model(
        cfg.model.type, c=cfg.model.image_channels, h=cfg.model.image_height, w=cfg.model.image_width,
    )
    noise_scheduler = NoiseScheduler(cfg.diffusion.num_timesteps)

    # 学習の進捗は train_webui に送る。WebUIから起動した場合はそのrunに、CLIから起動した場合は新しいrunとして記録される
    webui.init(new_run=True, source="train")

    TrainSchedule(
        net=net, noise_scheduler=noise_scheduler, cfg=cfg,
        save_path=save_path, load_path=load_path, device=device,
    )

    webui.finish()

    print(f"saved: {save_path}")


def run_infer(cfg: Config, device: torch.device, ckpt: str | None = None) -> None:
    load_path = ckpt or _save_path(cfg)
    if not Path(load_path).exists():
        raise FileNotFoundError(f"checkpoint がありません: {load_path}（先に --mode train / --ckpt を確認）")
    print(f"infer checkpoint: {load_path}")

    # 推論結果の画像は train_webui に送る。送信先はWebUIからの起動ならそのrun、CLIならgroupで最後に起動されたtrainのrun
    webui.init(source="infer")
    infer_mod.run(load_path, cfg, device)
    webui.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--ckpt", default=None,
                        help="inferで読むcheckpointパス（省略時は[wandb] groupから解決）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    cfg = load_config(CONFIG_PATH)

    if args.mode == "train":
        run_train(cfg, device)
    else:
        run_infer(cfg, device, args.ckpt)


if __name__ == "__main__":
    main()

"""エントリーポイント。--mode {train,infer} で学習/推論を呼び分ける（world_model_v1と同じ流儀）。

    venv/bin/python -m scratch.PolylineInterp.main --mode train
    venv/bin/python -m scratch.PolylineInterp.main --mode infer  [--ckpt path]

checkpointパスは[wandb] groupから解決する（checkpoints/model_{group}.pt）。headとして
アーキテクチャの違う実験を回すときは、上書き事故を防ぐためgroup名も変える運用にする
（詳細はdesign.md「checkpointファイル名」参照）。--ckpt省略時、inferはこの解決規則で
読みに行く。

l_maxの循環依存（train.pyの冒頭コメント参照）を解消するため、run_train内でnetを構築する前に
BuildDatasetでtrain/valデータを生成し、そこで確定したl_maxをcfg.model.l_maxへセットしてから
netを構築している。
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch

from train_webui import client as webui

from . import infer as infer_mod
from .config import Config, load_config
from .data import BuildDataset
from .model import PolylineTransformer
from .train import TrainSchedule

CONFIG_PATH = "scratch/PolylineInterp/config.toml"
_CKPT_DIR = Path("scratch/PolylineInterp/checkpoints")


def _save_path(cfg: Config) -> str:
    # WebUIから起動したら run ごとの専用ディレクトリ（model.pt）、手元で直接動かしたら従来の checkpoints/model_{group}.pt
    return webui.checkpoint_path(_CKPT_DIR / f"model_{cfg.wandb.group}.pt")


def _pad_to_l_max(targets: torch.Tensor, mask: torch.Tensor, l_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    train/valを別々にBuildDatasetした場合、それぞれのl_maxが異なりうる。
    netはtrain/val共通のl_max（両者の最大値）でクエリ数を固定するため、
    targets/maskをそちら側に合わせて0paddingし直す。
    """
    cur_l_max = targets.shape[1]
    if cur_l_max == l_max:
        return targets, mask
    pad = l_max - cur_l_max
    targets = torch.nn.functional.pad(targets, (0, 0, 0, pad))
    mask = torch.nn.functional.pad(mask, (0, pad))
    return targets, mask


def run_train(cfg: Config, device: torch.device) -> None:
    save_path = _save_path(cfg)
    load_path = webui.start_checkpoint(save_path)  # WebUIでは「この run から続きを学習」したときだけ。無ければ最初から
    if load_path is not None:
        print(f"loaded: {load_path}")

    torch.manual_seed(cfg.data.seed)
    checkpoints_all, targets, mask, l_max = BuildDataset(
        num_samples=cfg.data.num_train_samples,
        n_checkpoints=cfg.data.n_checkpoints,
        coord_min=cfg.data.coord_min,
        coord_max=cfg.data.coord_max,
        step=cfg.data.step,
        closed=cfg.data.closed,
        min_triangle_area=cfg.data.min_triangle_area,
    )
    val_checkpoints, val_targets, val_mask, val_l_max = BuildDataset(
        num_samples=cfg.data.num_val_samples,
        n_checkpoints=cfg.data.n_checkpoints,
        coord_min=cfg.data.coord_min,
        coord_max=cfg.data.coord_max,
        step=cfg.data.step,
        closed=cfg.data.closed,
        min_triangle_area=cfg.data.min_triangle_area,
    )
    l_max = max(l_max, val_l_max)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, l_max=l_max))
    targets, mask = _pad_to_l_max(targets, mask, l_max)
    val_targets, val_mask = _pad_to_l_max(val_targets, val_mask, l_max)

    net = PolylineTransformer(
        n_checkpoints=cfg.data.n_checkpoints,
        l_max=cfg.model.l_max,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_encoder_layers=cfg.model.n_encoder_layers,
        n_decoder_layers=cfg.model.n_decoder_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        head_type=cfg.model.head_type,
    )

    # 学習の進捗は train_webui に送る。WebUIから起動した場合はそのrunに、CLIから起動した場合は新しいrunとして記録される
    webui.init(new_run=True, source="train")

    TrainSchedule(
        net=net,
        cfg=cfg,
        checkpoints_all=checkpoints_all,
        targets=targets,
        mask=mask,
        val_checkpoints=val_checkpoints,
        val_targets=val_targets,
        val_mask=val_mask,
        save_path=save_path,
        load_path=load_path,
        device=device,
    )
    webui.log_artifact(save_path, kind="checkpoint")  # 重みの場所をWebUIに登録（本体は送らない）

    webui.finish()

    print(f"saved: {save_path}")


def run_infer(cfg: Config, device: torch.device, ckpt: str | None = None) -> None:
    load_path = ckpt or _save_path(cfg)
    if not Path(load_path).exists():
        raise FileNotFoundError(f"checkpoint がありません: {load_path}（先に --mode train / --ckpt を確認）")
    print(f"infer checkpoint: {load_path}")

    # 推論結果の画像は train_webui に送る。送信先はWebUIからの起動ならそのrun、CLIならgroupで最後に起動されたtrainのrun
    webui.init(source="infer")
    infer_mod.run(load_path, device)
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

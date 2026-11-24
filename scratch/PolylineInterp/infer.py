"""main.pyの--mode inferから呼ばれる推論・可視化ロジック。単体実行のエントリーポイントは
main.pyに一本化しているため、ここにはargparse/if __name__=="__main__"は持たせない。

checkpointに同梱されたconfig（n_checkpoints/l_max等）からモデル構造を復元するため、
手元のconfig.tomlには依存しない（model-dev-conventionの推論規約通り）。
予測はモデルが常にl_max点出力するが、意味があるのはSampleEquidistantPointsが実際に
生成する点数（L_actual）までなので、そこで切り出す。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from train_webui import client as webui

from .checkpoint import load_checkpoint
from .data import GenerateTriangleCheckpoints, SampleEquidistantPoints
from .model import PolylineTransformer

OUTPUT_IMAGE_PATH = "scratch/PolylineInterp/predicted.png"
NUM_SAMPLES_TO_SHOW = 4


def run(load_path: str, device: torch.device) -> None:
    state_dict, cfg = load_checkpoint(load_path, map_location=device)
    if cfg is None:
        raise RuntimeError(f"{load_path} にconfigが同梱されていません（旧形式のcheckpointは非対応）")

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
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    figure, axes = plt.subplots(1, NUM_SAMPLES_TO_SHOW, figsize=(4 * NUM_SAMPLES_TO_SHOW, 4))
    for axis in axes:
        checkpoints = GenerateTriangleCheckpoints(
            cfg.data.coord_min, cfg.data.coord_max, cfg.data.min_triangle_area,
        )
        l_actual = SampleEquidistantPoints(checkpoints, cfg.data.step, cfg.data.closed).shape[0]

        with torch.no_grad():
            pred = net(checkpoints.unsqueeze(0).to(device))[0].cpu()
        pred = pred[:l_actual]

        polygon = torch.cat([checkpoints, checkpoints[:1]]) if cfg.data.closed else checkpoints
        axis.plot(polygon[:, 0], polygon[:, 1], "o--", color="gray", label="checkpoints")
        axis.plot(pred[:, 0], pred[:, 1], "x-", color="tab:blue", label="predicted")
        axis.legend()
        axis.set_aspect("equal")

    figure.tight_layout()
    figure.savefig(OUTPUT_IMAGE_PATH)
    print(f"saved: {OUTPUT_IMAGE_PATH}")
    webui.log_image("infer/predicted", figure)

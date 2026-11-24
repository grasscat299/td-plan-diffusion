import pytest
import torch

from .config import config_from_dict
from .data import (
    BuildDataset,
    GenerateTriangleCheckpoints,
    PolylineArcLength,
    SampleEquidistantPoints,
)
from .model import PolylineTransformer
from .train import MaskedMSELoss, OneStep, TrainSchedule


def _make_cfg(**overrides):
    raw = {
        "data": {
            "n_checkpoints": 3,
            "closed": True,
            "coord_min": -1.0,
            "coord_max": 1.0,
            "step": 0.5,
            "min_triangle_area": 0.05,
            "num_train_samples": 8,
            "num_val_samples": 4,
            "seed": 0,
        },
        "model": {
            "d_model": 8,
            "n_heads": 2,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "dim_feedforward": 16,
            "head_type": "linear",
        },
        "train": {"lr": 1e-2, "epoch": 1, "batch": 4},
        "wandb": {"project": "test", "group": "test"},
    }
    for section, values in overrides.items():
        raw[section].update(values)
    return config_from_dict(raw)


def test_generate_triangle_checkpoints_shape_and_area():
    checkpoints = GenerateTriangleCheckpoints(coord_min=-1.0, coord_max=1.0, min_triangle_area=0.05)
    assert checkpoints.shape == (3, 2)
    assert torch.all(checkpoints >= -1.0) and torch.all(checkpoints <= 1.0)


def test_polyline_arc_length_closed_triangle():
    checkpoints = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    open_length = PolylineArcLength(checkpoints, closed=False)
    closed_length = PolylineArcLength(checkpoints, closed=True)
    assert open_length == 8.0  # 3 + 5
    assert closed_length == 12.0  # 3 + 5 + 4


def test_sample_equidistant_points_matches_segment_interval_counts():
    checkpoints = torch.tensor([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]])
    step = 1.0
    points = SampleEquidistantPoints(checkpoints, step, closed=False)

    # segment1(長さ4): round(4/1)=4区間, segment2(長さ3): round(3/1)=3区間 -> 1(始点) + 4 + 3
    assert points.shape == (8, 2)
    assert torch.allclose(points[0], checkpoints[0])


def test_sample_equidistant_points_includes_checkpoints_even_if_step_does_not_divide_evenly():
    checkpoints = torch.tensor([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]])
    step = 1.5  # 4/1.5, 3/1.5 とも割り切れない

    points = SampleEquidistantPoints(checkpoints, step, closed=False)

    assert torch.any(torch.all(torch.isclose(points, checkpoints[1]), dim=1))
    assert torch.any(torch.all(torch.isclose(points, checkpoints[2]), dim=1))


def test_sample_equidistant_points_handles_segment_shorter_than_step():
    checkpoints = torch.tensor([[0.0, 0.0], [0.1, 0.0], [4.0, 3.0]])
    step = 1.0  # segment1は長さ0.1でstepよりかなり短い

    points = SampleEquidistantPoints(checkpoints, step, closed=False)

    assert torch.any(torch.all(torch.isclose(points, checkpoints[1]), dim=1))
    assert torch.any(torch.all(torch.isclose(points, checkpoints[2]), dim=1))


def test_sample_equidistant_points_includes_all_checkpoints_when_closed():
    checkpoints = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    step = 0.7  # 3, 4, 5(閉路の戻り辺)のいずれもstepで割り切れない

    points = SampleEquidistantPoints(checkpoints, step, closed=True)

    for checkpoint in checkpoints:
        assert torch.any(torch.all(torch.isclose(points, checkpoint), dim=1))


def test_build_dataset_shapes_and_padding():
    checkpoints_all, targets, mask, l_max = BuildDataset(
        num_samples=6, n_checkpoints=3, coord_min=-1.0, coord_max=1.0,
        step=0.3, closed=True, min_triangle_area=0.05,
    )
    assert checkpoints_all.shape == (6, 3, 2)
    assert targets.shape == (6, l_max, 2)
    assert mask.shape == (6, l_max)
    assert mask.dtype == torch.bool
    # padding位置は0であるべき
    assert torch.all(targets[~mask] == 0.0)
    # 各サンプルで最低1点(始点)は有効
    assert torch.all(mask[:, 0])


def test_masked_mse_loss_ignores_padding():
    pred = torch.tensor([[[1.0, 1.0], [5.0, 5.0]]])
    target = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, False]])
    loss = MaskedMSELoss(pred, target, mask)
    assert loss.item() == 0.0


@pytest.mark.parametrize("head_type", ["linear", "interp"])
def test_model_forward_shape(head_type):
    net = PolylineTransformer(
        n_checkpoints=3, l_max=5, d_model=8, n_heads=2,
        n_encoder_layers=1, n_decoder_layers=1, dim_feedforward=16, head_type=head_type,
    )
    checkpoints = torch.rand(4, 3, 2)
    pred = net(checkpoints)
    assert pred.shape == (4, 5, 2)


def test_model_rejects_unknown_head_type():
    with pytest.raises(ValueError):
        PolylineTransformer(
            n_checkpoints=3, l_max=5, d_model=8, n_heads=2,
            n_encoder_layers=1, n_decoder_layers=1, dim_feedforward=16, head_type="no_such_head",
        )


def test_interp_head_output_is_convex_combination_of_checkpoints():
    net = PolylineTransformer(
        n_checkpoints=3, l_max=5, d_model=8, n_heads=2,
        n_encoder_layers=1, n_decoder_layers=1, dim_feedforward=16, head_type="interp",
    )
    checkpoints = torch.rand(4, 3, 2)
    pred = net(checkpoints)

    lower = checkpoints.min(dim=1, keepdim=True).values
    upper = checkpoints.max(dim=1, keepdim=True).values
    assert torch.all(pred >= lower - 1e-5)
    assert torch.all(pred <= upper + 1e-5)


@pytest.mark.parametrize("head_type", ["linear", "interp"])
def test_one_step_updates_weights(head_type):
    net = PolylineTransformer(
        n_checkpoints=3, l_max=5, d_model=8, n_heads=2,
        n_encoder_layers=1, n_decoder_layers=1, dim_feedforward=16, head_type=head_type,
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    checkpoints = torch.rand(4, 3, 2)
    targets = torch.rand(4, 5, 2)
    mask = torch.ones(4, 5, dtype=torch.bool)

    before = net.output_head.weight.clone()
    loss = OneStep(net, optimizer, checkpoints, targets, mask)
    after = net.output_head.weight

    assert loss.item() >= 0.0
    assert not torch.equal(before, after)


def test_train_schedule_runs_and_saves(tmp_path):
    cfg = _make_cfg()
    checkpoints_all, targets, mask, l_max = BuildDataset(
        num_samples=cfg.data.num_train_samples, n_checkpoints=cfg.data.n_checkpoints,
        coord_min=cfg.data.coord_min, coord_max=cfg.data.coord_max, step=cfg.data.step,
        closed=cfg.data.closed, min_triangle_area=cfg.data.min_triangle_area,
    )
    val_checkpoints, val_targets, val_mask, val_l_max = BuildDataset(
        num_samples=cfg.data.num_val_samples, n_checkpoints=cfg.data.n_checkpoints,
        coord_min=cfg.data.coord_min, coord_max=cfg.data.coord_max, step=cfg.data.step,
        closed=cfg.data.closed, min_triangle_area=cfg.data.min_triangle_area,
    )
    l_max = max(l_max, val_l_max)
    if val_targets.shape[1] != l_max:
        pad = l_max - val_targets.shape[1]
        val_targets = torch.nn.functional.pad(val_targets, (0, 0, 0, pad))
        val_mask = torch.nn.functional.pad(val_mask, (0, pad))
    if targets.shape[1] != l_max:
        pad = l_max - targets.shape[1]
        targets = torch.nn.functional.pad(targets, (0, 0, 0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad))

    net = PolylineTransformer(
        n_checkpoints=cfg.data.n_checkpoints, l_max=l_max, d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads, n_encoder_layers=cfg.model.n_encoder_layers,
        n_decoder_layers=cfg.model.n_decoder_layers, dim_feedforward=cfg.model.dim_feedforward,
        head_type=cfg.model.head_type,
    )
    save_path = tmp_path / "model.pt"

    TrainSchedule(
        net=net, cfg=cfg,
        checkpoints_all=checkpoints_all, targets=targets, mask=mask,
        val_checkpoints=val_checkpoints, val_targets=val_targets, val_mask=val_mask,
        save_path=str(save_path), load_path=None, device=torch.device("cpu"),
    )

    assert save_path.exists()

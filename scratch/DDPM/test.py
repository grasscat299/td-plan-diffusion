import pytest
import torch

from .config import config_from_dict
from .data import BuildSwissRoll, RandomPickedData
from .model import build_model
from .model.model_dit import DiTModel
from .model.model_transformer import TransformerModel
from .model.model_unet import UnetModel
from .noise_scheduler import NoiseScheduler
from .train import Loss, OneStep, TrainSchedule

NUM_TIMESTEPS = 10


def _make_cfg():
    return config_from_dict({
        "model": {"type": "transformer", "image_channels": 1, "image_height": 4, "image_width": 4},
        "diffusion": {"num_timesteps": NUM_TIMESTEPS},
        "train": {"lr": 1e-2, "epoch": 1, "steps_per_epoch": 2, "batch": 3},
        "wandb": {"project": "test", "group": "test"},
    })


def test_random_picked_data_samples_t_in_range():
    x0 = torch.zeros(8, 1, 4, 4)
    angle = torch.zeros(8)
    data = RandomPickedData(x0, NUM_TIMESTEPS, angle)
    assert data.t.shape == (8,)
    assert torch.all(data.t >= 1) and torch.all(data.t <= NUM_TIMESTEPS)


def test_build_swiss_roll_shapes():
    x0, angle = BuildSwissRoll(batch=5, c=1, h=16, w=16)
    assert x0.shape == (5, 1, 16, 16)
    assert angle.shape == (5,)
    assert set(torch.unique(x0).tolist()) <= {-1.0, 1.0}


def test_noise_scheduler_add_noise_shapes_and_reconstruction():
    scheduler = NoiseScheduler(NUM_TIMESTEPS)
    x0 = torch.rand(4, 1, 8, 8)
    angle = torch.rand(4) * 6.28
    random_picked_data = RandomPickedData(x0, NUM_TIMESTEPS, angle)

    input_data = scheduler.add_noise(random_picked_data)

    assert input_data.xt.shape == x0.shape
    assert input_data.eps.shape == x0.shape
    assert torch.equal(input_data.t, random_picked_data.t)

    a_bar_t = scheduler.a_t_bar[random_picked_data.t - 1].view(-1, 1, 1, 1)
    expected_xt = a_bar_t.sqrt() * x0 + (1.0 - a_bar_t).sqrt() * input_data.eps
    assert torch.allclose(input_data.xt, expected_xt)


def test_loss_is_mse():
    pred = torch.rand(4, 1, 8, 8)
    eps = torch.rand(4, 1, 8, 8)
    loss = Loss(pred, eps)
    assert loss.item() == torch.nn.functional.mse_loss(pred, eps).item()
    assert Loss(pred, pred).item() == 0.0


def test_model_forward_shape_and_gradient():
    model = TransformerModel(c=1, h=4, w=4)
    x0 = torch.rand(2, 1, 4, 4)
    angle = torch.rand(2) * 6.28
    scheduler = NoiseScheduler(NUM_TIMESTEPS)
    input_data = scheduler.add_noise(RandomPickedData(x0, NUM_TIMESTEPS, angle))

    noise_pred = model(input_data)
    assert noise_pred.shape == x0.shape

    loss = Loss(noise_pred, input_data.eps)
    loss.backward()
    assert model.input_linear.weight.grad is not None


def test_one_step_updates_weights():
    net = TransformerModel(c=1, h=4, w=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    scheduler = NoiseScheduler(NUM_TIMESTEPS)
    x0 = torch.rand(4, 1, 4, 4)
    angle = torch.rand(4) * 6.28
    random_picked_data = RandomPickedData(x0, NUM_TIMESTEPS, angle)

    before = net.input_linear.weight.clone()
    loss = OneStep(net, optimizer, scheduler, random_picked_data)
    after = net.input_linear.weight

    assert loss.item() >= 0.0
    assert not torch.equal(before, after)


def test_train_schedule_runs_and_saves(tmp_path):
    net = TransformerModel(c=1, h=4, w=4)
    scheduler = NoiseScheduler(NUM_TIMESTEPS)
    save_path = tmp_path / "model.pt"

    TrainSchedule(
        net=net,
        noise_scheduler=scheduler,
        cfg=_make_cfg(),
        save_path=str(save_path),
        load_path=None,
        device=torch.device("cpu"),
    )

    assert save_path.exists()


def test_build_model_selects_architecture_by_name():
    assert isinstance(build_model("transformer", c=1, h=4, w=4), TransformerModel)
    assert isinstance(build_model("unet", c=1, h=4, w=4), UnetModel)
    assert isinstance(build_model("dit", c=1, h=8, w=8), DiTModel)

    with pytest.raises(ValueError):
        build_model("no_such_model", c=1, h=4, w=4)


def test_unet_model_forward_shape_and_gradient():
    model = UnetModel(c=1, h=4, w=4)
    x0 = torch.rand(2, 1, 4, 4)
    angle = torch.rand(2) * 6.28
    scheduler = NoiseScheduler(NUM_TIMESTEPS)
    input_data = scheduler.add_noise(RandomPickedData(x0, NUM_TIMESTEPS, angle))

    noise_pred = model(input_data)
    assert noise_pred.shape == x0.shape

    loss = Loss(noise_pred, input_data.eps)
    loss.backward()
    assert model.in_conv.weight.grad is not None


def test_dit_model_forward_shape_and_gradient():
    # patch_size(4) で割り切れて、かつ2x2以上のパッチグリッドになるサイズを使う
    model = DiTModel(c=1, h=8, w=8)
    x0 = torch.rand(2, 1, 8, 8)
    angle = torch.rand(2) * 6.28
    scheduler = NoiseScheduler(NUM_TIMESTEPS)
    input_data = scheduler.add_noise(RandomPickedData(x0, NUM_TIMESTEPS, angle))

    noise_pred = model(input_data)
    assert noise_pred.shape == x0.shape

    loss = Loss(noise_pred, input_data.eps)
    loss.backward()
    assert model.patch_linear.weight.grad is not None


def test_dit_model_rejects_size_not_divisible_by_patch_size():
    with pytest.raises(ValueError):
        DiTModel(c=1, h=5, w=8)

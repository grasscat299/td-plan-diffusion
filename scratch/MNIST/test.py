import pytest
import torch

from scratch.DDPM.data import InputData
from scratch.DDPM.model import build_model
from scratch.DDPM.noise_scheduler import NoiseScheduler
from scratch.VAE.model import VAE

from .config import config_from_dict
from .data import BuildMNISTBatch
from .model_dit import DiTModel
from .train import BuildLatentMNISTBatch, Loss, TrainSchedule, build_net
from .transformer_train import TrainSchedule as PixelTransformerTrainSchedule
from .vae_train import VaeTrainSchedule

NUM_TIMESTEPS = 10


def _make_cfg(kind: str = "latent_dit", **section_overrides: dict) -> object:
    raw = {
        "experiment": {"kind": kind},
        "vae": {
            "image_channels": 1, "latent_height": 4, "latent_width": 4,
            "kl_weight": 0.1, "load_path": "unused",
        },
        "latent_dit": {"dit_variant": "local_dit", "loss_type": "mse"},
        "pixel_transformer": {
            "model_type": "transformer", "image_channels": 1, "image_height": 8, "image_width": 8,
        },
        "diffusion": {"num_timesteps": NUM_TIMESTEPS},
        "train": {"lr": 1e-2, "epoch": 1, "steps_per_epoch": 2, "batch": 3},
        "wandb": {"project": "test", "group": "test"},
    }
    for section, overrides in section_overrides.items():
        raw[section].update(overrides)
    return config_from_dict(raw)


def test_build_mnist_batch_shapes():
    x0, label = BuildMNISTBatch(batch=8, c=1, h=32, w=32)
    assert x0.shape == (8, 1, 32, 32)
    assert label.shape == (8,)
    assert x0.min() >= -1.0 and x0.max() <= 1.0
    assert torch.all(label >= 0.0) and torch.all(label <= 9.0)


def test_build_latent_mnist_batch_shapes():
    vae = VAE(c=1, h=4, w=4)
    z, label = BuildLatentMNISTBatch(vae, batch=3, c_img=1, h=4, w=4, device=torch.device("cpu"))
    assert z.shape == (3, 1, 4, 4)
    assert label.shape == (3,)


def test_build_net_switches_dit_variant():
    cfg_local = _make_cfg(latent_dit={"dit_variant": "local_dit"})
    assert isinstance(build_net(cfg_local), DiTModel)

    cfg_model = _make_cfg(latent_dit={"dit_variant": "model_dit"})
    assert not isinstance(build_net(cfg_model), DiTModel)

    cfg_unknown = _make_cfg(latent_dit={"dit_variant": "no_such_variant"})
    with pytest.raises(ValueError):
        build_net(cfg_unknown)


def test_vae_train_schedule_runs_and_saves(tmp_path):
    cfg = _make_cfg(kind="vae", vae={"latent_height": 2, "latent_width": 2})
    net = VAE(c=1, h=2, w=2)
    save_path = tmp_path / "vae.pt"

    VaeTrainSchedule(
        net=net,
        cfg=cfg,
        save_path=str(save_path),
        load_path=None,
        device=torch.device("cpu"),
    )

    assert save_path.exists()


def test_train_schedule_runs_and_saves(tmp_path):
    cfg = _make_cfg()
    net = build_model("transformer", c=1, h=4, w=4)
    vae = VAE(c=1, h=4, w=4)
    noise_scheduler = NoiseScheduler(NUM_TIMESTEPS)
    save_path = tmp_path / "model.pt"

    TrainSchedule(
        net=net,
        vae=vae,
        noise_scheduler=noise_scheduler,
        cfg=cfg,
        save_path=str(save_path),
        load_path=None,
        device=torch.device("cpu"),
    )

    assert save_path.exists()


def test_loss_switches_between_mse_and_l1():
    noise_pred = torch.tensor([0.0, 2.0])
    eps = torch.tensor([1.0, 1.0])

    assert Loss(noise_pred, eps, "mse").item() == torch.nn.functional.mse_loss(noise_pred, eps).item()
    assert Loss(noise_pred, eps, "l1").item() == torch.nn.functional.l1_loss(noise_pred, eps).item()

    with pytest.raises(ValueError):
        Loss(noise_pred, eps, "no_such_loss")


def test_dit_model_label_embedding_forward_shape_and_gradient():
    model = DiTModel(c=1, h=4, w=4)
    xt = torch.rand(3, 1, 4, 4)
    t = torch.randint(1, NUM_TIMESTEPS + 1, (3,))
    label = torch.randint(0, 10, (3,)).float()
    input_data = InputData(xt=xt, t=t, eps=torch.zeros_like(xt), angle=label)

    noise_pred = model(input_data)
    assert noise_pred.shape == xt.shape

    loss = torch.nn.functional.mse_loss(noise_pred, input_data.eps)
    loss.backward()
    assert model.label_embedding.weight.grad is not None


def test_train_schedule_runs_with_l1_loss(tmp_path):
    cfg = _make_cfg(latent_dit={"loss_type": "l1"})
    net = build_model("transformer", c=1, h=4, w=4)
    vae = VAE(c=1, h=4, w=4)
    noise_scheduler = NoiseScheduler(NUM_TIMESTEPS)
    save_path = tmp_path / "model.pt"

    TrainSchedule(
        net=net,
        vae=vae,
        noise_scheduler=noise_scheduler,
        cfg=cfg,
        save_path=str(save_path),
        load_path=None,
        device=torch.device("cpu"),
    )

    assert save_path.exists()


def test_pixel_transformer_train_schedule_runs_and_saves(tmp_path):
    cfg = _make_cfg(
        kind="pixel_transformer",
        pixel_transformer={"model_type": "transformer", "image_height": 8, "image_width": 8},
    )
    net = build_model("transformer", c=1, h=8, w=8)
    noise_scheduler = NoiseScheduler(NUM_TIMESTEPS)
    save_path = tmp_path / "model.pt"

    PixelTransformerTrainSchedule(
        net=net,
        noise_scheduler=noise_scheduler,
        cfg=cfg,
        save_path=str(save_path),
        load_path=None,
        device=torch.device("cpu"),
    )

    assert save_path.exists()

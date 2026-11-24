import torch

from scratch.DDPM.model import build_model
from scratch.DDPM.noise_scheduler import NoiseScheduler
from scratch.VAE.model import VAE

from .config import config_from_dict
from .train import BuildLatentBatch, TrainSchedule

NUM_TIMESTEPS = 10


def _make_cfg():
    return config_from_dict({
        "vae": {
            "load_path": "scratch/VAE/checkpoints/model_vae_128.pt",
            "image_channels": 1, "latent_height": 4, "latent_width": 4,
        },
        "model": {"type": "transformer"},
        "diffusion": {"num_timesteps": NUM_TIMESTEPS},
        "train": {"lr": 1e-2, "epoch": 1, "steps_per_epoch": 2, "batch": 3},
        "wandb": {"project": "test", "group": "test"},
    })


def test_build_latent_batch_shapes():
    vae = VAE(c=1, h=4, w=4)
    z, angle = BuildLatentBatch(vae, batch=3, c_img=1, h=4, w=4, device=torch.device("cpu"))
    assert z.shape == (3, 1, 4, 4)
    assert angle.shape == (3,)


def test_build_latent_batch_no_grad_for_vae():
    vae = VAE(c=1, h=4, w=4)
    z, _angle = BuildLatentBatch(vae, batch=2, c_img=1, h=4, w=4, device=torch.device("cpu"))
    assert not z.requires_grad


def test_train_schedule_runs_and_saves(tmp_path):
    cfg = _make_cfg()
    net = build_model(cfg.model.type, c=1, h=4, w=4)
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


def test_train_schedule_does_not_update_vae_weights(tmp_path):
    cfg = _make_cfg()
    net = build_model(cfg.model.type, c=1, h=4, w=4)
    vae = VAE(c=1, h=4, w=4)
    noise_scheduler = NoiseScheduler(NUM_TIMESTEPS)
    before = vae.encoder.conv_in.weight.clone()

    TrainSchedule(
        net=net,
        vae=vae,
        noise_scheduler=noise_scheduler,
        cfg=cfg,
        save_path=str(tmp_path / "model.pt"),
        load_path=None,
        device=torch.device("cpu"),
    )

    after = vae.encoder.conv_in.weight
    assert torch.equal(before, after)

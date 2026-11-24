import pytest
import torch

from .config import config_from_dict
from .model import Decoder, Encoder, EncoderOutput, Reparameterize, VAE, VAEOutput
from .train import KLLoss, Loss, OneStep, ReconLoss, TrainSchedule


def _make_cfg():
    return config_from_dict({
        "model": {"image_channels": 1, "latent_height": 2, "latent_width": 2},
        "train": {"lr": 1e-2, "epoch": 1, "steps_per_epoch": 2, "batch": 3, "kl_weight": 0.1},
        "wandb": {"project": "test", "group": "test"},
    })


def test_reparameterize_shape():
    mean = torch.zeros(2, 1, 4, 4)
    logvar = torch.zeros(2, 1, 4, 4)
    z = Reparameterize(mean, logvar)
    assert z.shape == mean.shape


def test_encoder_output_shape():
    encoder = Encoder(c=1, h=4, w=4)
    x = torch.rand(2, 1, 16, 16)
    output = encoder(x)
    assert isinstance(output, EncoderOutput)
    assert output.mean.shape == (2, 1, 4, 4)
    assert output.logvar.shape == (2, 1, 4, 4)


def test_decoder_output_shape():
    decoder = Decoder(c=1, h=4, w=4)
    z = torch.rand(2, 1, 4, 4)
    x_hat = decoder(z)
    assert x_hat.shape == (2, 1, 16, 16)


def test_vae_forward_shape_and_gradient():
    vae = VAE(c=1, h=4, w=4)
    x = torch.rand(2, 1, 16, 16)
    output = vae(x)
    assert isinstance(output, VAEOutput)
    assert output.x_hat.shape == x.shape
    assert output.mean.shape == (2, 1, 4, 4)
    assert output.logvar.shape == (2, 1, 4, 4)

    loss = Loss(output, x, kl_weight=0.1)
    loss.backward()
    assert vae.encoder.conv_in.weight.grad is not None


def test_kl_loss_zero_for_standard_normal():
    mean = torch.zeros(4, 1, 2, 2)
    logvar = torch.zeros(4, 1, 2, 2)
    assert KLLoss(mean, logvar).item() == pytest.approx(0.0, abs=1e-6)


def test_recon_loss_is_l1():
    x_hat = torch.rand(4, 1, 8, 8)
    x = torch.rand(4, 1, 8, 8)
    assert ReconLoss(x_hat, x).item() == torch.nn.functional.l1_loss(x_hat, x).item()
    assert ReconLoss(x, x).item() == 0.0


def test_one_step_updates_weights():
    net = VAE(c=1, h=4, w=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    x = torch.rand(4, 1, 16, 16)

    before = net.encoder.conv_in.weight.clone()
    loss = OneStep(net, optimizer, x, kl_weight=0.1)
    after = net.encoder.conv_in.weight

    assert loss.item() >= 0.0
    assert not torch.equal(before, after)


def test_train_schedule_runs_and_saves(tmp_path):
    net = VAE(c=1, h=2, w=2)
    save_path = tmp_path / "model.pt"

    TrainSchedule(
        net=net,
        cfg=_make_cfg(),
        save_path=str(save_path),
        load_path=None,
        device=torch.device("cpu"),
    )

    assert save_path.exists()

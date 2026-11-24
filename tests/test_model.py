"""乱数の入力で forward を通し、出力の形と、置けない升の扱いを確かめる。"""
from pathlib import Path

import torch

from world_model import ModelInput, NoiseScheduler, PlanDiT, legal_tile_mask, load_config, mask_logits

CFG = Path(__file__).resolve().parents[1] / "world_model" / "config.example.toml"


def random_input(cfg, B=2, n_en=5, K=6, seed=0) -> ModelInput:
    g = torch.Generator().manual_seed(seed)
    s = cfg.shape
    T, n_op, n_r = s.T, s.n_op, s.max_route_points   # 通り道は max_route_points まで詰め物で埋める
    route_pad = torch.arange(n_r).expand(B, n_en, n_r) >= torch.randint(2, 12, (B, n_en, 1), generator=g)
    t_s = torch.randint(0, T, (B, n_en, 1), generator=g)
    enemy_attr = torch.randn(B, n_en, s.enemy_attr_dim, generator=g)
    enemy_attr[:, -1] = 0                                  # 最後の行は敵がいない
    ope_attr = torch.rand(B, n_op, s.ope_attr_dim, generator=g)
    ope_attr[:, n_op // 2:] = 0                            # 後ろ半分は空きの枠
    ope_range = torch.randint(-2, 4, (B, n_op, K, 2), generator=g)
    ope_range[..., -1, :] = 99                             # 詰め物
    return ModelInput(
        plan_xt=torch.randn(B, n_op, s.c_op, generator=g),
        t=torch.randint(1, cfg.diffusion.num_timesteps + 1, (B,), generator=g),
        route=torch.randn(B, n_en, n_r, s.route_dim, generator=g),
        route_padding_mask=route_pad,
        known_en_x0=torch.randn(B, n_en, T, s.en_dim, generator=g),
        known_en_mask=torch.arange(T).view(1, 1, T) == t_s,
        enemy_attr=enemy_attr,
        tile_map=torch.randint(0, s.n_tile_types, (B, s.n_rows, s.n_cols), generator=g),
        tile_feat=torch.rand(B, s.n_tiles, s.tile_feat_dim, generator=g),
        avail=(torch.rand(B, n_op, generator=g) > 0.3).float(),
        ope_attr=ope_attr,
        ope_range=ope_range,
        ope_valid=ope_attr.abs().sum(-1) > 0,
    )


def test_forward_shapes():
    cfg = load_config(CFG)
    model = PlanDiT(cfg).eval()
    x = random_input(cfg)
    with torch.no_grad():
        y = model(x)
    s = cfg.shape
    B, n_en = x.enemy_attr.shape[:2]
    assert y.plan.shape == (B, s.n_op, s.c_op)
    assert y.enemy.shape == (B, n_en, s.T, s.en_dim)
    assert y.resource.shape == (B, s.T, s.n_resource)
    assert y.success.shape == (B,)
    for v in (y.plan, y.enemy, y.resource, y.success):
        assert torch.isfinite(v).all()


def test_legal_mask():
    cfg = load_config(CFG)
    s = cfg.shape
    x = random_input(cfg)
    legal = legal_tile_mask(x.tile_map, x.ope_attr)
    assert legal.shape == (x.tile_map.shape[0], s.n_op, s.n_tiles)
    logits = mask_logits(torch.randn(legal.shape), legal)
    pick = logits.argmax(-1, keepdim=True)
    assert legal.gather(-1, pick).all()


def test_noise_schedule_ends_at_zero():
    cfg = load_config(CFG)
    sch = NoiseScheduler(cfg.diffusion)
    assert sch.a_bar[-1].abs() < 1e-8
    assert torch.all(sch.a_bar[1:] <= sch.a_bar[:-1])

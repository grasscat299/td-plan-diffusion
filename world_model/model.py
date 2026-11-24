"""3 つのタワーの DiT（README の 2 章）。

- 敵タワー: 敵ごと・秒ごとのトークン。拡散しない。学習する問い合わせ（全部の敵で共有）と、出現の時点で分かっている値の埋め込みを、
  分かっている秒だけ入れ替えて作り、1 回の forward で敵の時系列を直接出す。
- 配置タワー: 配置の枠ごとのトークン。計画を拡散し、x0 を予測する。升は、枠のトークンと盤面のトークンの内積（ポインタ）に、
  射程の中を敵が通る延べ時間の項を足した logits で出す。
- 資源タワー: 秒ごとのトークン。配置コストと、枠ごとの HP・SP・スキルの発動回数の時系列を出す。

各層で、3 つのタワーが並列に相手を読む（敵 ← 通り道・配置、配置 ← 敵・資源（時間方向に圧縮）と盤面、資源 ← 配置）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import PLAN_DIR0, PLAN_TILE0, Config
from .inputs import ModelInput, ModelOutput
from .layers import (RANGE_GRID_N, BoardCA, FinalLayer, MapEncoder, RouteKVEncoder, TimeCompress, TowerBlock,
                     range_cov, range_grid, sinusoidal_embedding)


class PlanDiT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        s, m = cfg.shape, cfg.model
        d = m.d
        self.shape, self.d, self.t_c = s, d, m.t_c
        L = m.num_layers

        # 敵タワーの埋め込み
        self.query_en = nn.Parameter(torch.randn(s.T, d) * 0.02)
        self.known_linear_en = nn.Linear(s.en_dim, d)
        self.attr_lin_en = nn.Linear(s.enemy_attr_dim, d)
        # 配置タワーの埋め込み
        self.token_lin_op = nn.Linear(s.c_op, d)
        self.avail_lin = nn.Linear(s.n_op, d)
        self.ope_attr_lin = nn.Linear(s.ope_attr_dim, d)
        self.range_lin = nn.Linear(RANGE_GRID_N, d, bias=False)
        nn.init.zeros_(self.range_lin.weight)
        # キャラクターの属性の平均・標準偏差（学習データから求めて入れる。checkpoint に入る）
        self.register_buffer("ope_attr_mu", torch.zeros(s.ope_attr_dim))
        self.register_buffer("ope_attr_sd", torch.ones(s.ope_attr_dim))
        # 資源タワーの埋め込み
        self.query_dp = nn.Parameter(torch.randn(s.T, d) * 0.02)

        self.register_buffer("pe_time", sinusoidal_embedding(torch.arange(s.T), d))
        self.register_buffer("pe_op", sinusoidal_embedding(torch.arange(s.n_op), d))
        self.register_buffer("pe_tc", sinusoidal_embedding(torch.arange(m.t_c), d))
        self.step_mlp = nn.Sequential(nn.Linear(d, m.cond_mlp_mult * d), nn.SiLU(), nn.Linear(m.cond_mlp_mult * d, d))

        # 条件の符号化
        self.route_encoder = RouteKVEncoder(d, s.max_route_points, m.route_num_layers, m.route_num_heads, m.ffn_mult, s.route_dim)
        self.map_encoder = MapEncoder(d, s.n_rows, s.n_cols, m.num_heads, m.ffn_mult, s.n_tile_types, s.tile_feat_dim)

        # 3 つのタワー
        self.en_blocks = nn.ModuleList([TowerBlock(d, m.num_heads, m.ffn_mult) for _ in range(L)])
        self.op_blocks = nn.ModuleList([TowerBlock(d, m.num_heads, m.ffn_mult, m.op_dropout) for _ in range(L)])
        self.board_blocks = nn.ModuleList([BoardCA(d, m.num_heads, m.op_dropout) for _ in range(L)])
        self.dp_blocks = nn.ModuleList([TowerBlock(d, m.num_heads, m.ffn_mult) for _ in range(L)])
        self.compress = nn.ModuleList([TimeCompress(s.T, m.t_c) for _ in range(L)])
        self.compress_dp = nn.ModuleList([TimeCompress(s.T, m.t_c) for _ in range(L)])

        # 出力
        self.final_en = FinalLayer(d, s.en_dim)
        self.final_op = FinalLayer(d, PLAN_TILE0 + s.n_skill_cols)   # 升以外（時刻・使うか・向き・スキル）
        self.final_dp = FinalLayer(d, s.n_resource)
        self.ptr_q = nn.Linear(d, d)
        self.ptr_type = nn.Linear(s.ope_attr_dim, d, bias=False)
        self.ptr_k = nn.Linear(d, d)
        self.cov_w = nn.Parameter(torch.ones(1))
        self.success_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def set_ope_attr_stats(self, mean, std) -> None:
        self.ope_attr_mu.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.ope_attr_sd.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3))

    def _ope_attr_in(self, a: torch.Tensor) -> torch.Tensor:
        """属性を (a - 平均) / 標準偏差 にして [-4, 4] で切る。空きの枠（全部 0 の行）は 0 のまま。"""
        valid = (a.abs().sum(dim=-1, keepdim=True) > 0).to(a.dtype)
        z = ((a - self.ope_attr_mu.to(a.dtype)) / self.ope_attr_sd.to(a.dtype)).clamp(-4.0, 4.0)
        return z * valid

    def forward(self, x: ModelInput) -> ModelOutput:
        s, d, t_c = self.shape, self.d, self.t_c
        B, N_en, T, _ = x.known_en_x0.shape
        N_op = x.plan_xt.shape[1]
        attr = self._ope_attr_in(x.ope_attr.float())

        # 埋め込み
        known = self.known_linear_en(x.known_en_x0) + self.pe_time
        query = self.query_en.expand(B, N_en, T, d)
        h_en = torch.where(x.known_en_mask.unsqueeze(-1), known, query).reshape(B * N_en, T, d)
        h_en = h_en + self.attr_lin_en(x.enemy_attr).reshape(B * N_en, 1, d)
        h_op = self.token_lin_op(x.plan_xt) + self.pe_op
        h_op = h_op + self.avail_lin(x.avail).unsqueeze(1)
        h_op = h_op + self.ope_attr_lin(attr)
        h_op = h_op + self.range_lin(range_grid(x.ope_range))
        h_dp = self.query_dp.expand(B, T, d) + self.pe_time
        map_kv = self.map_encoder(x.tile_map, x.tile_feat)                         # (B, R*C, d)

        en_empty = x.enemy_attr.abs().sum(dim=-1) == 0                              # (B, N_en) 敵がいない行
        op_pad = ~x.ope_valid                                                       # (B, N_op) 使わない枠
        c = self.step_mlp(sinusoidal_embedding(x.t, d))                             # (B, d)
        c_en = c.repeat_interleave(N_en, dim=0)

        route = x.route.reshape(B * N_en, *x.route.shape[2:])
        r_pad = x.route_padding_mask.reshape(B * N_en, -1)
        route_kv = self.route_encoder(route, r_pad)                                 # (B*N_en, N_r, d)
        kv_en_pad = torch.cat([r_pad, op_pad.repeat_interleave(N_en, dim=0)], dim=1)
        pe_tc_en = self.pe_tc.repeat(N_en, 1).unsqueeze(0)
        kv_op_pad = torch.cat([en_empty.repeat_interleave(t_c, dim=1),
                               torch.zeros(B, t_c, dtype=torch.bool, device=en_empty.device)], dim=1)

        for l in range(len(self.en_blocks)):
            p_en = self.en_blocks[l].params(c_en)
            p_op = self.op_blocks[l].params(c)
            p_dp = self.dp_blocks[l].params(c)
            h_en_sa = self.en_blocks[l].sa(h_en, p_en)
            h_op_sa = self.op_blocks[l].sa(h_op, p_op, op_pad)
            h_dp_sa = self.dp_blocks[l].sa(h_dp, p_dp)

            enemy_kv = self.compress[l](h_en_sa).reshape(B, N_en * t_c, d) + pe_tc_en
            dp_kv = self.compress_dp[l](h_dp_sa) + self.pe_tc.unsqueeze(0)
            op_bc = h_op_sa.unsqueeze(1).expand(B, N_en, N_op, d).reshape(B * N_en, N_op, d)

            h_dp = self.dp_blocks[l].ca(h_dp_sa, p_dp, h_op_sa, op_pad)
            h_op = self.op_blocks[l].ca(h_op_sa, p_op, torch.cat([enemy_kv, dp_kv], dim=1), kv_op_pad)
            h_op = self.board_blocks[l](h_op, c, map_kv)
            h_en = self.en_blocks[l].ca(h_en_sa, p_en, torch.cat([route_kv, op_bc], dim=1), kv_en_pad)

            h_op = self.op_blocks[l].ff(h_op, p_op)
            h_en = self.en_blocks[l].ff(h_en, p_en)
            h_dp = self.dp_blocks[l].ff(h_dp, p_dp)

        enemy = self.final_en(h_en, c_en).reshape(B, N_en, T, s.en_dim)
        out = self.final_op(h_op, c)
        head, skill = out[..., :PLAN_TILE0], out[..., PLAN_TILE0:]

        # 升の logits: ポインタ（枠のトークンと盤面のトークンの内積）+ 射程の中を敵が通る延べ時間（向きの確率で期待値）
        q = self.ptr_q(h_op) + self.ptr_type(attr)
        logits = torch.einsum("bod,bnd->bon", q, self.ptr_k(map_kv)) / (d ** 0.5)
        tile_pass = torch.expm1(x.tile_feat[..., s.cov_channel].clamp_min(0.0))
        cov = range_cov(tile_pass, x.ope_range, s.n_rows, s.n_cols)                  # (B, N_op, 4, R*C)
        p_dir = torch.softmax(head[..., PLAN_DIR0:PLAN_TILE0], dim=-1)
        logits = logits + self.cov_w * torch.log1p((p_dir.unsqueeze(-1) * cov).sum(dim=2)).to(logits.dtype)
        plan = torch.cat([head, logits, skill], dim=-1)

        resource = self.final_dp(h_dp, c)

        # 成功の予測: 配置（使う枠）と敵（いる行）の表現の平均から
        w = (~op_pad).to(h_op.dtype).unsqueeze(-1)
        op_mean = (h_op * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        we = (~en_empty).to(h_en.dtype).view(B, N_en, 1, 1)
        en_mean = (h_en.reshape(B, N_en, T, d) * we).sum(dim=(1, 2)) / (we.sum(dim=(1, 2)) * T).clamp_min(1.0)
        success = self.success_head(torch.cat([en_mean, op_mean], dim=-1)).squeeze(-1)
        return ModelOutput(plan=plan, enemy=enemy, resource=resource, success=success)

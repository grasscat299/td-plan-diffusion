"""README の図を作る。値は評価の report.json（pass@8 の点推定と 95% bootstrap 区間）から写した物。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import glob
import os
from matplotlib import font_manager

# 日本語のフォント（Noto Sans JP）を登録する。無ければ既定のフォントで描く
for _d in ("/usr/share/fonts", os.path.expanduser("~/.local/share/fonts")):
    for _f in glob.glob(f"{_d}/**/NotoSans*JP*", recursive=True):
        font_manager.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Noto Sans JP"

SURFACE, INK, INK2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"

# 保留中の 132 件（知らない面・知らない編成）× 1 件 32 本の生成から求めた pass@8
rows = [
    ("B1: 手動スキルを押す秒も出す\n（自動スキルの列は 0）", 0.373, 0.302, 0.448),
    ("+ 推論の条件に、キャラクターごとの\n自動スキルの使用率（学習データから）", 0.444, 0.369, 0.521),
    ("B25: 敵・キャラクターの特徴量と\n射程の格子のトークンを追加", 0.466, 0.387, 0.543),
]

fig, ax = plt.subplots(figsize=(8, 3.4), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ys = list(range(len(rows)))[::-1]
for y, (label, v, lo, hi) in zip(ys, rows):
    ax.plot([lo, hi], [y, y], color=SERIES, linewidth=2, solid_capstyle="round")
    ax.scatter([v], [y], s=70, color=SERIES, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.text(hi + 0.012, y, f"{v:.3f}", va="center", ha="left", color=INK, fontsize=10)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], color=INK, fontsize=9.5)
ax.set_xlim(0.25, 0.62)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("pass@8（8 本の計画のどれかで勝てた割合）・横線は 95% 区間", color=INK2, fontsize=9)
ax.tick_params(axis="x", colors=INK2, labelsize=9)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_title("知らない面・知らない編成 132 件での pass@8", loc="left", color=INK, fontsize=11.5, pad=10)
fig.tight_layout()
fig.savefig("docs/images/pass8.png", facecolor=SURFACE)
print("saved docs/images/pass8.png")

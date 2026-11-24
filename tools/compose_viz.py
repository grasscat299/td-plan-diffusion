"""可視化のツールのスクショから、README の図を組む（切り抜いて並べ、見出しと凡例を付ける）。
使い方: python compose_viz.py pred <199 秒の png> <221 秒の png> <出力>
        python compose_viz.py plans <png|x0,y0,x1,y1;…> <見出し;…> <出力>"""
import glob
import os
import sys

from PIL import Image, ImageDraw, ImageFont

BG, INK, INK2 = (252, 252, 251), (11, 11, 11), (82, 81, 78)


def font(size, bold=False):
    for d in ("/usr/share/fonts", os.path.expanduser("~/.local/share/fonts")):
        for f in glob.glob(f"{d}/**/NotoSansJP*", recursive=True):
            ft = ImageFont.truetype(f, size)
            try:
                ft.set_variation_by_name("Bold" if bold else "Regular")
            except Exception:
                pass
            return ft
    return ImageFont.load_default()


def panel_row(crops, heads, title, legend, out, gap=24, pad=28):
    w = sum(c.width for c in crops) + gap * (len(crops) - 1) + pad * 2
    h_title, h_head = 44, 40
    h = pad + h_title + h_head + max(c.height for c in crops) + 16 + 30 * len(legend) + pad
    img = Image.new("RGB", (w, h), BG)
    g = ImageDraw.Draw(img)
    g.text((pad, pad), title, font=font(26, True), fill=INK)
    x, y0 = pad, pad + h_title
    for c, hd in zip(crops, heads):
        g.text((x, y0 + 4), hd, font=font(21, True), fill=INK)
        img.paste(c, (x, y0 + h_head))
        x += c.width + gap
    y = y0 + h_head + max(c.height for c in crops) + 16
    for line in legend:
        g.text((pad, y), line, font=font(19), fill=INK2)
        y += 30
    img.save(out, optimize=True)
    print("saved", out, img.size)


mode = sys.argv[1]
if mode == "pred":
    a, b, out = sys.argv[2:5]
    box = (8, 4, 780, 520)
    crops = [Image.open(a).convert("RGB").crop(box), Image.open(b).convert("RGB").crop(box)]
    panel_row(
        crops,
        ["199.6 秒", "221.4 秒"],
        "敵タワーの予測（シアン）と、シミュレーターでの実際の敵（赤）",
        ["赤い升 = 敵の出現、青い升 = 拠点、丸 = 置いたキャラクター（白い三角は向き）、薄い色の升 = 射程",
         "予測では、残りの 4 体は 198〜204 秒に盤面の左側で倒される。実際には 3 体は 197〜199 秒に中央で倒され、",
         "1 体（177 秒に出現）は倒されずに 227 秒に右上の拠点に入り、この計画は負けた"],
        out,
    )
elif mode == "plans":
    boxes, heads, out = sys.argv[2:5]
    crops = []
    for b in boxes.split(";"):
        src, box = b.split("|")
        crops.append(Image.open(src).convert("RGB").crop(tuple(int(v) for v in box.split(","))))
    crops = [c.resize((c.width * 3 // 2, c.height * 3 // 2), Image.LANCZOS) for c in crops]
    panel_row(
        crops,
        heads.split(";"),
        "1 件の条件から生成した計画のうち、勝った 4 本（学習に使っていない面）",
        ["丸 = 置いたキャラクター、白い三角 = 向き、薄い色の升 = 射程、赤い細い線 = シミュレーターでの敵の通り道",
         "この件では 32 本中 24 本が勝ち、勝った本の配置（キャラクターと升の組）は 19 通りあった"],
        out,
    )

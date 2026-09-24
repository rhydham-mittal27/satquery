"""Regenerate the README figures from the real pipeline output.

Run from the repo root:  uv run python docs/make_figures.py
Needs the datasets described in the README (RSVQA-LR, OSCD pairs).
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "datasets"
OUT = ROOT / "docs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vqa"))

from change.diff import detect_change  # noqa: E402
from grounding import bbox_from_heatmap, compute_heatmap, render_grounding_overlay  # noqa: E402

BG = (13, 19, 32)
INK = (226, 234, 246)
GOLD = (240, 161, 60)


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def panel_row(panels, titles, height=340, pad=14, title_h=34, caption=None):
    scaled = []
    for img in panels:
        w = round(img.width * height / img.height)
        scaled.append(img.convert("RGB").resize((w, height), Image.LANCZOS))
    total_w = sum(i.width for i in scaled) + pad * (len(scaled) + 1)
    cap_h = 40 if caption else 0
    canvas = Image.new("RGB", (total_w, height + title_h + pad * 2 + cap_h), BG)
    draw = ImageDraw.Draw(canvas)
    x = pad
    for img, title in zip(scaled, titles):
        draw.text((x, pad), title, fill=INK, font=font(17))
        canvas.paste(img, (x, pad + title_h))
        x += img.width + pad
    if caption:
        draw.text((pad, height + title_h + pad + 8), caption, fill=(150, 160, 178), font=font(14))
    return canvas


def mask_image(mask, size):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = GOLD
    return Image.fromarray(rgb).resize(size, Image.NEAREST)


def change_figure(idx, filename, note):
    d = DATA / "OSCD" / "pairs"
    before = Image.open(d / f"pair_{idx:02d}_before.png")
    after = Image.open(d / f"pair_{idx:02d}_after.png")
    gt = np.array(Image.open(d / f"pair_{idx:02d}_mask.png")) > 0
    pred = detect_change(before, after).mask

    tp = (pred & gt).sum()
    iou = tp / ((pred | gt).sum())
    caption = (
        f"OSCD test pair {idx:02d}. Ground truth {gt.mean()*100:.1f}% changed, predicted "
        f"{pred.mean()*100:.1f}%, IoU {iou:.2f} on this pair (mean over all 10: 0.27). {note}"
    )
    fig = panel_row(
        [before, after, mask_image(gt, before.size), mask_image(pred, before.size)],
        ["Before", "After", "Ground truth change", "Predicted change"],
        caption=caption,
    )
    fig.save(OUT / filename)


def grounding_figure():
    img = Image.open(DATA / "RSVQA-LR" / "Images_LR" / "0.tif").convert("RGB")
    panels, titles = [img], ["Input scene"]
    for question, target in (("Is there a water area?", "water"), ("Is there a forest?", "vegetation")):
        box = bbox_from_heatmap(compute_heatmap(img, target))
        data_url = render_grounding_overlay(img, box, target)
        panels.append(Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))))
        titles.append(f'"{question}" -> yes')
    fig = panel_row(
        panels,
        titles,
        caption="RSVQA-LR tile 0. Boxes come from RGB spectral indices (blue-dominance for water, greenness for vegetation), not a learned detector.",
    )
    fig.save(OUT / "vqa_grounding.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    change_figure(4, "change_detection.png", "Best of the 10.")
    change_figure(6, "change_detection_failure.png", "One of the 3 weakest (with pairs 01, 02).")
    grounding_figure()
    print("wrote figures to", OUT)

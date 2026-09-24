"""Change-detection VQA: answer_change(before_path, after_path, question) -> dict.

No trained model — deterministic diff-mask (change/diff.py) turned into a
templated natural-language answer (change/templates.py). Also renders a
before/after overlay preview so the evidence is visually checkable.
"""
import io
import base64

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from change.diff import detect_change
from change.templates import describe_change, short_headline

GOLD = (240, 161, 60)
PREVIEW_MIN_SIZE = 768


def _stretch_for_display(img: Image.Image) -> Image.Image:
    """Per-channel percentile contrast stretch + upscale, display-only."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    out = np.empty_like(arr)
    for c in range(arr.shape[-1]):
        chan = arr[..., c]
        lo, hi = np.percentile(chan, [2, 98])
        out[..., c] = chan if hi <= lo else np.clip((chan - lo) / (hi - lo), 0, 1) * 255
    img = Image.fromarray(out.astype(np.uint8))

    if max(img.size) < PREVIEW_MIN_SIZE:
        scale = PREVIEW_MIN_SIZE / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
    return img


def _draw_overlay(after: Image.Image, mask: np.ndarray, regions, orig_size) -> Image.Image:
    """Render the diff mask + region boxes on a display-quality (stretched/upscaled) image."""
    display_img = _stretch_for_display(after)
    scale_x = display_img.width / orig_size[0]
    scale_y = display_img.height / orig_size[1]

    mask_img = Image.fromarray((mask * 90).astype(np.uint8)).resize(display_img.size, Image.NEAREST)
    tint = Image.new("RGBA", display_img.size, GOLD + (0,))
    tint.putalpha(mask_img)
    img = Image.alpha_composite(display_img.convert("RGBA"), tint)
    draw = ImageDraw.Draw(img)

    for r in regions:
        x0, y0 = r.x * scale_x, r.y * scale_y
        x1, y1 = (r.x + r.w) * scale_x, (r.y + r.h) * scale_y
        draw.rectangle([x0, y0, x1, y1], outline=GOLD, width=3)
        draw.text((x0 + 4, max(0, y0 - 14)), r.id, fill=GOLD)

    return img.convert("RGB")


def answer_change(before_path: str, after_path: str, question: str) -> dict:
    before = Image.open(before_path)
    after = Image.open(after_path)

    result = detect_change(before, after)
    answer = describe_change(result, question)
    headline = short_headline(result, question)
    overlay = _draw_overlay(after, result.mask, result.regions, before.size)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    overlay_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    regions_out = [
        {
            "id": r.id,
            "area_pct": round(r.area_frac * 100, 2),
            "greenness_delta": round(r.greenness_delta, 3),
        }
        for r in result.regions
    ]

    return {
        "answer": answer,
        "headline": headline,
        "pct_changed": round(result.pct_changed, 2),
        "regions": regions_out,
        "overlay": overlay_b64,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vqa"))
    from prepare_data import IMAGES_DIR  # noqa: E402

    before_path = str(IMAGES_DIR / "0.tif")
    after_path = str(IMAGES_DIR / "1.tif")
    result = answer_change(before_path, after_path, "How much area changed?")
    print(result["answer"])
    print("pct_changed:", result["pct_changed"])
    print("regions:", result["regions"])

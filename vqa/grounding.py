"""Visual grounding for VQA answers using deterministic spectral indices.

CLIP-based patch-level zero-shot grounding was tried first (3 configs:
ViT-B/32 single prompt, ViT-B/32 5-prompt ensemble, ViT-B/16 finer grid)
and rejected — all three confidently highlighted the *wrong* region on
real test images (e.g. scored real water lowest in the grid, highest on
farmland). See satquery notes / commit history for the evidence. Dense
zero-shot CLIP similarity is unreliable on satellite imagery, which is
outside its training distribution.

This module instead computes real RGB-only spectral indices — the same
approach already validated for change detection's greenness proxy — so
grounding is transparent and checkable, not a black-box heatmap. It
only covers question types with a genuine spectral signal (water,
vegetation); other question types (buildings, roads, counts) return no
grounding rather than a fabricated box.
"""
import io
import base64
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

GOLD = (240, 161, 60)

WATER_TERMS = ["water area", "water"]
VEGETATION_TERMS = ["grass area", "grass", "forest", "farmland", "field"]


def extract_grounding_target(question: str) -> str | None:
    """Returns 'water', 'vegetation', or None (no reliable index available)."""
    q = question.lower()
    for term in WATER_TERMS:
        if term in q:
            return "water"
    for term in VEGETATION_TERMS:
        if term in q:
            return "vegetation"
    return None


def _water_index(rgb: np.ndarray) -> np.ndarray:
    """High where blue dominates over red+green — water/dark-blue areas.
    Same family of formula as the greenness index, just for blue."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (2 * b - r - g) / (2 * b + r + g + 1e-6)


def _greenness_index(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (2 * g - r - b) / (2 * g + r + b + 1e-6)


def compute_heatmap(image: Image.Image, target: str) -> np.ndarray:
    """Returns an (H, W) float heatmap in [0, 1]."""
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    index = _water_index(arr) if target == "water" else _greenness_index(arr)

    index = index - index.min()
    if index.max() > 0:
        index = index / index.max()
    return index


@dataclass
class GroundingBox:
    x: int
    y: int
    w: int
    h: int


def bbox_from_heatmap(heatmap: np.ndarray, percentile: float = 90.0) -> GroundingBox | None:
    threshold = np.percentile(heatmap, percentile)
    mask = heatmap > threshold
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))

    labeled, num = ndimage.label(mask)
    if num == 0:
        return None

    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labeled == biggest)
    return GroundingBox(x=int(xs.min()), y=int(ys.min()), w=int(xs.max() - xs.min() + 1), h=int(ys.max() - ys.min() + 1))


def render_grounding_overlay(image: Image.Image, box: GroundingBox, label: str) -> str:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([box.x, box.y, box.x + box.w, box.y + box.h], outline=GOLD, width=3)
    draw.text((box.x + 4, max(0, box.y - 14)), label, fill=GOLD)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

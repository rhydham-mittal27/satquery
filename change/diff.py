"""Diff-mask change detection between two aligned satellite images.

Two mask sources, both feeding the same region-extraction/stats logic:
- Trained: a RandomForest classifier (train_change_classifier.py)
  trained on OSCD's real 14-pair training set, evaluated on its 10-pair
  test set. Mean IoU 0.268 vs 0.205 for the unsupervised baseline (see
  change/EVAL.md) — used automatically when change/trained_classifier.joblib
  exists.
- Fallback: Change Vector Analysis + Otsu threshold (unsupervised,
  no training data needed) — used if the trained model isn't present.

Kept separate from templates.py (which turns these stats into an answer)
so the numeric pipeline stays independently testable/verifiable.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.exposure import match_histograms
from skimage.filters import threshold_otsu

from change.features import extract_features

_TRAINED_MODEL_PATH = Path(__file__).resolve().parent / "trained_classifier.joblib"
_trained_model = None
_trained_model_loaded = False


def _get_trained_model():
    global _trained_model, _trained_model_loaded
    if not _trained_model_loaded:
        _trained_model_loaded = True
        if _TRAINED_MODEL_PATH.exists():
            import joblib

            _trained_model = joblib.load(_TRAINED_MODEL_PATH)
    return _trained_model


@dataclass
class ChangeRegion:
    id: str
    x: int
    y: int
    w: int
    h: int
    area_frac: float  # fraction of full image area
    greenness_delta: float  # after - before greenness (proxy for vegetation change)


@dataclass
class ChangeResult:
    pct_changed: float
    regions: list[ChangeRegion]
    mask: np.ndarray  # boolean, shape (H, W)


def _greenness(rgb: np.ndarray) -> np.ndarray:
    """Cheap vegetation-greenness proxy from RGB only (no NIR band available)."""
    r, g, b = (
        rgb[..., 0].astype(np.float32),
        rgb[..., 1].astype(np.float32),
        rgb[..., 2].astype(np.float32),
    )
    return (2 * g - r - b) / (2 * g + r + b + 1e-6)


def detect_change(
    before: Image.Image,
    after: Image.Image,
    min_region_frac: float = 0.005,
) -> ChangeResult:
    """Compare two images of the same scene and localize changed regions.

    Uses Change Vector Analysis (CVA) + Otsu thresholding — classical
    unsupervised change-detection technique (see change/EVAL.md for the
    comparison against a plain mean+std threshold and against PCA+k-means,
    both of which this outperformed on real OSCD ground truth).
    """
    size = before.size
    after = after.resize(size) if after.size != size else after

    before_arr = np.asarray(before.convert("RGB")).astype(np.float32)
    after_arr = np.asarray(after.convert("RGB")).astype(np.float32)

    model = _get_trained_model()
    if model is not None:
        feats = extract_features(before, after)
        h, w = feats.shape[:2]
        probs = model.predict_proba(feats.reshape(-1, feats.shape[-1]))[:, 1]
        mask = (probs > 0.5).reshape(h, w)
    else:
        # Two acquisitions of the same place still differ globally (sun
        # angle, season, sensor calibration) even where nothing actually
        # changed. Match "after"'s per-channel histogram to "before" before
        # differencing so global illumination/color shift isn't read as
        # change, then threshold the CVA magnitude with Otsu instead of a
        # fixed mean+k*std heuristic.
        after_matched = match_histograms(after_arr, before_arr, channel_axis=-1)
        diff_vec = after_matched - before_arr
        magnitude = np.sqrt((diff_vec**2).sum(axis=-1))
        otsu_threshold = threshold_otsu(magnitude)
        mask = magnitude > otsu_threshold

    # Clean up isolated noise pixels before measuring regions.
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))

    labeled, num_features = ndimage.label(mask)
    h, w = mask.shape
    total_area = h * w
    min_pixels = min_region_frac * total_area

    before_green = _greenness(before_arr)
    after_green = _greenness(after_arr)

    regions = []
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < min_pixels:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        region_mask = labeled == i
        green_delta = float(
            after_green[region_mask].mean() - before_green[region_mask].mean()
        )
        regions.append(
            ChangeRegion(
                id=chr(ord("A") + len(regions)),
                x=int(x0),
                y=int(y0),
                w=int(x1 - x0 + 1),
                h=int(y1 - y0 + 1),
                area_frac=len(xs) / total_area,
                greenness_delta=green_delta,
            )
        )
        if len(regions) >= 8:
            break

    pct_changed = float(mask.sum() / total_area * 100)
    return ChangeResult(pct_changed=pct_changed, regions=regions, mask=mask)

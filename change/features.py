"""Per-pixel feature extraction for the trained change classifier.

RGB-only by design (no NIR) so it works on any uploaded image at
inference time, not just multi-band Sentinel-2 tiles like OSCD's own
training data. Shared between train_change_classifier.py and
change/diff.py (trained-model path).

Kept to this original 6-feature set deliberately: expanding to 10
features (multi-scale magnitude + local NCC) was tried and evaluated
against real OSCD ground truth — it consistently performed *worse*
(mean IoU 0.254 vs 0.269), independent of training-set size, i.e. pure
overfitting on this problem, not an underfitting issue. See
change/EVAL.md.
"""
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.exposure import match_histograms

N_FEATURES = 6
FEATURE_NAMES = ["cva_magnitude", "r_diff", "g_diff", "b_diff", "greenness_delta", "local_mean_magnitude"]


def _greenness(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (2 * g - r - b) / (2 * g + r + b + 1e-6)


def extract_features(before: Image.Image, after: Image.Image) -> np.ndarray:
    """Returns an (H, W, N_FEATURES) float32 array."""
    size = before.size
    after = after.resize(size) if after.size != size else after

    before_arr = np.asarray(before.convert("RGB")).astype(np.float32)
    after_arr = np.asarray(after.convert("RGB")).astype(np.float32)
    after_matched = match_histograms(after_arr, before_arr, channel_axis=-1)

    diff_vec = after_matched - before_arr
    magnitude = np.sqrt((diff_vec**2).sum(axis=-1))
    local_mean_mag = ndimage.uniform_filter(magnitude, size=5)

    green_before = _greenness(before_arr)
    green_after = _greenness(after_arr)
    green_delta = green_after - green_before

    feats = np.stack(
        [
            magnitude,
            diff_vec[..., 0],
            diff_vec[..., 1],
            diff_vec[..., 2],
            green_delta,
            local_mean_mag,
        ],
        axis=-1,
    ).astype(np.float32)
    return feats

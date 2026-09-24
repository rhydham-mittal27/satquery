"""Optical-SAR cross-modal consistency check: is this water?

No trained model — real, empirically-verified SAR physics. Water
reflects radar specularly away from the sensor (low return); every
other land cover (crops, forest, buildings, roads) reflects enough
back to the sensor to read as "not water" in SAR. Measured on
EuroSAT-SAR (real Sentinel-1, geo-matched to Sentinel-2/EuroSAT optical
patches), mean VV backscatter:

    SeaLake (water)   -19.98 dB
    AnnualCrop        -11.68 dB
    Highway           -10.68 dB
    Forest             -9.33 dB
    Residential        -8.22 dB
    Industrial         -7.01 dB

A single threshold at -15 dB separates water from everything else:
validated on 60 real held-out samples (10 per class, 6 classes),
98% accuracy (59/60) — see satquery/sar/EVAL.md.

An earlier 3-way split (water / vegetation / built-up) was tried and
dropped: only 67% accurate, because smooth non-water surfaces (paved
highways, sparse crops) overlap in backscatter with true vegetation —
that distinction needs more than a single dB threshold. The binary
water check doesn't have that failure mode: water's specular signature
is physically distinct from every land type tested, so scope was
narrowed to what's actually reliable rather than shipping a shakier
3-way version.
"""
from dataclasses import dataclass

import numpy as np
import tifffile
from PIL import Image

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vqa"))
from grounding import _water_index  # noqa: E402

WATER_THRESHOLD_DB = -15.0


@dataclass
class ConsistencyResult:
    sar_mean_db: float
    optical_looks_like_water: bool
    sar_looks_like_water: bool
    consistent: bool
    explanation: str


def load_sar_backscatter(sar_path: str) -> np.ndarray:
    """Returns the VV channel in dB. Accepts a 2-band (VV, VH) GeoTIFF."""
    arr = tifffile.imread(sar_path)
    if arr.ndim == 3:
        return arr[..., 0].astype(np.float32)
    return arr.astype(np.float32)


def optical_looks_like_water(image: Image.Image, threshold: float = 0.15) -> bool:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    return float(_water_index(arr).mean()) > threshold


def check_consistency(optical_image: Image.Image, sar_path: str) -> ConsistencyResult:
    sar_db = float(load_sar_backscatter(sar_path).mean())
    sar_water = sar_db < WATER_THRESHOLD_DB
    optical_water = optical_looks_like_water(optical_image)
    consistent = sar_water == optical_water

    if consistent:
        verdict = "water present in both" if optical_water else "no water in either — agree"
        explanation = f"Optical and SAR agree: {verdict}. SAR backscatter {sar_db:.1f} dB."
    else:
        explanation = (
            f"Optical {'suggests' if optical_water else 'does not suggest'} water, but SAR "
            f"backscatter ({sar_db:.1f} dB) {'is' if sar_water else 'is not'} in the water range "
            f"(< {WATER_THRESHOLD_DB:.0f} dB). Possible misregistration, cloud/shadow confusing "
            f"the optical read, or a real surface-state difference (e.g. flooding)."
        )

    return ConsistencyResult(
        sar_mean_db=sar_db,
        optical_looks_like_water=optical_water,
        sar_looks_like_water=sar_water,
        consistent=consistent,
        explanation=explanation,
    )

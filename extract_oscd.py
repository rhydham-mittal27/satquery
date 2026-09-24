"""Extract OSCD before/after RGB pairs + ground-truth change masks to PNG.

Source: HF parquet mirror (blanchon/OSCD_MSI), 13-band Sentinel-2 arrays
stored as nested lists per row. Band order is standard S2 L1C:
B01,B02,B03,B04,B05,B06,B07,B08,B08A,B09,B10,B11,B12 (indices 0-12).
True colour RGB = B04,B03,B02 = indices 3,2,1.
"""
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "OSCD" / "pairs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RGB_BANDS = [3, 2, 1]  # B04 (red), B03 (green), B02 (blue)


def bands_to_rgb(nested_list) -> np.ndarray:
    """nested_list: 13 x H x W uint16 -> HxWx3 uint8 (percentile-stretched)."""
    bands = [np.array(nested_list[i].as_py() if hasattr(nested_list[i], "as_py") else nested_list[i], dtype=np.float32) for i in RGB_BANDS]
    rgb = np.stack(bands, axis=-1)
    out = np.empty_like(rgb)
    for c in range(3):
        chan = rgb[..., c]
        lo, hi = np.percentile(chan, [2, 98])
        out[..., c] = chan if hi <= lo else np.clip((chan - lo) / (hi - lo), 0, 1) * 255
    return out.astype(np.uint8)


def extract(split: str, prefix: str):
    parquet_path = OUT_DIR.parent / "data" / f"{split}-00000-of-00001.parquet"
    pf = pq.ParquetFile(parquet_path)

    idx = 0
    for batch in pf.iter_batches(batch_size=1, columns=["image1", "image2", "mask"]):
        img1 = batch.column("image1")[0]
        img2 = batch.column("image2")[0]
        mask_struct = batch.column("mask")[0]

        rgb1 = bands_to_rgb(img1)
        rgb2 = bands_to_rgb(img2)
        Image.fromarray(rgb1).save(OUT_DIR / f"{prefix}_{idx:02d}_before.png")
        Image.fromarray(rgb2).save(OUT_DIR / f"{prefix}_{idx:02d}_after.png")

        mask_bytes = mask_struct["bytes"].as_py()
        mask_img = Image.open(io.BytesIO(mask_bytes))
        mask_img.save(OUT_DIR / f"{prefix}_{idx:02d}_mask.png")

        print(f"{prefix}_{idx:02d}: before {rgb1.shape}, after {rgb2.shape}, mask {mask_img.size}")
        idx += 1


def main():
    extract("test", "pair")
    extract("train", "trainpair")


if __name__ == "__main__":
    main()

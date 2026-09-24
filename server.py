"""SatQuery API server.

Serves the static frontend and exposes /api/vqa for single-image
visual question answering (Part A).
"""
import base64
import io
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter
import numpy as np

PREVIEW_MIN_SIZE = 768


def stretch_for_display(img: Image.Image) -> Image.Image:
    """Per-channel percentile contrast stretch, display-only (doesn't touch model input)."""
    arr = np.asarray(img).astype(np.float32)
    out = np.empty_like(arr)
    for c in range(arr.shape[-1]):
        chan = arr[..., c]
        lo, hi = np.percentile(chan, [2, 98])
        if hi <= lo:
            out[..., c] = chan
        else:
            out[..., c] = np.clip((chan - lo) / (hi - lo), 0, 1) * 255
    img = Image.fromarray(out.astype(np.uint8))

    # Source tiles are tiny (e.g. 256px) — upscale with Lanczos + a light
    # unsharp mask so the preview doesn't look soft when stretched to fill
    # the viewport. Display-only; the model still sees the original pixels.
    if max(img.size) < PREVIEW_MIN_SIZE:
        scale = PREVIEW_MIN_SIZE / max(img.size)
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
    return img

import sys
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent / "vqa"))
from infer import answer_vqa  # noqa: E402
from change.infer import answer_change  # noqa: E402
from router import route  # noqa: E402
from sar.consistency import check_consistency  # noqa: E402

app = FastAPI(title="SatQuery API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.post("/api/preview")
async def preview_endpoint(image: UploadFile = File(...)):
    suffix = Path(image.filename).suffix or ".tif"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        preview = stretch_for_display(Image.open(tmp_path).convert("RGB"))
        buf = io.BytesIO()
        preview.save(buf, format="PNG")
        return {"preview": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/api/vqa")
async def vqa_endpoint(image: UploadFile = File(...), question: str = Form(...)):
    suffix = Path(image.filename).suffix or ".tif"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        result = answer_vqa(tmp_path, question)

        preview = stretch_for_display(Image.open(tmp_path).convert("RGB"))
        buf = io.BytesIO()
        preview.save(buf, format="PNG")
        result["preview"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


@app.post("/api/change")
async def change_endpoint(
    before: UploadFile = File(...),
    after: UploadFile = File(...),
    question: str = Form(...),
):
    paths = {}
    for key, upload in [("before", before), ("after", after)]:
        suffix = Path(upload.filename).suffix or ".tif"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            shutil.copyfileobj(upload.file, tmp)
            paths[key] = tmp.name

    try:
        result = answer_change(paths["before"], paths["after"], question)

        before_preview = stretch_for_display(Image.open(paths["before"]).convert("RGB"))
        buf = io.BytesIO()
        before_preview.save(buf, format="PNG")
        result["before_preview"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    finally:
        for p in paths.values():
            Path(p).unlink(missing_ok=True)

    return result


@app.post("/api/query")
async def query_endpoint(
    images: List[UploadFile] = File(...),
    question: str = Form(...),
):
    """Single entry point: routes to VQA or change-detection based on how
    many images were provided (router.py), instead of the caller having
    to pick a mode. Agentic task orchestration, per the PS."""
    if len(images) not in (1, 2):
        return {"error": f"Expected 1 or 2 images, got {len(images)}."}

    paths = []
    try:
        for upload in images:
            suffix = Path(upload.filename).suffix or ".tif"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(upload.file, tmp)
                paths.append(tmp.name)

        result = route(paths, question)

        # Attach a display preview of the primary/first image regardless
        # of task type, so the frontend always has something to show.
        preview = stretch_for_display(Image.open(paths[0]).convert("RGB"))
        buf = io.BytesIO()
        preview.save(buf, format="PNG")
        result.setdefault("preview", "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    finally:
        for p in paths:
            Path(p).unlink(missing_ok=True)

    return result


@app.post("/api/sar_check")
async def sar_check_endpoint(
    optical: UploadFile = File(...),
    sar: UploadFile = File(...),
):
    """Optical-SAR cross-modal consistency check (sar/consistency.py):
    does the SAR backscatter agree with what the optical image suggests
    about water presence? Real Sentinel-1 physics, 98% validated on real
    EuroSAT-SAR ground truth — see sar/EVAL.md."""
    paths = {}
    try:
        opt_suffix = Path(optical.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=opt_suffix, delete=False) as tmp:
            shutil.copyfileobj(optical.file, tmp)
            paths["optical"] = tmp.name

        sar_suffix = Path(sar.filename).suffix or ".tif"
        with tempfile.NamedTemporaryFile(suffix=sar_suffix, delete=False) as tmp:
            shutil.copyfileobj(sar.file, tmp)
            paths["sar"] = tmp.name

        optical_img = Image.open(paths["optical"])
        result = check_consistency(optical_img, paths["sar"])

        preview = stretch_for_display(optical_img.convert("RGB"))
        buf = io.BytesIO()
        preview.save(buf, format="PNG")
        preview_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        return {
            "sar_mean_db": round(result.sar_mean_db, 2),
            "optical_looks_like_water": result.optical_looks_like_water,
            "sar_looks_like_water": result.sar_looks_like_water,
            "consistent": result.consistent,
            "explanation": result.explanation,
            "preview": preview_b64,
        }
    finally:
        for p in paths.values():
            Path(p).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

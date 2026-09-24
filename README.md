# SatQuery

Natural-language question answering over satellite imagery: single-scene VQA,
before/after change detection, and an optical-SAR consistency check, behind
one auto-routing query box. Built for the ISRO/SAC problem statement
SIH26167 (SatQuery AI), aligned to SDG 13 (Climate Action).

## What's in it

| Piece | What it does | Measured result |
|---|---|---|
| `vqa/` | Frozen CLIP ViT-B/32 embeddings + small MLP head, trained on RSVQA-LR. Answers yes/no, rural/urban, count-range questions. | 86.2% validation accuracy |
| `vqa/grounding.py` | Draws an evidence box for water/vegetation "yes" answers using RGB spectral indices. No box for question types without a reliable index. | Verified visually on RSVQA-LR tiles |
| `change/` | Change detection between two scenes. RandomForest on 6 RGB-derived per-pixel features, trained on OSCD's 14 training pairs; falls back to CVA + Otsu if the model file is missing. | Mean IoU 0.269 / precision 0.433 / recall 0.526 on OSCD's 10 held-out test pairs |
| `sar/` | Optical-SAR check: does Sentinel-1 backscatter agree with what the optical image says about water? | 98% with ground-truth labels (59/60); 78% end-to-end with the RGB water heuristic |
| `router.py` | One entry point: 1 image goes to VQA, 2 images go to change detection. Warns if the question sounds like a comparison but only one image was given. | n/a |
| `frontend/` | Single-page UI (plain HTML/JS) served by the FastAPI app. | n/a |

## Read this before trusting the numbers

The evaluation write-ups record what did *not* work as well as what did:

- `change/EVAL.md` - v1 to v4, plus rejected experiments (PCA + k-means, edge
  diff, 10-feature model, adding 512 LEVIR-CD pairs). More features and more
  out-of-domain data both made the OSCD score worse. The weakest pairs are
  scenes with strong seasonal farmland change, which RGB alone can't separate
  from real construction.
- `sar/EVAL.md` - a 3-way land-cover split was tried and dropped (67%); the
  shipped check is binary water / not-water. The optical side inherits the
  RGB water index's sensitivity to imagery source (Forest is misread as water
  on EuroSAT JPEGs).
- VQA grounding: CLIP patch-token zero-shot grounding was tried in three
  configurations and rejected because it pointed at the wrong region on real
  tiles.

The change-detection model is trained on Sentinel-2 (10 m/px). On inputs unlike
that (e.g. a flat synthetic patch) it under-detects; the CVA + Otsu fallback
does better there.

## Run it

```bash
uv sync
uv run python server.py     # http://127.0.0.1:8000
```

The trained artifacts (`vqa/vqa_head.pt`, `vqa/vocab.json`,
`change/trained_classifier.joblib`) are included. CLIP weights download from
Hugging Face on first run.

API: `POST /api/query` (1-2 images + `question`), `POST /api/sar_check`
(`optical` + `sar` GeoTIFF), plus `/api/vqa`, `/api/change`, `/api/preview`.
The SAR check is not wired into the UI yet.

## Data (not in the repo)

The scripts read datasets from a `datasets/` folder next to this repo's folder
(i.e. `../datasets/...`):

- RSVQA-LR - Zenodo 10.5281/zenodo.6344334
- OSCD - `blanchon/OSCD_MSI` on Hugging Face (`extract_oscd.py` turns it into PNG pairs)
- LEVIR-CD - `sy2002123/levir-cd` (only used in the rejected experiment)
- EuroSAT-SAR - `wangyi111/EuroSAT-SAR`, paired with EuroSAT optical (`torchgeo/eurosat`)

Rebuild the VQA feature cache and retrain: `vqa/prepare_data.py` ->
`vqa/extract_features.py` -> `vqa/train_classifier.py`. Retrain the change
model with `train_change_classifier.py`; score it with `eval_oscd.py`.

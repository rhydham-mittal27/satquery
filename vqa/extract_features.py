"""Extract frozen CLIP image + text embeddings for RSVQA-LR and cache to disk.

Image embeddings are cached per unique image id (a .tif is shared by ~100
questions, so we never encode the same image twice). Text embeddings are
computed per split since each question is unique.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from prepare_data import load_split

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32


def load_clip():
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


@torch.no_grad()
def encode_images(model, processor, image_paths):
    embeds = {}
    unique_paths = sorted(set(image_paths))
    for i in range(0, len(unique_paths), BATCH_SIZE):
        batch_paths = unique_paths[i : i + BATCH_SIZE]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt")
        feats = model.get_image_features(**inputs).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        for p, f in zip(batch_paths, feats):
            embeds[p] = f.numpy()
        print(f"  images {i + len(batch_paths)}/{len(unique_paths)}", end="\r")
    print()
    return embeds


@torch.no_grad()
def encode_texts(model, processor, texts):
    embeds = np.zeros((len(texts), model.config.projection_dim), dtype=np.float32)
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = processor(
            text=batch, return_tensors="pt", padding=True, truncation=True
        )
        feats = model.get_text_features(**inputs).pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeds[i : i + len(batch)] = feats.numpy()
        print(f"  texts {i + len(batch)}/{len(texts)}", end="\r")
    print()
    return embeds


def process_split(split, model, processor, image_cache):
    samples = load_split(split)
    print(f"[{split}] {len(samples)} samples")

    image_paths = [s["image_path"] for s in samples]
    new_paths = [p for p in set(image_paths) if p not in image_cache]
    if new_paths:
        image_cache.update(encode_images(model, processor, new_paths))

    img_feats = np.stack([image_cache[p] for p in image_paths])
    questions = [s["question"] for s in samples]
    text_feats = encode_texts(model, processor, questions)
    answers = np.array([s["answer"] for s in samples])

    np.savez(
        CACHE_DIR / f"{split}.npz",
        img_feats=img_feats,
        text_feats=text_feats,
        answers=answers,
    )
    print(f"[{split}] saved -> {CACHE_DIR / f'{split}.npz'}")


def main():
    print("Loading CLIP...")
    model, processor = load_clip()
    image_cache = {}
    for split in ["train", "val", "test"]:
        process_split(split, model, processor, image_cache)


if __name__ == "__main__":
    main()

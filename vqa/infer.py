"""Single-image VQA inference: answer_vqa(image_path, question) -> answer string.

Loads the frozen CLIP encoders once plus the trained MLP head, and answers
natural-language questions about a remote-sensing image from a small
closed vocabulary (yes/no, land-cover type, count buckets).
"""
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from train_classifier import VQAHead
from grounding import extract_grounding_target, compute_heatmap, bbox_from_heatmap, render_grounding_overlay

MODEL_DIR = Path(__file__).resolve().parent
CLIP_NAME = "openai/clip-vit-base-patch32"

# RSVQA-LR's rural/urban question has exactly one fixed phrasing in
# training ("Is it a rural or an urban area"), unlike presence/count/
# comparison questions which have thousands of paraphrases. A reworded
# rural/urban question is out-of-distribution for the classifier and can
# land on any class (e.g. "yes"), so we detect question intent from
# keywords and mask the vocabulary to the plausible answer group.
ANSWER_GROUPS = {
    "rural_urban": {"rural", "urban"},
    "yes_no": {"yes", "no"},
    "count": {"0", "between 1 and 10", "between 11 and 100", "between 101 and 1000", "more than 1000"},
}


def _detect_group(question: str) -> set | None:
    q = question.lower()
    if "rural" in q or "urban" in q:
        return ANSWER_GROUPS["rural_urban"]
    if q.startswith(("how many", "what is the number", "what is the amount", "what amount", "what number")):
        return ANSWER_GROUPS["count"]
    if q.startswith(("is ", "are ", "does ", "do ")):
        return ANSWER_GROUPS["yes_no"]
    return None


_clip_model = None
_clip_processor = None
_head = None
_idx_to_answer = None


def _load():
    global _clip_model, _clip_processor, _head, _idx_to_answer
    if _head is not None:
        return

    _clip_model = CLIPModel.from_pretrained(CLIP_NAME).eval()
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME)

    vocab = json.load(open(MODEL_DIR / "vocab.json"))
    _idx_to_answer = {i: a for a, i in vocab.items()}

    ckpt = torch.load(MODEL_DIR / "vqa_head.pt", map_location="cpu")
    _head = VQAHead(ckpt["embed_dim"], ckpt["num_classes"])
    _head.load_state_dict(ckpt["state_dict"])
    _head.eval()


@torch.no_grad()
def answer_vqa(image_path: str, question: str) -> dict:
    """Answer a natural-language question about a single remote-sensing image."""
    _load()

    image = Image.open(image_path).convert("RGB")
    img_inputs = _clip_processor(images=[image], return_tensors="pt")
    img_feat = _clip_model.get_image_features(**img_inputs).pooler_output
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    text_inputs = _clip_processor(text=[question], return_tensors="pt", padding=True, truncation=True)
    text_feat = _clip_model.get_text_features(**text_inputs).pooler_output
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    logits = _head(img_feat, text_feat)

    group = _detect_group(question)
    if group is not None:
        mask = torch.tensor([_idx_to_answer[i] in group for i in range(logits.shape[-1])])
        logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))

    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = probs.argmax().item()
    answer = _idx_to_answer[pred_idx]

    result = {
        "answer": answer,
        "confidence": probs[pred_idx].item(),
    }

    # Only draw grounding evidence for affirmative presence answers with
    # a real spectral signal (water/vegetation) — see grounding.py for
    # why CLIP-based dense grounding was rejected. No box for "no", and
    # no box for question types we don't have a reliable index for.
    if answer == "yes":
        target = extract_grounding_target(question)
        if target is not None:
            heatmap = compute_heatmap(image, target)
            box = bbox_from_heatmap(heatmap)
            if box is not None:
                result["grounding"] = {
                    "target": target,
                    "box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
                    "overlay": render_grounding_overlay(image, box, target),
                }

    return result


if __name__ == "__main__":
    import sys

    from prepare_data import IMAGES_DIR

    image_path = str(IMAGES_DIR / "0.tif")
    for q in [
        "Is it a rural or an urban area",
        "Is there a water area?",
        "What is the amount of buildings?",
    ]:
        result = answer_vqa(image_path, q)
        print(f"Q: {q}\nA: {result['answer']}  (confidence={result['confidence']:.2f})\n")

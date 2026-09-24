"""Load RSVQA-LR splits into flat (image_path, question, answer) samples.

Count-type answers are bucketed into RSVQA-style ranges so the
classification vocabulary stays small instead of one class per raw count.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "RSVQA-LR"
IMAGES_DIR = DATA_DIR / "Images_LR"


def bucket_count(answer: str) -> str:
    if not answer.isdigit():
        return answer
    n = int(answer)
    if n == 0:
        return "0"
    if n <= 10:
        return "between 1 and 10"
    if n <= 100:
        return "between 11 and 100"
    if n <= 1000:
        return "between 101 and 1000"
    return "more than 1000"


def load_split(split: str):
    images = json.load(open(DATA_DIR / f"LR_split_{split}_images.json"))["images"]
    questions = json.load(open(DATA_DIR / f"LR_split_{split}_questions.json"))["questions"]
    answers = json.load(open(DATA_DIR / f"LR_split_{split}_answers.json"))["answers"]

    active_img_ids = {img["id"] for img in images if img["active"]}
    ans_by_qid = {a["question_id"]: a["answer"] for a in answers if a["active"]}

    samples = []
    for q in questions:
        if not q["active"] or q["img_id"] not in active_img_ids:
            continue
        raw_answer = ans_by_qid.get(q["id"])
        if raw_answer is None:
            continue
        answer = bucket_count(raw_answer) if q["type"] == "count" else raw_answer
        samples.append(
            {
                "image_path": str(IMAGES_DIR / f"{q['img_id']}.tif"),
                "question": q["question"],
                "answer": answer,
                "type": q["type"],
            }
        )
    return samples


def build_vocab(train_samples):
    answers = sorted({s["answer"] for s in train_samples})
    return {a: i for i, a in enumerate(answers)}


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        samples = load_split(split)
        print(split, len(samples), "samples")
    train_samples = load_split("train")
    vocab = build_vocab(train_samples)
    print("vocab size:", len(vocab))
    print(list(vocab.items())[:10])

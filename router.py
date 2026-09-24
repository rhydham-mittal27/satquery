"""Agentic task routing: one entry point, dispatches to the right
specialist pipeline based on what was actually provided.

This is deliberately simple and inspectable rather than an LLM-based
router — the routing signal here is unambiguous (how many images came
in), so a classifier would just be adding a point of failure. Question
text is used only as a secondary consistency check, to warn rather than
silently answer the wrong question.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "vqa"))
from infer import answer_vqa  # noqa: E402
from change.infer import answer_change  # noqa: E402

CHANGE_KEYWORDS = ["changed", "change", "compare", "between the two dates", "before and after", "increase", "decrease"]


def classify_query(num_images: int, question: str) -> str:
    """Returns 'vqa' or 'change'. num_images is the real routing signal."""
    if num_images >= 2:
        return "change"
    return "vqa"


def question_hints_change(question: str) -> bool:
    """Secondary check: does the question's own wording suggest a
    comparison, even though only one image was provided? Used to warn
    the user rather than guess."""
    q = question.lower()
    return any(kw in q for kw in CHANGE_KEYWORDS)


def route(image_paths: list[str], question: str) -> dict:
    """image_paths: 1 path -> single-image VQA, 2 paths -> change detection."""
    task = classify_query(len(image_paths), question)

    if task == "vqa":
        result = answer_vqa(image_paths[0], question)
        result["task_type"] = "vqa"
        if question_hints_change(question):
            result["router_note"] = (
                "This question sounds like it's asking about a comparison between two dates, "
                "but only one image was provided — answered as a single-scene question instead. "
                "Upload a second (before/after) image to run change detection."
            )
        return result

    result = answer_change(image_paths[0], image_paths[1], question)
    result["task_type"] = "change"
    return result

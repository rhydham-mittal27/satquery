"""Turn ChangeResult stats into a natural-language answer.

Templated on purpose: the numbers (pct_changed, region deltas) come from
deterministic image processing in diff.py, not a language model, so
there's nothing to hallucinate. This keeps the "zero-fabrication" answer
the PS asks for.
"""
from change.diff import ChangeResult


def describe_change(result: ChangeResult, question: str) -> str:
    q = question.lower()
    pct = result.pct_changed
    n = len(result.regions)

    if n == 0:
        if pct > 1.0:
            return (
                f"{pct:.1f}% of pixels differ between the two dates, but no single region large enough "
                f"to localize passed the detection threshold — likely diffuse noise (illumination/seasonal) "
                f"rather than one clear change."
            )
        return "No significant change detected between the two dates (less than the detection threshold)."

    avg_green_delta = sum(r.greenness_delta for r in result.regions) / n
    veg_word = "an increase in vegetation" if avg_green_delta > 0.02 else (
        "a decrease in vegetation" if avg_green_delta < -0.02 else "no clear vegetation shift"
    )

    if "how much" in q or "percent" in q or "area" in q:
        return f"{pct:.1f}% of the scene changed between the two dates, across {n} localized region(s)."

    if "increase" in q or "decrease" in q or "vegetation" in q or "grow" in q:
        return f"Detected {veg_word} in the changed regions (avg. greenness delta {avg_green_delta:+.3f}), covering {pct:.1f}% of the scene."

    region_labels = ", ".join(r.id for r in result.regions)
    return (
        f"Detected {n} changed region(s) [{region_labels}] covering {pct:.1f}% of the scene "
        f"between the two dates, with {veg_word}."
    )


def short_headline(result: ChangeResult, question: str) -> str:
    """A short, question-appropriate label for the answer card's headline.

    describe_change() gives the full sentence; the UI also wants a
    2-4 word headline that actually matches what was asked, instead of
    always showing the raw % regardless of question type.
    """
    q = question.lower()
    pct = result.pct_changed
    n = len(result.regions)

    veg_question = "increase" in q or "decrease" in q or "vegetation" in q or "grow" in q

    if n == 0:
        if veg_question:
            return "Inconclusive" if pct > 1.0 else "No vegetation shift"
        return f"{pct:.1f}% changed (diffuse)" if pct > 1.0 else "No significant change"

    if veg_question:
        avg_green_delta = sum(r.greenness_delta for r in result.regions) / n
        if avg_green_delta > 0.02:
            return "Vegetation increase"
        if avg_green_delta < -0.02:
            return "Vegetation decrease"
        return "No clear vegetation shift"

    return f"{pct:.1f}% changed"

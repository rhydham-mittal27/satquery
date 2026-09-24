"""Evaluate change/diff.py's predicted mask against real OSCD ground truth."""
import numpy as np
from PIL import Image

from change.diff import detect_change

PAIRS_DIR = "../datasets/OSCD/pairs" if __name__ != "__main__" else "datasets/OSCD/pairs"


def iou_precision_recall(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    union = tp + fp + fn
    iou = tp / union if union > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return iou, precision, recall


def main():
    rows = []
    for i in range(10):
        before = Image.open(f"../datasets/OSCD/pairs/pair_{i:02d}_before.png")
        after = Image.open(f"../datasets/OSCD/pairs/pair_{i:02d}_after.png")
        gt = np.array(Image.open(f"../datasets/OSCD/pairs/pair_{i:02d}_mask.png")) > 0

        result = detect_change(before, after)
        pred_mask = result.mask

        iou, prec, rec = iou_precision_recall(pred_mask, gt)
        gt_pct = gt.mean() * 100
        rows.append((i, gt_pct, result.pct_changed, iou, prec, rec))

    print(f"{'pair':>4} {'gt%':>7} {'pred%':>7} {'iou':>7} {'prec':>7} {'recall':>7}")
    for i, gt_pct, pred_pct, iou, prec, rec in rows:
        print(f"{i:4d} {gt_pct:7.2f} {pred_pct:7.2f} {iou:7.3f} {prec:7.3f} {rec:7.3f}")

    mean_iou = sum(r[3] for r in rows) / len(rows)
    mean_prec = sum(r[4] for r in rows) / len(rows)
    mean_rec = sum(r[5] for r in rows) / len(rows)
    print(f"\nmean IoU={mean_iou:.3f}  mean precision={mean_prec:.3f}  mean recall={mean_rec:.3f}")


if __name__ == "__main__":
    main()

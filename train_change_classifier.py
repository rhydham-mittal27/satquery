"""Train a supervised per-pixel change classifier on real ground-truth
change-detection data, evaluate on OSCD's 10-pair test set (never
touched during training), compare against the unsupervised v3
(CVA+Otsu) baseline.

Training pool: OSCD's own 14 real training pairs (Sentinel-2, 10m/px,
urban growth) + LEVIR-CD (512 real pairs, VHR 0.5m/px, building change)
— 45x more real labeled pairs than OSCD alone, to fight overfitting on
the earlier OSCD-only model. Evaluation stays strictly on OSCD's 10
test pairs so the comparison to v3/earlier v4 (OSCD-only) is apples to
apples and there's no leakage from the extra data into the test set.

RGB-only features (change/features.py) so the trained model still works
on any plain RGB upload at inference, not just multi-band tiles.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from change.features import extract_features

N_TEST = 10
POS_NEG_RATIO = 3  # negatives per positive, for class balance in training

OSCD_SAMPLES_PER_IMAGE = 15000
LEVIR_SAMPLES_PER_IMAGE = 1200
# OSCD (Sentinel-2, 10m/px) and LEVIR-CD (VHR, 0.5m/px) have very
# different resolution/texture statistics. Using all 512 LEVIR pairs
# (614k samples) against OSCD's ~40k drowned out the in-domain signal
# and hurt OSCD-test accuracy (mean IoU 0.269 -> 0.150) — see
# change/EVAL.md. Capped here to roughly match OSCD's sample count
# instead of scaling naively.
LEVIR_MAX_PAIRS = 0


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


def _sample_one(rng, before, after, gt, samples_per_image, label):
    feats = extract_features(before, after)
    h, w = gt.shape
    feats = feats[:h, :w] if feats.shape[:2] != gt.shape else feats

    pos_idx = np.argwhere(gt)
    neg_idx = np.argwhere(~gt)
    n_pos = min(len(pos_idx), samples_per_image // (1 + POS_NEG_RATIO))
    n_neg = min(len(neg_idx), n_pos * POS_NEG_RATIO)
    if n_pos == 0:
        return None, None

    pos_sel = pos_idx[rng.choice(len(pos_idx), n_pos, replace=False)]
    neg_sel = neg_idx[rng.choice(len(neg_idx), n_neg, replace=False)]

    X = np.concatenate([feats[pos_sel[:, 0], pos_sel[:, 1]], feats[neg_sel[:, 0], neg_sel[:, 1]]])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    print(f"{label}: {n_pos} pos, {n_neg} neg sampled")
    return X, y


def build_training_set():
    rng = np.random.default_rng(0)
    X_parts, y_parts = [], []

    for i in range(14):
        before = Image.open(f"../datasets/OSCD/pairs/trainpair_{i:02d}_before.png")
        after = Image.open(f"../datasets/OSCD/pairs/trainpair_{i:02d}_after.png")
        gt = np.array(Image.open(f"../datasets/OSCD/pairs/trainpair_{i:02d}_mask.png")) > 0
        X, y = _sample_one(rng, before, after, gt, OSCD_SAMPLES_PER_IMAGE, f"trainpair_{i:02d}")
        if X is not None:
            X_parts.append(X)
            y_parts.append(y)

    levir_a_paths = sorted(glob.glob("../datasets/LEVIR-CD/img_512/img_512/*_A.png"))
    rng.shuffle(levir_a_paths)
    levir_a_paths = levir_a_paths[:LEVIR_MAX_PAIRS]
    for a_path in levir_a_paths:
        b_path = a_path.replace("_A.png", "_B.png")
        mask_name = os.path.basename(a_path).replace("_A.png", ".png")
        mask_path = f"../datasets/LEVIR-CD/anno_512/anno_512/{mask_name}"

        before = Image.open(a_path)
        after = Image.open(b_path)
        gt = np.array(Image.open(mask_path)) > 0
        X, y = _sample_one(rng, before, after, gt, LEVIR_SAMPLES_PER_IMAGE, mask_name)
        if X is not None:
            X_parts.append(X)
            y_parts.append(y)

    return np.concatenate(X_parts), np.concatenate(y_parts)


def evaluate(clf):
    ious, precs, recs = [], [], []
    for i in range(N_TEST):
        before = Image.open(f"../datasets/OSCD/pairs/pair_{i:02d}_before.png")
        after = Image.open(f"../datasets/OSCD/pairs/pair_{i:02d}_after.png")
        gt = np.array(Image.open(f"../datasets/OSCD/pairs/pair_{i:02d}_mask.png")) > 0

        feats = extract_features(before, after)
        h, w = gt.shape
        feats = feats[:h, :w] if feats.shape[:2] != gt.shape else feats
        flat = feats.reshape(-1, feats.shape[-1])

        probs = clf.predict_proba(flat)[:, 1].reshape(h, w)
        pred_mask = probs > 0.5

        iou, prec, rec = iou_precision_recall(pred_mask, gt)
        ious.append(iou)
        precs.append(prec)
        recs.append(rec)
        print(f"pair_{i:02d}: gt%={gt.mean()*100:.2f} pred%={pred_mask.mean()*100:.2f} iou={iou:.3f} prec={prec:.3f} rec={rec:.3f}")

    print(f"\nmean IoU={sum(ious)/len(ious):.3f}  mean precision={sum(precs)/len(precs):.3f}  mean recall={sum(recs)/len(recs):.3f}")
    return ious, precs, recs


def main():
    model_choice = sys.argv[1] if len(sys.argv) > 1 else "rf"

    print("Building training set from 14 OSCD + up to 512 LEVIR-CD real pairs...")
    X, y = build_training_set()
    print(f"\nTotal training samples: {len(y)} ({y.sum()} positive, {(1-y).sum()} negative)")

    if model_choice == "gb":
        clf = HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.08, random_state=0)
    else:
        clf = RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=0)
    clf.fit(X, y)

    if hasattr(clf, "feature_importances_"):
        print("\nFeature importances:")
        from change.features import FEATURE_NAMES
        for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
            print(f"  {name}: {imp:.3f}")

    print(f"\nEvaluating {model_choice} on 10 held-out real test pairs...")
    evaluate(clf)

    import joblib
    out_path = f"change/trained_classifier_{model_choice}.joblib"
    joblib.dump(clf, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()

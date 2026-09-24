# Optical-SAR Consistency Check — Evaluation

Real data: [EuroSAT-SAR](https://huggingface.co/datasets/wangyi111/EuroSAT-SAR)
(Sentinel-1, dual-pol VV/VH, geo-matched to the original EuroSAT/
Sentinel-2 optical patches by coordinates) +
[EuroSAT optical](https://huggingface.co/datasets/torchgeo/eurosat)
(same tile IDs, so real, exact, filename-matched pairs — `SeaLake_5.jpg`
↔ `SeaLake_5.tif`). 27,000 pairs across 10 land-cover classes; 60 used
for evaluation here (10 samples × 6 classes).

## SAR-side physics (verified, not assumed)

Mean VV backscatter (dB), 30 real samples per class:

| Class | Mean VV (dB) |
|---|---|
| SeaLake (water) | -19.98 |
| AnnualCrop | -11.68 |
| Highway | -10.68 |
| Forest | -9.33 |
| Residential | -8.22 |
| Industrial | -7.01 |

Textbook-correct ordering: water reflects radar specularly away from
the sensor (low return); built structures cause double-bounce
reflection back to the sensor (high return); vegetation/smooth surfaces
sit in between.

## Attempt 1: 3-way split (water / vegetation / built-up) — rejected

Using ground-truth optical labels (not our own classifier — isolating
just the SAR-threshold logic): **67% (40/60)**. Per-class breakdown:
SeaLake 9/10, AnnualCrop 9/10, Industrial 8/10, Residential 6/10,
**Forest 5/10, Highway 3/10**. Smooth non-water surfaces (paved
highways, sparse crops) overlap in backscatter with actual vegetation —
a single dB threshold can't cleanly separate 3 classes.

## Attempt 2: binary water / not-water — shipped

Same real data, same ground-truth labels, single threshold at -15 dB:
**98% (59/60)**. SeaLake 9/10, everything else 10/10. Water's specular
signature is physically distinct from every other land type tested;
narrowed scope to what's actually reliable instead of shipping the
shakier 3-way version.

## End-to-end (using our own optical water heuristic, not ground truth)

The above used the real EuroSAT class *labels* to isolate the SAR
threshold's accuracy. Running the full pipeline — our own RGB
water-index classifier (`vqa/grounding.py`, already used for VQA
grounding) deciding "does the optical image look like water" — against
the same real pairs:

**78% (47/60)**. Breakdown: Residential 10/10, Industrial 10/10,
Highway 10/10, AnnualCrop 9/10, SeaLake 7/10, **Forest 1/10**.

Forest is the failure mode: the RGB water index
(`(2B-R-G)/(2B+R+G)`) was calibrated informally on RSVQA-LR's
Sentinel-2 true-color tiles, where it worked correctly (verified
earlier for VQA grounding — the box landed exactly on a real water
body). It doesn't transfer to EuroSAT's JPEG-compressed imagery, whose
different color/gamma handling pushes Forest's blue channel high enough
to look "water-like" under the same threshold. Same domain-transfer
issue documented elsewhere in this project (change-detection's
raw-diff threshold, CLIP zero-shot grounding) — RGB heuristics tuned on
one imagery source don't automatically generalize to another.

## Conclusion

SAR-side binary water detection is genuinely strong and shipped as-is
(98%, real physics, real data). The optical-side classifier inherits a
known, documented limitation rather than a new one — same fix path as
before (recalibrate per imagery source, or use real NDWI from a NIR
band instead of an RGB proxy) if pursued further.

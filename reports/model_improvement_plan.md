# Model Improvement Plan from Dashboard Insights

## Current Finding

The best fast experiment is `F3_lightgbm_weighted`.

- Holdout F1: 0.9913
- Holdout recall: 0.9905
- Holdout false negative rate: 0.0095
- Holdout errors: 11 images, with 6 false negatives and 5 false positives

Feature importance suggests that both pixel regions and global image conditions matter.

- Pixel feature importance share: 87.3%
- Metadata feature importance share: 12.7%
- Top metadata features: contrast, brightness

## Improvement Plan

1. Review high-confidence errors first.
   These are strong candidates for label noise, outlier images, or systematic shortcuts such as lighting or background artifacts.

2. Tune the operating threshold separately from model selection.
   For quality inspection, false negatives are more expensive than false positives. Compare F1-optimal, recall-first, and business-cost thresholds.

3. Run the CNN/transfer learning CV pipeline.
   Use `resnet18`, `mobilenet_v3_small`, and `efficientnet_b0` with the existing 5-fold split so results remain comparable.

4. Add GradCAM for CNN models.
   GradCAM should confirm whether the model focuses on defect regions instead of product borders, shadows, or background patterns.

5. Add lighting-robust preprocessing ablations.
   Compare ImageNet normalization, dataset-specific normalization, brightness/contrast jitter, and histogram equalization.

6. Add hard-example mining.
   Use near-threshold errors and high-confidence errors as hard examples in the next training pass through weighting, sampling, or targeted augmentation.

7. Strengthen leakage checks.
   SHA1 duplicate checks found exact duplicates. Add perceptual hashing to catch near-duplicate images across folds.

# Reproducibility notes

This document describes the supplied code and data snapshot. It does not modify the published paper, retrospectively change the original analysis, or assert that all published results have been independently reproduced.

## Verified frozen Coswara results

The included March 18 corrected Coswara CSV contains 24,322 recordings: 12,161 validation and 12,161 test. Filtering `split == "test"` gives 1,899 true positives, 1,269 false positives, 805 false negatives, and 8,188 true negatives.

Read-only recomputation from that test split gives:

| Measurement | Recomputed value |
| --- | ---: |
| FP-detection AUROC, with increasing distance flagging FPs | 0.8585579238 |
| FN-detection AUROC among predicted negatives, with decreasing distance flagging FNs | 0.8197275516 |
| Mean TP Mahalanobis distance | 44.7152431770 |
| Mean FP Mahalanobis distance | 63.4094926327 |
| Cohen's d, pooled sample standard deviation | 0.9442057417 |
| Welch's t, TP minus FP | -23.2513505233 |
| Welch-Satterthwaite degrees of freedom | 1739.5140896761 |

The empirical 90th percentile of the baseline TP distances, using linear quantile interpolation, is approximately 55.7007392091. Rejecting predicted coughs above that distance gives TP=1,709, FP=440, FN=995, TN=9,017, precision=79.5254%, recall=63.2027%. This is a descriptive point selected using labeled test data, not an independently validated deployment threshold. Exactly 56 gives slightly different counts.

The accompanying Coswara summary records temperature 8.5709 and a classification probability threshold of 0.54. The ESC-50 March summary records temperature 3.8174 and a probability threshold of 0.60. Those files are separate runs; they should not be merged with older-run calibration constants.

## Original implementation limitations

- The corrected main script fits its temperature, probability threshold, and Mahalanobis reference on validation data. Its split is recording-level, not participant-disjoint. A no-test-fitting statement does not establish participant independence.
- Only date folders are sorted in the Coswara collector. Participant and audio listing order is not fixed, and the MC Dropout random seed is not set. `random_state=42` for splitting alone does not make a fresh run identical.
- Per-pass logits and the full MC-averaged embedding matrix were not saved by the corrected main script. The frozen CSV retains scores, outcomes, and split assignments, but is not a complete intermediate cache for refitting the entire pipeline.
- The historical `generate_statistics.py` points at nonexistent old filenames. The README's explicit configuration uses the included March CSVs without modifying the source.
- The historical report generator is not a complete implementation of the final ROC analyses, all reported intervals, exact retention-quantile calculations, or the 39.5 threshold-transfer analysis. The main inference script's validation threshold grid is `[30, 35, 40, 45, 50]` and cannot itself select 39.5. A calculation from saved scores is distinct from the grid search that the archived script runs.
- Some scripts suppress exceptions or proceed without a `split` column. Do not interpret an incomplete run or an unfiltered CSV as equivalent to the held-out test analysis.
- The t-SNE extractor uses `model.eval()` features, while the corrected inference script averages features across dropout passes. Plotting freshly extracted features does not reconstruct the original Mahalanobis reference.

## Calibration terminology and historical values

The published manuscript defines a confidence gap as accuracy minus mean confidence. The archived calibration implementations instead calculate mean assigned-class confidence on correct predictions minus that on incorrect predictions. These are different quantities; zero for the latter is not an average-calibration-bias statement.

The March summary reports the implementation's confidence difference as -0.1632 before and -0.0260 after calibration. Historical manuscript values and high-confidence error percentages are not all identical to the March snapshot. No code has been changed to force agreement. Future corrected analyses should state the exact formula, population, confidence definition, and source run.

In `run_temp_scaling.py`, "high confidence" means above that evaluated population's median assigned-class confidence. It does not mean a fixed probability such as 0.9. Moving toward zero on the correct-versus-incorrect confidence difference, by itself, does not establish improved probabilistic calibration.

## Taxonomy dashboard

The original dashboard requires `outputs/Cough Detection - Error Taxonomy Summary (1).xlsx`, sheet `UQ_Coswara`. It is not included because a shareable minimal annotation export and redistribution review have not been completed.

The unchanged dashboard does not restrict its prediction CSV to `split == "test"`. It labels mutual-information epistemic uncertainty as "variance", calls raw predictive entropy "via Temp Scaling", and draws a historical 35.25 distance threshold. Its output should not be represented as a verified reproduction of the final diagnostic table without a separately documented revision.

## Prototype code

The prototype uses different calibration constants, adaptive gates, live-audio preprocessing, and an uncertainty gate. Its exported reference uses high-confidence true positives from an older unsplit CSV and a different covariance regularization. It is supplied only as a historical demonstration, not as the source of the offline test metrics. See `examples/prototype/README.md`.

## Verification scope for this release

Selection and packaging checks cover source identity, dependency closure, syntax parsing, required frozen-file schemas, split counts, core frozen-score statistics, and manifest integrity. The full audio inference pipeline, live microphone capture, and manual-taxonomy dashboard have not been rerun as part of packaging. No new experiments are claimed.

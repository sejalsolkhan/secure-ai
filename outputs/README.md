# Frozen March result files

Only four original result files are included:

- `all_samples_T8p57_th0p54_coswara_20260318_145415_mahal_corrected.csv`
- `summary_temp_scaling_coswara_20260318_145415_mahal_corrected.csv`
- `all_samples_T3p82_th0p60_esc50_20260317_234844_mahal_corrected.csv`
- `summary_temp_scaling_esc50_20260317_234844_mahal_corrected.csv`

The per-recording files include validation and test rows. Use `split == "test"` for held-out evaluation. Outcome labels describe calibrated predictions before Mahalanobis rejection. The summary files evaluate the corresponding held-out test splits.

These files preserve original dataset recording identifiers and paths so scores can be traced to inputs. They have not been anonymized or repackaged. Confirm the dataset providers' terms permit redistribution of these derived records before uploading this snapshot publicly.

These CSVs do not contain the complete embedding matrix, original per-pass logits, or saved fitted reference parameters. They support score-level checks but do not replace all original inputs.

## Manual annotations

The taxonomy dashboard additionally expects `Cough Detection - Error Taxonomy Summary (1).xlsx`, sheet `UQ_Coswara`, with fields including `filename`, `file_path`, and `Category Code`. This workbook is not bundled. A reviewed export of the relevant manually assigned labels is needed to make that part of the analysis independently reproducible. Such an export would require a documented adaptation of the original dashboard input loader.

New generated output files are ignored by Git by default; the original four files are explicitly allowlisted.

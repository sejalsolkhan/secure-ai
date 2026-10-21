# SECURE-AI: code accompanying the paper

SECURE-AI evaluates uncertainty signals around a frozen MobileNetV3 cough/non-cough classifier. The study uses Coswara and ESC-50 recordings, Monte Carlo Dropout, temperature scaling, and Mahalanobis distance.

This is a selected snapshot of the existing research code. All included Python files and result CSVs are unchanged copies of the original files. It is not a rewritten implementation or a corrected edition of the paper. Consult [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md) before interpreting the outputs.

## Included code

| File | Role |
| --- | --- |
| `run_all_lab_model.py` | Standard frozen-classifier baseline on Coswara, ESC-50, or both |
| `run_mc_dropout.py` | Standalone MC Dropout and selective-prediction analysis on either dataset |
| `run_temp_scaling_mahalanobis_new.py` | Main corrected validation/test pipeline, including temperature scaling and validation-only Mahalanobis reference fitting |
| `run_temp_scaling.py` | Standalone temperature-scaling analysis, including historical confidence and selective-prediction calculations |
| `extract_embeddings_test_set.py` | Test-set feature extraction for visualization |
| `generate_tsne_publication_test_set.py` | t-SNE and supporting exploratory visualizations |
| `generate_statistics.py` | Historical statistical report generator; its input paths require configuration |
| `error_taxonomy_uq_metrics_new.py` | Historical manual-taxonomy dashboard; requires the separate annotation workbook |
| `models/mn/` and `helpers/` | Source dependencies imported by the classifier |
| `examples/prototype/` | Historical live-demo and parameter-export scripts, not the evaluated offline pipeline |

The `outputs/` directory contains only the two corrected March per-recording result files and their matching summaries. These retain the original validation/test assignments. See [outputs/README.md](outputs/README.md).

## Requirements and inputs

Use Python 3.12 or newer to parse all included scripts. The original March environment was not recorded as a dependency lock file. [requirements.in](requirements.in) lists imported packages, not a tested or pinned environment. In particular, use mutually compatible PyTorch, TorchVision, and TorchAudio versions, and account for the older APIs used in the scripts.

Required external inputs are deliberately not bundled:

- The lab-provided binary classifier checkpoint. See [models/README.md](models/README.md).
- Coswara and ESC-50 audio and ESC-50 metadata. See [datasets/README.md](datasets/README.md).
- The manually labeled taxonomy workbook for the taxonomy dashboard. See [outputs/README.md](outputs/README.md).

Do not substitute an AudioSet-pretrained 527-class model for the lab's two-class checkpoint.

## Offline inference

Run commands from the repository root after placing the required model and datasets in their expected locations:

```bash
python run_all_lab_model.py coswara
python run_all_lab_model.py esc50
python run_mc_dropout.py coswara
python run_mc_dropout.py esc50
python run_temp_scaling_mahalanobis_new.py coswara
python run_temp_scaling_mahalanobis_new.py esc50
```

These commands run inference over the audio and write new timestamped result files. The original scripts do not fully fix input ordering and dropout randomness, so a fresh run is not guaranteed to reproduce the frozen CSVs exactly.

## Statistics from the frozen results

The unchanged `generate_statistics.py` contains outdated input filenames. To analyze the included March files without editing the archived script, invoke it with explicit configuration:

```bash
python -c 'import generate_statistics as s; s.COSWARA_CSV="outputs/all_samples_T8p57_th0p54_coswara_20260318_145415_mahal_corrected.csv"; s.ESC50_CSV="outputs/all_samples_T3p82_th0p60_esc50_20260317_234844_mahal_corrected.csv"; s.OUTPUT_MD="outputs/statistical_analysis_generated.md"; s.OUTPUT_CSV="outputs/statistical_summary_generated.csv"; s.main()'
```

This runs the historical report generator, not a complete reproduction of every paper table, figure, or confidence interval. The important gaps are listed in [REPRODUCIBILITY_NOTES.md](REPRODUCIBILITY_NOTES.md).

## Test-set visualization

```bash
python extract_embeddings_test_set.py --csv outputs/all_samples_T8p57_th0p54_coswara_20260318_145415_mahal_corrected.csv --model models/mn/mn10_mydata_epoch_13.pt --output embeddings_test_set
python generate_tsne_publication_test_set.py --input embeddings_test_set --output figures_generated
```

The extractor performs a new deterministic-model feature extraction. It does not recover the original in-memory MC-averaged embedding matrix used to fit the saved Mahalanobis reference. The plotting script creates supporting visualizations, not the exact final paper figure set.

## Scope and provenance

EAT/AudioSet experiments, TBscreen follow-up studies, old Mahalanobis-fitting implementations, raw audio, binary checkpoints, caches, and manuscript-working folders are not part of this snapshot. Prototype scripts are isolated under `examples/prototype/` and documented separately.

`SOURCE_MANIFEST.tsv` records the original relative path, release path, size, and SHA-256 digest of every copied source or result file. `SHA256SUMS` covers all release files other than itself. No Git repository has been initialized and no commit or upload is implied by this folder.

Model-source attribution is documented in [models/README.md](models/README.md). Applicable source-code and dataset licenses and sharing permissions still apply.

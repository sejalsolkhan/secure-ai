# Historical prototype scripts

These unchanged scripts are supplied as supporting wearable/live-demo examples:

- `live_demo.py`: microphone capture, dropout prediction averaging, temperature scaling, and adaptive decision gates.
- `save_maha_params.py`: historical parameter export for the live demo.

They are not the evaluated offline system and are not part of the default reproduction commands.

Both refer to `models/mn10_mydata_epoch_13.pt`, whereas the offline scripts refer to `models/mn/mn10_mydata_epoch_13.pt`. Their expected root-level model path must be supplied explicitly in a configured working copy. The parameter exporter also expects an older CSV, `outputs/all_samples_T8p49_th0p54_coswara_20260117_230855.csv`, which is not included in this selected snapshot.

The exporter selects high-confidence true positives without a validation-split restriction and uses a regularized sample covariance. This is different from the corrected offline reference fit. The live demo uses temperature 8.54, adaptive thresholds, peak normalization, and a logit-variation gate rather than the exact evaluated configuration.

They also require `colorama`, `sounddevice`, and compatible audio hardware/system libraries. Those optional dependencies are not part of the offline package list.

When working with these scripts, keep the repository root as the current directory and on `PYTHONPATH`. Do not describe their output as reproducing the paper's held-out metrics.

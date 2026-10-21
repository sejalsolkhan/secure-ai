# External datasets

Obtain the datasets from their original providers under the applicable data-use terms. No raw audio or participant health metadata is bundled.

The unchanged scripts expect:

```text
datasets/Coswara-Data/Extracted_data/<date>/<participant>/*.wav
datasets/ESC-50/audio/*.wav
datasets/ESC-50/meta/esc50.csv
```

Coswara recordings named `cough-heavy.wav` and `cough-shallow.wav` are treated as coughs. ESC-50 rows whose category is `coughing` are treated as coughs. Other recording types are treated as non-coughs by the original scripts.

The included March Coswara CSV has 24,322 successfully processed recordings with saved split assignments. A different dataset revision, participant subset, traversal order, failed-file policy, or dropout state can change the rerun. Use the saved CSV's recording keys and `split` column when analyzing the frozen results.

The main inference scripts do not use the separate TBscreen follow-up dataset. That exploratory work is intentionally outside this paper-code snapshot.

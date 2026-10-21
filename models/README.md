# Classifier source and checkpoint

The `mn/` files and top-level `helpers/` files are the source dependencies imported by the research scripts. They are unchanged copies of the lab-supplied project files.

The main offline scripts require the exact binary cough/non-cough checkpoint at:

```text
models/mn/mn10_mydata_epoch_13.pt
```

The checkpoint is not included. Confirm with the providing lab whether it can be redistributed and supply approved access instructions before advertising end-to-end reproducibility. Do not use the separate AudioSet checkpoint as a substitute.

SHA-256 of the checkpoint present in the original project at packaging time:

```text
cd1a5f29ef87235be866cec938b8ba60ad4850322e59daba71210c75eddb0670
```

The lab-supplied model source includes references to the EfficientAT model release in `mn/model.py`; `mn/utils.py` attributes its `make_divisible` implementation to TensorFlow's MobileNet code. Existing source attribution has been preserved unchanged. Retain applicable upstream license notices and confirm the lab's sharing permissions before redistribution. No blanket license has been applied to these files.

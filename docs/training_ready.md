# Prepared detector experiment

`configs/detector_ready.yaml` is the new default training configuration. It uses
`data/training/pothole_v1/`, with explicit matching images and labels.
Original datasets, manifests, and annotations are unchanged.

This is the folder to use for the current detector experiment:

```text
data/training/pothole_v1/
  images/
    train/       # 2,815 training images
    val/         # 260 validation images
    test/        # 221 held-out test images
  labels/
    train/       # matching YOLO labels (empty files are negatives)
    val/
    test/
  train.txt      # paths used by the training loader
  val.txt
  test.txt
  manifest.csv
  excluded.csv
  dataset_summary.json
```

The training script selects these splits through `configs/detector_ready.yaml`.
`data/processed/` contains source-preparation inputs and older outputs;
`data/raw/` contains original downloads and footage. See [the data folder guide](../data/README.md).

| Split | Pothole-positive images | Negative images | Total | Pothole boxes |
| --- | ---: | ---: | ---: | ---: |
| Train | 2253 | 562 | 2815 | 3660 |
| Validation | 179 | 81 | 260 | 287 |
| Test | 177 | 44 | 221 | 266 |

All training positives from the supplied sources are kept. Training negatives
are capped at one per four positive images within each source (approximately
20% background overall). Negatives are sampled deterministically across
available hard-negative strata: RDD crack types, Kaggle crack/manhole class
combinations, and annotated local confuser tags. These tags are not a measured
model-specific hardness score. The remaining 1270 training negatives remain
available in the original sources for future train-only hard-negative mining.
Training data includes Japan, Norway, US, local reviewed images, and Kaggle.

This ratio is an experimental starting point, not a promise of improved accuracy
or an official universal recommendation. Too few negatives can increase false
alarms. Compare ratios on validation data using separate versioned outputs.
Undersampling is training-only, as described in
[Ultralytics preprocessing guidance](https://docs.ultralytics.com/guides/preprocessing-annotated-data/).

## Commands

```powershell
# Already prepared; verify actual image/label loading without training:
.venv\Scripts\python.exe scripts/prepare_training_dataset.py --verify-only
.venv\Scripts\python.exe scripts/train_detector.py --dry-run

# Train YOLO26x with augmentation, CUDA, batch 4, 640 pixels, 50 epochs:
.venv\Scripts\python.exe scripts/train_detector.py --device 0

# After choosing and freezing the model and pipeline settings, evaluate once:
.venv\Scripts\python.exe scripts/train_detector.py --evaluate-only --model models/yolo26x_ready_v1.pt --data configs/detector_ready.yaml --split test
```

Training saves to a unique run directory and exports
`models/yolo26x_ready_v1.pt`. Existing checkpoints cannot be overwritten by
default. Training and final validation do not run test evaluation unless
explicitly requested. The dashboard must be configured to use this new checkpoint.
Training has not been started as part of dataset preparation.

To reproduce preparation, run `python scripts/prepare_training_dataset.py` after
preparing the V3 RDD, reviewed local export, and Kaggle source. Outputs must not
already exist. For a ratio comparison, use e.g.:

```powershell
.venv\Scripts\python.exe scripts/prepare_training_dataset.py --negative-ratio 0.1 --output data/training/pothole_ratio01 --config-output configs/detector_ready_ratio01.yaml
```

Generated configs contain machine-specific absolute paths; regenerate on another
machine. `manifest.csv` records image, matching label, source, split, group,
stratum, decoded-image hash, and box count. `excluded.csv` records removals.
`dataset_summary.json` records counts, settings, and limitations.

## Split integrity and evaluation limits

- Existing RDD V3 assignments and reviewed local assignments are retained.
  Local recording/sequence groups cannot span splits; preparation stops if they do.
- Kaggle is training-only, avoiding random separation of neighboring video frames.
- Exact decoded-image duplicates are checked globally, giving test then validation
  precedence. Conflicting annotations on identical images stop preparation.
  The current build has no exact overlap among the selected sources.
- RDD image/label pairing is repaired by staging split-specific processed labels
  beside copied images in standard YOLO layout. Missing labels stop preparation
  instead of silently turning positives into background images.
- Validation/test negatives are not further subsampled. RDD V3 had already been
  balanced upstream, so these metrics do not represent natural road prevalence.
- RDD route identifiers are unavailable; sample-level grouping does not prove
  route independence. Near-duplicates with different encoding are not ruled out.
- Test is RDD-only. It is held out for this new recipe, not necessarily for older
  checkpoints trained on other RDD splits. Do not benchmark those as unseen data.
- All reviewed Green Bay frames come from one recording assigned to training.
  A new separately annotated recording is required for a local end-to-end test.

## Detector plus verification layers

Choose detector confidence, verifier threshold, and temporal policies using
validation data only. Favor detector recall while measuring false candidates per
minute and verifier workload. The verifier can remove false positives but cannot
recover a pothole missed by the detector. Track mAP50-95, precision, recall, and
event-level recall/precision rather than calling everything accuracy.

Generate classifier/verifier training crops only from detector training images
and training recordings. Keep validation/test crops, neighboring frames, and
physical events out of verifier training. Keep the 1270 omitted negatives for
train-only false-positive mining; never mine the test split for training examples.
Freeze the full pipeline before final test evaluation. Real video event testing
requires annotated event identities and separate recordings, not only still-image
boxes.

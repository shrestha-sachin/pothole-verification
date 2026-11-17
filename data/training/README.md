# Current detector dataset

Use **`pothole_v1/`**. It contains matching `images/` and `labels/` folders,
each divided into `train/`, `val/`, and `test/`.

| Split | Images | Pothole-positive | Negative |
| --- | ---: | ---: | ---: |
| Train | 2815 | 2253 | 562 |
| Validation | 260 | 179 | 81 |
| Test | 221 | 177 | 44 |

`configs/detector_ready.yaml` is the default used by `scripts/train_detector.py`.
There is no need to select individual source folders.

The dataset's `manifest.csv` traces each image back to its source and label;
`dataset_summary.json` contains balance settings and limitations. Files in
`data/processed/` and `data/raw/` are preserved preparation inputs, not additional
images to add manually to the current training run.

See [the training guide](../../docs/training_ready.md).

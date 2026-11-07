# Safe dataset and experiment workflow

This branch introduces non-destructive defaults. **Do not reuse an existing output directory** unless you explicitly intend to replace its contents.

## Prepare a new RDD2022 dataset

```bash
python scripts/convert_rdd2022.py \
  --input /path/to/RDD2022 \
  --countries United_States \
  --output data/processed/rdd2022_v3 \
  --config-yaml configs/detector_rdd_v3.yaml \
  --neg-ratio 0.25 --seed 42
```

The converter now indexes images once, protects existing split manifests, records exact duplicate/sequence connected components, and writes `split_metadata.csv` and `dataset_summary.json`. Standard RDD2022 filenames without sequence metadata still use sample-level grouping; **do not claim geographic/route independence from this fallback**. If you have real sequence or route identifiers, provide `--metadata-csv` with `image_id,sequence_id` columns. Keep every recording or route entirely in one split.

The `--overwrite` flag is an explicit escape hatch; back up existing experiments before using it.

## Optional negative balancing without modifying original manifests

```bash
python scripts/subsample_rdd_negatives.py \
  --proc-dir data/processed/rdd2022_v3 \
  --train-negatives 25 --val-negatives 5 --test-negatives 5 \
  --manifest-csv data/metadata/rdd_v3_negatives.csv \
  --yaml-config configs/detector_rdd_v3_balanced.yaml
```

This writes `train_balanced.txt`, `val_balanced.txt`, and `test_balanced.txt`. Primary `train.txt`, `val.txt`, and `test.txt` are preserved. Repeating the same command against existing outputs requires `--overwrite`. Replacing primary manifests requires both `--update-primary-manifests` and `--overwrite`.

## Train an isolated experiment

```bash
python scripts/train_detector.py \
  --data configs/detector_rdd_v3_balanced.yaml \
  --experiment-name rdd_v3_baseline \
  --output models/rdd_v3_baseline.pt \
  --epochs 50
```

Each experiment uses its own `runs/detect/<experiment-name>` directory and checkpoint. The script refuses to overwrite either by default. Avoid using `models/best.pt` as a training output if that file already exists.

## Final evaluation without training

```bash
python scripts/train_detector.py \
  --evaluate-only \
  --model models/rdd_v3_baseline.pt \
  --data configs/detector_rdd_v3_balanced.yaml \
  --split test
```

Use validation data for tuning and run the protected test evaluation **once after choosing the final checkpoint**. A separate recording/route-level local test set is still necessary for the capstone's real-world generalization claims.

## Local video annotation export

```bash
python scripts/export_yolo.py --output exports/yolo_labels_v3
```

All reviewed inventory entries must have an explicit split. If frames with the same `recording_id` appear in different splits, export stops before writing. Existing output directories are protected unless `--overwrite` is explicitly supplied.

## Validation

```bash
python -m pytest tests/test_dataset_safety.py tests/test_rdd2022_conversion.py -q
```

The repository's generated RDD manifests and model checkpoints are not committed. Run the tests and check the resulting `dataset_summary.json` on the computer containing the actual dataset before retraining.

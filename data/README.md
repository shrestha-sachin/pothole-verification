# Data folders

**Use `data/training/pothole_v1/` for the current YOLO26x detector experiment.**
Training automatically selects it through `configs/detector_ready.yaml`.

| Folder | Purpose |
| --- | --- |
| `training/pothole_v1/` | Final balanced dataset: training, validation, and held-out test images with matching labels |
| `raw/` | Original RDD2022, Kaggle, Mapillary, and local video sources; preserve these to rebuild data |
| `annotations/` | Human annotation records; preserve these |
| `metadata/` | Image inventory, source metadata, and split assignments used by preparation scripts |
| `processed/rdd2022_v3/` | Multi-country RDD labels and split metadata used to build the final dataset |
| `processed/kaggle_arcioni_v4/` | Converted Kaggle source and audit records used to build the final dataset |
| `processed/video_frames/` | Extracted local frames referenced by reviewed annotations |
| `processed/verifier_crops/` | Separate verifier/classifier data; not the detector dataset |
| `processed/rdd2022/` | Earlier RDD preparation, kept for existing experiments |
| `processes/` | Legacy preparation folder; not used by the current training script |

`exports/yolo_labels/` outside this directory supplies reviewed local/Mapillary
labels to the builder. Keep it for rebuilding. The final training folder contains
its own image/label copies, so training uses that folder rather than the raw or
intermediate image locations.

Temporary converter test folders were removed from `processed/`. The final
dataset was moved from `processed/pothole_ready_v1/` to `training/pothole_v1/`;
its split manifests, audit paths, preparation defaults, and training config were
updated together.

Run from the project root:

```powershell
.venv\Scripts\python.exe scripts/train_detector.py --dry-run
.venv\Scripts\python.exe scripts/train_detector.py --device 0
```

See [the full training guide](../docs/training_ready.md) for dataset counts,
reproduction, validation, final testing, and limitations.

# Additional road-damage training data

Source: https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes

Publication and attribution: Giordani et al., *Real-world road damage dataset
with potholes, cracks, and maintenance holes* (2026),
https://doi.org/10.1038/s41598-026-46679-4.
Dataset DOI: https://doi.org/10.5281/zenodo.17834373.
The publication specifies CC BY 4.0. Preserve attribution when sharing results.

The downloaded release is version 4. To reproduce in PowerShell:

```powershell
$env:KAGGLEHUB_CACHE = "$PWD/data/raw/kaggle_cache"
.venv\Scripts\python.exe -c "import kagglehub; print(kagglehub.dataset_download('lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes/versions/4'))"
.venv\Scripts\python.exe scripts/prepare_kaggle_roads.py --input data/raw/kaggle_cache/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes/versions/4/data
.venv\Scripts\python.exe scripts/train_detector.py --data configs/detector_kaggle.yaml --dry-run
.venv\Scripts\python.exe scripts/train_detector.py --data configs/detector_kaggle.yaml --device 0 --output models/yolo26x_combined_kaggle.pt
```

The converter validates supplied `labels-YOLO` bounding boxes, clips boxes at image
boundaries (some source annotations extend outside the frame), retains potholes
as class 0, and removes crack/manhole annotations. Images without annotated
potholes receive empty labels as background negatives. This assumes the source
pothole annotations are complete; review samples before drawing conclusions.
The version-4 audit found four pothole boxes crossing image boundaries and one
zero-height crack box. Pothole boxes are clipped; crack annotations are discarded.
Original images and labels are preserved. Converted images/labels, an audit CSV,
and counts are written under `data/processed/kaggle_arcioni_v4`.

All new images go into training. Existing validation/test manifests are preserved
from `configs/detector.yaml`; new source frames never get randomly divided across
evaluation splits. Decoded-pixel hashes remove exact duplicates within the new
source and against all existing splits. Resized, recompressed, and adjacent-frame
near-duplicates are not comprehensively detected. If base test manifests are
missing, the combined config omits the test split and records the missing paths
in its summary; it does not substitute validation data. Source-independent local
evaluation remains necessary. No route-independence claim is made.

The generated `configs/detector_kaggle.yaml` uses absolute paths for this machine;
regenerate it on another computer. Outputs are protected against overwriting;
choose new `--output` and `--config-output` paths for a different experiment.
Use `--base-config configs/detector_v3.yaml` with new output paths to extend V3
instead, after preparing and validating that base dataset.

Compare the baseline and added-data experiments on the same validation split,
using mAP50-95, recall, and false positives on local footage. The currently empty
held-out test split must be addressed before reporting a final benchmark.

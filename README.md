# Pothole Verification — Data Preparation & Annotation

**CS Capstone:** *Reducing False Pothole Reports from Smartphone Video via Visual Verification*

This repository contains:
- A YOLOv8n detector for potholes in full road images
- An ImageNet-pretrained MobileNetV3-Small verifier to filter false detections (shadows, patches, manholes)
- A local Streamlit annotation tool for the Green Bay pilot dataset
- Scripts for inventory building, frame extraction, YOLO export, and verifier crop generation

---

## Quick Start (Mac, local annotation workflow)

### 1. Activate the virtual environment
```bash
source .venv/bin/activate
```

### 2. Place your images and video

| Data | Where to place it |
|------|-------------------|
| Mapillary images | Already in `data/raw/mapillary/segment_001/images/` and `segment_002/images/` |
| Additional Green Bay images | `data/raw/green_bay_images/` |
| Road video (~3 min) | `data/raw/local_video/your_video.mp4` (or `.mov`) |

> **Originals are never overwritten.** All scripts read from `data/raw/` and write outputs elsewhere.

### 3. Generate the image inventory
```bash
python scripts/build_inventory.py
```
Writes `data/metadata/image_inventory.csv`. Safe to rerun — existing review labels are preserved.

### 4. Extract video frames
```bash
python scripts/extract_frames.py --video data/raw/local_video/your_video.mp4
```
- Saves ~1 frame/sec to `data/processed/video_frames/<video-stem>/`
- Frame filenames encode the video ID and timestamp: `<video-stem>_t<timestamp_ms>.jpg`
- Writes a `frames.csv` manifest in the output directory
- Safe to rerun — existing frames are skipped, not overwritten

For filenames with spaces, quote the path:
```bash
python scripts/extract_frames.py --video "data/raw/local_video/Green Bay Road.mp4"
```

Dry-run (inspect video metadata without extracting):
```bash
python scripts/extract_frames.py --video data/raw/local_video/your_video.mp4 --dry-run
```

### 5. Launch the annotation interface
```bash
# Install once (requires network):
pip install streamlit streamlit-drawable-canvas

# Launch:
streamlit run dashboard/annotate.py
```
Opens at `http://localhost:8501`.

### 6. Annotate images

**Annotation rules:**
- A **pothole** = clearly visible missing pavement or open cavity
- **NOT potholes:** shadows on intact pavement, intact repaired patches, manhole covers
- **Damaged patches** that contain an open cavity *may* be potholes
- **Uncertain** = mark as uncertain; excluded from training and scored evaluation

**Workflow:**
1. Browse images in the sidebar; use filters to find unreviewed images
2. Draw tight bounding boxes around every clearly visible pothole on the canvas
3. Check label checkboxes: shadows present, patches present, manholes present
4. If unsure, tick "Uncertain — exclude from export"
5. Click **Save Annotation**, then **Mark as Reviewed**
6. For images with no potholes: click **No Potholes (reviewed)**

**Negative regions** (for the verifier classifier):
- Scroll to "Negative Regions"
- Select subtype: shadow / repaired_patch / manhole / ordinary_road
- Draw the region (or import JSON)

Annotations save to `data/annotations/green_bay/<image_id>.json`.

### 7. Export YOLO detection labels
```bash
python scripts/export_yolo.py
```
Output: `exports/yolo_labels/labels/<image_id>.txt`
- Reviewed + potholes → lines: `0 cx cy w h` (normalized [0,1])
- Reviewed + no potholes → empty `.txt` (true negative)
- Uncertain or unreviewed → **not exported** (never used as negatives)

Validate coordinate math without touching source data:
```bash
python scripts/export_yolo.py --validate
```

### 8. Generate verifier crops
```bash
python scripts/generate_crops.py --context 20
```
Cuts crops from reviewed images with 20 px surrounding road context.
Writes to `data/processed/verifier_crops/` and `data/metadata/crop_inventory.csv`.

> **Important:** These manual crops are for inspection and prototype development only.
> Final pipeline evaluation must use **detector-generated** candidate crops,
> with held-out ground truth to separately measure missed potholes and false positives.

---

## Why frames from the same recording must stay together

Video frames from a continuous recording are spatially correlated — adjacent frames show
the same road section from slightly different angles. Randomly splitting them across
train/val/test causes **data leakage**: the model sees nearly identical views in both
training and evaluation, producing inflated metrics that don't reflect real-world performance.

The same applies to nearby Mapillary images from the same sequence.

**Rule: all frames from one recording go into one split.**

---

## Why pilot images become development data

Once you use the 31 Green Bay images to select a detection threshold or tune any parameter,
they are part of your development loop. Using the same images as your final test set gives
an optimistic and invalid performance estimate.

**You need a separate, never-touched final test set** — collected and held out before any
model decisions are made.

---

## Lab computer (RTX 4090) — RDD2022 integration

The same code runs on the lab computer with no path changes committed:

1. `git clone` (or `git pull`) the repository
2. Place RDD2022 at any path (e.g., `/datasets/RDD2022/`)
3. Run `scripts/convert_rdd2022.py --input /datasets/RDD2022 --output data/raw/rdd2022` (to be implemented) to convert annotations to YOLO format
4. Run `python scripts/build_inventory.py` to add RDD2022 images to the inventory
5. Train the detector: `yolo train model=yolov8n.pt data=configs/detector.yaml`

No machine-specific absolute paths are ever committed to the repository.

---

## Repository layout

```
pothole-verification/
├── .env.example              # Credential template (committed)
├── .gitignore
├── README.md
├── requirements.txt
│
├── dashboard/
│   └── annotate.py           # Streamlit annotation tool
│
├── scripts/
│   ├── build_inventory.py    # Generate data/metadata/image_inventory.csv
│   ├── collect_mapillary.py  # Download Mapillary images
│   ├── convert_rdd2022.py    # (stub) RDD2022 → YOLO conversion
│   ├── export_yolo.py        # Export YOLO detection labels
│   ├── extract_frames.py     # Extract video frames (~1/sec)
│   └── generate_crops.py     # Generate verifier crops from annotations
│
├── src/
│   ├── annotations.py        # Annotation I/O module (shared)
│   ├── detector/             # YOLOv8 detector code
│   ├── verifier/             # MobileNetV3 verifier code
│   ├── tracking/             # Tracking utilities
│   └── evaluation/           # Evaluation metrics
│
├── configs/                  # Training and model configs
├── models/                   # Model weights (git-ignored)
├── results/                  # Evaluation results
│
└── data/                     # All data (git-ignored except manifests)
    ├── raw/
    │   ├── mapillary/
    │   │   ├── segment_001/manifest.csv   ← committed
    │   │   └── segment_002/manifest.csv   ← committed
    │   ├── local_video/                   ← git-ignored (place video here)
    │   └── green_bay_images/              ← git-ignored
    ├── annotations/green_bay/             ← git-ignored
    ├── processed/
    │   ├── video_frames/                  ← git-ignored
    │   └── verifier_crops/                ← git-ignored
    ├── metadata/                          ← git-ignored
    └── splits/
```

## Files committed to GitHub

| File/Path | Reason |
|-----------|--------|
| `scripts/*.py` | Source code |
| `src/annotations.py` | Shared annotation module |
| `dashboard/annotate.py` | Annotation app |
| `data/raw/mapillary/*/manifest.csv` | Reproducibility — documents Mapillary IDs collected |
| `.env.example` | Documents required credentials |
| `requirements.txt` | Dependency list |
| `README.md` | Documentation |
| `.gitignore` | Git protection rules |

## Files NOT committed (git-ignored)

| Pattern | Reason |
|---------|--------|
| `data/raw/mapillary/*/images/` | Large binary images |
| `data/raw/local_video/` | Large video file |
| `data/processed/` | Generated outputs |
| `data/metadata/` | Generated inventory CSVs |
| `data/annotations/` | Human labels (commit with `git add -f` if desired) |
| `exports/` | Regenerated from annotations |
| `*.pt`, `*.pth` | Model weights |
| `.venv/` | Virtual environment |
| `.env` | Secrets |

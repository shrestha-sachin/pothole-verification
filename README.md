# Pothole Verification — Detection Dashboard & Annotation Tool

**CS Capstone:** *Reducing False Pothole Reports from Smartphone Video via Visual Verification*

This repository contains the full system for detecting, grouping, reviewing, and comparing pothole
candidate events from road video using YOLO11n + BoT-SORT + MobileNetV3-Small.

---

## Quick Start — Main Dashboard (Phase 1)

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Launch the detection/review dashboard
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

### Workflow
1. **Process Video** — select your `best.pt` checkpoint and a road video, adjust FPS / confidence, click **Analyze Video**
2. **Review Events** — inspect each grouped candidate event (image, timestamp, confidence, policy flags), confirm / reject / mark uncertain
3. **Export** — download a CSV report with all events and review decisions
4. **Overview** — summary statistics and charts across all processed videos

> **Note:** The existing annotation tool is still at `streamlit run dashboard/annotate.py`

---

## Architecture

```
Video → Frame sampler (5 FPS) → YOLO11n detector → BoT-SORT tracker
                                                         ↓
                                               Candidate events grouped
                                                         ↓
                              ┌──────────────────────────┤
                              │ Policy A: conf threshold  │
                              │ Policy B: ≥2 detections   │
                              │ Policy C: MobileNetV3 (*)  │
                              └──────────────────────────┘
                                                         ↓
                                         SQLite DB + Evidence images
                                                         ↓
                                    Human review → CSV export
```

(*) Policy C verifier not trained yet; dashboard shows "Verifier not configured" gracefully.

---

## Module Layout (new files)

| File | Purpose |
|------|---------|
| `dashboard/app.py` | **New** — Main detection/review Streamlit dashboard |
| `src/detector/inference.py` | YOLO11n wrapper (load-once singleton) |
| `src/detector/video_sampler.py` | Frame sampler preserving original timestamps |
| `src/tracking/event_grouper.py` | BoT-SORT event grouper + evidence extractor |
| `src/verifier/interface.py` | MobileNetV3-Small verifier (graceful stub) |
| `src/evaluation/policies.py` | Policy A/B/C application on shared events |
| `src/database.py` | SQLite persistence (videos, events, detections, reviews) |

---

## Model checkpoints

| Checkpoint | Location | Purpose |
|-----------|----------|---------|
| `yolo11n.pt` | project root | Base COCO YOLO11n (autodownloaded) |
| `models/best.pt` | `models/` (git-ignored) | **Your fine-tuned pothole detector** — place here |
| `models/verifier_v1.pt` | `models/` (git-ignored) | MobileNetV3-Small verifier (not trained yet) |

---

## Installation

```bash
# Python 3.11+ recommended
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install lapx          # BoT-SORT Hungarian solver
pip install torchvision   # for MobileNetV3 verifier (when ready)
```

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
pip install "streamlit>=1.45" Pillow

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
4. If unsure, choose **Uncertain — exclude from training**
5. Choose a review decision and click **Save review** or **Save review & next**
6. For images with no potholes: save a **Reviewed** decision with no pothole boxes

**Negative regions** (for the verifier classifier):
- Choose Shadow, Repaired patch, Manhole, or Ordinary road in the canvas category selector
- Draw the region, then adjust it with Select if necessary

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

## RDD2022 Dataset Preparation & Splitting Pipeline

The Road Damage Dataset 2022 (RDD2022) is used to augment the training data for single-class pothole detection (`D40` mapped to YOLO class 0).

### 1. Splitting Strategy (80 / 10 / 10 Default)

- **Default Split:** 80% Training, 10% Validation, and 10% Held-Out Test.
- **Reproducibility:** Configurable random seed (defaults to `42`).
- **Stratification:** Maintains representation of pothole-positive (`D40` present) and negative background road images across splits.
- **Group Protection:**
  - Prefers verified route, sequence, location, or camera metadata when available (`--metadata-csv`).
  - Automatically identifies exact content duplicates (SHA-256 match) and guarantees that duplicate images are placed in the **exact same split** to prevent data leakage.
  - **Honest Limitation:** Standard RDD2022 Pascal VOC XML files lack GPS or physical road sequence tags. When no external sequence logs are supplied, the converter falls back to sample grouping. Geographic independence is **not** claimed unless verified external sequence metadata is provided.
- **Zero Leakage:** Asserts that train, validation, and test splits have 0% image ID overlap and 0% content hash overlap.
- **Empty Split Protection:** Guarantees non-empty train, validation, and test sets.

### 2. Single-Class Pothole & Negative Sample Handling

- **Pothole Positive:** Any image containing `D40` (pothole), even if co-occurring with cracks (`D00`, `D10`, `D20`), remains classified as pothole-positive. In pothole-only mode, only `D40` bounding boxes are exported as class 0.
- **Candidate Negatives:** Images containing road distress without `D40` (or clean road images) are candidate negatives **only when their XML annotations are valid**. Corrupt or malformed XML files are rejected, never treated as true negatives.
- **Hard Negative Tiers:**
  - **Tier 1:** Alligator crack (`D20`) + Transverse crack (`D10`)
  - **Tier 2:** Alligator crack (`D20`)
  - **Tier 3:** Transverse crack (`D10`)
  - **Tier 4:** Longitudinal crack (`D00`) or clean road background
  > **Research Note:** Crack annotations (`D10`, `D20`) serve as road distress negatives to teach YOLO not to false-alarm on surface cracks. They are **not** equivalent to the project's target video look-alikes (shadows, repaired asphalt patches, and manhole covers), which are supplied by our human-reviewed Green Bay dataset.
- **In-Split Negative Subsampling:**
  - Negative subsampling occurs **strictly within each assigned split** (train negatives from train candidates, val negatives from val candidates, test negatives from test candidates).
  - Images **never cross split boundaries** to satisfy class-balance targets.
  - Configurable ratio (e.g. `--neg-ratio 0.25` for ~20% background images, matching Ultralytics recommendations) or explicit counts (`--train-negatives`, `--val-negatives`, `--test-negatives`).
  - Unselected candidate negatives remain in their assigned split and are tracked in `split_metadata.csv` with `selected_in_manifest=0`. Full candidate lists are preserved in `*_all_negatives.txt`.

### 3. Preparation Commands (Lab Computer)

Run these commands on your lab computer where RDD2022 is stored:

```bash
# Activate virtual environment
source .venv/bin/activate

# 1. Prepare US-only RDD2022 subset with 80/10/10 split and 20% background negatives
python scripts/convert_rdd2022.py \
    --input /path/to/RDD2022 \
    --output data/processed/rdd2022 \
    --countries United_States \
    --train-split 0.80 \
    --val-split 0.10 \
    --test-split 0.10 \
    --neg-ratio 0.25 \
    --seed 42

# 2. (Optional) Multi-country preparation (e.g. US + Japan)
python scripts/convert_rdd2022.py \
    --input /path/to/RDD2022 \
    --output data/processed/rdd2022 \
    --countries United_States,Japan \
    --neg-ratio 0.25 \
    --seed 42

# 3. Dry-run inspection (inspect counts without writing files)
python scripts/convert_rdd2022.py --input /path/to/RDD2022 --countries United_States --dry-run
```

### 4. Output Manifests & Audit Artifacts

The conversion generates:
- `data/processed/rdd2022/train.txt`: List of absolute paths for training images
- `data/processed/rdd2022/val.txt`: List of absolute paths for validation images
- `data/processed/rdd2022/test.txt`: List of absolute paths for held-out test images
- `data/processed/rdd2022/split_metadata.csv`: Full provenance audit (image ID, country, group ID, grouping method, split, selected status, bounding box counts, negative tier, SHA-256)
- `data/processed/rdd2022/dataset_summary.json`: Summary statistics (total images, class distribution, D40 bounding box totals per split, duplicate clusters)
- `configs/detector_rdd_balanced.yaml`: RDD2022 dataset YAML for Ultralytics YOLO

### 5. Training & Evaluating YOLO11n

```bash
# 1. Export local Green Bay human-reviewed annotations
python scripts/export_yolo.py

# 2. Dry-run validation of dataset configuration and paths
python scripts/train_detector.py --data configs/detector_combined.yaml --dry-run

# 3. Train YOLO11n on GPU with the combined dataset (50 epochs, batch 16)
python scripts/train_detector.py \
    --data configs/detector_combined.yaml \
    --model yolo11n.pt \
    --epochs 50 \
    --batch 16 \
    --device 0

# 4. Train only on balanced RDD2022 US data
python scripts/train_detector.py \
    --data configs/detector_rdd_balanced.yaml \
    --epochs 50 \
    --batch 16 \
    --device 0
```

### 6. Held-Out Test Set Protection

> [!IMPORTANT]
> The test split (`test.txt`) is strictly held out. Training, validation loss monitoring, early stopping, and hyperparameter tuning evaluate **exclusively** on `val.txt`.
> The test split is never evaluated during model training to avoid data snooping and maintain research-grade evaluation integrity.
> To run final evaluation on the held-out test set only after training is completely frozen:
> `python scripts/train_detector.py --data configs/detector_combined.yaml --evaluate-test`

---

## Lab computer (RTX 4090) — GPU training setup

The same code runs on the lab computer with no path changes committed:

1. `git clone` (or `git pull`) the repository
2. Place RDD2022 at any path (e.g., `/datasets/RDD2022/`)
3. Run `python scripts/convert_rdd2022.py --input /datasets/RDD2022 --countries United_States --neg-ratio 0.25`
4. Run `python scripts/export_yolo.py` to ensure local annotations are exported
5. Train the detector: `python scripts/train_detector.py --data configs/detector_combined.yaml --device 0`

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
│   ├── build_inventory.py        # Generate data/metadata/image_inventory.csv
│   ├── collect_mapillary.py      # Download Mapillary images
│   ├── convert_rdd2022.py        # RDD2022 → YOLO conversion & 80/10/10 splitting
│   ├── subsample_rdd_negatives.py# Subsample hard crack negatives within splits
│   ├── export_yolo.py            # Export YOLO detection labels
│   ├── extract_frames.py         # Extract video frames (~1/sec)
│   ├── generate_crops.py         # Generate verifier crops from annotations
│   └── train_detector.py         # Train and evaluate YOLO11n on GPU/CPU
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


## Dashboard safety and evaluation workflow

Start the research dashboard with `python -m streamlit run dashboard/app.py`.
Streamlit 1.45 or newer is required for native Material icons. The interface uses
system typography, a neutral light theme, and explicit text status labels. Users
can select another Streamlit theme from Settings.

Replacement runs write evidence to unique `results/evidence/runs/<run_id>/`
directories. Existing events and reviews remain active until the replacement
completes and its database transaction commits. Previous records, observations,
review decisions and evidence references are retained in `run_archive` and can
be downloaded as JSON from Export. Evidence files from archived runs are retained;
archive downloads contain references, not the image/video bytes. Review transfer
uses timestamp proximity and IoU >= 0.5, with each new event assigned at most one
old review. This is heuristic: inspect transferred decisions before research use.

Evaluation JSON must include exact `video_id`, numeric `timestamp_ms`,
`review_status: "reviewed"`, and `ground_truth_verified: true`. Each pothole box
must have a human-verified `gt_event_id`. Reuse that ID for all sightings of the
same physical pothole. Legacy frame annotations are still training data but are
not automatically scored as evaluation ground truth. Filename guesses and
unverified automatic clustering are no longer accepted.

CSV evaluation requires timestamps, boxes, and `gt_event_id` for potholes. An
explicit `video_id`, when supplied, must equal the selected video's ID. The UI
requires confirmation of full-video annotation coverage, a validation/test split,
and frozen thresholds for held-out tests. Timestamp matching defaults to exact
alignment; a nonzero tolerance requires a documented justification. Policy C
headline metrics require every candidate to have a verifier score. Predictions
and human reviews remain separate.

Run settings (including checkpoint SHA-256 identities, sampling and policy
thresholds) are saved in `videos.run_config` and included in event CSV exports.
Legacy settings are displayed as unknown rather than inferred.

### Focused regression checks

Install development test support with `python -m pip install pytest` and run:

```bash
python -m pytest tests/test_run_safety.py tests/test_dashboard_flows.py tests/test_database_cleanup.py tests/test_event_evaluation.py tests/test_rdd2022_conversion.py tests/test_ground_truth_loader.py tests/test_policies.py tests/test_evidence_and_crop.py -q
```

Synthetic model services exercise detector, processing, verifier and persistence
failure paths without downloading checkpoints or starting training. Streamlit
AppTest covers navigation and review controls. These tests do not establish real
model accuracy or replace a visual browser check on your own footage.


## Annotation workspace

Launch with `python -m streamlit run dashboard/annotate.py`. The editor uses a
bundled, dependency-free canvas; `streamlit-drawable-canvas` is no longer required.
There is one navigation pair, one drawing toolbar, and one review form.

- **Draw**: drag in either direction. Choose pothole, shadow, repaired patch,
  manhole, or ordinary road before drawing.
- **Select**: click a box to move it, or drag one of its eight handles to resize.
  The selected-box inspector supports exact original-image pixel coordinates,
  reclassification, and optional physical pothole IDs.
- **Pan**: drag the image, or hold Space temporarily. Wheel zooms at the pointer;
  Fit restores the complete frame. Fullscreen uses the same editor and controls.
- Undo/redo and Delete work locally. D/V/H switch tools; Ctrl/Cmd+Z undoes.
  Shortcuts are scoped to the editor and never intercept text fields or page buttons.
- Completed box edits save automatically without resetting the canvas or marking
  the image reviewed. Review decisions, flags and notes save only when the review
  form is submitted. Unsaved form changes are not carried across images.
- Annotation metadata and physical IDs survive edits. Any box edit invalidates
  prior ground-truth verification; verify the frame again before scored evaluation.
  Existing annotation JSON paths and coordinate conventions are preserved.

The old batch actions that automatically assigned repaired-road labels or invented
central patch crops have been removed. No existing annotations are rewritten by
opening the tool. Errors are reported with a contextual Retry save control.

Optional browser interaction regression:

```bash
python -m pip install pytest playwright
python -m playwright install chromium
python -m pytest tests/test_annotation_editor.py tests/test_annotation_canvas_browser.py -q
```

The browser test uses a synthetic image and checks reverse-direction drawing,
local pointer motion, selection, moving, resizing, category changes, undo/redo,
deleting and toolbar click targets. It never reads or modifies real footage.

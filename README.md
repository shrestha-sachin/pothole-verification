# Reducing False Pothole Reports from Smartphone & Dashcam Video via Visual Verification

A two-stage computer vision pipeline for road pothole detection and false-positive suppression:
1. **Stage 1 — Detector (YOLO11 / YOLO26):** High-recall candidate pothole detection on road imagery and video frames.
2. **Stage 2 — Tracking & Visual Verifier (BoT-SORT + MobileNetV3-Small):** Groups frame-level detections into physical road events and filters out false positives (shadows, repaired asphalt patches, manhole covers, cracks).

---

## Approved Data Sources

The active repository exclusively uses three US/local road data sources:
1. **RDD2022 USA (`rdd2022_usa`):** United States subset of RDD2022 (`D40` mapped to class `0: pothole`; crack-only images used as hard negatives).
2. **Local Footage (`local/smartphone`, `local/dashcam`):** Locally recorded smartphone and dashcam videos/images from Green Bay, WI.
3. **Mapillary (`mapillary`):** Locally relevant road imagery collected via the Mapillary API.

---

## Repository Structure

```text
pothole-verification/
├── configs/                 # Portable YAML configurations (detector, verifier, training, inference, tracking)
├── data/
│   ├── raw/                 # Untouched original images, videos, and XMLs (checksum-verified)
│   ├── annotated/           # Canonical reviewed images, YOLO labels, and metadata
│   ├── processed/           # Reproducible training-ready detector & verifier splits
│   ├── manifests/           # Migration & master manifests
│   └── splits/              # Split definitions and dataset statistics
├── src/
│   ├── data/                # Annotation I/O, validation, and canonical dataset pipeline
│   ├── detector/            # YOLO inference and video frame sampling
│   ├── verifier/            # MobileNetV3-Small interface, preprocessing, and training
│   ├── tracking/            # BoT-SORT event grouping
│   ├── preprocessing/       # Preprocessing helpers
│   ├── evaluation/          # Event matching, policies, and metrics
│   ├── video/               # Annotated video rendering & review metadata
│   └── utils/               # Centralized config, paths, and SQLite database
├── scripts/                 # Minimal CLI entrypoints
├── dashboard/               # Streamlit detection & annotation dashboards
├── models/                  # Organized detector and verifier checkpoints
├── runs/                    # Experiment logs, curves, and confusion matrices
├── results/                 # Evaluations, rendered predictions, reports, SQLite DB
└── tests/                   # Automated unit & integration tests
```

---

## Environment Setup

### Windows (Lab Workstation — NVIDIA RTX 4090)
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\pip install -r requirements.txt
```

### macOS (Apple Silicon / MPS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt
```

### Modal (Cloud GPU)
Mount the repository and `configs/` + `data/processed/` into the container, and set `POTHOLE_DEVICE=0` (or pass `--device 0`).

Copy `.env.example` to `.env` and add `MAPILLARY_ACCESS_TOKEN` if collecting new Mapillary imagery.

---

## Core Workflows

### 1. Prepare & Validate the Canonical Dataset
Builds `data/processed/detector/` and `data/manifests/master_manifest.csv` deterministically from `data/annotated/`, enforcing recording-level and sequence-level split isolation:
```powershell
.venv\Scripts\python.exe scripts/prepare_dataset.py --prepare-verifier
.venv\Scripts\python.exe scripts/validate_dataset.py
```

### 2. Extract Video Frames & Annotate
```powershell
# Extract frames from a new video
.venv\Scripts\python.exe scripts/extract_frames.py --video data/raw/local/dashcam/videos/<video>.mp4 --fps 0.33

# Launch the annotation dashboard
.venv\Scripts\python.exe -m streamlit run dashboard/annotate.py
```

### 3. Train Detector (YOLO11 or YOLO26)
```powershell
# Dry-run validation
.venv\Scripts\python.exe scripts/train_detector.py --dry-run

# Train YOLO26x on GPU (defaults read from configs/training.yaml & configs/detector.yaml)
.venv\Scripts\python.exe scripts/train_detector.py --device 0 --experiment-name yolo26x_us_run --output models/detector/yolo26x_us_run.pt

# Or train YOLO11n
.venv\Scripts\python.exe scripts/train_detector.py --model models/detector/yolo11n.pt --batch 16 --device 0 --experiment-name yolo11n_us_run --output models/detector/yolo11n_us_run.pt
```

### 4. Mine Hard Negatives & Train Visual Verifier
```powershell
# Mine false-positive crops strictly from the training split
.venv\Scripts\python.exe scripts/mine_hard_negatives.py

# Train MobileNetV3-Small verifier
.venv\Scripts\python.exe scripts/train_verifier.py --data data/processed/verifier --output models/verifier/verifier_v2.pt
```

### 5. Inference, Evaluation & Detection Dashboard
```powershell
# Run end-to-end video analysis
.venv\Scripts\python.exe scripts/analyze_video.py --video data/raw/local/smartphone/videos/local_smartphone_video_0002.mp4

# Evaluate on validation or held-out test split
.venv\Scripts\python.exe scripts/evaluate.py --split test

# Launch main detection & evaluation dashboard
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

---

## Split Protection & Research Integrity
- Held-out `test` images (RDD2022 USA) and held-out test video (`local_smartphone_video_0002.mp4`) are never used for training, hyperparameter tuning, or hard-negative mining.
- Frames from the same driving session or video stay in the same split.
- Unreviewed frames are excluded and never treated as negative examples.

# Pothole Verification — Detection Dashboard & Annotation Tool

**CS Capstone:** *Reducing False Pothole Reports from Smartphone Video via Visual Verification*

This repository contains the full system for detecting, grouping, reviewing, and comparing pothole
candidate events from road video using YOLO11n + BoT-SORT + MobileNetV3-Small.


Experiment 2 uses `data/training/pothole_experiment_2_v2/` and `configs/detector_experiment_2.yaml`. Follow the [Experiment 2 workflow](docs/experiment_2_workflow.md) for collection, review, preparation, validation, and the later RTX 4090 training command. The [historical V1 guide](docs/training_ready.md) records the first experiment.

---

## Quick Start — Main Dashboard (Phase 1)

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Launch the detection/review dashboard
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

### Everyday workflow

1. **Analyze a video** - upload road footage or choose a saved recording. Use **Find potholes to review** and enter a road or area name to create a review list.
2. **Review findings** - check the close-up, road view, and video; choose Pothole, Not a pothole, Unsure, or a possible duplicate/grouping issue. Save your decision or save and move to the next unreviewed finding.
3. **Reports** - download a CSV with readable columns. Confirmed findings are included by default; other decisions can be included explicitly.
4. **Watch videos** - play recordings created with **Create a video with marked potholes**. These videos show possible detections and do not create a review list or report.

**Settings** holds model paths and analysis controls. Defaults use `models/best.pt` as the pothole detector; the additional image-check model is optional. Settings are saved for the installation in `configs/dashboard_settings.json` and survive navigation and app restarts. Enable **Show research tools** to access evaluation, technical scores, full analysis exports, training crop inventories, and previous run archives. This display option is not an access-control mechanism.

Road names are entered by the operator. No GPS location is recorded. Findings may include repeated views of the same physical pothole, so finding totals do not establish a unique defect count. Reports retain stable finding IDs for reference.

The separate annotation tool remains available with `streamlit run dashboard/annotate.py`.

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


## Collect and annotate road images

The [Experiment 2 workflow](docs/experiment_2_workflow.md) covers new recordings and the existing annotation editor. The original inventory can still be opened with `streamlit run dashboard/annotate.py`. Use the collection option to review a new recording in its own inventory.

The existing YOLO export and verifier crop utilities remain available for historical reproduction and verifier development. Experiment 2 reads the original labels and image reviews directly.

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

## Prepare and verify Experiment 2

```powershell
.venv\Scripts\python.exe scripts/prepare_experiment_2.py --dry-run
.venv\Scripts\python.exe scripts/prepare_experiment_2.py --verify-only
.venv\Scripts\python.exe scripts/train_detector.py --data configs/detector_experiment_2.yaml --dry-run
```

Follow the Experiment 2 workflow for dataset builds, negative ratios, review queues, and training. [Experiment 2 workflow](docs/experiment_2_workflow.md) · [Historical preparation notes](docs/safe_training_workflow.md)

## Repository layout

| Location | What belongs here |
| --- | --- |
| `dashboard/` | Road review app, annotation editor, and dataset review |
| `src/`, `scripts/`, `tests/` | Shared code, command-line tools, and automated tests |
| `configs/` | Portable model and dataset configurations |
| `data/raw/`, `data/annotations/`, `data/metadata/` | Original media, manual labels, and source inventories |
| `data/processed/`, `data/training/`, `exports/` | Derived inputs, prepared datasets, and exported labels |
| `models/`, `weights/` | Model weights and checkpoints |
| `runs/detect/` | Training runs, metrics, and run-specific checkpoints |
| `results/dataset_audits/` | Current dataset audits and review queues |
| `results/annotated_videos/`, `results/evidence/` | Dashboard videos and evaluation evidence |
| `results/machine_configs/` | Machine-local configs used by training runs |
| `docs/` | Workflows and reference material |

[Data guide](data/README.md)

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

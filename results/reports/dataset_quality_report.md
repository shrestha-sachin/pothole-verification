# Pothole Detection Dataset Quality & Curation Report

**Active Experiment Profile:** `exp_b_roi_90_10` (`roi`)

## 1. Original Annotated Inventory (Approved Sources Only)

| Source | Total | Reviewed | Unreviewed | Uncertain | Positives | Reviewed Negatives | Boxes |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rdd2022_usa` | 4805 | 4805 | 0 | 0 | 116 | 4689 | 135 |
| `local_smartphone` | 154 | 154 | 0 | 0 | 38 | 116 | 90 |
| `local_dashcam` | 840 | 103 | 736 | 1 | 58 | 45 | 128 |
| `mapillary` | 206 | 206 | 0 | 0 | 17 | 189 | 27 |

## 1B. Object vs. Physical Pothole Accounting (Explicit Distinction)

> [!IMPORTANT]
> Annotated 2D bounding boxes and heuristic temporal encounters are **not** verified physical pothole counts. Verified physical counts require human `gt_event_id` linking across frames.

| Source | Images | Positive Images | Bounding-Box Instances | Verified Physical Potholes (`gt_event_id`) | Est. Temporal Encounters (Unverified <=4s heuristic) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `rdd2022_usa` | 4805 | 116 | 135 | 0 | N/A (still images) |
| `local_smartphone` | 154 | 38 | 90 | 0 | 16 |
| `local_dashcam` | 840 | 58 | 128 | 0 | 36 |
| `mapillary` | 206 | 17 | 27 | 0 | N/A (still images) |

## 2. Active Curated Detector Splits

| Split | Images | Positive Images | Negative Images | Pothole Boxes |
| :--- | ---: | ---: | ---: | ---: |
| **train** | 201 | 181 | 20 | 292 |
| **val** | 107 | 36 | 71 | 76 |
| **test** | 15 | 12 | 3 | 12 |

### Training Composition (Target vs. Achieved)
- **Target:** Positive `90%`, Hard-Negative `8%`, Ordinary-Negative `2%`
- **Achieved:** Positive `90.05%`, Hard-Negative `7.96%`, Ordinary-Negative `1.99%`
- **Unused reviewed negatives preserved:** 873 (plus full RDD2022 USA pool)

## 3. Small-Object Effective Pixel Resolution (Full Frame vs. ROI vs. Horizontal 2-Tile)

| Mode | imgsz=640 Median (W×H px) | imgsz=960 Median (W×H px) | imgsz=1280 Median (W×H px) | % <16px @960 |
| :--- | :---: | :---: | :---: | ---: |
| `full_frame` | 22.0×12.5 | 33.0×18.8 | 44.0×25.0 | 43.4% |
| `roi` | 24.0×13.4 | 36.0×20.0 | 48.0×26.7 | 37.9% |
| `roi_tiled` | 33.8×18.6 | 50.7×28.0 | 67.5×37.3 | 13.2% |

## 4. ROI Coverage, Deduplication & Exclusions
- **ROI Box Retention:** 380 / 380 boxes retained (0 dropped, 0 truncated).
- **Deduplication:** {'exact_duplicates': 0, 'cross_split_near_duplicates': 0, 'adjacent_video_negative_duplicates': 1}
- **Excluded/Flagged records logged:** 738 (see `data/manifests/excluded_images.csv`).

## 5. Visual Previews
- `results/reports/visuals/roi_before_after.png`
- `results/reports/visuals/roi_tiled_preview.png`
- `results/reports/visuals/positive_samples.png`
- `results/reports/visuals/hard_negative_samples.png`
- `results/reports/visuals/augmentation_preview.png`
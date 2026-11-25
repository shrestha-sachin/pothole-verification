# Detector Experiment Comparison Report (`val` split, Original-Image Space)

All models are compared against unchanged original images and ground-truth bounding boxes.
- **mAP50 / mAP50-95:** Confidence-ranked PR integration down to `conf_floor=0.001`.
- **Precision / Recall / F1 / FP-per-image:** Evaluated separately at fixed operating threshold (`conf=0.25`, `IoU=0.50`).

## 1. Overall Results

| Checkpoint / Profile | Preprocessing | imgsz | mAP50 | mAP50-95 | Precision | Recall | F1 | FP / img | Latency (ms/img) | Status |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `yolo11n_exp_a.pt` (`exp_a_full_90_10`) | `full_frame` | 960 | 0.1150 | 0.0559 | 0.3214 | 0.1184 | 0.1731 | 0.1776 | 22.7 | Evaluated |
| `yolo11n_exp_b.pt` (`exp_b_roi_90_10`) | `roi` | 960 | 0.1128 | 0.0502 | 0.2619 | 0.1447 | 0.1864 | 0.2897 | 20.2 | Evaluated |
| `yolo11n_exp_c.pt` (`exp_c_tiled_90_10`) | `roi_tiled` | 960 | 0.1686 | 0.0591 | 0.2299 | 0.2632 | 0.2454 | 0.6262 | 30.9 | Evaluated |
| `yolo26x_us_v5_960.pt` (`legacy_us_v5_full_frame`) | `full_frame` | 960 | 0.1226 | 0.0451 | 0.3000 | 0.1184 | 0.1698 | 0.1963 | 31.4 | Evaluated |
| `yolo11n_exp_d_best_1280` (`exp_d_best_1280`) | `roi` | 1280 | - | - | - | - | - | - | - | Pending training |
| `yolo11n_exp_e_best_80_20` (`exp_e_best_80_20`) | `roi` | 960 | - | - | - | - | - | - | - | Pending training |

## 2. Metrics by Data Source (Evaluated Checkpoints)

| Checkpoint | Source | Images | GT Boxes | mAP50 | mAP50-95 | Precision | Recall | F1 | FP / img | Latency (ms/img) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo11n_exp_a.pt` | `local_dashcam` | 53 | 61 | 0.0086 | 0.0024 | 0.0000 | 0.0000 | 0.0000 | 0.1321 | 33.3 |
| `yolo11n_exp_a.pt` | `mapillary` | 39 | 3 | 0.5556 | 0.1111 | 0.2000 | 0.6667 | 0.3077 | 0.2051 | 12.7 |
| `yolo11n_exp_a.pt` | `rdd2022_usa` | 15 | 12 | 0.6350 | 0.3581 | 0.6364 | 0.5833 | 0.6087 | 0.2667 | 11.6 |
| `yolo11n_exp_b.pt` | `local_dashcam` | 53 | 61 | 0.0330 | 0.0084 | 0.0667 | 0.0164 | 0.0263 | 0.2642 | 30.7 |
| `yolo11n_exp_b.pt` | `mapillary` | 39 | 3 | 0.0754 | 0.0175 | 0.0714 | 0.3333 | 0.1176 | 0.3333 | 9.0 |
| `yolo11n_exp_b.pt` | `rdd2022_usa` | 15 | 12 | 0.6164 | 0.3245 | 0.6923 | 0.7500 | 0.7200 | 0.2667 | 12.3 |
| `yolo11n_exp_c.pt` | `local_dashcam` | 53 | 61 | 0.0983 | 0.0352 | 0.2222 | 0.1639 | 0.1887 | 0.6604 | 38.4 |
| `yolo11n_exp_c.pt` | `mapillary` | 39 | 3 | 0.6667 | 0.1167 | 0.0667 | 0.6667 | 0.1212 | 0.7179 | 23.3 |
| `yolo11n_exp_c.pt` | `rdd2022_usa` | 15 | 12 | 0.6454 | 0.2696 | 0.6667 | 0.6667 | 0.6667 | 0.2667 | 24.6 |
| `yolo26x_us_v5_960.pt` | `local_dashcam` | 53 | 61 | 0.0743 | 0.0204 | 0.2353 | 0.0656 | 0.1026 | 0.2453 | 44.6 |
| `yolo26x_us_v5_960.pt` | `mapillary` | 39 | 3 | 0.0316 | 0.0065 | 0.0000 | 0.0000 | 0.0000 | 0.0513 | 17.5 |
| `yolo26x_us_v5_960.pt` | `rdd2022_usa` | 15 | 12 | 0.4213 | 0.2226 | 0.4545 | 0.4167 | 0.4348 | 0.4000 | 21.1 |

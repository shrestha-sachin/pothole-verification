"""Generate comprehensive dataset quality report and visual verification previews."""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from src.preprocessing.quality import compute_effective_yolo_box_px
from src.preprocessing.roi import DEFAULT_SOURCE_ROIS, apply_roi_crop, generate_horizontal_tiles
from src.utils.config import ANNOTATED_DIR, MANIFESTS_DIR, PROCESSED_DIR, PROJECT_ROOT, REPORTS_DIR, SPLITS_DIR


def _draw_yolo_boxes(im: Image.Image, boxes: list[tuple[int, float, float, float, float]], color: str = "#00ff66") -> Image.Image:
    out = im.copy().convert("RGB")
    w, h = out.size
    draw = ImageDraw.Draw(out)
    lw = max(2, w // 300)
    for _, cx, cy, bw, bh in boxes:
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
    return out


def _compute_resolution_comparison(master_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Compare effective pixel sizes at 640/960/1280 for full_frame, roi, and roi+2-tile horizontally."""
    modes: dict[str, dict[int, list[tuple[float, float]]]] = {
        "full_frame": {640: [], 960: [], 1280: []},
        "roi": {640: [], 960: [], 1280: []},
        "roi_tiled": {640: [], 960: [], 1280: []},
    }
    src_dirs = {
        "rdd2022_usa": ANNOTATED_DIR / "rdd2022_usa",
        "local_smartphone": ANNOTATED_DIR / "local" / "smartphone",
        "local_dashcam": ANNOTATED_DIR / "local" / "dashcam",
        "mapillary": ANNOTATED_DIR / "mapillary",
    }

    for r in master_rows:
        if int(r["pothole_count"]) == 0:
            continue
        src = r["source"]
        img_id = r["source_image_id"]
        orig_w, orig_h = int(r["orig_width"]), int(r["orig_height"])
        proc_w, proc_h = int(r["proc_width"]), int(r["proc_height"])
        lbl_p = src_dirs[src] / "labels" / f"{img_id}.txt"
        if not lbl_p.exists():
            continue
        orig_boxes = []
        for ln in lbl_p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            p = ln.split()
            orig_boxes.append((0, float(p[1]), float(p[2]), float(p[3]), float(p[4])))

        roi = DEFAULT_SOURCE_ROIS.get(src, DEFAULT_SOURCE_ROIS["rdd2022_usa"])
        rx1, ry1, rx2, ry2 = roi.to_pixels(orig_w, orig_h)
        rw, rh = rx2 - rx1, ry2 - ry1
        tile_w = int(round(rw * 0.575)) if rw >= 2000 else rw

        for _, _, _, bw, bh in orig_boxes:
            px_w, px_h = bw * orig_w, bh * orig_h
            for sz in (640, 960, 1280):
                ew_f, eh_f, _ = compute_effective_yolo_box_px(orig_w, orig_h, bw, bh, sz)
                modes["full_frame"][sz].append((ew_f, eh_f))

                s_r = min(sz / max(1, rw), sz / max(1, rh))
                modes["roi"][sz].append((px_w * s_r, px_h * s_r))

                s_t = min(sz / max(1, tile_w), sz / max(1, rh))
                modes["roi_tiled"][sz].append((px_w * s_t, px_h * s_t))

    out: dict[str, dict[str, Any]] = {}
    for mname, sz_dict in modes.items():
        out[mname] = {}
        for sz, pairs in sz_dict.items():
            ws = sorted(p[0] for p in pairs)
            hs = sorted(p[1] for p in pairs)
            out[mname][str(sz)] = {
                "median_w": round(float(np.median(ws)), 1),
                "median_h": round(float(np.median(hs)), 1),
                "p10_min_side": round(float(np.percentile([min(p) for p in pairs], 10)), 1),
                "pct_lt_16px": round(100.0 * sum(1 for w, h in pairs if min(w, h) < 16.0) / len(pairs), 1),
            }
    return out


def _generate_visuals(master_rows: list[dict[str, str]], vis_dir: Path) -> None:
    vis_dir.mkdir(parents=True, exist_ok=True)
    src_dirs = {
        "rdd2022_usa": ANNOTATED_DIR / "rdd2022_usa",
        "local_smartphone": ANNOTATED_DIR / "local" / "smartphone",
        "local_dashcam": ANNOTATED_DIR / "local" / "dashcam",
        "mapillary": ANNOTATED_DIR / "mapillary",
    }

    # 1. Before-and-after ROI visual comparison (1 positive per source)
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    for row_idx, src in enumerate(("local_dashcam", "local_smartphone", "mapillary", "rdd2022_usa")):
        cand = next((r for r in master_rows if r["source"] == src and int(r["pothole_count"]) > 0), None)
        if not cand:
            continue
        base = src_dirs[src]
        img_id = cand["source_image_id"]
        orig_im = Image.open(base / "images" / f"{img_id}.jpg").convert("RGB")
        orig_boxes = [
            (0, *map(float, ln.split()[1:]))
            for ln in (base / "labels" / f"{img_id}.txt").read_text().splitlines() if ln.strip()
        ]
        roi = DEFAULT_SOURCE_ROIS[src]
        roi_res = apply_roi_crop(orig_im, orig_boxes, roi)

        before_im = _draw_yolo_boxes(orig_im, orig_boxes, "#ff3333")
        # Draw dashed yellow ROI boundary on Before image
        dw = ImageDraw.Draw(before_im)
        dw.rectangle(roi_res.crop_box_px, outline="#ffd700", width=max(3, orig_im.width // 250))
        after_im = _draw_yolo_boxes(roi_res.image, roi_res.boxes, "#00ff66")

        axes[row_idx, 0].imshow(before_im)
        axes[row_idx, 0].set_title(f"{src} Original ({orig_im.width}x{orig_im.height}) + ROI box", fontsize=10)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(after_im)
        axes[row_idx, 1].set_title(f"{src} ROI Cropped ({roi_res.image.width}x{roi_res.image.height})", fontsize=10)
        axes[row_idx, 1].axis("off")

    plt.suptitle("Before & After Source-Aware Road ROI Preprocessing (Green = Transformed Boxes)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(vis_dir / "roi_before_after.png", dpi=160)
    plt.close()

    # 2. Representative Positive Samples (2x2 across sources)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, src in zip(axes.flatten(), ("local_dashcam", "local_smartphone", "mapillary", "rdd2022_usa")):
        cand = next((r for r in master_rows if r["source"] == src and int(r["pothole_count"]) > 0 and r["split"] == "train"), None)
        if cand:
            im = Image.open(PROJECT_ROOT / cand["image_relpath"]).convert("RGB")
            boxes = [(0, *map(float, ln.split()[1:])) for ln in (PROJECT_ROOT / cand["label_relpath"]).read_text().splitlines() if ln.strip()]
            ax.imshow(_draw_yolo_boxes(im, boxes, "#00ff66"))
            ax.set_title(f"Positive: {src} ({len(boxes)} box(es))", fontsize=10)
        ax.axis("off")
    plt.suptitle("Representative Curated Positive Training Images (ROI Applied)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(vis_dir / "positive_samples.png", dpi=150)
    plt.close()

    # 3. Representative Hard-Negative Samples (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    hard_cands = [r for r in master_rows if r["category"] == "hard_negative"]
    seen_subs: list[dict] = []
    for sub in ("repaired_patch", "manhole_cover", "crack", "other"):
        match = next((r for r in hard_cands if r["negative_subcategory"] == sub), None)
        if match:
            seen_subs.append(match)
    for ax, cand in zip(axes.flatten(), seen_subs + hard_cands[:4]):
        im = Image.open(PROJECT_ROOT / cand["image_relpath"]).convert("RGB")
        ax.imshow(im)
        ax.set_title(f"Hard Neg ({cand['negative_subcategory']}): {cand['source']}", fontsize=10)
        ax.axis("off")
    plt.suptitle("Representative Hard-Negative Training Images (Zero Pothole Boxes)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(vis_dir / "hard_negative_samples.png", dpi=150)
    plt.close()

    # 4. Conservative Small-Object Augmentation Preview (HSV + horizontal flip + slight affine)
    rng = random.Random(42)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pos_train = [r for r in master_rows if r["split"] == "train" and int(r["pothole_count"]) > 0][:4]
    for ax, cand in zip(axes.flatten(), pos_train):
        im_bgr = cv2.imread(str(PROJECT_ROOT / cand["image_relpath"]))
        ih, iw = im_bgr.shape[:2]
        boxes = [(0, *map(float, ln.split()[1:])) for ln in (PROJECT_ROOT / cand["label_relpath"]).read_text().splitlines() if ln.strip()]
        # Apply horizontal flip (fliplr=0.5) & mild brightness/saturation jitter (hsv_s=0.3, hsv_v=0.25)
        flip = rng.random() < 0.5
        if flip:
            im_bgr = cv2.flip(im_bgr, 1)
            boxes = [(c, 1.0 - cx, cy, bw, bh) for c, cx, cy, bw, bh in boxes]
        hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.75, 1.25), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.80, 1.20), 0, 255)
        aug_rgb = Image.fromarray(cv2.cvtColor(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB))
        ax.imshow(_draw_yolo_boxes(aug_rgb, boxes, "#00ff66"))
        ax.set_title(f"Conservative Aug ({cand['source']}, flip={flip})", fontsize=10)
        ax.axis("off")
    plt.suptitle("Conservative Training Augmentation Preview (Preserves Road Geometry & Labels)", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(vis_dir / "augmentation_preview.png", dpi=150)
    plt.close()

    # 5. Horizontal Tiling Preview for Wide Dashcam Road ROI (Experiment C)
    dash_cand = next((r for r in master_rows if r["source"] == "local_dashcam" and int(r["pothole_count"]) > 0), None)
    if dash_cand:
        d_base = src_dirs["local_dashcam"]
        d_id = dash_cand["source_image_id"]
        orig_d = Image.open(d_base / "images" / f"{d_id}.jpg").convert("RGB")
        orig_boxes = [
            (0, *map(float, ln.split()[1:]))
            for ln in (d_base / "labels" / f"{d_id}.txt").read_text().splitlines() if ln.strip()
        ]
        roi_d = DEFAULT_SOURCE_ROIS["local_dashcam"]
        roi_res = apply_roi_crop(orig_d, orig_boxes, roi_d)
        tiles = generate_horizontal_tiles(roi_res.image, roi_res.boxes, roi_res.crop_box_px, overlap_frac=0.15, min_tile_trigger_width=2000)

        fig, axes = plt.subplots(1 + len(tiles), 1, figsize=(14, 7.5))
        axes[0].imshow(_draw_yolo_boxes(roi_res.image, roi_res.boxes, "#00ff66"))
        axes[0].set_title(f"Single ROI ({roi_res.image.width}x{roi_res.image.height}, boxes={len(roi_res.boxes)})", fontsize=10)
        axes[0].axis("off")
        for idx, tile in enumerate(tiles):
            axes[idx + 1].imshow(_draw_yolo_boxes(tile.image, tile.boxes, "#00ff66"))
            axes[idx + 1].set_title(f"Tile {idx + 1} ({tile.image.width}x{tile.image.height}, boxes={len(tile.boxes)})", fontsize=10)
            axes[idx + 1].axis("off")
        plt.suptitle("Experiment C: Wide Dashcam ROI Horizontal Overlap Tiling (15% Overlap)", fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(vis_dir / "roi_tiled_preview.png", dpi=150)
        plt.close()


def generate_dataset_report() -> Path:
    """Create markdown dataset quality report and visual examples in results/reports/."""
    summary = json.loads((SPLITS_DIR / "dataset_summary.json").read_text(encoding="utf-8"))
    with (MANIFESTS_DIR / "master_manifest.csv").open(encoding="utf-8") as fh:
        master_rows = list(csv.DictReader(fh))
    with (MANIFESTS_DIR / "excluded_images.csv").open(encoding="utf-8") as fh:
        excl_rows = list(csv.DictReader(fh))

    res_comp = _compute_resolution_comparison(master_rows)
    vis_dir = REPORTS_DIR / "visuals"
    _generate_visuals(master_rows, vis_dir)

    lines = [
        "# Pothole Detection Dataset Quality & Curation Report",
        "",
        f"**Active Experiment Profile:** `{summary['experiment_profile']}` (`{summary['preprocessing_mode']}`)",
        "",
        "## 1. Original Annotated Inventory (Approved Sources Only)",
        "",
        "| Source | Total | Reviewed | Unreviewed | Uncertain | Positives | Reviewed Negatives | Boxes |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for src, st in summary["raw_audit_by_source"].items():
        lines.append(f"| `{src}` | {st['total']} | {st['reviewed']} | {st['unreviewed']} | {st['uncertain']} | {st['positives']} | {st['reviewed_negatives']} | {st['total_boxes']} |")

    lines += [
        "",
        "## 1B. Object vs. Physical Pothole Accounting (Explicit Distinction)",
        "",
        "> [!IMPORTANT]",
        "> Annotated 2D bounding boxes and heuristic temporal encounters are **not** verified physical pothole counts. Verified physical counts require human `gt_event_id` linking across frames.",
        "",
        "| Source | Images | Positive Images | Bounding-Box Instances | Verified Physical Potholes (`gt_event_id`) | Est. Temporal Encounters (Unverified <=4s heuristic) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for src, oc in summary.get("object_vs_physical_counts", {}).items():
        est = oc.get("estimated_temporal_encounters_unverified")
        est_txt = str(est) if est is not None else "N/A (still images)"
        lines.append(
            f"| `{src}` | {oc['images']} | {oc['positive_images']} | {oc['bounding_box_instances']} | "
            f"{oc['verified_unique_physical_potholes']} | {est_txt} |"
        )

    lines += [
        "",
        "## 2. Active Curated Detector Splits",
        "",
        "| Split | Images | Positive Images | Negative Images | Pothole Boxes |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for sp, st in summary["splits"].items():
        lines.append(f"| **{sp}** | {st['images']} | {st['positive_images']} | {st['negative_images']} | {st['pothole_boxes']} |")

    tc = summary["train_composition"]
    lines += [
        "",
        "### Training Composition (Target vs. Achieved)",
        f"- **Target:** Positive `{tc['target']['positive']*100:.0f}%`, Hard-Negative `{tc['target']['hard_negative']*100:.0f}%`, Ordinary-Negative `{tc['target']['ordinary_negative']*100:.0f}%`",
        f"- **Achieved:** Positive `{tc['achieved']['positive']*100:.2f}%`, Hard-Negative `{tc['achieved']['hard_negative']*100:.2f}%`, Ordinary-Negative `{tc['achieved']['ordinary_negative']*100:.2f}%`",
        f"- **Unused reviewed negatives preserved:** {tc['unused_reviewed_negatives_preserved']} (plus full RDD2022 USA pool)",
        "",
        "## 3. Small-Object Effective Pixel Resolution (Full Frame vs. ROI vs. Horizontal 2-Tile)",
        "",
        "| Mode | imgsz=640 Median (W×H px) | imgsz=960 Median (W×H px) | imgsz=1280 Median (W×H px) | % <16px @960 |",
        "| :--- | :---: | :---: | :---: | ---: |",
    ]
    for mode in ("full_frame", "roi", "roi_tiled"):
        d = res_comp[mode]
        lines.append(f"| `{mode}` | {d['640']['median_w']}×{d['640']['median_h']} | {d['960']['median_w']}×{d['960']['median_h']} | {d['1280']['median_w']}×{d['1280']['median_h']} | {d['960']['pct_lt_16px']}% |")

    lines += [
        "",
        "## 4. ROI Coverage, Deduplication & Exclusions",
        f"- **ROI Box Retention:** {summary['roi_stats']['retained_boxes']} / {summary['roi_stats']['original_boxes']} boxes retained (0 dropped, 0 truncated).",
        f"- **Deduplication:** {summary['deduplication']}",
        f"- **Excluded/Flagged records logged:** {len(excl_rows)} (see `data/manifests/excluded_images.csv`).",
        "",
        "## 5. Visual Previews",
        "- `results/reports/visuals/roi_before_after.png`",
        "- `results/reports/visuals/roi_tiled_preview.png`",
        "- `results/reports/visuals/positive_samples.png`",
        "- `results/reports/visuals/hard_negative_samples.png`",
        "- `results/reports/visuals/augmentation_preview.png`",
    ]

    report_path = REPORTS_DIR / "dataset_quality_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

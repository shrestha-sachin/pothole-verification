"""Model-assisted annotation proposal generator (excludes protected test data; never auto-confirms labels)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2

from src.detector.inference import PotholeDetector
from src.preprocessing.quality import assess_image_quality
from src.preprocessing.roi import DEFAULT_SOURCE_ROIS
from src.utils.config import ANNOTATED_DIR, MANIFESTS_DIR, MODELS_DIR, PROJECT_ROOT, resolve_device

SUBTYPE_CATEGORY_MAP = {
    "repaired_patch": "repaired_patch",
    "manhole": "manhole_cover",
    "manhole_cover": "manhole_cover",
    "shadow": "shadow",
    "road_marking": "road_marking",
    "crack": "crack",
    "ordinary_road": "ordinary_background",
}


def generate_assisted_proposals(
    model_path: Path | None = None,
    *,
    low_conf: float = 0.15,
    high_conf: float = 0.40,
    imgsz: int = 960,
    device: str = "auto",
) -> Path:
    """Run YOLO26x V5 across unreviewed non-test dashcam frames to prioritize diverse annotation candidates."""
    ckpt = model_path or (MODELS_DIR / "detector" / "yolo26x_us_v5_960.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

    det = PotholeDetector(ckpt, conf=low_conf, iou=0.50, imgsz=imgsz, device=resolve_device(device))
    base = ANNOTATED_DIR / "local" / "dashcam"
    csv_path = base / "metadata" / "annotations.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    proposals: dict[str, dict[str, Any]] = {}
    prev_dh_by_vid: dict[str, int] = {}
    roi = DEFAULT_SOURCE_ROIS["local_dashcam"]

    unreviewed_rows = [
        r for r in rows
        if r.get("review_status", "unreviewed") != "reviewed"
        and r.get("split", "") != "test"  # Strictly exclude protected test data!
    ]
    print(f"[Assist] Scoring {len(unreviewed_rows)} unreviewed non-test dashcam frames using {ckpt.name}...")

    for idx, r in enumerate(unreviewed_rows):
        if idx % 100 == 0:
            print(f"  Progress: {idx}/{len(unreviewed_rows)}...", flush=True)
        img_id = r["image_id"]
        img_p = base / "images" / f"{img_id}.jpg"
        bgr = cv2.imread(str(img_p))
        if bgr is None:
            continue

        q = assess_image_quality(img_p)
        vid = r.get("parent_video", "")
        is_near_dup = False
        if vid in prev_dh_by_vid and (q.dhash_int ^ prev_dh_by_vid[vid]).bit_count() <= 2:
            is_near_dup = True
        else:
            prev_dh_by_vid[vid] = q.dhash_int

        # Run ROI-focused inference & map predictions back to original frame coordinates
        dets = det.run_on_frame(bgr, frame_index=int(r.get("frame_index") or 0), roi=roi)
        boxes = [
            {
                "id": f"assist_{img_id}_{b_i}",
                "category": "pothole",
                "x": int(d.x1),
                "y": int(d.y1),
                "w": max(1, int(d.x2 - d.x1)),
                "h": max(1, int(d.y2 - d.y1)),
                "confidence": round(float(d.conf), 4),
            }
            for b_i, d in enumerate(dets)
        ]
        max_conf = max((b["confidence"] for b in boxes), default=0.0)
        if max_conf >= high_conf:
            bucket = "high_conf_candidate"
        elif boxes:
            bucket = "uncertain_detection"
        elif any(f in q.flags for f in ("low_sharpness_warn", "low_brightness_warn")) or (q.contrast_std >= 48.0):
            # High-contrast shadows / surface texture with 0 detections -> likely hard negative!
            bucket = "potential_hard_negative"
        else:
            bucket = "clean_background"

        proposals[img_id] = {
            "source": "local_dashcam",
            "parent_video": vid,
            "frame_index": int(r.get("frame_index") or 0),
            "assist_bucket": bucket,
            "max_conf": round(max_conf, 4),
            "boxes": boxes,
            "quality_flags": q.flags,
            "near_duplicate": is_near_dup,
            "laplacian_var": q.laplacian_var,
            "brightness_mean": q.brightness_mean,
            "model_used": ckpt.name,
            "note": "Model draft proposal only — never treated as ground truth until human-reviewed.",
        }

    out_path = MANIFESTS_DIR / "assisted_proposals.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(proposals, indent=2), encoding="utf-8")
    print(f"[OK] Saved {len(proposals)} assisted proposals -> {out_path}")
    return out_path

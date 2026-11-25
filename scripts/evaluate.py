"""Evaluate detector checkpoints on validation or protected held-out test split.

Supports raw Ultralytics evaluation and --original-space evaluation, which
evaluates full_frame, roi, and roi_tiled models on the exact same uncropped
original images and ground-truth boxes. Computes standard confidence-ranked PR
integration mAP50 and mAP50-95 from a low confidence floor (default 0.001)
separately from fixed operating-threshold Precision, Recall, F1, and FP/image,
plus per-source breakdowns and a concise experiment comparison report.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.inference import predict_frame_detections
from src.preprocessing.roi import (
    DEFAULT_SOURCE_ROIS,
    SourceROI,
    apply_roi_crop,
)
from src.utils.config import (
    ANNOTATED_DIR,
    CONFIGS_DIR,
    EVALUATIONS_DIR,
    MANIFESTS_DIR,
    REPORTS_DIR,
    load_checkpoint_metadata,
    load_yaml_config,
    resolve_device,
)


_SOURCE_DIRS = {
    "rdd2022_usa": ANNOTATED_DIR / "rdd2022_usa",
    "local_smartphone": ANNOTATED_DIR / "local" / "smartphone",
    "local_dashcam": ANNOTATED_DIR / "local" / "dashcam",
    "mapillary": ANNOTATED_DIR / "mapillary",
}

_COCO_IOU_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]


def compute_box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0.0 else 0.0


def _compute_ap_at_iou(
    flat_preds: list[tuple[str, float, tuple[float, float, float, float]]],
    gt_by_image: dict[str, list[tuple[float, float, float, float]]],
    iou_thresh: float,
) -> float:
    """Compute continuous all-point interpolated Precision-Recall AP at one IoU threshold."""
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt <= 0 or not flat_preds:
        return 0.0

    matched: dict[str, set[int]] = {k: set() for k in gt_by_image}
    tp_flags = np.zeros(len(flat_preds), dtype=np.float64)
    fp_flags = np.zeros(len(flat_preds), dtype=np.float64)
    for idx, (image_id, _conf, pbox) in enumerate(flat_preds):
        gts = gt_by_image.get(image_id, [])
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gbox in enumerate(gts):
            if gi in matched[image_id]:
                continue
            ov = compute_box_iou(pbox, gbox)
            if ov > best_iou:
                best_iou = ov
                best_gt_idx = gi
        if best_gt_idx >= 0 and best_iou >= iou_thresh:
            matched[image_id].add(best_gt_idx)
            tp_flags[idx] = 1.0
        else:
            fp_flags[idx] = 1.0

    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(fp_flags)
    recalls = tp_cum / total_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx_change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx_change + 1] - mrec[idx_change]) * mpre[idx_change + 1]))


def evaluate_predictions_original_space(
    predictions_by_image: dict[str, list[dict[str, Any]]],
    gt_by_image: dict[str, list[tuple[float, float, float, float]]],
    *,
    conf_threshold: float = 0.25,
    conf_floor: float = 0.001,
    iou_threshold: float = 0.50,
    roi_excluded_gt_count: int = 0,
) -> dict[str, Any]:
    """Compute confidence-ranked mAP50 & mAP50-95 (down to conf_floor) + separate fixed-threshold metrics."""
    total_gt = sum(len(v) for v in gt_by_image.values())
    n_imgs = len(gt_by_image)
    flat_preds: list[tuple[str, float, tuple[float, float, float, float]]] = []
    for image_id, preds in predictions_by_image.items():
        for p in preds:
            c = float(p["conf"])
            if c >= conf_floor:
                flat_preds.append((image_id, c, (float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"]))))
    flat_preds.sort(key=lambda item: item[1], reverse=True)

    ap50 = _compute_ap_at_iou(flat_preds, gt_by_image, 0.50)
    aps_coco = [_compute_ap_at_iou(flat_preds, gt_by_image, thr) for thr in _COCO_IOU_THRESHOLDS]
    map50_95 = float(np.mean(aps_coco)) if aps_coco else 0.0

    # Operating-point metrics at conf >= conf_threshold strictly separated from mAP
    matched_op: dict[str, set[int]] = {k: set() for k in gt_by_image}
    tp_op = 0
    fp_op = 0
    for image_id, conf, pbox in flat_preds:
        if conf < conf_threshold:
            continue
        gts = gt_by_image.get(image_id, [])
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gbox in enumerate(gts):
            if gi in matched_op[image_id]:
                continue
            ov = compute_box_iou(pbox, gbox)
            if ov > best_iou:
                best_iou = ov
                best_gt_idx = gi
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            matched_op[image_id].add(best_gt_idx)
            tp_op += 1
        else:
            fp_op += 1

    fn_op = max(0, total_gt - tp_op)
    precision = tp_op / (tp_op + fp_op) if (tp_op + fp_op) > 0 else 0.0
    recall = tp_op / total_gt if total_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fp_per_image = fp_op / max(1, n_imgs)
    return {
        "images_evaluated": n_imgs,
        "total_gt_boxes": total_gt,
        "roi_excluded_gt_boxes": roi_excluded_gt_count,
        "map_conf_floor": conf_floor,
        "operating_conf": conf_threshold,
        "iou_threshold": iou_threshold,
        "tp": tp_op,
        "fp": fp_op,
        "fn": fn_op,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positives_per_image": round(fp_per_image, 4),
        "mAP50": round(ap50, 4),
        "mAP50_95": round(map50_95, 4),
    }


def _find_original_image_and_labels(source: str, source_image_id: str) -> tuple[Path, Path]:
    base = _SOURCE_DIRS[source]
    img_dir = base / "images"
    lbl_path = base / "labels" / f"{source_image_id}.txt"
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        cand = img_dir / f"{source_image_id}{ext}"
        if cand.exists():
            return cand, lbl_path
    raise FileNotFoundError(f"Original image not found for {source}/{source_image_id} in {img_dir}")


def _resolve_profile_rois(ckpt_meta: dict[str, Any]) -> dict[str, SourceROI]:
    rois = dict(DEFAULT_SOURCE_ROIS)
    raw = ckpt_meta.get("source_rois") or {}
    for src, coords in raw.items():
        if isinstance(coords, (list, tuple)) and len(coords) == 4:
            rois[src] = SourceROI(*[float(v) for v in coords])
    return rois


def _resolve_eval_manifest(ckpt_meta: dict[str, Any], explicit_manifest: Path | None) -> Path:
    if explicit_manifest is not None:
        return explicit_manifest
    frozen_rel = (ckpt_meta.get("manifests") or {}).get("frozen_manifest")
    if frozen_rel:
        cand = PROJECT_ROOT / frozen_rel if not Path(frozen_rel).is_absolute() else Path(frozen_rel)
        if cand.exists():
            return cand
    return MANIFESTS_DIR / "master_manifest.csv"


def run_original_space_evaluation(
    model_path: Path,
    split: str,
    *,
    profile: str = "auto",
    imgsz: int | None = None,
    conf_threshold: float = 0.25,
    conf_floor: float = 0.001,
    nms_iou: float = 0.70,
    merge_iou: float = 0.50,
    iou_threshold: float = 0.50,
    device: str = "auto",
    manifest_path: Path | None = None,
    max_images: int | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint against original uncropped images & ground-truth boxes."""
    import cv2

    if conf_floor <= 0.0 or conf_floor > 0.05:
        raise ValueError(f"mAP conf_floor must be in (0, 0.05], got {conf_floor}")

    ckpt_meta = load_checkpoint_metadata(model_path)
    resolved_profile = ckpt_meta.get("preprocessing_profile", "full_frame") if profile == "auto" else profile
    resolved_imgsz = int(imgsz or ckpt_meta.get("input_imgsz", 960))
    tile_wide = resolved_profile == "roi_tiled" or bool(profile == "auto" and ckpt_meta.get("tile_wide_roi"))
    rois = _resolve_profile_rois(ckpt_meta)
    resolved_dev = resolve_device(device)
    chosen_manifest = _resolve_eval_manifest(ckpt_meta, manifest_path)

    with chosen_manifest.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == split]
    if max_images is not None:
        rows = rows[:max_images]
    if not rows:
        raise ValueError(f"No rows found for split='{split}' in {chosen_manifest}")

    model = YOLO(str(model_path))
    preds_by_image: dict[str, list[dict[str, Any]]] = {}
    gt_by_image: dict[str, list[tuple[float, float, float, float]]] = {}
    sources_by_key: dict[str, str] = {}
    roi_excluded_by_key: dict[str, int] = {}
    latencies_ms_by_key: dict[str, float] = {}
    seen_sources: set[tuple[str, str]] = set()

    for r in rows:
        src = r["source"]
        src_img_id = r["source_image_id"]
        if (src, src_img_id) in seen_sources:
            continue
        seen_sources.add((src, src_img_id))

        orig_img_p, orig_lbl_p = _find_original_image_and_labels(src, src_img_id)
        bgr = cv2.imread(str(orig_img_p))
        if bgr is None:
            raise IOError(f"Unable to read original image {orig_img_p}")
        oh, ow = bgr.shape[:2]

        orig_boxes_norm: list[tuple[int, float, float, float, float]] = []
        gt_px: list[tuple[float, float, float, float]] = []
        if orig_lbl_p.exists():
            for ln in orig_lbl_p.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                parts = ln.split()
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                orig_boxes_norm.append((0, cx, cy, bw, bh))
                gt_px.append(((cx - bw / 2.0) * ow, (cy - bh / 2.0) * oh, (cx + bw / 2.0) * ow, (cy + bh / 2.0) * oh))

        key = f"{src}:{src_img_id}"
        sources_by_key[key] = src
        gt_by_image[key] = gt_px

        roi: SourceROI | None = None
        excluded_here = 0
        if resolved_profile != "full_frame":
            roi = rois.get(src, DEFAULT_SOURCE_ROIS["rdd2022_usa"])
            with Image.open(orig_img_p) as pil_im:
                roi_check = apply_roi_crop(pil_im, orig_boxes_norm, roi, min_visibility=0.35)
                excluded_here = max(0, len(orig_boxes_norm) - len(roi_check.boxes))
        roi_excluded_by_key[key] = excluded_here

        t0 = time.perf_counter()
        preds_by_image[key] = predict_frame_detections(
            model,
            bgr,
            conf=conf_floor,
            iou=nms_iou,
            imgsz=resolved_imgsz,
            device=resolved_dev,
            roi=roi,
            tile_wide_roi=tile_wide,
            merge_iou=merge_iou,
            target_class_id=0,
        )
        latencies_ms_by_key[key] = (time.perf_counter() - t0) * 1000.0

    overall_metrics = evaluate_predictions_original_space(
        preds_by_image,
        gt_by_image,
        conf_threshold=conf_threshold,
        conf_floor=conf_floor,
        iou_threshold=iou_threshold,
        roi_excluded_gt_count=sum(roi_excluded_by_key.values()),
    )
    lat_vals = list(latencies_ms_by_key.values())
    overall_metrics["latency_ms_mean"] = round(float(np.mean(lat_vals)), 2) if lat_vals else 0.0
    overall_metrics["latency_ms_p95"] = round(float(np.percentile(lat_vals, 95)), 2) if lat_vals else 0.0

    metrics_by_source: dict[str, Any] = {}
    for src_name in sorted(set(sources_by_key.values())):
        keys = [k for k, s in sources_by_key.items() if s == src_name]
        s_preds = {k: preds_by_image[k] for k in keys}
        s_gts = {k: gt_by_image[k] for k in keys}
        s_metrics = evaluate_predictions_original_space(
            s_preds,
            s_gts,
            conf_threshold=conf_threshold,
            conf_floor=conf_floor,
            iou_threshold=iou_threshold,
            roi_excluded_gt_count=sum(roi_excluded_by_key[k] for k in keys),
        )
        s_lats = [latencies_ms_by_key[k] for k in keys]
        s_metrics["latency_ms_mean"] = round(float(np.mean(s_lats)), 2) if s_lats else 0.0
        metrics_by_source[src_name] = s_metrics

    try:
        manifest_rel = chosen_manifest.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        manifest_rel = str(chosen_manifest)

    return {
        "evaluation_mode": "original_space",
        "model": str(model_path),
        "model_stem": model_path.stem,
        "experiment_profile": ckpt_meta.get("experiment_profile", "unknown"),
        "split": split,
        "manifest_used": manifest_rel,
        "preprocessing_profile": resolved_profile,
        "tile_wide_roi": tile_wide,
        "imgsz": resolved_imgsz,
        "nms_iou": nms_iou,
        "merge_iou": merge_iou,
        "metrics": overall_metrics,
        "metrics_by_source": metrics_by_source,
    }


def write_experiment_comparison_report(split: str = "val") -> Path:
    """Build concise comparison Markdown & JSON across all evaluated checkpoints and pending A-E runs."""
    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_cfg = load_yaml_config("training.yaml")
    exp_defs = train_cfg.get("experiments", {})

    evaluated: list[dict[str, Any]] = []
    for p in sorted(EVALUATIONS_DIR.glob(f"eval_*_{split}_orig.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            if doc.get("evaluation_mode") == "original_space":
                evaluated.append(doc)
        except Exception:
            continue

    lines = [
        f"# Detector Experiment Comparison Report (`{split}` split, Original-Image Space)",
        "",
        "All models are compared against unchanged original images and ground-truth bounding boxes.",
        "- **mAP50 / mAP50-95:** Confidence-ranked PR integration down to `conf_floor=0.001`.",
        "- **Precision / Recall / F1 / FP-per-image:** Evaluated separately at fixed operating threshold (`conf=0.25`, `IoU=0.50`).",
        "",
        "## 1. Overall Results",
        "",
        "| Checkpoint / Profile | Preprocessing | imgsz | mAP50 | mAP50-95 | Precision | Recall | F1 | FP / img | Latency (ms/img) | Status |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    seen_profiles: set[str] = set()
    for doc in evaluated:
        m = doc["metrics"]
        prof = doc.get("experiment_profile", "unknown")
        seen_profiles.add(prof)
        lines.append(
            f"| `{Path(doc['model']).name}` (`{prof}`) | `{doc['preprocessing_profile']}` | {doc['imgsz']} | "
            f"{m['mAP50']:.4f} | {m.get('mAP50_95', 0.0):.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
            f"{m['f1']:.4f} | {m.get('false_positives_per_image', 0.0):.4f} | {m.get('latency_ms_mean', 0.0):.1f} | Evaluated |"
        )

    for exp_key, spec in exp_defs.items():
        if exp_key in seen_profiles:
            continue
        pre = "roi_tiled" if spec.get("tile_wide_roi") else spec.get("preprocessing", "roi")
        lines.append(
            f"| `yolo11n_{exp_key}` (`{exp_key}`) | `{pre}` | {spec.get('imgsz', 960)} | "
            f"- | - | - | - | - | - | - | Pending training |"
        )

    lines += ["", "## 2. Metrics by Data Source (Evaluated Checkpoints)", ""]
    if evaluated:
        lines += [
            "| Checkpoint | Source | Images | GT Boxes | mAP50 | mAP50-95 | Precision | Recall | F1 | FP / img | Latency (ms/img) |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for doc in evaluated:
            name = Path(doc["model"]).name
            for src_name, sm in doc.get("metrics_by_source", {}).items():
                lines.append(
                    f"| `{name}` | `{src_name}` | {sm['images_evaluated']} | {sm['total_gt_boxes']} | "
                    f"{sm['mAP50']:.4f} | {sm.get('mAP50_95', 0.0):.4f} | {sm['precision']:.4f} | "
                    f"{sm['recall']:.4f} | {sm['f1']:.4f} | {sm.get('false_positives_per_image', 0.0):.4f} | "
                    f"{sm.get('latency_ms_mean', 0.0):.1f} |"
                )
    else:
        lines.append("_No original-space evaluations recorded yet._")

    md_path = REPORTS_DIR / "experiment_comparison.md"
    json_path = REPORTS_DIR / "experiment_comparison.json"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps({"split": split, "evaluated": evaluated, "defined_experiments": exp_defs}, indent=2), encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> None:
    inf_cfg = load_yaml_config("inference.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / inf_cfg.get("detector_checkpoint", "models/detector/yolo26x_us_v5_960.pt"))
    parser.add_argument("--data", type=Path, default=CONFIGS_DIR / "detector.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--imgsz", type=int, default=int(inf_cfg.get("imgsz", 960)))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--original-space", action="store_true", help="Evaluate fairly on uncropped original images & original GT boxes")
    parser.add_argument("--profile", choices=["auto", "full_frame", "roi", "roi_tiled"], default="auto")
    parser.add_argument("--manifest", type=Path, default=None, help="Explicit master/frozen manifest CSV override")
    parser.add_argument("--conf", type=float, default=float(inf_cfg.get("confidence_threshold", 0.25)), help="Operating confidence threshold for P/R/F1/FP-per-image")
    parser.add_argument("--conf-floor", type=float, default=0.001, help="Low confidence floor for mAP50 / mAP50-95 PR integration")
    parser.add_argument("--nms-iou", type=float, default=0.70, help="Detector NMS IoU threshold during prediction")
    parser.add_argument("--iou", type=float, default=float(inf_cfg.get("iou_threshold", 0.50)), help="GT matching IoU threshold for P/R/F1")
    parser.add_argument("--merge-iou", type=float, default=0.50, help="Tile overlap merge NMS IoU threshold")
    parser.add_argument("--max-images", type=int, default=None, help="Optional subset limit for quick smoke checks")
    parser.add_argument("--no-comparison-report", action="store_true")
    args = parser.parse_args(argv)

    if args.original_space:
        summary = run_original_space_evaluation(
            model_path=args.model,
            split=args.split,
            profile=args.profile,
            imgsz=args.imgsz,
            conf_threshold=args.conf,
            conf_floor=args.conf_floor,
            nms_iou=args.nms_iou,
            merge_iou=args.merge_iou,
            iou_threshold=args.iou,
            device=args.device,
            manifest_path=args.manifest,
            max_images=args.max_images,
        )
        suffix = "_orig"
    else:
        model = YOLO(str(args.model))
        metrics = model.val(
            data=str(args.data),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=resolve_device(args.device),
            verbose=False,
        )
        summary = {
            "evaluation_mode": "ultralytics_val",
            "model": str(args.model),
            "split": args.split,
            "imgsz": args.imgsz,
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
        }
        suffix = ""

    EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = EVALUATIONS_DIR / f"eval_{args.model.stem}_{args.split}{suffix}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.original_space and not args.no_comparison_report and args.max_images is None:
        comp_path = write_experiment_comparison_report(split=args.split)
        summary["comparison_report"] = str(comp_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

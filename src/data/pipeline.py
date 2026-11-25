"""Canonical curated dataset preparation, quality filtering, ROI/tiling, and verifier crop builder."""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml
from PIL import Image

from src.data.validation import sha256_file, validate_detector_processed, validate_yolo_label_file
from src.preprocessing.quality import QualityThresholds, assess_image_quality, compute_effective_yolo_box_px
from src.preprocessing.roi import (
    DEFAULT_SOURCE_ROIS,
    SourceROI,
    apply_roi_crop,
    generate_horizontal_tiles,
)
from src.utils.config import (
    ANNOTATED_DIR,
    MANIFESTS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    REPORTS_DIR,
    SPLITS_DIR,
    load_yaml_config,
)
from src.verifier.preprocessing import DEFAULT_PAD_FRAC, expand_and_clamp_box

HARD_NEG_SUBCATEGORIES = {"shadow", "repaired_patch", "manhole_cover", "road_marking", "crack", "other"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _load_source_rois(train_cfg: dict[str, Any]) -> dict[str, SourceROI]:
    rois = dict(DEFAULT_SOURCE_ROIS)
    for src, coords in (train_cfg.get("source_rois") or {}).items():
        if isinstance(coords, (list, tuple)) and len(coords) == 4:
            rois[src] = SourceROI(*(float(v) for v in coords))
    return rois


def collect_and_audit_all_sources(
    thresholds: QualityThresholds = QualityThresholds(),
    *,
    scan_all_extra_rdd_pixels: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inspect all approved images in data/annotated/, computing quality, box metrics, and flags."""
    records: list[dict[str, Any]] = []
    source_stats: dict[str, dict[str, Any]] = {}

    sources = [
        ("rdd2022_usa", ANNOTATED_DIR / "rdd2022_usa"),
        ("local_smartphone", ANNOTATED_DIR / "local" / "smartphone"),
        ("local_dashcam", ANNOTATED_DIR / "local" / "dashcam"),
        ("mapillary", ANNOTATED_DIR / "mapillary"),
    ]

    for src_name, base in sources:
        meta_rows = _read_csv(base / "metadata" / "annotations.csv")
        st = {
            "total": len(meta_rows),
            "reviewed": 0,
            "unreviewed": 0,
            "uncertain": 0,
            "positives": 0,
            "reviewed_negatives": 0,
            "total_boxes": 0,
            "corrupt": 0,
            "missing_label": 0,
            "invalid_boxes": 0,
            "neg_subcategories": Counter(),
        }
        for r in meta_rows:
            img_id = r["image_id"]
            img_p = base / "images" / f"{img_id}.jpg"
            lbl_p = base / "labels" / f"{img_id}.txt"
            rev = r.get("review_status", "unreviewed").strip() or "unreviewed"
            split = (r.get("split") or r.get("historical_v3_split") or "").strip()
            if src_name == "rdd2022_usa" and split not in {"train", "val", "test"}:
                split = "train"
            if split not in {"train", "val", "test"}:
                split = "train"

            if rev == "reviewed":
                st["reviewed"] += 1
            elif rev == "uncertain":
                st["uncertain"] += 1
            else:
                st["unreviewed"] += 1

            boxes: list[tuple[int, float, float, float, float]] = []
            lbl_errs: list[str] = []
            if rev == "reviewed":
                if not lbl_p.exists():
                    st["missing_label"] += 1
                    lbl_errs.append("missing_label")
                else:
                    boxes, lbl_errs = validate_yolo_label_file(lbl_p)
                    if lbl_errs:
                        st["invalid_boxes"] += len(lbl_errs)

            neg_sub = r.get("negative_subcategory", "").strip()
            prior_fp = r.get("prior_detector_fp", "false").lower() == "true"
            if rev == "reviewed":
                if boxes:
                    st["positives"] += 1
                    st["total_boxes"] += len(boxes)
                    category = "positive"
                else:
                    st["reviewed_negatives"] += 1
                    if not neg_sub:
                        neg_sub = "crack" if (src_name == "rdd2022_usa" and r.get("other_classes")) else "ordinary_background"
                    st["neg_subcategories"][neg_sub] += 1
                    category = "hard_negative" if (neg_sub in HARD_NEG_SUBCATEGORIES or prior_fp) else "ordinary_negative"
            else:
                category = "excluded_" + rev

            # Quality inspection: skip heavy pixel decoding on excess unchosen RDD negatives unless requested
            hist = r.get("historical_v3_split", "").strip()
            should_decode = (
                scan_all_extra_rdd_pixels
                or src_name != "rdd2022_usa"
                or bool(boxes)
                or hist in {"train", "val", "test"}
                or int(img_id.rsplit("_", 1)[-1]) <= 600
            )
            if rev == "reviewed" and should_decode and not lbl_errs:
                q = assess_image_quality(img_p, boxes, thresholds)
                if not q.valid_image and "corrupt_image" in q.flags:
                    st["corrupt"] += 1
            elif not img_p.exists():
                q = assess_image_quality(img_p, boxes, thresholds)
                st["corrupt"] += 1
            else:
                q = None

            records.append({
                "image_id": img_id,
                "source": src_name,
                "image_path": img_p,
                "label_path": lbl_p,
                "split": split,
                "historical_v3_split": hist,
                "group_id": r.get("group_id") or r.get("parent_video") or img_id,
                "parent_video": r.get("parent_video", ""),
                "frame_index": r.get("frame_index", ""),
                "timestamp_ms": r.get("timestamp_ms", ""),
                "review_status": rev,
                "category": category,
                "negative_subcategory": neg_sub,
                "prior_detector_fp": prior_fp,
                "boxes": boxes,
                "label_errors": lbl_errs,
                "quality": q,
            })

        st["neg_subcategories"] = dict(st["neg_subcategories"])
        source_stats[src_name] = st
    return records, source_stats


def _round_robin_source_sample(
    candidates: list[dict[str, Any]],
    target_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample negatives round-robin across sources so local footage and RDD both have representation,
    prioritizing prior detector false positives first."""
    if target_count <= 0 or not candidates:
        return []
    if len(candidates) <= target_count:
        return list(candidates)

    rng = random.Random(seed)
    by_src: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_src[c["source"]].append(c)

    for src in by_src:
        rng.shuffle(by_src[src])
        # Prioritize prior detector false positives & local sources before general negatives
        by_src[src].sort(key=lambda r: (0 if r["prior_detector_fp"] else 1, r["image_id"]))

    # Order sources prioritizing local_dashcam, local_smartphone, mapillary, then rdd2022_usa
    src_order = [s for s in ("local_dashcam", "local_smartphone", "mapillary", "rdd2022_usa") if s in by_src]
    selected: list[dict[str, Any]] = []
    while len(selected) < target_count and any(by_src[s] for s in src_order):
        for s in src_order:
            if len(selected) >= target_count:
                break
            if by_src[s]:
                selected.append(by_src[s].pop(0))
    return selected


def _compute_object_vs_physical_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Explicitly separate images, 2D bounding boxes, verified physical IDs, and heuristic encounters."""
    src_dirs = {
        "rdd2022_usa": ANNOTATED_DIR / "rdd2022_usa",
        "local_smartphone": ANNOTATED_DIR / "local" / "smartphone",
        "local_dashcam": ANNOTATED_DIR / "local" / "dashcam",
        "mapillary": ANNOTATED_DIR / "mapillary",
    }
    out: dict[str, dict[str, Any]] = {}
    for src_name, base in src_dirs.items():
        sub_recs = [r for r in records if r["source"] == src_name]
        pos_recs = [r for r in sub_recs if r["boxes"]]
        box_count = sum(len(r["boxes"]) for r in sub_recs)

        verified_ids: set[str] = set()
        meta_dir = base / "metadata"
        if meta_dir.exists():
            for jp in meta_dir.glob("*.json"):
                try:
                    doc = json.loads(jp.read_text(encoding="utf-8"))
                    for b in doc.get("pothole_boxes", []):
                        ge = str(b.get("gt_event_id", "")).strip()
                        if ge:
                            verified_ids.add(ge)
                except Exception:
                    continue

        est_encounters: int | None = None
        if src_name in {"local_dashcam", "local_smartphone"}:
            by_vid: dict[str, list[int]] = defaultdict(list)
            for r in pos_recs:
                ts_raw = str(r.get("timestamp_ms") or "").strip()
                ts = int(ts_raw) if ts_raw.isdigit() else 0
                by_vid[r.get("parent_video") or src_name].append(ts)
            enc = 0
            for ts_list in by_vid.values():
                ts_sorted = sorted(ts_list)
                last_ts: int | None = None
                for ts in ts_sorted:
                    if last_ts is None or (ts - last_ts) > 4000:
                        enc += 1
                    last_ts = ts
            est_encounters = enc

        out[src_name] = {
            "images": len(sub_recs),
            "positive_images": len(pos_recs),
            "bounding_box_instances": box_count,
            "verified_unique_physical_potholes": len(verified_ids),
            "estimated_temporal_encounters_unverified": est_encounters,
            "note": (
                "Estimated temporal encounters (<=4s video gap) are NOT verified unique physical pothole IDs."
                if est_encounters is not None
                else "Static imagery without multi-frame physical pothole tracking IDs."
            ),
        }
    return out


def build_curated_dataset(
    *,
    experiment: str | None = None,
    prepare_verifier: bool = True,
    seed: int | None = None,
    force_refreeze: bool = False,
) -> dict[str, Any]:
    """Build curated, reproducible detector (and optional verifier) datasets for a given experiment profile."""
    train_cfg = load_yaml_config("training.yaml")
    exp_name = experiment or train_cfg.get("active_experiment", "exp_b_roi_90_10")
    exp_cfg = (train_cfg.get("experiments") or {}).get(exp_name)
    if not exp_cfg:
        raise ValueError(f"Unknown experiment profile '{exp_name}'. Available: {list((train_cfg.get('experiments') or {}).keys())}")

    use_seed = seed if seed is not None else int(train_cfg.get("seed", 42))
    preprocessing_mode = exp_cfg.get("preprocessing", "roi")
    tile_wide = bool(exp_cfg.get("tile_wide_roi", False))
    pos_target = float(exp_cfg.get("positive_ratio", 0.90))
    hard_target = float(exp_cfg.get("hard_negative_ratio", 0.08))
    ord_target = float(exp_cfg.get("ordinary_negative_ratio", 0.02))
    rois = _load_source_rois(train_cfg)
    thresholds = QualityThresholds()

    records, raw_audit_stats = collect_and_audit_all_sources(thresholds)

    # Filter out unreviewed, uncertain, label errors, quality failures, and duplicates
    excluded_rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []

    for r in records:
        if r["review_status"] != "reviewed":
            excluded_rows.append({"image_id": r["image_id"], "source": r["source"], "split": r["split"], "reason": f"review_status:{r['review_status']}", "laplacian_var": "", "brightness": ""})
            continue
        if r["label_errors"]:
            excluded_rows.append({"image_id": r["image_id"], "source": r["source"], "split": r["split"], "reason": ";".join(r["label_errors"]), "laplacian_var": "", "brightness": ""})
            continue
        q = r["quality"]
        if q is None:
            continue  # Unscanned surplus RDD2022 negatives beyond initial 600-item pool
        if not q.valid_image:
            # Do not silently discard dark or small-box *positives* in val/test unless corrupt
            if r["category"] == "positive" and "corrupt_image" not in q.flags and r["split"] in {"val", "test"}:
                pass
            else:
                excluded_rows.append({"image_id": r["image_id"], "source": r["source"], "split": r["split"], "reason": q.rejection_reason, "laplacian_var": q.laplacian_var, "brightness": q.brightness_mean})
                continue
        eligible.append(r)

    # Enforce strict group split isolation
    grp_splits: dict[str, set[str]] = defaultdict(set)
    for r in eligible:
        grp_splits[r["group_id"]].add(r["split"])
    conflicts = {g: sorted(s) for g, s in grp_splits.items() if len(s) > 1}
    if conflicts:
        raise ValueError(f"Recording/sequence group split overlap: {conflicts}")

    # Deduplication (exact SHA-256 + near-duplicate dHash across splits & adjacent negative video frames)
    eligible.sort(key=lambda r: ({"test": 0, "val": 1, "train": 2}[r["split"]], r["image_id"]))
    seen_sha: dict[str, tuple[str, str]] = {}
    kept_dhash_by_group: dict[str, list[int]] = defaultdict(list)
    seen_dhash_global: dict[int, str] = {}
    dedup_eligible: list[dict[str, Any]] = []
    dup_stats = {"exact_duplicates": 0, "cross_split_near_duplicates": 0, "adjacent_video_negative_duplicates": 0}

    for r in eligible:
        sha = sha256_file(r["image_path"])
        dh = r["quality"].dhash_int
        r["sha256"] = sha
        if sha in seen_sha:
            dup_stats["exact_duplicates"] += 1
            excluded_rows.append({"image_id": r["image_id"], "source": r["source"], "split": r["split"], "reason": f"exact_duplicate_of:{seen_sha[sha][0]}", "laplacian_var": r["quality"].laplacian_var, "brightness": r["quality"].brightness_mean})
            continue
        # Cross-split near-duplicate guard
        cross_hit = any((dh ^ g_dh).bit_count() <= thresholds.near_dup_hamming and g_sp != r["split"] for g_dh, g_sp in seen_dhash_global.items())
        if cross_hit:
            dup_stats["cross_split_near_duplicates"] += 1
            excluded_rows.append({"image_id": r["image_id"], "source": r["source"], "split": r["split"], "reason": "cross_split_near_duplicate", "laplacian_var": r["quality"].laplacian_var, "brightness": r["quality"].brightness_mean})
            continue
        # Within-video adjacent near-duplicate negative pruning (preserve positives!)
        if r["category"] != "positive" and r["parent_video"]:
            adj_hit = any((dh ^ prev_dh).bit_count() <= thresholds.near_dup_hamming for prev_dh in kept_dhash_by_group[r["group_id"]])
            if adj_hit:
                dup_stats["adjacent_video_negative_duplicates"] += 1
                excluded_rows.append({"image_id": r["image_id"], "source": r["source"], "split": r["split"], "reason": "adjacent_near_duplicate_negative", "laplacian_var": r["quality"].laplacian_var, "brightness": r["quality"].brightness_mean})
                continue
        seen_sha[sha] = (r["image_id"], r["split"])
        seen_dhash_global[dh] = r["split"]
        kept_dhash_by_group[r["group_id"]].append(dh)
        dedup_eligible.append(r)

    # Controlled training composition (applied strictly to train; val/test untouched!)
    train_pool = [r for r in dedup_eligible if r["split"] == "train"]
    eval_pool = [r for r in dedup_eligible if r["split"] in {"val", "test"}]

    train_pos = [r for r in train_pool if r["category"] == "positive"]
    train_hard = [r for r in train_pool if r["category"] == "hard_negative"]
    train_ord = [r for r in train_pool if r["category"] == "ordinary_negative"]

    # Calculate target negative counts relative to all genuine training positives
    target_total = len(train_pos) / max(0.01, pos_target)
    target_hard = int(round(target_total * hard_target))
    target_ord = int(round(target_total * ord_target))

    sel_hard = _round_robin_source_sample(train_hard, target_hard, use_seed)
    sel_ord = _round_robin_source_sample(train_ord, target_ord, use_seed + 1)

    selected_train_ids = {r["image_id"] for r in train_pos + sel_hard + sel_ord}
    unused_reviewed_negatives = [r for r in train_hard + train_ord if r["image_id"] not in selected_train_ids]
    selected_records = train_pos + sel_hard + sel_ord + eval_pool

    # Stage data/processed/detector/ idempotently with optional ROI / Tiling
    det_dir = PROCESSED_DIR / "detector"
    if det_dir.exists():
        shutil.rmtree(det_dir)
    for sp in ("train", "val", "test"):
        (det_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (det_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

    master_rows: list[dict[str, Any]] = []
    splits_map: dict[str, list[str]] = defaultdict(list)
    roi_stats = {"original_boxes": 0, "retained_boxes": 0, "truncated_boxes": 0, "dropped_boxes_train": 0, "dropped_boxes_eval_misses": 0}
    eff_res_stats: dict[int, list[tuple[float, float]]] = {640: [], 960: [], 1280: []}
    box_areas: list[float] = []

    for r in selected_records:
        split = r["split"]
        img_id = r["image_id"]
        with Image.open(r["image_path"]) as im_raw:
            im = im_raw.convert("RGB")
            orig_w, orig_h = im.size
            orig_boxes = r["boxes"]
            roi_stats["original_boxes"] += len(orig_boxes)

            if preprocessing_mode == "roi":
                roi = rois.get(r["source"], DEFAULT_SOURCE_ROIS["rdd2022_usa"])
                cropped_res = apply_roi_crop(im, orig_boxes, roi)
                if cropped_res.dropped_boxes > 0:
                    if split == "train":
                        roi_stats["dropped_boxes_train"] += cropped_res.dropped_boxes
                    else:
                        roi_stats["dropped_boxes_eval_misses"] += cropped_res.dropped_boxes
                    # Never convert an accidentally cropped-out positive into a negative in train!
                    if split == "train" and len(orig_boxes) > 0 and len(cropped_res.boxes) == 0:
                        excluded_rows.append({"image_id": img_id, "source": r["source"], "split": split, "reason": "roi_excluded_all_boxes", "laplacian_var": r["quality"].laplacian_var, "brightness": r["quality"].brightness_mean})
                        continue
                outputs = (
                    generate_horizontal_tiles(cropped_res.image, cropped_res.boxes, cropped_res.crop_box_px)
                    if tile_wide
                    else [cropped_res]
                )
            else:
                from src.preprocessing.roi import CropTransformResult
                outputs = [CropTransformResult(im, (0, 0, orig_w, orig_h), (orig_w, orig_h), (orig_w, orig_h), orig_boxes, len(orig_boxes), 0, 0)]

            for tile_idx, out_res in enumerate(outputs):
                out_stem = img_id if len(outputs) == 1 else f"{img_id}_tile{tile_idx}"
                dst_img = det_dir / "images" / split / f"{out_stem}.jpg"
                dst_lbl = det_dir / "labels" / split / f"{out_stem}.txt"
                out_res.image.save(dst_img, format="JPEG", quality=94)

                yolo_lines = [f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for c, cx, cy, bw, bh in out_res.boxes]
                dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

                roi_stats["retained_boxes"] += len(out_res.boxes)
                roi_stats["truncated_boxes"] += out_res.truncated_boxes
                cw, ch = out_res.crop_wh
                for _, _, _, bw, bh in out_res.boxes:
                    box_areas.append(bw * bh)
                    for sz in (640, 960, 1280):
                        ew, eh, _ = compute_effective_yolo_box_px(cw, ch, bw, bh, sz)
                        eff_res_stats[sz].append((ew, eh))

                splits_map[split].append(out_stem)
                master_rows.append({
                    "sample_id": out_stem,
                    "source_image_id": img_id,
                    "source": r["source"],
                    "split": split,
                    "category": r["category"],
                    "negative_subcategory": r["negative_subcategory"],
                    "prior_detector_fp": r["prior_detector_fp"],
                    "group_id": r["group_id"],
                    "parent_video": r["parent_video"],
                    "frame_index": r["frame_index"],
                    "timestamp_ms": r["timestamp_ms"],
                    "orig_width": orig_w,
                    "orig_height": orig_h,
                    "proc_width": cw,
                    "proc_height": ch,
                    "roi_crop_px": json.dumps(list(out_res.crop_box_px)),
                    "pothole_count": len(out_res.boxes),
                    "laplacian_var": r["quality"].laplacian_var,
                    "brightness_mean": r["quality"].brightness_mean,
                    "quality_flags": ";".join(r["quality"].flags),
                    "dhash_hex": r["quality"].dhash_hex,
                    "sha256": r["sha256"],
                    "image_relpath": dst_img.relative_to(PROJECT_ROOT).as_posix(),
                    "label_relpath": dst_lbl.relative_to(PROJECT_ROOT).as_posix(),
                })

    # Write dataset.yaml inside data/processed/detector/ (portable relative to project root)
    ds_yaml = {
        "path": "data/processed/detector",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "pothole"},
    }
    (det_dir / "dataset.yaml").write_text(yaml.safe_dump(ds_yaml, sort_keys=False), encoding="utf-8")

    # Save manifests & exclusions
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    if master_rows:
        with (MANIFESTS_DIR / "master_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(master_rows[0]))
            w.writeheader()
            w.writerows(master_rows)

    with (MANIFESTS_DIR / "excluded_images.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["image_id", "source", "split", "reason", "laplacian_var", "brightness"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(excluded_rows)

    with (MANIFESTS_DIR / "unused_reviewed_negatives.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["image_id", "source", "split", "category", "negative_subcategory", "prior_detector_fp", "group_id", "image_path"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for u in unused_reviewed_negatives:
            w.writerow({k: str(u.get(k, "")) for k in fields})

    (SPLITS_DIR / "detector_splits.json").write_text(json.dumps({k: sorted(v) for k, v in splits_map.items()}, indent=2), encoding="utf-8")

    val_report = validate_detector_processed(det_dir)
    train_cnt = max(1, val_report["splits"]["train"]["images"])
    achieved_train_pos_ratio = val_report["splits"]["train"]["positive_images"] / train_cnt
    achieved_train_hard_ratio = sum(1 for m in master_rows if m["split"] == "train" and m["category"] == "hard_negative") / train_cnt
    achieved_train_ord_ratio = sum(1 for m in master_rows if m["split"] == "train" and m["category"] == "ordinary_negative") / train_cnt

    eff_summary = {}
    for sz, pairs in eff_res_stats.items():
        if pairs:
            ws = sorted(p[0] for p in pairs)
            hs = sorted(p[1] for p in pairs)
            eff_summary[str(sz)] = {
                "median_w_px": round(float(np.median(ws)), 1),
                "median_h_px": round(float(np.median(hs)), 1),
                "p10_w_px": round(float(np.percentile(ws, 10)), 1),
                "p10_h_px": round(float(np.percentile(hs, 10)), 1),
                "pct_boxes_min_side_lt_16px": round(100.0 * sum(1 for w, h in pairs if min(w, h) < 16.0) / len(pairs), 1),
            }

    ds_version = str((train_cfg.get("dataset") or {}).get("version", "curated_v1"))
    frozen_manifest_p = MANIFESTS_DIR / f"{ds_version}_{exp_name}_manifest.csv"
    frozen_summary_p = MANIFESTS_DIR / f"{ds_version}_{exp_name}_summary.json"
    active_manifest_p = MANIFESTS_DIR / "master_manifest.csv"

    if (not frozen_manifest_p.exists()) or force_refreeze:
        shutil.copy2(active_manifest_p, frozen_manifest_p)
        frozen_status = "created_or_updated"
    else:
        frozen_status = "preserved_immutable"

    frozen_manifest_sha = sha256_file(frozen_manifest_p)
    obj_vs_phys = _compute_object_vs_physical_counts(records)

    pad_frac = float((train_cfg.get("dataset") or {}).get("verifier_pad_fraction", DEFAULT_PAD_FRAC))
    verifier_summary = _build_verifier_dataset(dedup_eligible, train_cfg, use_seed, pad_frac=pad_frac) if prepare_verifier else {}

    summary = {
        "dataset_version": ds_version,
        "experiment_profile": exp_name,
        "preprocessing_mode": preprocessing_mode,
        "tile_wide_roi": tile_wide,
        "seed": use_seed,
        "frozen_manifests": {
            "version": ds_version,
            "status": frozen_status,
            "manifest_csv": frozen_manifest_p.relative_to(PROJECT_ROOT).as_posix(),
            "manifest_sha256": frozen_manifest_sha,
            "summary_json": frozen_summary_p.relative_to(PROJECT_ROOT).as_posix(),
        },
        "raw_audit_by_source": raw_audit_stats,
        "object_vs_physical_counts": obj_vs_phys,
        "splits": val_report["splits"],
        "by_source_and_split": {
            src: {
                sp: {
                    "positive": sum(1 for m in master_rows if m["source"] == src and m["split"] == sp and m["pothole_count"] > 0),
                    "negative": sum(1 for m in master_rows if m["source"] == src and m["split"] == sp and m["pothole_count"] == 0),
                }
                for sp in ("train", "val", "test")
            }
            for src in sorted({m["source"] for m in master_rows})
        },
        "train_composition": {
            "target": {"positive": pos_target, "hard_negative": hard_target, "ordinary_negative": ord_target},
            "achieved": {
                "positive": round(achieved_train_pos_ratio, 4),
                "hard_negative": round(achieved_train_hard_ratio, 4),
                "ordinary_negative": round(achieved_train_ord_ratio, 4),
            },
            "unused_reviewed_negatives_preserved": len(unused_reviewed_negatives),
        },
        "unique_groups": len({m["group_id"] for m in master_rows}),
        "unique_videos": len({m["parent_video"] for m in master_rows if m["parent_video"]}),
        "deduplication": dup_stats,
        "roi_stats": roi_stats,
        "effective_resolution_at_imgsz": eff_summary,
        "verifier_dataset": verifier_summary,
        "validation_passed": val_report["valid"],
    }
    summary_text = json.dumps(summary, indent=2)
    (SPLITS_DIR / "dataset_summary.json").write_text(summary_text, encoding="utf-8")
    if (not frozen_summary_p.exists()) or force_refreeze:
        frozen_summary_p.write_text(summary_text, encoding="utf-8")
    return summary


def _build_verifier_dataset(
    dedup_eligible: list[dict[str, Any]],
    train_cfg: dict[str, Any],
    seed: int,
    pad_frac: float = DEFAULT_PAD_FRAC,
) -> dict[str, Any]:
    """Build verifier positive (40%) and hard/ordinary negative (60%) crops per split using
    15% bounding-box context padding (matching inference). Strictly prioritizes human-annotated
    negative_regions (repaired patches, manholes, shadows) and RDD crack boxes over random crops."""
    ver_dir = PROCESSED_DIR / "verifier"
    if ver_dir.exists():
        shutil.rmtree(ver_dir)
    for sp in ("train", "val", "test"):
        (ver_dir / sp / "pothole").mkdir(parents=True, exist_ok=True)
        (ver_dir / sp / "non_pothole").mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    counts: dict[str, dict[str, int]] = {sp: {"pothole": 0, "non_pothole": 0} for sp in ("train", "val", "test")}
    subcats: dict[str, Counter] = {sp: Counter() for sp in ("train", "val", "test")}
    ver_rows: list[dict[str, str]] = []

    subtype_norm = {
        "manhole": "manhole_cover",
        "manhole_cover": "manhole_cover",
        "repaired_patch": "repaired_patch",
        "shadow": "shadow",
        "road_marking": "road_marking",
        "ordinary_road": "ordinary_background",
        "ordinary_background": "ordinary_background",
        "crack": "crack",
        "D00": "crack",
        "D10": "crack",
        "D20": "crack",
        "D43": "road_marking",
        "D44": "road_marking",
    }

    def _add_row(crop_path: Path, split: str, label: str, sub: str, r: dict[str, Any]) -> None:
        ver_rows.append({
            "crop_id": crop_path.stem,
            "crop_filepath": crop_path.relative_to(PROJECT_ROOT).as_posix(),
            "label": label,
            "subtype": sub,
            "split": split,
            "pad_frac": str(pad_frac),
            "review_status": "reviewed",
            "uncertain": "false",
            "source": r["source"],
            "source_image_id": r["image_id"],
            "group_id": r.get("group_id", ""),
            "recording_id": r.get("group_id", ""),
            "video_id": r.get("parent_video", ""),
            "sequence_id": r.get("group_id", "") if r["source"] == "mapillary" else "",
        })

    # 1. Extract ALL reviewed pothole positive crops (with 15% surrounding pavement context)
    for r in dedup_eligible:
        if not r["boxes"]:
            continue
        with Image.open(r["image_path"]) as im_raw:
            rgb = im_raw.convert("RGB")
            iw, ih = rgb.size
            for idx, (_, cx, cy, bw, bh) in enumerate(r["boxes"]):
                x1 = int(round((cx - bw / 2.0) * iw))
                y1 = int(round((cy - bh / 2.0) * ih))
                x2 = int(round((cx + bw / 2.0) * iw))
                y2 = int(round((cy + bh / 2.0) * ih))
                px1, py1, px2, py2 = expand_and_clamp_box(x1, y1, x2, y2, iw, ih, pad_frac=pad_frac)
                if px2 - px1 >= 16 and py2 - py1 >= 16:
                    dst = ver_dir / r["split"] / "pothole" / f"pos_{r['image_id']}_b{idx:02d}.jpg"
                    rgb.crop((px1, py1, px2, py2)).save(dst, quality=92)
                    counts[r["split"]]["pothole"] += 1
                    _add_row(dst, r["split"], "pothole", "pothole", r)

    # 2. Extract reviewed negative crops per split (targeting ~1.5x positives -> 40% pos / 60% neg)
    src_dirs = {
        "local_smartphone": ANNOTATED_DIR / "local" / "smartphone",
        "local_dashcam": ANNOTATED_DIR / "local" / "dashcam",
        "mapillary": ANNOTATED_DIR / "mapillary",
    }
    rdd_xml_dir = RAW_DIR / "rdd2022_usa" / "annotations"

    for sp in ("train", "val", "test"):
        target_neg = int(round(counts[sp]["pothole"] * 1.5))
        neg_candidates: list[tuple[int, dict[str, Any], tuple[int, int, int, int], str]] = []

        # Priority A: Human-annotated negative_regions (repaired_patch, manhole, shadow) from ANY reviewed image in split
        for r in dedup_eligible:
            if r["split"] != sp or r["source"] not in src_dirs:
                continue
            meta_json = src_dirs[r["source"]] / "metadata" / f"{r['image_id']}.json"
            if not meta_json.exists():
                continue
            doc = json.loads(meta_json.read_text(encoding="utf-8"))
            if doc.get("review_status") != "reviewed" or doc.get("uncertain"):
                continue
            for nr in doc.get("negative_regions", []):
                sub = subtype_norm.get(str(nr.get("subtype", "other")).strip().lower(), "other")
                bx, by, bw, bh = int(nr.get("x", 0)), int(nr.get("y", 0)), int(nr.get("w", 0)), int(nr.get("h", 0))
                if bw >= 16 and bh >= 16:
                    neg_candidates.append((0, r, (bx, by, bx + bw, by + bh), sub))

        # Priority B: RDD2022 USA annotated non-D40 crack / road_marking boxes on reviewed pothole-free images
        for r in dedup_eligible:
            if r["split"] != sp or r["source"] != "rdd2022_usa" or r["boxes"]:
                continue
            xml_p = rdd_xml_dir / f"{r['image_id']}.xml"
            if not xml_p.exists():
                continue
            try:
                rt = ET.parse(xml_p).getroot()
                for obj in rt.findall("object"):
                    cn = (obj.find("name").text or "").strip()
                    if cn == "D40":
                        continue
                    bnd = obj.find("bndbox")
                    x1 = int(float(bnd.find("xmin").text))
                    y1 = int(float(bnd.find("ymin").text))
                    x2 = int(float(bnd.find("xmax").text))
                    y2 = int(float(bnd.find("ymax").text))
                    if x2 - x1 >= 16 and y2 - y1 >= 16:
                        sub = subtype_norm.get(cn, "crack")
                        neg_candidates.append((1, r, (x1, y1, x2, y2), sub))
            except Exception:
                continue

        # Priority C: Prior detector FP reviewed frames (crop lower road band only if needed to reach 60%)
        fp_neg_imgs = [r for r in dedup_eligible if r["split"] == sp and not r["boxes"] and r["prior_detector_fp"]]
        rng.shuffle(fp_neg_imgs)

        # Balance across negative subcategories & sources within Priority A before taking Priority B cracks
        p_a = [c for c in neg_candidates if c[0] == 0]
        p_b = [c for c in neg_candidates if c[0] == 1]
        rng.shuffle(p_a)
        rng.shuffle(p_b)

        # Round-robin Priority A by subtype so patches, manholes, and shadows are all well represented
        by_sub: dict[str, list] = defaultdict(list)
        for item in p_a:
            by_sub[item[3]].append(item)
        ordered_p_a: list = []
        sub_keys = sorted(by_sub)
        while any(by_sub.values()):
            for k in sub_keys:
                if by_sub[k]:
                    ordered_p_a.append(by_sub[k].pop(0))

        # Keep RDD cracks capped at <=50% of target_neg when local confusers/FPs exist
        ordered_all = ordered_p_a + p_b[: max(10, target_neg // 2)] + p_b[max(10, target_neg // 2) :]

        for c_idx, (_, r, (bx1, by1, bx2, by2), sub) in enumerate(ordered_all):
            if counts[sp]["non_pothole"] >= target_neg:
                break
            with Image.open(r["image_path"]) as im_raw:
                rgb = im_raw.convert("RGB")
                iw, ih = rgb.size
                px1, py1, px2, py2 = expand_and_clamp_box(bx1, by1, bx2, by2, iw, ih, pad_frac=pad_frac)
                if px2 - px1 >= 16 and py2 - py1 >= 16:
                    dst = ver_dir / sp / "non_pothole" / f"neg_{sub}_{r['image_id']}_{c_idx:03d}.jpg"
                    rgb.crop((px1, py1, px2, py2)).save(dst, quality=92)
                    counts[sp]["non_pothole"] += 1
                    subcats[sp][sub] += 1
                    _add_row(dst, sp, "non_pothole", sub, r)

        # Fallback Priority C (only reviewed prior_detector_fp frames) if still short of target_neg
        for r in fp_neg_imgs:
            if counts[sp]["non_pothole"] >= target_neg:
                break
            with Image.open(r["image_path"]) as im_raw:
                rgb = im_raw.convert("RGB")
                iw, ih = rgb.size
                pw, ph = max(64, min(iw // 4, 280)), max(48, min(ih // 5, 180))
                xs = rng.randint(int(iw * 0.15), max(int(iw * 0.15), int(iw * 0.80) - pw))
                ys = rng.randint(int(ih * 0.68), max(int(ih * 0.68), int(ih * 0.94) - ph))
                dst = ver_dir / sp / "non_pothole" / f"neg_fp_{r['image_id']}.jpg"
                rgb.crop((xs, ys, xs + pw, ys + ph)).save(dst, quality=92)
                counts[sp]["non_pothole"] += 1
                sub = r["negative_subcategory"] or "other"
                subcats[sp][sub] += 1
                _add_row(dst, sp, "non_pothole", sub, r)

    if ver_rows:
        with (ver_dir / "verifier_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ver_rows[0]))
            w.writeheader()
            w.writerows(ver_rows)

    return {
        "pad_fraction": pad_frac,
        "splits": counts,
        "negative_subcategories_by_split": {sp: dict(subcats[sp]) for sp in ("train", "val", "test")},
    }


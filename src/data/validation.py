"""Dataset validation, box verification, and duplicate/near-duplicate detection."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_image(path: Path, hash_size: int = 8) -> str:
    """Compute a 64-bit difference hash (dHash) for near-duplicate detection."""
    with Image.open(path) as img:
        gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits = (bits << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return f"{bits:016x}"


def validate_yolo_label_file(label_path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """Validate a YOLO label file. Returns (parsed_boxes, errors)."""
    errors: list[str] = []
    boxes: list[tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return boxes, [f"missing_label:{label_path.name}"]

    text = label_path.read_text(encoding="utf-8")
    for idx, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"invalid_columns:{label_path.name}:L{idx}")
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            errors.append(f"non_numeric:{label_path.name}:L{idx}")
            continue
        if cls_id != 0:
            errors.append(f"invalid_class_{cls_id}:{label_path.name}:L{idx}")
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            errors.append(f"out_of_bounds:{label_path.name}:L{idx}")
            continue
        boxes.append((cls_id, cx, cy, w, h))
    return boxes, errors


def validate_detector_processed(processed_dir: Path) -> dict[str, Any]:
    """Validate the final processed YOLO detector directory for split isolation and label integrity."""
    report: dict[str, Any] = {"splits": {}, "overlap_errors": [], "label_errors": [], "valid": True}
    split_names: dict[str, set[str]] = {}

    for split in ("train", "val", "test"):
        img_dir = processed_dir / "images" / split
        lbl_dir = processed_dir / "labels" / split
        imgs = sorted(p for p in img_dir.iterdir() if p.is_file()) if img_dir.exists() else []
        lbls = sorted(p for p in lbl_dir.iterdir() if p.is_file() and p.suffix == ".txt") if lbl_dir.exists() else []

        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}
        split_names[split] = img_stems

        orphaned_imgs = sorted(img_stems - lbl_stems)
        orphaned_lbls = sorted(lbl_stems - img_stems)
        if orphaned_imgs:
            report["label_errors"].extend([f"{split}:orphaned_image:{s}" for s in orphaned_imgs[:20]])
        if orphaned_lbls:
            report["label_errors"].extend([f"{split}:orphaned_label:{s}" for s in orphaned_lbls[:20]])

        pos = 0
        neg = 0
        total_boxes = 0
        for img_p in imgs:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            boxes, errs = validate_yolo_label_file(lbl_p)
            if errs:
                report["label_errors"].extend(errs[:20])
            if boxes:
                pos += 1
                total_boxes += len(boxes)
            else:
                neg += 1

        report["splits"][split] = {
            "images": len(imgs),
            "positive_images": pos,
            "negative_images": neg,
            "pothole_boxes": total_boxes,
        }

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = split_names[a] & split_names[b]
        if shared:
            report["overlap_errors"].append(f"{a}_{b}_overlap:{len(shared)}")

    report["valid"] = not report["overlap_errors"] and not report["label_errors"]
    return report

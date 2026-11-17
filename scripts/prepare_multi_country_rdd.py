"""Prepare and split multi-country RDD2022 dataset (United States, Norway, Japan).

Applies leak-free stratified splitting and in-split negative subsampling:
  • Positives (pothole class 'D40'): 100% preserved (or capped if specified).
  • Negatives (background): Subsampled strictly within splits to 10%–20% of the dataset
    (~0.25 negative-to-positive ratio), prioritizing visual confusers:
      - Tier 1: Alligator cracks (D20) + Transverse cracks (D10) (40%)
      - Tier 2: Alligator cracks (D20) (40%)
      - Tier 3: Transverse cracks (D10) (20%)
  • Generates YOLO-compliant label files (0 cx cy w h for potholes; empty file for negatives).
  • Preserves split isolation: no images cross train/val boundaries.
  • Updates data/processed/rdd2022/train.txt and val.txt.
  • Writes comprehensive audit logs to data/metadata/.

Usage:
    # Dry-run inspection:
    python scripts/prepare_multi_country_rdd.py --dry-run

    # Execute conversion for all three countries:
    python scripts/prepare_multi_country_rdd.py

    # Optional cap on Japanese positives if desired (e.g. 500):
    python scripts/prepare_multi_country_rdd.py --max-japan-positives 500
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RDD_ROOT = PROJECT_ROOT / "data" / "raw" / "RDD2022"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "rdd2022"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


@dataclass
class ImageRecord:
    image_id: str
    stem: str
    country: str
    img_path: Path
    xml_path: Path
    img_width: int
    img_height: int
    has_pothole: bool
    pothole_boxes: list[tuple[float, float, float, float]]  # (cx, cy, nw, nh) in [0, 1]
    classes: list[str]
    negative_tier: str = ""
    split: str = ""


def parse_voc_xml(xml_path: Path) -> tuple[int, int, list[tuple[float, float, float, float]], list[str]] | None:
    """Parse Pascal VOC XML, extracting image dimensions, normalized pothole boxes, and all class names."""
    if not xml_path.exists():
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_elem = root.find("size")
        if size_elem is None:
            return None
        w_text = size_elem.findtext("width", "").strip()
        h_text = size_elem.findtext("height", "").strip()
        if not w_text or not h_text:
            return None
        width = int(w_text)
        height = int(h_text)
        if width <= 0 or height <= 0:
            return None

        pothole_boxes = []
        all_classes = []

        for obj in root.findall("object"):
            name = obj.findtext("name", "").strip()
            if not name:
                continue
            all_classes.append(name)
            if name == "D40":
                bnd = obj.find("bndbox")
                if bnd is None:
                    continue
                xmin = float(bnd.findtext("xmin", "0").strip())
                ymin = float(bnd.findtext("ymin", "0").strip())
                xmax = float(bnd.findtext("xmax", "0").strip())
                ymax = float(bnd.findtext("ymax", "0").strip())

                # Validate coordinates
                xmin = max(0.0, min(float(width), xmin))
                xmax = max(0.0, min(float(width), xmax))
                ymin = max(0.0, min(float(height), ymin))
                ymax = max(0.0, min(float(height), ymax))

                box_w = xmax - xmin
                box_h = ymax - ymin
                if box_w <= 1.0 or box_h <= 1.0:
                    continue

                cx = (xmin + xmax) / (2.0 * width)
                cy = (ymin + ymax) / (2.0 * height)
                nw = box_w / width
                nh = box_h / height

                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                pothole_boxes.append((cx, cy, nw, nh))

        return width, height, pothole_boxes, all_classes
    except Exception as exc:
        print(f"  [WARN] Failed to parse XML {xml_path.name}: {exc}")
        return None


def classify_tier(classes: list[str]) -> str:
    """Classify negative image into crack hardness tiers."""
    has_d20 = "D20" in classes
    has_d10 = "D10" in classes
    if has_d20 and has_d10:
        return "Tier 1: D20 (alligator) + D10 (transverse)"
    elif has_d20:
        return "Tier 2: D20 (alligator crack)"
    elif has_d10:
        return "Tier 3: D10 (transverse crack)"
    else:
        return "Tier 4: Other background"


def discover_country_records(rdd_root: Path, country: str) -> list[ImageRecord]:
    """Scan and parse all image/XML pairs for a specific country."""
    # Handle both nested (<Country>/<Country>/train) and flat (<Country>/train) structures
    train_dir = rdd_root / country / country / "train"
    if not train_dir.exists():
        train_dir = rdd_root / country / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Cannot find train directory for country '{country}' in {rdd_root}")

    images_dir = train_dir / "images"
    xml_dir = train_dir / "annotations" / "xmls"

    if not images_dir.exists() or not xml_dir.exists():
        raise FileNotFoundError(f"Missing images or annotations/xmls in {train_dir}")

    records = []
    xml_files = sorted(xml_dir.glob("*.xml"))
    for xml_p in xml_files:
        stem = xml_p.stem
        # Preferred image formats
        img_p = None
        for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                img_p = cand
                break
        if img_p is None:
            continue

        parsed = parse_voc_xml(xml_p)
        if parsed is None:
            continue
        w, h, pboxes, classes = parsed
        has_pothole = len(pboxes) > 0
        tier = classify_tier(classes) if not has_pothole else ""

        records.append(
            ImageRecord(
                image_id=stem,
                stem=stem,
                country=country,
                img_path=img_p,
                xml_path=xml_p,
                img_width=w,
                img_height=h,
                has_pothole=has_pothole,
                pothole_boxes=pboxes,
                classes=classes,
                negative_tier=tier,
            )
        )

    return records


def subsample_negatives(
    candidates: list[ImageRecord],
    target_count: int,
    seed: int,
    both_ratio: float = 0.40,
    d20_ratio: float = 0.40,
) -> list[ImageRecord]:
    """Select target_count negatives prioritizing Tier 1, then Tier 2, then Tier 3."""
    t1 = [r for r in candidates if "Tier 1" in r.negative_tier]
    t2 = [r for r in candidates if "Tier 2" in r.negative_tier]
    t3 = [r for r in candidates if "Tier 3" in r.negative_tier]
    t4 = [r for r in candidates if "Tier 4" in r.negative_tier]

    rng = random.Random(seed)
    rng.shuffle(t1)
    rng.shuffle(t2)
    rng.shuffle(t3)
    rng.shuffle(t4)

    n_both = min(len(t1), int(round(target_count * both_ratio)))
    n_d20 = min(len(t2), int(round(target_count * d20_ratio)))
    remaining = target_count - n_both - n_d20
    n_d10 = min(len(t3), remaining)

    selected = t1[:n_both] + t2[:n_d20] + t3[:n_d10]
    if len(selected) < target_count:
        leftover = t1[n_both:] + t2[n_d20:] + t3[n_d10:] + t4
        rng.shuffle(leftover)
        needed = target_count - len(selected)
        selected.extend(leftover[:needed])

    return selected[:target_count]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and split multi-country RDD2022 dataset for YOLO pothole detection."
    )
    parser.add_argument(
        "--rdd-root",
        type=Path,
        default=DEFAULT_RDD_ROOT,
        help="Path to RDD2022 root directory containing country subfolders",
    )
    parser.add_argument(
        "--countries",
        type=str,
        default="United_States,Norway,Japan",
        help="Comma-separated countries to include (default: United_States,Norway,Japan)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of dataset to allocate to validation split (default: 0.15)",
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=0.25,
        help="Negative background ratio relative to positives (default: 0.25 -> 20%% background)",
    )
    parser.add_argument(
        "--max-japan-positives",
        type=int,
        default=None,
        help="Optional upper limit on Japan positive pothole images (default: None, use all 1,390)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print detailed dataset statistics and split plan without writing files",
    )
    args = parser.parse_args(argv)

    rdd_root = args.rdd_root.resolve()
    countries = [c.strip() for c in args.countries.split(",") if c.strip()]

    print("================================================================================")
    print("Multi-Country RDD2022 Dataset Preparation & Negative Balancing")
    print(f"Countries:    {', '.join(countries)}")
    print(f"Val Fraction: {args.val_fraction * 100:.1f}%")
    print(f"Neg Ratio:    {args.neg_ratio:.2f} (~{args.neg_ratio / (1.0 + args.neg_ratio) * 100:.1f}% background)")
    if args.max_japan_positives:
        print(f"Japan Positives Capped: {args.max_japan_positives}")
    print(f"Random Seed:  {args.seed}")
    print("================================================================================\n")

    all_selected_train: list[ImageRecord] = []
    all_selected_val: list[ImageRecord] = []
    all_selected_negatives: list[ImageRecord] = []

    country_stats = {}

    for country in countries:
        print(f"Scanning and analyzing annotations for {country}...")
        records = discover_country_records(rdd_root, country)
        positives = [r for r in records if r.has_pothole]
        negatives = [r for r in records if not r.has_pothole]

        # Optional cap on Japan positives
        if country == "Japan" and args.max_japan_positives and len(positives) > args.max_japan_positives:
            rng_cap = random.Random(args.seed)
            rng_cap.shuffle(positives)
            positives = sorted(positives[: args.max_japan_positives], key=lambda r: r.stem)

        # Deterministic shuffle for splitting
        rng_pos = random.Random(args.seed + hash(country))
        rng_pos.shuffle(positives)

        n_val_pos = max(1, int(round(len(positives) * args.val_fraction)))
        n_train_pos = len(positives) - n_val_pos

        train_pos = positives[:n_train_pos]
        val_pos = positives[n_train_pos:]

        # Deterministic shuffle for negative candidates
        rng_neg = random.Random(args.seed + hash(country) + 100)
        rng_neg.shuffle(negatives)

        n_val_neg_pool = int(round(len(negatives) * args.val_fraction))
        val_neg_candidates = negatives[:n_val_neg_pool]
        train_neg_candidates = negatives[n_val_neg_pool:]

        # Calculate negative targets based on neg_ratio
        train_neg_target = int(round(len(train_pos) * args.neg_ratio))
        val_neg_target = int(round(len(val_pos) * args.neg_ratio))

        # Subsample negatives strictly within split
        train_neg = subsample_negatives(train_neg_candidates, train_neg_target, seed=args.seed + 1)
        val_neg = subsample_negatives(val_neg_candidates, val_neg_target, seed=args.seed + 2)

        # Assign split tags
        for r in train_pos:
            r.split = "train"
        for r in train_neg:
            r.split = "train"
        for r in val_pos:
            r.split = "val"
        for r in val_neg:
            r.split = "val"

        selected_train = train_pos + train_neg
        selected_val = val_pos + val_neg

        all_selected_train.extend(selected_train)
        all_selected_val.extend(selected_val)
        all_selected_negatives.extend(train_neg + val_neg)

        country_stats[country] = {
            "total_available": len(records),
            "pos_total": len(positives),
            "pos_train": len(train_pos),
            "pos_val": len(val_pos),
            "neg_train": len(train_neg),
            "neg_val": len(val_neg),
            "total_used": len(selected_train) + len(selected_val),
            "bg_percent": (len(train_neg) + len(val_neg)) / (len(selected_train) + len(selected_val)) * 100.0,
        }

        print(f"  {country} Summary:")
        print(f"    Available images: {len(records):>5} (Positives: {len(positives)}, Negatives: {len(negatives)})")
        print(f"    Selected Train:   {len(selected_train):>5} (Positives: {len(train_pos)}, Negatives: {len(train_neg)})")
        print(f"    Selected Val:     {len(selected_val):>5} (Positives: {len(val_pos)}, Negatives: {len(val_neg)})")
        print(f"    Background Ratio: {country_stats[country]['bg_percent']:.1f}%")

    total_pos = sum(1 for r in all_selected_train + all_selected_val if r.has_pothole)
    total_neg = len(all_selected_negatives)
    total_all = len(all_selected_train) + len(all_selected_val)

    print("\n--------------------------------------------------------------------------------")
    print("Combined Multi-Country Dataset Statistics:")
    print(f"  Total Images Selected:    {total_all:>5} (out of {sum(s['total_available'] for s in country_stats.values())})")
    print(f"  Train Split:              {len(all_selected_train):>5} ({sum(1 for r in all_selected_train if r.has_pothole)} pos + {sum(1 for r in all_selected_train if not r.has_pothole)} neg)")
    print(f"  Val Split:                {len(all_selected_val):>5} ({sum(1 for r in all_selected_val if r.has_pothole)} pos + {sum(1 for r in all_selected_val if not r.has_pothole)} neg)")
    print(f"  Total Pothole Positives:  {total_pos:>5} ({total_pos / total_all * 100:.1f}%)")
    print(f"  Total Crack Negatives:    {total_neg:>5} ({total_neg / total_all * 100:.1f}%)")
    print("--------------------------------------------------------------------------------")

    if args.dry_run:
        print("\n[Dry-run completed. No files were written to disk.]")
        return

    # Write YOLO label files
    print("\nWriting YOLO label files to raw dataset directories and processed directories...")
    DEFAULT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    labels_proc_train = DEFAULT_PROCESSED_DIR / "labels" / "train"
    labels_proc_val = DEFAULT_PROCESSED_DIR / "labels" / "val"
    labels_proc_train.mkdir(parents=True, exist_ok=True)
    labels_proc_val.mkdir(parents=True, exist_ok=True)

    # Ensure each country's train/labels directory exists
    country_label_dirs = {}
    for country in countries:
        train_dir = rdd_root / country / country / "train"
        if not train_dir.exists():
            train_dir = rdd_root / country / "train"
        lbl_dir = train_dir / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        country_label_dirs[country] = lbl_dir

    written_labels = 0
    for r in all_selected_train + all_selected_val:
        lines = []
        if r.has_pothole:
            for cx, cy, nw, nh in r.pothole_boxes:
                lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        content = "\n".join(lines) + ("\n" if lines else "")

        # 1. Write to raw country labels directory (for direct Ultralytics resolution)
        lbl_raw_path = country_label_dirs[r.country] / f"{r.stem}.txt"
        lbl_raw_path.write_text(content, encoding="utf-8")

        # 2. Write to processed labels directory (for split backup)
        target_proc_dir = labels_proc_train if r.split == "train" else labels_proc_val
        lbl_proc_path = target_proc_dir / f"{r.stem}.txt"
        lbl_proc_path.write_text(content, encoding="utf-8")

        written_labels += 1

    print(f"  [OK] Successfully wrote {written_labels} label files.")

    # Write train.txt and val.txt manifests
    print("\nUpdating train.txt and val.txt manifests...")
    train_txt = DEFAULT_PROCESSED_DIR / "train.txt"
    val_txt = DEFAULT_PROCESSED_DIR / "val.txt"

    # Backup existing manifests
    if train_txt.exists():
        shutil.copy2(train_txt, DEFAULT_PROCESSED_DIR / "train.txt.bak")
    if val_txt.exists():
        shutil.copy2(val_txt, DEFAULT_PROCESSED_DIR / "val.txt.bak")

    train_lines = [r.img_path.resolve().as_posix() for r in all_selected_train]
    val_lines = [r.img_path.resolve().as_posix() for r in all_selected_val]

    train_txt.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    print(f"  [OK] Wrote {len(train_lines)} entries to {train_txt}")
    print(f"  [OK] Wrote {len(val_lines)} entries to {val_txt}")

    # Write audit manifest CSVs
    DEFAULT_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_csv = DEFAULT_METADATA_DIR / "rdd_dataset_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "country",
                "split",
                "has_pothole",
                "pothole_count",
                "tier",
                "classes",
                "image_width",
                "image_height",
                "filepath",
            ],
        )
        writer.writeheader()
        for r in all_selected_train + all_selected_val:
            writer.writerow({
                "image_id": r.stem,
                "country": r.country,
                "split": r.split,
                "has_pothole": r.has_pothole,
                "pothole_count": len(r.pothole_boxes),
                "tier": r.negative_tier,
                "classes": "+".join(sorted(set(r.classes))),
                "image_width": r.img_width,
                "image_height": r.img_height,
                "filepath": r.img_path.resolve().as_posix(),
            })
    print(f"  [OK] Saved complete dataset audit log to: {manifest_csv}")

    negatives_csv = DEFAULT_METADATA_DIR / "rdd_subsampled_negatives.csv"
    with negatives_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "country",
                "split",
                "tier",
                "d20_count",
                "d10_count",
                "d00_count",
                "total_crack_boxes",
                "classes_str",
            ],
        )
        writer.writeheader()
        for r in all_selected_negatives:
            writer.writerow({
                "image_id": r.stem,
                "country": r.country,
                "split": r.split,
                "tier": r.negative_tier,
                "d20_count": r.classes.count("D20"),
                "d10_count": r.classes.count("D10"),
                "d00_count": r.classes.count("D00"),
                "total_crack_boxes": len(r.classes),
                "classes_str": "+".join(sorted(set(r.classes))),
            })
    print(f"  [OK] Updated subsampled negatives audit log to: {negatives_csv}")

    # Remove old YOLO dataset cache files to force clean re-index
    print("\nClearing old YOLO cache files...")
    for cache_p in rdd_root.rglob("labels.cache"):
        try:
            cache_p.unlink()
            print(f"  Removed old cache: {cache_p}")
        except Exception:
            pass

    print("\n================================================================================")
    print("Multi-Country RDD2022 Dataset successfully prepared!")
    print("================================================================================")


if __name__ == "__main__":
    main()


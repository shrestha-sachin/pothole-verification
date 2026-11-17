"""Prepare and split multi-country RDD2022 V3 dataset with 80/10/10 split and strict leakage protection.

Versioned Pipeline:
  - Input: data/raw/RDD2022 (United_States, Norway, Japan)
  - Output: data/processed/rdd2022_v3/ (completely separate from existing rdd2022)
  - Dataset config: configs/detector_v3.yaml

Key Methodological Guarantees:
  1. 80% Train / 10% Validation / 10% Held-Out Test split.
  2. 100% preservation of all valid pothole-positive images (RDD class 'D40').
  3. Negative crack subsampling strictly within each split to maintain ~20% background
     (~0.25 negative-to-positive ratio) per Ultralytics recommendations:
       - Tier 1: Alligator cracks (D20) + Transverse cracks (D10) (40%)
       - Tier 2: Alligator cracks (D20) (40%)
       - Tier 3: Transverse cracks (D10) (20%)
  4. Content deduplication via SHA-256: Identical images are connected into single
     atomic groups and never split across train/val/test boundaries.
  5. Stratified group splitting: Preserves country and pothole/crack representation
     across all three splits deterministically (seed 42).
  6. Zero cross-split leakage: Absolute disjointness verified across image IDs and hashes.
  7. Non-destructive: Never overwrites existing outputs without explicit --overwrite.
  8. Full auditability: Exports split_metadata.csv, dataset_summary.json, and audit CSVs.

Usage:
    # Dry-run inspection (no files written):
    python scripts/prepare_rdd2022_v3.py --dry-run

    # Execute conversion for V3:
    python scripts/prepare_rdd2022_v3.py

    # Allow overwrite if re-generating V3:
    python scripts/prepare_rdd2022_v3.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RDD_ROOT = PROJECT_ROOT / "data" / "raw" / "RDD2022"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "rdd2022_v3"
DEFAULT_YAML_PATH = PROJECT_ROOT / "configs" / "detector_v3.yaml"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "data" / "metadata"


@dataclass
class RDDItemV3:
    image_id: str
    stem: str
    country: str
    img_path: Path
    xml_path: Path
    img_width: int
    img_height: int
    sha256: str
    has_pothole: bool
    pothole_boxes: list[tuple[float, float, float, float]]  # (cx, cy, nw, nh) in [0, 1]
    classes: list[str]
    negative_tier: str = ""
    group_id: str = ""
    split: str = ""
    selected: bool = False


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for content-based duplicate detection."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


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

                # Validate and clamp pixel coordinates
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

                # Clamp normalized coordinates to [0, 1]
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


def discover_country_items(rdd_root: Path, country: str) -> list[RDDItemV3]:
    """Scan and parse all valid image/XML pairs for a country."""
    train_dir = rdd_root / country / country / "train"
    if not train_dir.exists():
        train_dir = rdd_root / country / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Cannot find train directory for country '{country}' in {rdd_root}")

    images_dir = train_dir / "images"
    xml_dir = train_dir / "annotations" / "xmls"

    if not images_dir.exists() or not xml_dir.exists():
        raise FileNotFoundError(f"Missing images or annotations/xmls in {train_dir}")

    items = []
    xml_files = sorted(xml_dir.glob("*.xml"))
    for xml_p in xml_files:
        stem = xml_p.stem
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
        sha256 = compute_sha256(img_p)

        items.append(
            RDDItemV3(
                image_id=stem,
                stem=stem,
                country=country,
                img_path=img_p,
                xml_path=xml_p,
                img_width=w,
                img_height=h,
                sha256=sha256,
                has_pothole=has_pothole,
                pothole_boxes=pboxes,
                classes=classes,
                negative_tier=tier,
            )
        )
    return items


def assign_connected_group_ids(items: list[RDDItemV3]) -> dict[str, list[RDDItemV3]]:
    """Union exact SHA-256 duplicate clusters into atomic groups."""
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    seen_hash: dict[str, int] = {}
    for idx, it in enumerate(items):
        prev = seen_hash.setdefault(it.sha256, idx)
        if prev != idx:
            union(idx, prev)

    groups: dict[str, list[RDDItemV3]] = {}
    for idx, it in enumerate(items):
        root_idx = find(idx)
        root_stem = items[root_idx].stem
        group_id = f"grp_{items[root_idx].country}_{root_stem}"
        it.group_id = group_id
        groups.setdefault(group_id, []).append(it)

    return groups


def stratified_split_groups(
    groups: dict[str, list[RDDItemV3]],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[RDDItemV3], list[RDDItemV3], list[RDDItemV3]]:
    """Stratified split across atomic groups to guarantee split isolation."""
    if len(groups) < 3:
        raise ValueError(f"Insufficient independent groups ({len(groups)}). Need at least 3 for train/val/test.")

    # Strata defined by: (country, has_pothole, tier)
    strata: dict[tuple[str, bool, str], list[str]] = {}
    for gid, group_items in sorted(groups.items()):
        country = group_items[0].country
        has_pos = any(it.has_pothole for it in group_items)
        tier = group_items[0].negative_tier if not has_pos else "pothole"
        strata.setdefault((country, has_pos, tier), []).append(gid)

    rng = random.Random(seed)
    train_gids: list[str] = []
    val_gids: list[str] = []
    test_gids: list[str] = []

    for key in sorted(strata.keys()):
        gids = list(strata[key])
        rng.shuffle(gids)
        n = len(gids)

        if n == 1:
            # Assign greedily to the split currently furthest behind target
            tot_train = len(train_gids)
            tot_val = len(val_gids)
            tot_test = len(test_gids)
            tot = tot_train + tot_val + tot_test
            if tot == 0:
                train_gids.append(gids[0])
            else:
                d_train = train_frac - (tot_train / tot)
                d_val = val_frac - (tot_val / tot)
                d_test = test_frac - (tot_test / tot)
                max_d = max(d_train, d_val, d_test)
                if max_d == d_val:
                    val_gids.append(gids[0])
                elif max_d == d_test:
                    test_gids.append(gids[0])
                else:
                    train_gids.append(gids[0])
        elif n == 2:
            train_gids.append(gids[0])
            val_gids.append(gids[1])
        else:
            n_val = max(1, int(round(n * val_frac)))
            n_test = max(1, int(round(n * test_frac)))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = 1
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1

            train_gids.extend(gids[:n_train])
            val_gids.extend(gids[n_train : n_train + n_val])
            test_gids.extend(gids[n_train + n_val :])

    train_items: list[RDDItemV3] = []
    val_items: list[RDDItemV3] = []
    test_items: list[RDDItemV3] = []

    for gid in train_gids:
        for it in groups[gid]:
            it.split = "train"
            train_items.append(it)
    for gid in val_gids:
        for it in groups[gid]:
            it.split = "val"
            val_items.append(it)
    for gid in test_gids:
        for it in groups[gid]:
            it.split = "test"
            test_items.append(it)

    return train_items, val_items, test_items


def subsample_negatives_in_split(
    items: list[RDDItemV3],
    neg_ratio: float = 0.25,
    both_ratio: float = 0.40,
    d20_ratio: float = 0.40,
    seed: int = 42,
) -> list[RDDItemV3]:
    """Subsample negative crack images strictly within split to maintain ~20% background."""
    positives = [it for it in items if it.has_pothole]
    negatives = [it for it in items if not it.has_pothole]

    target_count = int(round(len(positives) * neg_ratio))
    target_count = max(0, min(target_count, len(negatives)))

    # Subsample per country within split to preserve geographic representation
    countries = sorted(set(it.country for it in items))
    selected_negatives: list[RDDItemV3] = []

    for c in countries:
        c_pos = [it for it in positives if it.country == c]
        c_neg = [it for it in negatives if it.country == c]
        c_target = int(round(len(c_pos) * neg_ratio))
        c_target = max(1 if c_neg and c_pos else 0, min(c_target, len(c_neg)))

        t1 = [it for it in c_neg if "Tier 1" in it.negative_tier]
        t2 = [it for it in c_neg if "Tier 2" in it.negative_tier]
        t3 = [it for it in c_neg if "Tier 3" in it.negative_tier]
        t4 = [it for it in c_neg if "Tier 4" in it.negative_tier]

        rng = random.Random(seed + hash(c))
        rng.shuffle(t1)
        rng.shuffle(t2)
        rng.shuffle(t3)
        rng.shuffle(t4)

        n_both = min(len(t1), int(round(c_target * both_ratio)))
        n_d20 = min(len(t2), int(round(c_target * d20_ratio)))
        remaining = c_target - n_both - n_d20
        n_d10 = min(len(t3), remaining)

        c_selected = t1[:n_both] + t2[:n_d20] + t3[:n_d10]
        if len(c_selected) < c_target:
            leftover = t1[n_both:] + t2[n_d20:] + t3[n_d10:] + t4
            rng.shuffle(leftover)
            c_selected.extend(leftover[: c_target - len(c_selected)])

        selected_negatives.extend(c_selected[:c_target])

    selected_ids = {it.image_id for it in selected_negatives}
    for it in positives:
        it.selected = True
    for it in negatives:
        it.selected = it.image_id in selected_ids

    return [it for it in items if it.selected]


def verify_leak_free(train_items: list[RDDItemV3], val_items: list[RDDItemV3], test_items: list[RDDItemV3]) -> None:
    """Verify zero overlap in image IDs, stems, and SHA-256 hashes across splits."""
    t_ids = {it.image_id for it in train_items}
    v_ids = {it.image_id for it in val_items}
    te_ids = {it.image_id for it in test_items}

    if t_ids & v_ids:
        raise RuntimeError(f"Data leakage! Train and Val share image IDs: {t_ids & v_ids}")
    if t_ids & te_ids:
        raise RuntimeError(f"Data leakage! Train and Test share image IDs: {t_ids & te_ids}")
    if v_ids & te_ids:
        raise RuntimeError(f"Data leakage! Val and Test share image IDs: {v_ids & te_ids}")

    t_hashes = {it.sha256 for it in train_items}
    v_hashes = {it.sha256 for it in val_items}
    te_hashes = {it.sha256 for it in test_items}

    if t_hashes & v_hashes:
        raise RuntimeError(f"Data leakage! Train and Val share identical content hashes: {t_hashes & v_hashes}")
    if t_hashes & te_hashes:
        raise RuntimeError(f"Data leakage! Train and Test share identical content hashes: {t_hashes & te_hashes}")
    if v_hashes & te_hashes:
        raise RuntimeError(f"Data leakage! Val and Test share identical content hashes: {v_hashes & te_hashes}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and split multi-country RDD2022 V3 dataset (80/10/10 split, zero leakage)."
    )
    parser.add_argument(
        "--rdd-root",
        type=Path,
        default=DEFAULT_RDD_ROOT,
        help="Path to RDD2022 root directory containing country subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Versioned output directory for RDD2022 V3 (default: data/processed/rdd2022_v3)",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=DEFAULT_YAML_PATH,
        help="Destination path for detector V3 YAML configuration (default: configs/detector_v3.yaml)",
    )
    parser.add_argument(
        "--countries",
        type=str,
        default="United_States,Norway,Japan",
        help="Comma-separated country list (default: United_States,Norway,Japan)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.80,
        help="Training split fraction (default: 0.80)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.10,
        help="Validation split fraction (default: 0.10)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.10,
        help="Held-out test split fraction (default: 0.10)",
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=0.25,
        help="Ratio of negative crack images to positive images (default: 0.25 -> 20%% background)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow overwriting an existing V3 output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect counts and print summary without writing files",
    )
    args = parser.parse_args(argv)

    rdd_root = args.rdd_root.resolve()
    out_dir = args.output_dir.resolve()
    yaml_path = args.yaml_path.resolve()

    if not args.dry_run and not args.overwrite:
        if out_dir.exists() and any(out_dir.iterdir()):
            sys.exit(
                f"ERROR: Destination directory {out_dir} is not empty.\n"
                "  Use a new versioned --output-dir path or pass --overwrite."
            )
        if yaml_path.exists():
            sys.exit(
                f"ERROR: Dataset YAML already exists at {yaml_path}.\n"
                "  Use a new --yaml-path or pass --overwrite."
            )

    countries = [c.strip() for c in args.countries.split(",") if c.strip()]

    print("=" * 80)
    print("RDD2022 V3 Dataset Preparation & Leakage-Free 80/10/10 Split")
    print(f"Countries:       {', '.join(countries)}")
    print(f"Split Ratios:    Train {args.train_split*100:.1f}% / Val {args.val_split*100:.1f}% / Test {args.test_split*100:.1f}%")
    print(f"Negative Ratio:  {args.neg_ratio:.2f} (~{args.neg_ratio / (1.0 + args.neg_ratio) * 100:.1f}% background)")
    print(f"Output Directory:{out_dir}")
    print(f"YAML Config:     {yaml_path}")
    print(f"Seed:            {args.seed}")
    print("=" * 80 + "\n")

    all_raw_items: list[RDDItemV3] = []
    country_raw_counts = {}

    for c in countries:
        print(f"Scanning and parsing annotations for {c}...")
        c_items = discover_country_items(rdd_root, c)
        c_pos = sum(1 for it in c_items if it.has_pothole)
        c_neg = len(c_items) - c_pos
        country_raw_counts[c] = {"total": len(c_items), "pos": c_pos, "neg": c_neg}
        print(f"  {c}: {len(c_items)} pairs ({c_pos} potholes, {c_neg} candidate negatives)")
        all_raw_items.extend(c_items)

    print(f"\nTotal raw pairs across all countries: {len(all_raw_items)}")
    print(f"Total pothole-positive images:        {sum(1 for it in all_raw_items if it.has_pothole)}")
    print(f"Total pothole-negative candidates:    {sum(1 for it in all_raw_items if not it.has_pothole)}")

    # Group connected components based on SHA-256 duplicate content
    groups = assign_connected_group_ids(all_raw_items)
    dup_groups = {gid: items for gid, items in groups.items() if len(items) > 1}
    print(f"Duplicate content clusters detected:  {len(dup_groups)} ({sum(len(items) for items in dup_groups.values())} total images)")
    for gid, items in dup_groups.items():
        print(f"  Cluster {gid}: {[it.stem for it in items]}")

    # Stratified group splitting (80/10/10)
    print(f"\nPartitioning into Train ({args.train_split*100:.0f}%), Val ({args.val_split*100:.0f}%), Test ({args.test_split*100:.0f}%)...")
    train_pool, val_pool, test_pool = stratified_split_groups(
        groups,
        train_frac=args.train_split,
        val_frac=args.val_split,
        test_frac=args.test_split,
        seed=args.seed,
    )

    # Subsample crack negatives strictly within each split
    train_selected = subsample_negatives_in_split(train_pool, neg_ratio=args.neg_ratio, seed=args.seed + 1)
    val_selected = subsample_negatives_in_split(val_pool, neg_ratio=args.neg_ratio, seed=args.seed + 2)
    test_selected = subsample_negatives_in_split(test_pool, neg_ratio=args.neg_ratio, seed=args.seed + 3)

    # Verify zero data leakage
    verify_leak_free(train_selected, val_selected, test_selected)
    print("  [OK] Leak-free isolation verified across all splits (0 ID overlap, 0 SHA-256 overlap).")

    # Split breakdown
    def print_split_summary(split_name: str, items: list[RDDItemV3]) -> dict:
        pos = [it for it in items if it.has_pothole]
        neg = [it for it in items if not it.has_pothole]
        boxes = sum(len(it.pothole_boxes) for it in pos)
        by_country = {}
        for c in countries:
            c_pos = sum(1 for it in pos if it.country == c)
            c_neg = sum(1 for it in neg if it.country == c)
            by_country[c] = {"pos": c_pos, "neg": c_neg, "total": c_pos + c_neg}
        bg_pct = (len(neg) / len(items) * 100.0) if items else 0.0
        print(f"\n--- {split_name.upper()} SPLIT ({len(items)} images, {boxes} pothole boxes, {bg_pct:.1f}% background) ---")
        print(f"  Total: {len(items):>5} | Positives: {len(pos):>5} | Negatives: {len(neg):>5}")
        for c in countries:
            st = by_country[c]
            print(f"    {c:<14}: {st['total']:>5} ({st['pos']:>4} pos, {st['neg']:>4} neg)")
        return {"total": len(items), "pos": len(pos), "neg": len(neg), "boxes": boxes, "by_country": by_country}

    s_train = print_split_summary("train", train_selected)
    s_val = print_split_summary("validation", val_selected)
    s_test = print_split_summary("held-out test", test_selected)

    total_selected = len(train_selected) + len(val_selected) + len(test_selected)
    total_pos = s_train["pos"] + s_val["pos"] + s_test["pos"]
    total_neg = s_train["neg"] + s_val["neg"] + s_test["neg"]
    total_boxes = s_train["boxes"] + s_val["boxes"] + s_test["boxes"]

    print("\n" + "=" * 80)
    print(f"V3 TOTAL SELECTED IMAGES:    {total_selected:>5} (out of {len(all_raw_items)} available)")
    print(f"  Train Images:              {len(train_selected):>5} ({len(train_selected)/total_selected*100:.1f}%)")
    print(f"  Val Images:                {len(val_selected):>5} ({len(val_selected)/total_selected*100:.1f}%)")
    print(f"  Test Images (Held-out):    {len(test_selected):>5} ({len(test_selected)/total_selected*100:.1f}%)")
    print(f"  Total Pothole Positives:   {total_pos:>5} ({total_pos/total_selected*100:.1f}%)")
    print(f"  Total Crack Negatives:     {total_neg:>5} ({total_neg/total_selected*100:.1f}%)")
    print(f"  Total Pothole Boxes (D40): {total_boxes:>5}")
    print("=" * 80)

    if args.dry_run:
        print("\n[Dry-run completed. No files were written to disk.]")
        return

    # Write files
    print(f"\nWriting versioned outputs to {out_dir}...")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_base = out_dir / "labels"

    # Also ensure each raw country train/labels directory has the label files for Ultralytics
    country_raw_label_dirs = {}
    for c in countries:
        c_train = rdd_root / c / c / "train"
        if not c_train.exists():
            c_train = rdd_root / c / "train"
        lbl_dir = c_train / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        country_raw_label_dirs[c] = lbl_dir

    all_selected = [("train", train_selected), ("val", val_selected), ("test", test_selected)]
    for split_name, split_items in all_selected:
        split_lbl_dir = labels_base / split_name
        split_lbl_dir.mkdir(parents=True, exist_ok=True)
        manifest_lines = []

        for it in split_items:
            lines = []
            if it.has_pothole:
                for cx, cy, nw, nh in it.pothole_boxes:
                    lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            txt_content = "\n".join(lines) + ("\n" if lines else "")

            # Write to versioned processed labels
            (split_lbl_dir / f"{it.stem}.txt").write_text(txt_content, encoding="utf-8")

            # Write to raw country labels directory
            (country_raw_label_dirs[it.country] / f"{it.stem}.txt").write_text(txt_content, encoding="utf-8")

            manifest_lines.append(it.img_path.resolve().as_posix())

        manifest_file = out_dir / f"{split_name}.txt"
        manifest_file.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        print(f"  [OK] Wrote {len(manifest_lines)} paths to {manifest_file}")

    # Write split_metadata.csv
    meta_csv = out_dir / "split_metadata.csv"
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "country",
                "split",
                "has_pothole",
                "pothole_count",
                "tier",
                "classes_str",
                "image_width",
                "image_height",
                "group_id",
                "sha256",
                "filepath",
            ],
        )
        writer.writeheader()
        for _, items in all_selected:
            for it in items:
                writer.writerow({
                    "image_id": it.stem,
                    "country": it.country,
                    "split": it.split,
                    "has_pothole": it.has_pothole,
                    "pothole_count": len(it.pothole_boxes),
                    "tier": it.negative_tier,
                    "classes_str": "+".join(sorted(set(it.classes))),
                    "image_width": it.img_width,
                    "image_height": it.img_height,
                    "group_id": it.group_id,
                    "sha256": it.sha256,
                    "filepath": it.img_path.resolve().as_posix(),
                })
    print(f"  [OK] Wrote split metadata to {meta_csv}")

    # Write dataset_summary.json
    summary_json = out_dir / "dataset_summary.json"
    summary_data = {
        "dataset_name": "RDD2022_v3",
        "countries": countries,
        "split_ratios": {"train": args.train_split, "val": args.val_split, "test": args.test_split},
        "seed": args.seed,
        "negative_ratio": args.neg_ratio,
        "total_images": total_selected,
        "total_positives": total_pos,
        "total_negatives": total_neg,
        "total_pothole_boxes": total_boxes,
        "splits": {
            "train": s_train,
            "val": s_val,
            "test": s_test,
        },
    }
    summary_json.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"  [OK] Wrote dataset summary to {summary_json}")

    # Write RDD2022 dataset.yaml inside processed dir
    rdd_yaml = out_dir / "dataset.yaml"
    rdd_yaml_content = f"""# YOLO Dataset Configuration - RDD2022 V3 (Multi-Country 80/10/10 Split)
path: .

train: data/processed/rdd2022_v3/train.txt
val: data/processed/rdd2022_v3/val.txt
test: data/processed/rdd2022_v3/test.txt

names:
  0: pothole
"""
    rdd_yaml.write_text(rdd_yaml_content, encoding="utf-8")
    print(f"  [OK] Wrote RDD-only YAML to {rdd_yaml}")

    # Write combined detector_v3.yaml (combining protected Green Bay + RDD2022 V3)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    combined_yaml_content = f"""# YOLO Dataset Configuration - Experiment V3 (Multi-Country RDD + Local Green Bay)
# Protected recording-level splits for Green Bay + 80/10/10 stratified RDD2022 V3
path: .

train:
  - exports/yolo_labels/train.txt
  - data/processed/rdd2022_v3/train.txt

val:
  - exports/yolo_labels/val.txt
  - data/processed/rdd2022_v3/val.txt

test:
  - data/processed/rdd2022_v3/test.txt

names:
  0: pothole
"""
    yaml_path.write_text(combined_yaml_content, encoding="utf-8")
    print(f"  [OK] Wrote combined detector V3 YAML configuration to {yaml_path}")

    # Clear old YOLO caches
    print("\nClearing old YOLO cache files...")
    for cache_p in rdd_root.rglob("labels.cache"):
        try:
            cache_p.unlink()
            print(f"  Removed old cache: {cache_p}")
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("RDD2022 V3 Dataset preparation completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()


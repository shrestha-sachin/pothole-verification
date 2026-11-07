"""Convert RDD2022 (Road Damage Dataset) annotations to YOLO format with reproducible, leakage-free splitting.

Supports the multi-country RDD2022 structure:
    RDD2022/
    ├── Country_Name/ (e.g. United_States, Japan, India, Czech, Norway)
    │   └── train/
    │       ├── images/
    │       └── annotations/xmls/
    ...

Also supports flat VOC structures where images and XMLs are in the same or paired folders.

By default:
  - Extracts potholes (RDD2022 class 'D40') as YOLO class 0 ('pothole').
  - Images containing D40 (alone or with other classes) are pothole-positive.
  - Images without D40 with valid XML are candidate negatives (useful hard crack negatives D20/D10).
  - Default split: 80% train, 10% validation, 10% test.
  - Reproducible with configurable seed (default: 42).
  - Content deduplication (exact duplicates stay in same split).
  - Preserves verified metadata when available; falls back to sample grouping with honest documentation.
  - In-split negative subsampling: negatives are subsampled strictly within their protected splits.
  - Generates train.txt, val.txt, test.txt, split_metadata.csv, dataset_summary.json, and YOLO YAML.

Usage:
    # Full conversion (US-only, 80/10/10 split, balanced negatives ~20% background)
    python scripts/convert_rdd2022.py --input /path/to/RDD2022 --countries United_States --neg-ratio 0.25

    # Multi-country conversion
    python scripts/convert_rdd2022.py --input /path/to/RDD2022 --countries United_States,Japan

    # Dry-run inspection without writing files
    python scripts/convert_rdd2022.py --input /path/to/RDD2022 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RDD2022 damage classes:
# D00: Longitudinal Crack
# D10: Transverse Crack
# D20: Alligator Crack
# D40: Pothole
RDD_CLASSES = {
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
}

DEFAULT_COUNTRIES = [
    "United_States",
    "Japan",
    "India",
    "Czech",
    "Norway",
    "China_MotorBike",
    "China_Drone",
]


@dataclass
class RDDItem:
    """Represents a discovered image-annotation pair in RDD2022."""
    image_id: str
    img_path: Path
    xml_path: Path
    country: str
    img_w: int
    img_h: int
    boxes: list[tuple[int, float, float, float, float]]  # [cls_id, cx, cy, nw, nh]
    class_counts: dict[str, int]
    sha256: str
    has_pothole: bool
    has_crack: bool
    negative_tier: str = ""
    group_id: str = ""
    grouping_method: str = ""
    split: str = ""
    selected_in_manifest: bool = True


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for content-based duplicate detection."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_voc_xml(
    xml_path: Path,
    target_classes: dict[str, int],
    return_metadata: bool = False,
) -> tuple[int, int, list[tuple[int, float, float, float, float]]] | tuple[int, int, list[tuple[int, float, float, float, float]], dict[str, int]] | None:
    """Parse a Pascal VOC XML file and return (width, height, list of [class_id, cx, cy, w, h]).

    Returns None if the XML is corrupt or invalid, preventing it from being silently
    treated as a true negative road image.
    If return_metadata=True, also returns class_counts dictionary for all detected classes.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as exc:
        print(f"  [ERROR] Failed to parse XML {xml_path.name}: {exc}")
        return None

    size_elem = root.find("size")
    if size_elem is None:
        print(f"  [ERROR] Missing <size> element in {xml_path.name}")
        return None

    w_val = size_elem.find("width")
    h_val = size_elem.find("height")
    if w_val is None or h_val is None:
        print(f"  [ERROR] Missing width/height in {xml_path.name}")
        return None

    try:
        img_w = int(w_val.text.strip())
        img_h = int(h_val.text.strip())
    except (ValueError, AttributeError) as exc:
        print(f"  [ERROR] Non-numeric width/height in {xml_path.name}: {exc}")
        return None

    if img_w <= 0 or img_h <= 0:
        print(f"  [ERROR] Non-positive dimensions ({img_w}x{img_h}) in {xml_path.name}")
        return None

    boxes = []
    class_counts: dict[str, int] = {k: 0 for k in RDD_CLASSES}

    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue
        cls_name = name_elem.text.strip()
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        if cls_name not in target_classes:
            continue

        cls_id = target_classes[cls_name]
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        try:
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
        except (ValueError, AttributeError, TypeError):
            continue

        # Clamp to image boundaries
        xmin = max(0.0, min(xmin, float(img_w)))
        xmax = max(0.0, min(xmax, float(img_w)))
        ymin = max(0.0, min(ymin, float(img_h)))
        ymax = max(0.0, min(ymax, float(img_h)))

        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 1.0 or bh <= 1.0:
            continue

        cx = (xmin + bw / 2.0) / img_w
        cy = (ymin + bh / 2.0) / img_h
        nw = bw / img_w
        nh = bh / img_h

        boxes.append((cls_id, cx, cy, nw, nh))

    if return_metadata:
        return img_w, img_h, boxes, class_counts
    return img_w, img_h, boxes


def discover_pairs(input_dir: Path, countries: list[str] | None = None) -> list[tuple[Path, Path, str]]:
    """Locate all (image_path, xml_path, country) pairs in the RDD2022 dataset."""
    pairs: list[tuple[Path, Path, str]] = []

    # Check for country subdirectories
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    country_dirs = []
    if countries:
        country_set = {c.strip().lower() for c in countries if c.strip()}
        for d in subdirs:
            if d.name.lower() in country_set:
                country_dirs.append(d)
    else:
        country_dirs = subdirs

    search_dirs = country_dirs if country_dirs else [input_dir]

    for cdir in search_dirs:
        country_name = cdir.name if cdir != input_dir else "general"

        # Search for XML folders
        xml_candidates = list(cdir.rglob("annotations/xmls/*.xml"))
        if not xml_candidates:
            xml_candidates = list(cdir.rglob("*.xml"))

        for xml_file in xml_candidates:
            stem = xml_file.stem
            parent = xml_file.parent
            # Typical locations: ../images/{stem}.jpg, or same dir, or ../../images/
            img_candidates = [
                parent.parent / "images" / f"{stem}.jpg",
                parent.parent / "images" / f"{stem}.png",
                parent.parent / "images" / f"{stem}.jpeg",
                parent / f"{stem}.jpg",
                parent / f"{stem}.png",
                parent / f"{stem}.jpeg",
            ]
            img_file = next((p for p in img_candidates if p.exists()), None)
            if not img_file:
                # Broader search in cdir
                found = (
                    list(cdir.rglob(f"{stem}.jpg"))
                    + list(cdir.rglob(f"{stem}.png"))
                    + list(cdir.rglob(f"{stem}.jpeg"))
                )
                if found:
                    img_file = found[0]

            if img_file and img_file.exists():
                pairs.append((img_file, xml_file, country_name))

    return pairs


def classify_negative_tier(class_counts: dict[str, int]) -> str:
    """Classify negative image into hardness tiers based on crack damage types.

    Note: D10/D20 cracks teach the detector not to trigger false alarms on road distress,
    but are distinct from smartphone video look-alikes (shadows, asphalt patches, manhole covers).
    """
    d20 = class_counts.get("D20", 0)
    d10 = class_counts.get("D10", 0)
    if d20 > 0 and d10 > 0:
        return "Tier 1: D20 (alligator) + D10 (transverse)"
    elif d20 > 0:
        return "Tier 2: D20 (alligator crack)"
    elif d10 > 0:
        return "Tier 3: D10 (transverse crack)"
    else:
        return "Tier 4: Other background"


def resolve_group_id(
    item_id: str,
    country: str,
    sha256: str,
    duplicates_map: dict[str, list[str]],
    external_metadata: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve grouping identifier and document the exact method used.

    Priority:
    1. External verified sequence/route metadata if provided.
    2. Explicit filename sequence token (e.g. seq01, segment_02, route_A).
    3. Exact content duplicates grouped together (keeps duplicate content in same split).
    4. Fallback: individual sample grouping with documented limitation.
    """
    if external_metadata and item_id in external_metadata:
        return f"{country}_{external_metadata[item_id]}", "verified_metadata"

    # Check exact content duplicate cluster (identical images must remain in the same split)
    if len(duplicates_map.get(sha256, [])) > 1:
        return f"dup_{sha256[:12]}", "exact_content_duplicate"

    # Check for explicit sequence or segment pattern in filename
    seq_match = re.search(r"(?:seq|sequence|segment|track|route)[-_]?([a-zA-Z0-9]+)", item_id, re.IGNORECASE)
    if seq_match:
        return f"{country}_seq_{seq_match.group(1)}", "filename_sequence_tag"

    # Fallback: Individual sample grouping (honest documentation: no physical sequence assumed)
    return f"{country}_{item_id}", "sample_fallback_no_sequence_metadata"


def stratified_group_split(
    items: list[RDDItem],
    train_split: float = 0.80,
    val_split: float = 0.10,
    test_split: float = 0.10,
    seed: int = 42,
) -> tuple[list[RDDItem], list[RDDItem], list[RDDItem]]:
    """Partition items into train, validation, and test splits with group protection and stratification.

    Guarantees:
    - Entire groups remain together within a single split (no group leakage).
    - Preserves representation of pothole-positive and negative groups across splits.
    - Deterministic and reproducible with seed.
    - Prevents empty splits (enforces len(train) > 0, len(val) > 0, len(test) > 0).
    - Enforces zero ID and content hash overlap across splits.
    """
    if not items:
        raise ValueError("Cannot split empty item list.")

    total_frac = train_split + val_split + test_split
    if not (0.99 <= total_frac <= 1.01):
        raise ValueError(f"Split proportions must sum to 1.0 (got {train_split}+{val_split}+{test_split} = {total_frac:.3f})")

    if len(items) < 3:
        raise ValueError(f"Dataset has only {len(items)} items; at least 3 items are required for train/val/test splits.")

    # Group items by group_id
    groups: dict[str, list[RDDItem]] = {}
    for it in items:
        groups.setdefault(it.group_id, []).append(it)

    group_keys = sorted(groups.keys())

    # Build group-level strata: (country, has_pothole, has_crack)
    strata: dict[tuple[str, bool, bool], list[str]] = {}
    for gid in group_keys:
        grp = groups[gid]
        has_pothole = any(x.has_pothole for x in grp)
        has_crack = any(x.has_crack for x in grp)
        country = grp[0].country
        key = (country, has_pothole, has_crack)
        strata.setdefault(key, []).append(gid)

    rng = random.Random(seed)

    train_groups: list[str] = []
    val_groups: list[str] = []
    test_groups: list[str] = []

    # Stratified distribution across groups
    for stratum_key in sorted(strata.keys()):
        stratum_gids = list(strata[stratum_key])
        rng.shuffle(stratum_gids)

        n = len(stratum_gids)
        if n == 1:
            # Single group: assign greedily to split currently most below its overall target fraction
            n_tot_train = sum(len(groups[g]) for g in train_groups)
            n_tot_val = sum(len(groups[g]) for g in val_groups)
            n_tot_test = sum(len(groups[g]) for g in test_groups)
            tot = n_tot_train + n_tot_val + n_tot_test
            if tot == 0:
                train_groups.append(stratum_gids[0])
            else:
                train_deficit = train_split - (n_tot_train / tot)
                val_deficit = val_split - (n_tot_val / tot)
                test_deficit = test_split - (n_tot_test / tot)
                max_deficit = max(train_deficit, val_deficit, test_deficit)
                if max_deficit == val_deficit and len(val_groups) <= len(test_groups):
                    val_groups.append(stratum_gids[0])
                elif max_deficit == test_deficit:
                    test_groups.append(stratum_gids[0])
                else:
                    train_groups.append(stratum_gids[0])
        else:
            n_val = max(0, int(round(n * val_split)))
            n_test = max(0, int(round(n * test_split)))
            n_train = n - n_val - n_test
            if n_train < 1 and n >= 3:
                n_train = 1
                if n_val > 0:
                    n_val -= 1
                elif n_test > 0:
                    n_test -= 1

            train_groups.extend(stratum_gids[:n_train])
            val_groups.extend(stratum_gids[n_train : n_train + n_val])
            test_groups.extend(stratum_gids[n_train + n_val :])

    # Ensure no split is empty if total groups >= 3
    all_assigned_groups = [train_groups, val_groups, test_groups]
    for split_idx, split_name in enumerate(["train", "val", "test"]):
        if not all_assigned_groups[split_idx]:
            # Borrow one group from the largest split with > 1 group
            largest_split = max(range(3), key=lambda idx: len(all_assigned_groups[idx]))
            if len(all_assigned_groups[largest_split]) > 1:
                moved_group = all_assigned_groups[largest_split].pop()
                all_assigned_groups[split_idx].append(moved_group)

    train_groups, val_groups, test_groups = all_assigned_groups

    train_items: list[RDDItem] = []
    val_items: list[RDDItem] = []
    test_items: list[RDDItem] = []

    for gid in train_groups:
        for it in groups[gid]:
            it.split = "train"
            train_items.append(it)

    for gid in val_groups:
        for it in groups[gid]:
            it.split = "val"
            val_items.append(it)

    for gid in test_groups:
        for it in groups[gid]:
            it.split = "test"
            test_items.append(it)

    # Empty split validation
    if not train_items:
        raise ValueError("Train split is empty after sequence grouping. Supply more data or adjust split fractions.")
    if not val_items:
        raise ValueError("Validation split is empty after sequence grouping. Supply more data or adjust split fractions.")
    if not test_items:
        raise ValueError("Test split is empty after sequence grouping. Supply more data or adjust split fractions.")

    # Overlap and leakage validation
    train_ids = {it.image_id for it in train_items}
    val_ids = {it.image_id for it in val_items}
    test_ids = {it.image_id for it in test_items}

    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    if overlap_tv:
        raise RuntimeError(f"Data leakage detected! Train and Val share image IDs: {list(overlap_tv)[:5]}")
    if overlap_tt:
        raise RuntimeError(f"Data leakage detected! Train and Test share image IDs: {list(overlap_tt)[:5]}")
    if overlap_vt:
        raise RuntimeError(f"Data leakage detected! Val and Test share image IDs: {list(overlap_vt)[:5]}")

    train_hashes = {it.sha256 for it in train_items}
    val_hashes = {it.sha256 for it in val_items}
    test_hashes = {it.sha256 for it in test_items}

    hash_overlap_tv = train_hashes & val_hashes
    hash_overlap_tt = train_hashes & test_hashes
    hash_overlap_vt = val_hashes & test_hashes
    if hash_overlap_tv:
        raise RuntimeError(f"Data leakage detected! Train and Val share identical content hashes: {list(hash_overlap_tv)[:3]}")
    if hash_overlap_tt:
        raise RuntimeError(f"Data leakage detected! Train and Test share identical content hashes: {list(hash_overlap_tt)[:3]}")
    if hash_overlap_vt:
        raise RuntimeError(f"Data leakage detected! Val and Test share identical content hashes: {list(hash_overlap_vt)[:3]}")

    return train_items, val_items, test_items


def subsample_negatives_in_split(
    items: list[RDDItem],
    target_count: int | None = None,
    neg_ratio: float | None = None,
    both_ratio: float = 0.40,
    d20_ratio: float = 0.40,
    seed: int = 42,
) -> list[RDDItem]:
    """Subsample negative background road images strictly within an already-assigned split.

    CRITICAL RULES:
    1. Images NEVER move across splits.
    2. Positives (has_pothole == True) are 100% preserved.
    3. Prioritizes hard crack negatives: Tier 1 (D20+D10) -> Tier 2 (D20) -> Tier 3 (D10) -> Tier 4.
    4. Unselected negatives remain assigned to this split with selected_in_manifest=False.
    """
    positives = [it for it in items if it.has_pothole]
    candidates = [it for it in items if not it.has_pothole]

    # If no subsampling requested, mark all as selected
    if target_count is None and neg_ratio is None:
        for it in items:
            it.selected_in_manifest = True
        return items

    if target_count is None and neg_ratio is not None:
        if neg_ratio < 0:
            for it in items:
                it.selected_in_manifest = True
            return items
        target_count = int(round(len(positives) * neg_ratio)) if positives else len(candidates)

    target_count = max(0, min(target_count, len(candidates)))

    tier1 = [it for it in candidates if "Tier 1" in it.negative_tier]
    tier2 = [it for it in candidates if "Tier 2" in it.negative_tier]
    tier3 = [it for it in candidates if "Tier 3" in it.negative_tier]
    tier4 = [it for it in candidates if "Tier 4" in it.negative_tier]

    rng = random.Random(seed)
    rng.shuffle(tier1)
    rng.shuffle(tier2)
    rng.shuffle(tier3)
    rng.shuffle(tier4)

    n_both = min(len(tier1), int(round(target_count * both_ratio)))
    n_d20 = min(len(tier2), int(round(target_count * d20_ratio)))
    remaining = target_count - n_both - n_d20
    n_d10 = min(len(tier3), remaining)

    selected_negs: list[RDDItem] = tier1[:n_both] + tier2[:n_d20] + tier3[:n_d10]
    if len(selected_negs) < target_count:
        leftover = tier1[n_both:] + tier2[n_d20:] + tier3[n_d10:] + tier4
        rng.shuffle(leftover)
        needed = target_count - len(selected_negs)
        selected_negs.extend(leftover[:needed])

    selected_ids = {it.image_id for it in selected_negs}

    for it in positives:
        it.selected_in_manifest = True

    for it in candidates:
        it.selected_in_manifest = it.image_id in selected_ids

    return items


def write_yolo_yaml(
    yaml_path: Path,
    output_dir: Path,
    train_list: Path,
    val_list: Path,
    test_list: Path,
    class_names: list[str],
) -> None:
    """Generate YOLO data configuration yaml with portable paths including test manifest."""
    try:
        rel_train = train_list.relative_to(PROJECT_ROOT).as_posix()
        rel_val = val_list.relative_to(PROJECT_ROOT).as_posix()
        rel_test = test_list.relative_to(PROJECT_ROOT).as_posix()
        root_path = "."
    except ValueError:
        rel_train = train_list.resolve().as_posix()
        rel_val = val_list.resolve().as_posix()
        rel_test = test_list.resolve().as_posix()
        root_path = output_dir.resolve().as_posix()

    content = [
        "# YOLO Dataset Configuration - Generated from RDD2022",
        "# Split: 80% train, 10% validation, 10% test (held-out)",
        f"path: {root_path}",
        f"train: {rel_train}",
        f"val: {rel_val}",
        f"test: {rel_test}",
        "",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        content.append(f"  {idx}: {name}")
    content.append("")

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(content), encoding="utf-8")
    print(f"[OK] Wrote YOLO dataset config to: {yaml_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert RDD2022 Pascal VOC annotations to YOLO format with leak-free splitting.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to RDD2022 root directory")
    parser.add_argument("--output", "-o", type=Path, default=PROJECT_ROOT / "data" / "processed" / "rdd2022",
                        help="Output directory for YOLO labels and manifest")
    parser.add_argument("--countries", type=str, default="",
                        help="Comma-separated country list (e.g. 'United_States,Japan') or empty for all")
    parser.add_argument("--pothole-only", action="store_true", default=True,
                        help="Map only D40 (pothole) to class 0 (default: True)")
    parser.add_argument("--all-classes", action="store_true",
                        help="Map all damage classes: D00:0, D10:1, D20:2, D40:3")
    parser.add_argument("--train-split", type=float, default=None,
                        help="Fraction for training (default: 0.80, or 1.0 - val_split - test_split)")
    parser.add_argument("--val-split", type=float, default=0.10, help="Fraction for validation (default: 0.10)")
    parser.add_argument("--test-split", type=float, default=0.10, help="Fraction for held-out test (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split (default: 42)")
    parser.add_argument("--neg-ratio", type=float, default=None,
                        help="Ratio of negative background images to positive images per split (e.g. 0.25 for ~20%% background)")
    parser.add_argument("--train-negatives", type=int, default=None, help="Explicit negative count for train split")
    parser.add_argument("--val-negatives", type=int, default=None, help="Explicit negative count for val split")
    parser.add_argument("--test-negatives", type=int, default=None, help="Explicit negative count for test split")
    parser.add_argument("--keep-all-negatives", action="store_true", help="Keep 100%% of candidate negative road images")
    parser.add_argument("--metadata-csv", type=Path, default=None,
                        help="Optional CSV file mapping image_id to sequence_id/route_id for verified grouping")
    parser.add_argument("--split-metadata-csv", type=Path, default=None,
                        help="Destination for split metadata CSV audit file")
    parser.add_argument("--summary-json", type=Path, default=None,
                        help="Destination for dataset summary JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and count annotations without saving")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images to output directory (default: symlink to avoid disk bloat)")
    parser.add_argument("--config-yaml", type=Path, default=PROJECT_ROOT / "configs" / "detector_rdd_balanced.yaml",
                        help="Path to generated detector YAML config (default: configs/detector_rdd_balanced.yaml)")

    args = parser.parse_args(argv)

    input_dir: Path = args.input.resolve()
    if not input_dir.exists():
        sys.exit(f"ERROR: Input directory does not exist: {input_dir}")

    countries = [c.strip() for c in args.countries.split(",") if c.strip()] if args.countries else None

    # Define class mapping
    if args.all_classes:
        target_classes = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
        class_names = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]
    else:
        target_classes = {"D40": 0}
        class_names = ["pothole"]

    print(f"Scanning RDD2022 at: {input_dir}")
    pairs = discover_pairs(input_dir, countries)
    print(f"Found {len(pairs)} image-annotation pairs.")

    if not pairs:
        sys.exit("No valid pairs found. Please check the dataset directory structure.")

    # Load external metadata if provided
    external_metadata: dict[str, str] = {}
    if args.metadata_csv and args.metadata_csv.exists():
        with args.metadata_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row.get("image_id", "").strip()
                seq_id = row.get("sequence_id", row.get("route_id", row.get("group_id", ""))).strip()
                if img_id and seq_id:
                    external_metadata[img_id] = seq_id

    # First pass: parse XMLs, compute SHA-256, detect duplicates
    raw_items = []
    hash_to_ids: dict[str, list[str]] = {}
    corrupt_count = 0
    total_boxes = 0
    class_box_totals: dict[str, int] = {k: 0 for k in RDD_CLASSES}

    print("Parsing annotations and checking content integrity...")
    for img_path, xml_path, country in pairs:
        parsed = parse_voc_xml(xml_path, target_classes, return_metadata=True)
        if parsed is None:
            corrupt_count += 1
            continue
        img_w, img_h, boxes, class_counts = parsed
        for c_name, count in class_counts.items():
            class_box_totals[c_name] = class_box_totals.get(c_name, 0) + count

        sha256 = compute_file_sha256(img_path)
        item_id = img_path.stem
        hash_to_ids.setdefault(sha256, []).append(item_id)

        has_pothole = class_counts.get("D40", 0) > 0
        has_crack = class_counts.get("D20", 0) > 0 or class_counts.get("D10", 0) > 0
        neg_tier = classify_negative_tier(class_counts) if not has_pothole else ""

        total_boxes += len(boxes)

        it = RDDItem(
            image_id=item_id,
            img_path=img_path,
            xml_path=xml_path,
            country=country,
            img_w=img_w,
            img_h=img_h,
            boxes=boxes,
            class_counts=class_counts,
            sha256=sha256,
            has_pothole=has_pothole,
            has_crack=has_crack,
            negative_tier=neg_tier,
        )
        raw_items.append(it)

    # Assign group IDs
    dup_clusters = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}
    grouping_methods_used: set[str] = set()

    for it in raw_items:
        gid, gmethod = resolve_group_id(
            item_id=it.image_id,
            country=it.country,
            sha256=it.sha256,
            duplicates_map=hash_to_ids,
            external_metadata=external_metadata,
        )
        it.group_id = gid
        it.grouping_method = gmethod
        grouping_methods_used.add(gmethod)

    pos_count = sum(1 for it in raw_items if it.has_pothole)
    neg_count = sum(1 for it in raw_items if not it.has_pothole)

    print("\n--- Summary of Discovered Data ---")
    print(f"Total pairs scanned:         {len(pairs)}")
    print(f"Valid images processed:      {len(raw_items)}")
    print(f"Corrupt/invalid annotations: {corrupt_count}")
    print(f"Images with potholes (D40):  {pos_count}")
    print(f"Valid negative road images:  {neg_count}")
    print(f"Total target bounding boxes: {total_boxes}")
    print(f"Duplicate content clusters:  {len(dup_clusters)} ({sum(len(ids) for ids in dup_clusters.values())} total images)")
    print(f"Grouping methods applied:    {sorted(grouping_methods_used)}")

    if "sample_fallback_no_sequence_metadata" in grouping_methods_used:
        print("  [NOTE] Standard RDD2022 Pascal VOC XMLs do not contain GPS/route sequence logs.")
        print("  Deduplication and sample grouping applied; geographic independence cannot be claimed.")

    if args.dry_run:
        print("\n[Dry-run completed. No files written.]")
        return

    # Split into Train / Val / Test (default 80/10/10)
    val_split = args.val_split
    test_split = args.test_split
    if args.train_split is not None:
        train_split = args.train_split
    else:
        train_split = max(0.0, round(1.0 - val_split - test_split, 4))

    print(f"\nPartitioning data into Train ({train_split*100:.0f}%), Val ({val_split*100:.0f}%), Test ({test_split*100:.0f}%)...")
    train_items, val_items, test_items = stratified_group_split(
        raw_items,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        seed=args.seed,
    )

    # Apply in-split negative subsampling
    train_items = subsample_negatives_in_split(
        train_items,
        target_count=args.train_negatives,
        neg_ratio=None if args.keep_all_negatives else args.neg_ratio,
        seed=args.seed,
    )
    val_items = subsample_negatives_in_split(
        val_items,
        target_count=args.val_negatives,
        neg_ratio=None if args.keep_all_negatives else args.neg_ratio,
        seed=args.seed + 1,
    )
    test_items = subsample_negatives_in_split(
        test_items,
        target_count=args.test_negatives,
        neg_ratio=None if args.keep_all_negatives else args.neg_ratio,
        seed=args.seed + 2,
    )

    all_processed_items = train_items + val_items + test_items

    # Export images and YOLO labels
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def export_split(split_items: list[RDDItem], split_name: str) -> tuple[list[str], list[str]]:
        labels_dir = out_dir / "labels" / split_name
        images_dir = out_dir / "images" / split_name
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        manifest_paths: list[str] = []
        all_neg_paths: list[str] = []

        for it in split_items:
            lbl_file = labels_dir / f"{it.image_id}.txt"
            lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in it.boxes]
            lbl_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            target_img = images_dir / it.img_path.name
            if not target_img.exists() and not target_img.is_symlink():
                if args.copy_images:
                    import shutil
                    shutil.copy2(it.img_path, target_img)
                else:
                    try:
                        rel_src = os.path.relpath(it.img_path, images_dir)
                        os.symlink(rel_src, target_img)
                    except OSError:
                        import shutil
                        shutil.copy2(it.img_path, target_img)

            img_entry = target_img.absolute().as_posix()
            all_neg_paths.append(img_entry)
            if it.selected_in_manifest:
                manifest_paths.append(img_entry)

        return manifest_paths, all_neg_paths

    print("\nWriting YOLO split directories and label files...")
    train_paths, train_all_paths = export_split(train_items, "train")
    val_paths, val_all_paths = export_split(val_items, "val")
    test_paths, test_all_paths = export_split(test_items, "test")

    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    test_txt = out_dir / "test.txt"

    train_txt.write_text("\n".join(train_paths) + ("\n" if train_paths else ""), encoding="utf-8")
    val_txt.write_text("\n".join(val_paths) + ("\n" if val_paths else ""), encoding="utf-8")
    test_txt.write_text("\n".join(test_paths) + ("\n" if test_paths else ""), encoding="utf-8")

    # If negative subsampling filtered images, also write full un-subsampled lists
    if len(train_paths) != len(train_all_paths):
        (out_dir / "train_all_negatives.txt").write_text("\n".join(train_all_paths) + "\n", encoding="utf-8")
    if len(val_paths) != len(val_all_paths):
        (out_dir / "val_all_negatives.txt").write_text("\n".join(val_all_paths) + "\n", encoding="utf-8")
    if len(test_paths) != len(test_all_paths):
        (out_dir / "test_all_negatives.txt").write_text("\n".join(test_all_paths) + "\n", encoding="utf-8")

    # Verify Ultralytics label resolution
    def verify_label_resolution(manifest_path: Path) -> None:
        missing = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            parts = list(p.parts)
            if "images" in parts:
                idx = len(parts) - 1 - parts[::-1].index("images")
                parts[idx] = "labels"
                lbl_file = Path(*parts).with_suffix(".txt")
                if not lbl_file.exists():
                    missing.append(f"{line} -> {lbl_file}")
            else:
                missing.append(f"No 'images' directory in path: {line}")
        if missing:
            raise RuntimeError(
                f"Ultralytics label resolution failed for {len(missing)} images in {manifest_path.name}! "
                f"Sample missing: {missing[:3]}"
            )

    verify_label_resolution(train_txt)
    verify_label_resolution(val_txt)
    verify_label_resolution(test_txt)
    print("✓ Verified Ultralytics img2label_paths: 100% of train, val, and test images resolve to labels.")

    # Write Split Metadata CSV
    meta_csv_path = args.split_metadata_csv or (out_dir / "split_metadata.csv")
    meta_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id",
            "country",
            "group_id",
            "grouping_method",
            "split",
            "selected_in_manifest",
            "has_pothole",
            "d40_count",
            "d20_count",
            "d10_count",
            "d00_count",
            "total_boxes",
            "classes_str",
            "negative_tier",
            "sha256",
            "source_image_path",
            "source_xml_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in all_processed_items:
            writer.writerow({
                "image_id": it.image_id,
                "country": it.country,
                "group_id": it.group_id,
                "grouping_method": it.grouping_method,
                "split": it.split,
                "selected_in_manifest": "1" if it.selected_in_manifest else "0",
                "has_pothole": "1" if it.has_pothole else "0",
                "d40_count": it.class_counts.get("D40", 0),
                "d20_count": it.class_counts.get("D20", 0),
                "d10_count": it.class_counts.get("D10", 0),
                "d00_count": it.class_counts.get("D00", 0),
                "total_boxes": len(it.boxes),
                "classes_str": "+".join(sorted(k for k, v in it.class_counts.items() if v > 0)),
                "negative_tier": it.negative_tier,
                "sha256": it.sha256,
                "source_image_path": it.img_path.as_posix(),
                "source_xml_path": it.xml_path.as_posix(),
            })
    print(f"[OK] Wrote split metadata CSV to: {meta_csv_path}")

    # Write Dataset Summary JSON
    summary_path = args.summary_json or (out_dir / "dataset_summary.json")
    train_sel = [it for it in train_items if it.selected_in_manifest]
    val_sel = [it for it in val_items if it.selected_in_manifest]
    test_sel = [it for it in test_items if it.selected_in_manifest]

    summary_data = {
        "dataset_name": "RDD2022",
        "random_seed": args.seed,
        "total_images_processed": len(all_processed_items),
        "corrupt_annotations_rejected": corrupt_count,
        "images_with_potholes": sum(1 for it in all_processed_items if it.has_pothole),
        "candidate_negatives_total": sum(1 for it in all_processed_items if not it.has_pothole),
        "manifest_counts": {
            "train": len(train_sel),
            "val": len(val_sel),
            "test": len(test_sel),
            "total_manifest": len(train_sel) + len(val_sel) + len(test_sel),
        },
        "d40_bounding_boxes": {
            "train": sum(it.class_counts.get("D40", 0) for it in train_sel),
            "val": sum(it.class_counts.get("D40", 0) for it in val_sel),
            "test": sum(it.class_counts.get("D40", 0) for it in test_sel),
            "total": sum(it.class_counts.get("D40", 0) for it in all_processed_items),
        },
        "country_distribution": {
            c: sum(1 for it in all_processed_items if it.country == c)
            for c in sorted(set(it.country for it in all_processed_items))
        },
        "duplicate_detection": {
            "clusters_count": len(dup_clusters),
            "total_duplicate_instances": sum(len(ids) for ids in dup_clusters.values()),
        },
        "grouping_methods_applied": sorted(grouping_methods_used),
        "class_box_totals": class_box_totals,
        "negative_tiers_selected": {
            tier: sum(1 for it in all_processed_items if it.selected_in_manifest and it.negative_tier == tier)
            for tier in [
                "Tier 1: D20 (alligator) + D10 (transverse)",
                "Tier 2: D20 (alligator crack)",
                "Tier 3: D10 (transverse crack)",
                "Tier 4: Other background",
            ]
        },
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"[OK] Wrote dataset statistics summary to: {summary_path}")

    # Generate detector YAML config
    config_yaml = Path(args.config_yaml).resolve()
    write_yolo_yaml(config_yaml, out_dir, train_txt, val_txt, test_txt, class_names)

    print("\n[SUCCESS] RDD2022 dataset conversion and splitting complete:")
    print(f"  Train: {len(train_paths)} images (list: {train_txt})")
    print(f"  Val:   {len(val_paths)} images (list: {val_txt})")
    print(f"  Test:  {len(test_paths)} images (list: {test_txt}) [HELD-OUT]")
    print(f"  Labels dir:  {out_dir / 'labels'}")
    print(f"  YOLO config: {config_yaml}")


if __name__ == "__main__":
    main()

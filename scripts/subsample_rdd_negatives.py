"""Subsample negative (background) road images from the RDD2022 dataset with split protection.

Specifically targets hard negatives containing alligator cracks (D20) and
transverse cracks (D10) to teach the YOLO detector not to trigger false alarms
on cracked road surfaces, adhering to Ultralytics' recommendation of keeping
background images at roughly 10% to 20% of the dataset.

Preserves split isolation:
  - Train negatives come strictly from train candidate images.
  - Validation negatives come strictly from validation candidate images.
  - Test negatives come strictly from test candidate images.
  - Images NEVER cross split boundaries to satisfy class-balance targets.

Usage:
    # Subsample 25 train negatives (~20% background), 5 val negatives, 5 test negatives
    python scripts/subsample_rdd_negatives.py

    # Custom counts and random seed
    python scripts/subsample_rdd_negatives.py --train-negatives 30 --val-negatives 6 --test-negatives 6 --seed 42

    # Negative-to-positive ratio (e.g. 0.25 negatives per positive)
    python scripts/subsample_rdd_negatives.py --neg-ratio 0.25

    # Dry-run inspection
    python scripts/subsample_rdd_negatives.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PROC_DIR = PROJECT_ROOT / "data" / "processed" / "rdd2022"
DEFAULT_METADATA_CSV = PROJECT_ROOT / "data" / "metadata" / "rdd_subsampled_negatives.csv"
DEFAULT_YAML_PATH = PROJECT_ROOT / "configs" / "detector_rdd_balanced.yaml"


def parse_xml_classes(xml_path: Path) -> list[str]:
    """Extract list of object damage class names from a Pascal VOC XML."""
    if not xml_path.exists():
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        names = []
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is not None and name_elem.text:
                names.append(name_elem.text.strip())
        return names
    except Exception as exc:
        print(f"  [WARN] Failed to parse XML {xml_path.name}: {exc}")
        return []


def find_xml_for_stem(stem: str, search_dirs: list[Path]) -> Path | None:
    """Find XML file corresponding to an image stem across candidate directories."""
    for d in search_dirs:
        candidate = d / f"{stem}.xml"
        if candidate.exists():
            return candidate
        # Try annotations/xmls subdirectory
        sub = d / "annotations" / "xmls" / f"{stem}.xml"
        if sub.exists():
            return sub
        # Try rglob within directory
        found = list(d.rglob(f"{stem}.xml"))
        if found:
            return found[0]
    return None


def subsample_split_negatives(
    stems: list[str],
    image_classes: dict[str, list[str]],
    target_count: int,
    split_name: str,
    seed: int,
    both_ratio: float = 0.40,
    d20_ratio: float = 0.40,
) -> list[dict]:
    """Subsample negative images prioritizing D20 + D10, D20-only, then D10-only."""
    # Filter candidates: strictly NO D40 (pothole)
    tier1_both = []
    tier2_d20 = []
    tier3_d10 = []
    tier4_other = []

    for s in stems:
        classes = image_classes.get(s, [])
        if "D40" in classes:
            continue
        has_d20 = "D20" in classes
        has_d10 = "D10" in classes

        if has_d20 and has_d10:
            tier1_both.append(s)
        elif has_d20:
            tier2_d20.append(s)
        elif has_d10:
            tier3_d10.append(s)
        else:
            tier4_other.append(s)

    rng = random.Random(seed)
    rng.shuffle(tier1_both)
    rng.shuffle(tier2_d20)
    rng.shuffle(tier3_d10)
    rng.shuffle(tier4_other)

    # Compute targets per tier
    n_both = min(len(tier1_both), int(round(target_count * both_ratio)))
    n_d20 = min(len(tier2_d20), int(round(target_count * d20_ratio)))
    remaining = target_count - n_both - n_d20
    n_d10 = min(len(tier3_d10), remaining)

    # If any tier couldn't fulfill its share, backfill from other tiers
    selected_stems = tier1_both[:n_both] + tier2_d20[:n_d20] + tier3_d10[:n_d10]
    if len(selected_stems) < target_count:
        leftover_pool = (
            tier1_both[n_both:] + tier2_d20[n_d20:] + tier3_d10[n_d10:] + tier4_other
        )
        rng.shuffle(leftover_pool)
        needed = target_count - len(selected_stems)
        selected_stems.extend(leftover_pool[:needed])

    results = []
    for s in selected_stems:
        classes = image_classes.get(s, [])
        has_d20 = "D20" in classes
        has_d10 = "D10" in classes
        if has_d20 and has_d10:
            tier = "Tier 1: D20 (alligator) + D10 (transverse)"
        elif has_d20:
            tier = "Tier 2: D20 (alligator crack)"
        elif has_d10:
            tier = "Tier 3: D10 (transverse crack)"
        else:
            tier = "Tier 4: Other road background"

        results.append({
            "image_id": s,
            "split": split_name,
            "tier": tier,
            "d20_count": classes.count("D20"),
            "d10_count": classes.count("D10"),
            "d00_count": classes.count("D00"),
            "total_crack_boxes": len(classes),
            "classes_str": "+".join(sorted(set(classes))),
        })

    return results


def write_balanced_yaml(
    yaml_path: Path,
    train_manifest: Path,
    val_manifest: Path,
    test_manifest: Path | None = None,
) -> None:
    """Generate YOLO data configuration yaml for the balanced dataset."""
    try:
        rel_train = train_manifest.relative_to(PROJECT_ROOT).as_posix()
        rel_val = val_manifest.relative_to(PROJECT_ROOT).as_posix()
        rel_test = test_manifest.relative_to(PROJECT_ROOT).as_posix() if test_manifest and test_manifest.exists() else None
        root_path = "."
    except ValueError:
        rel_train = train_manifest.resolve().as_posix()
        rel_val = val_manifest.resolve().as_posix()
        rel_test = test_manifest.resolve().as_posix() if test_manifest and test_manifest.exists() else None
        root_path = yaml_path.parent.resolve().as_posix()

    content = [
        "# YOLO Dataset Configuration - RDD2022 US (Balanced Background Negatives)",
        "# Subsampled hard negatives targeting alligator cracks (D20) & transverse cracks (D10)",
        f"path: {root_path}",
        f"train: {rel_train}",
        f"val: {rel_val}",
    ]
    if rel_test:
        content.append(f"test: {rel_test}")
    content.extend([
        "",
        "names:",
        "  0: pothole",
        "",
    ])
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(content), encoding="utf-8")
    print(f"  [OK] Wrote YOLO dataset configuration to: {yaml_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Subsample negative crack images from RDD2022 dataset with split protection."
    )
    parser.add_argument(
        "--proc-dir",
        type=Path,
        default=DEFAULT_PROC_DIR,
        help="Path to processed RDD2022 directory containing train.txt, val.txt, and optional test.txt",
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=None,
        help="Optional path to RDD2022 raw XML annotations directory",
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=None,
        help="Ratio of negative images to positive images per split (e.g. 0.25 for ~20%% background)",
    )
    parser.add_argument(
        "--train-negatives",
        type=int,
        default=None,
        help="Number of negative images to sample for train split (default: 25 or derived from --neg-ratio)",
    )
    parser.add_argument(
        "--val-negatives",
        type=int,
        default=None,
        help="Number of negative images to sample for val split (default: 5 or derived from --neg-ratio)",
    )
    parser.add_argument(
        "--test-negatives",
        type=int,
        default=None,
        help="Number of negative images to sample for test split (default: 5 or derived from --neg-ratio)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help="Output CSV path for the audit manifest of sampled negatives",
    )
    parser.add_argument(
        "--yaml-config",
        type=Path,
        default=DEFAULT_YAML_PATH,
        help="Output path for detector YAML config",
    )
    parser.add_argument(
        "--update-primary-manifests",
        action="store_true",
        default=True,
        help="Update train.txt, val.txt, test.txt directly in proc-dir (backing up originals as *_all_negatives.txt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected samples and statistics without writing files",
    )

    args = parser.parse_args(argv)

    proc_dir: Path = args.proc_dir.resolve()
    train_txt = proc_dir / "train.txt"
    val_txt = proc_dir / "val.txt"
    test_txt = proc_dir / "test.txt"

    if not train_txt.exists() or not val_txt.exists():
        sys.exit(f"ERROR: Missing train.txt or val.txt in {proc_dir}")

    has_test = test_txt.exists()

    # Read existing manifests (or full backup manifests if already run before)
    train_backup = proc_dir / "train_all_negatives.txt"
    val_backup = proc_dir / "val_all_negatives.txt"
    test_backup = proc_dir / "test_all_negatives.txt"

    source_train_txt = train_backup if train_backup.exists() else train_txt
    source_val_txt = val_backup if val_backup.exists() else val_txt
    source_test_txt = (test_backup if test_backup.exists() else test_txt) if has_test else None

    train_lines = [l.strip() for l in source_train_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    val_lines = [l.strip() for l in source_val_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    test_lines = [l.strip() for l in source_test_txt.read_text(encoding="utf-8").splitlines() if l.strip()] if source_test_txt else []

    train_stem_to_line = {Path(l).stem: l for l in train_lines}
    val_stem_to_line = {Path(l).stem: l for l in val_lines}
    test_stem_to_line = {Path(l).stem: l for l in test_lines}

    all_stems = list(train_stem_to_line.keys()) + list(val_stem_to_line.keys()) + list(test_stem_to_line.keys())
    print(f"Scanning annotations for {len(all_stems)} total images across splits...")

    # Candidate XML search directories
    xml_search_dirs = []
    if args.xml_dir:
        xml_search_dirs.append(args.xml_dir.resolve())
    # Standard locations
    xml_search_dirs.extend([
        PROJECT_ROOT / "data" / "raw" / "RDD2022" / "United_States" / "United_States" / "train" / "annotations" / "xmls",
        PROJECT_ROOT / "data" / "raw" / "RDD2022" / "United_States" / "train" / "annotations" / "xmls",
        PROJECT_ROOT / "data" / "raw" / "RDD2022",
        PROJECT_ROOT / "data" / "raw" / "rdd2022",
    ])

    # Check if split_metadata.csv exists in proc_dir
    metadata_csv_path = proc_dir / "split_metadata.csv"
    image_classes: dict[str, list[str]] = {}
    if metadata_csv_path.exists():
        with metadata_csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stem = row.get("image_id", "")
                classes_str = row.get("classes_str", "")
                cls_list = classes_str.split("+") if classes_str else []
                if stem:
                    image_classes[stem] = cls_list

    # Fill any missing from XML files directly
    missing_stems = [s for s in all_stems if s not in image_classes]
    if missing_stems:
        for s in missing_stems:
            xml_p = find_xml_for_stem(s, xml_search_dirs)
            if xml_p:
                image_classes[s] = parse_xml_classes(xml_p)
            else:
                image_classes[s] = []

    # Positives (D40)
    train_pos = [s for s in train_stem_to_line if "D40" in image_classes.get(s, [])]
    val_pos = [s for s in val_stem_to_line if "D40" in image_classes.get(s, [])]
    test_pos = [s for s in test_stem_to_line if "D40" in image_classes.get(s, [])]

    print(f"\nPositive pothole images found:")
    print(f"  Train positives: {len(train_pos)}")
    print(f"  Val positives:   {len(val_pos)}")
    if has_test:
        print(f"  Test positives:  {len(test_pos)}")

    # Target negative counts
    if args.neg_ratio is not None:
        target_train_neg = int(round(len(train_pos) * args.neg_ratio))
        target_val_neg = int(round(len(val_pos) * args.neg_ratio))
        target_test_neg = int(round(len(test_pos) * args.neg_ratio))
    else:
        target_train_neg = args.train_negatives if args.train_negatives is not None else 25
        target_val_neg = args.val_negatives if args.val_negatives is not None else 5
        target_test_neg = args.test_negatives if args.test_negatives is not None else 5

    # Subsample negatives independently within each split
    sampled_train_neg = subsample_split_negatives(
        list(train_stem_to_line.keys()),
        image_classes,
        target_count=target_train_neg,
        split_name="train",
        seed=args.seed,
    )

    sampled_val_neg = subsample_split_negatives(
        list(val_stem_to_line.keys()),
        image_classes,
        target_count=target_val_neg,
        split_name="val",
        seed=args.seed + 1,
    )

    sampled_test_neg = []
    if has_test:
        sampled_test_neg = subsample_split_negatives(
            list(test_stem_to_line.keys()),
            image_classes,
            target_count=target_test_neg,
            split_name="test",
            seed=args.seed + 2,
        )

    total_sampled_neg = sampled_train_neg + sampled_val_neg + sampled_test_neg

    print(f"\nSubsampled Negatives Summary:")
    print(f"  Train negatives sampled: {len(sampled_train_neg)} (target: {target_train_neg})")
    print(f"  Val negatives sampled:   {len(sampled_val_neg)} (target: {target_val_neg})")
    if has_test:
        print(f"  Test negatives sampled:  {len(sampled_test_neg)} (target: {target_test_neg})")
    print(f"  Total negatives sampled: {len(total_sampled_neg)}")

    # Calculate final dataset balance
    final_train_count = len(train_pos) + len(sampled_train_neg)
    train_neg_pct = (len(sampled_train_neg) / final_train_count) * 100 if final_train_count else 0

    final_val_count = len(val_pos) + len(sampled_val_neg)
    val_neg_pct = (len(sampled_val_neg) / final_val_count) * 100 if final_val_count else 0

    print(f"\nFinal Dataset Balance:")
    print(f"  Train set: {final_train_count} images (Positives: {len(train_pos)}, Negatives: {len(sampled_train_neg)} -> {train_neg_pct:.1f}% background)")
    print(f"  Val set:   {final_val_count} images (Positives: {len(val_pos)}, Negatives: {len(sampled_val_neg)} -> {val_neg_pct:.1f}% background)")
    if has_test:
        final_test_count = len(test_pos) + len(sampled_test_neg)
        test_neg_pct = (len(sampled_test_neg) / final_test_count) * 100 if final_test_count else 0
        print(f"  Test set:  {final_test_count} images (Positives: {len(test_pos)}, Negatives: {len(sampled_test_neg)} -> {test_neg_pct:.1f}% background)")

    if args.dry_run:
        print("\n[Dry-run completed. No files modified.]")
        return

    # Write audit CSV manifest
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_id",
            "split",
            "tier",
            "d20_count",
            "d10_count",
            "d00_count",
            "total_crack_boxes",
            "classes_str",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(total_sampled_neg)
    print(f"\n  [OK] Wrote audit manifest to: {args.manifest_csv}")

    # Build new manifest file lists
    new_train_lines = [train_stem_to_line[s] for s in train_pos] + [
        train_stem_to_line[item["image_id"]] for item in sampled_train_neg
    ]
    new_val_lines = [val_stem_to_line[s] for s in val_pos] + [
        val_stem_to_line[item["image_id"]] for item in sampled_val_neg
    ]
    new_test_lines = []
    if has_test:
        new_test_lines = [test_stem_to_line[s] for s in test_pos] + [
            test_stem_to_line[item["image_id"]] for item in sampled_test_neg
        ]

    # Save explicit balanced manifests
    train_balanced_txt = proc_dir / "train_balanced.txt"
    val_balanced_txt = proc_dir / "val_balanced.txt"
    train_balanced_txt.write_text("\n".join(new_train_lines) + "\n", encoding="utf-8")
    val_balanced_txt.write_text("\n".join(new_val_lines) + "\n", encoding="utf-8")
    print(f"  [OK] Wrote balanced train list to: {train_balanced_txt}")
    print(f"  [OK] Wrote balanced val list to:   {val_balanced_txt}")

    if has_test:
        test_balanced_txt = proc_dir / "test_balanced.txt"
        test_balanced_txt.write_text("\n".join(new_test_lines) + "\n", encoding="utf-8")
        print(f"  [OK] Wrote balanced test list to:  {test_balanced_txt}")

    # Backup originals and update train.txt / val.txt / test.txt
    if args.update_primary_manifests:
        if not train_backup.exists():
            shutil.copy2(train_txt, train_backup)
            print(f"  [OK] Backed up original full train list to: {train_backup}")
        if not val_backup.exists():
            shutil.copy2(val_txt, val_backup)
            print(f"  [OK] Backed up original full val list to:   {val_backup}")
        if has_test and not test_backup.exists():
            shutil.copy2(test_txt, test_backup)
            print(f"  [OK] Backed up original full test list to:  {test_backup}")

        train_txt.write_text("\n".join(new_train_lines) + "\n", encoding="utf-8")
        val_txt.write_text("\n".join(new_val_lines) + "\n", encoding="utf-8")
        print(f"  [OK] Updated primary {train_txt.name} ({len(new_train_lines)} images)")
        print(f"  [OK] Updated primary {val_txt.name} ({len(new_val_lines)} images)")
        if has_test:
            test_txt.write_text("\n".join(new_test_lines) + "\n", encoding="utf-8")
            print(f"  [OK] Updated primary {test_txt.name} ({len(new_test_lines)} images)")

    # Write detector YAML
    write_balanced_yaml(args.yaml_config, train_txt, val_txt, test_txt if has_test else None)

    print("\n[SUCCESS] Negatives subsampled and dataset configuration ready!")


if __name__ == "__main__":
    main()

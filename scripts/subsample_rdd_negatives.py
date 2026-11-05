"""Subsample negative (background) road images from the RDD2022 US dataset.

Specifically targets hard negatives containing alligator cracks (D20) and
transverse cracks (D10) to teach the YOLO detector not to trigger false alarms
on cracked road surfaces, adhering to Ultralytics' recommendation of keeping
background images at roughly 10% to 20% of the dataset.

Preserves split isolation: train negatives come strictly from train sequences;
validation negatives come strictly from validation sequences.

Usage:
    # Subsample 25 train negatives (~20% background) and 5 val negatives
    python scripts/subsample_rdd_negatives.py

    # Custom counts and random seed
    python scripts/subsample_rdd_negatives.py --train-negatives 30 --val-negatives 6 --seed 42

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
DEFAULT_XML_DIR = (
    PROJECT_ROOT / "data" / "raw" / "RDD2022" / "United_States" / "United_States" / "train" / "annotations" / "xmls"
)
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

    rng = random.Random(seed)
    rng.shuffle(tier1_both)
    rng.shuffle(tier2_d20)
    rng.shuffle(tier3_d10)

    # Compute targets per tier
    n_both = min(len(tier1_both), int(target_count * both_ratio))
    n_d20 = min(len(tier2_d20), int(target_count * d20_ratio))
    remaining = target_count - n_both - n_d20
    n_d10 = min(len(tier3_d10), remaining)

    # If any tier couldn't fulfill its share, backfill from other tiers
    selected_stems = tier1_both[:n_both] + tier2_d20[:n_d20] + tier3_d10[:n_d10]
    if len(selected_stems) < target_count:
        leftover_pool = (
            tier1_both[n_both:] + tier2_d20[n_d20:] + tier3_d10[n_d10:]
        )
        rng.shuffle(leftover_pool)
        needed = target_count - len(selected_stems)
        selected_stems.extend(leftover_pool[:needed])

    results = []
    for s in selected_stems:
        classes = image_classes[s]
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
) -> None:
    """Generate YOLO data configuration yaml for the balanced dataset."""
    try:
        rel_train = train_manifest.relative_to(PROJECT_ROOT).as_posix()
        rel_val = val_manifest.relative_to(PROJECT_ROOT).as_posix()
        root_path = "."
    except ValueError:
        rel_train = train_manifest.resolve().as_posix()
        rel_val = val_manifest.resolve().as_posix()
        root_path = yaml_path.parent.resolve().as_posix()

    content = [
        "# YOLO Dataset Configuration - RDD2022 US (Balanced 10-20% Background Negatives)",
        "# Subsampled hard negatives targeting alligator cracks (D20) & transverse cracks (D10)",
        f"path: {root_path}",
        f"train: {rel_train}",
        f"val: {rel_val}",
        "",
        "names:",
        "  0: pothole",
        "",
    ]
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(content), encoding="utf-8")
    print(f"  [OK] Wrote YOLO dataset configuration to: {yaml_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Subsample negative crack images from RDD2022 US dataset to achieve 10-20% background balance."
    )
    parser.add_argument(
        "--proc-dir",
        type=Path,
        default=DEFAULT_PROC_DIR,
        help="Path to processed RDD2022 directory containing train.txt and val.txt",
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=DEFAULT_XML_DIR,
        help="Path to RDD2022 raw XML annotations directory",
    )
    parser.add_argument(
        "--train-negatives",
        type=int,
        default=25,
        help="Number of negative images to sample for train split (default: 25, ~20% of train set)",
    )
    parser.add_argument(
        "--val-negatives",
        type=int,
        default=5,
        help="Number of negative images to sample for val split (default: 5, ~22% of val set)",
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
        help="Update train.txt and val.txt directly in proc-dir (backing up originals as *_all_negatives.txt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected samples and statistics without writing files",
    )

    args = parser.parse_args(argv)

    proc_dir: Path = args.proc_dir.resolve()
    xml_dir: Path = args.xml_dir.resolve()

    train_txt = proc_dir / "train.txt"
    val_txt = proc_dir / "val.txt"

    if not train_txt.exists() or not val_txt.exists():
        sys.exit(f"ERROR: Missing train.txt or val.txt in {proc_dir}")

    # Read existing manifests (or full backup manifests if already run before)
    train_backup = proc_dir / "train_all_negatives.txt"
    val_backup = proc_dir / "val_all_negatives.txt"

    source_train_txt = train_backup if train_backup.exists() else train_txt
    source_val_txt = val_backup if val_backup.exists() else val_txt

    train_lines = [l.strip() for l in source_train_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    val_lines = [l.strip() for l in source_val_txt.read_text(encoding="utf-8").splitlines() if l.strip()]

    train_stem_to_line = {Path(l).stem: l for l in train_lines}
    val_stem_to_line = {Path(l).stem: l for l in val_lines}

    all_stems = list(train_stem_to_line.keys()) + list(val_stem_to_line.keys())
    print(f"Scanning XML annotations for {len(all_stems)} total images...")

    image_classes = {}
    for s in all_stems:
        xml_p = xml_dir / f"{s}.xml"
        image_classes[s] = parse_xml_classes(xml_p)

    # Positives (D40)
    train_pos = [s for s in train_stem_to_line if "D40" in image_classes.get(s, [])]
    val_pos = [s for s in val_stem_to_line if "D40" in image_classes.get(s, [])]

    print(f"\nPositive pothole images found:")
    print(f"  Train positives: {len(train_pos)}")
    print(f"  Val positives:   {len(val_pos)}")

    # Subsample negatives
    sampled_train_neg = subsample_split_negatives(
        list(train_stem_to_line.keys()),
        image_classes,
        target_count=args.train_negatives,
        split_name="train",
        seed=args.seed,
    )

    sampled_val_neg = subsample_split_negatives(
        list(val_stem_to_line.keys()),
        image_classes,
        target_count=args.val_negatives,
        split_name="val",
        seed=args.seed + 1,  # independent seed for val
    )

    total_sampled_neg = sampled_train_neg + sampled_val_neg

    print(f"\nSubsampled Negatives Summary:")
    print(f"  Train negatives sampled: {len(sampled_train_neg)} (target: {args.train_negatives})")
    print(f"  Val negatives sampled:   {len(sampled_val_neg)} (target: {args.val_negatives})")
    print(f"  Total negatives sampled: {len(total_sampled_neg)}")

    # Calculate final dataset balance
    final_train_count = len(train_pos) + len(sampled_train_neg)
    train_neg_pct = (len(sampled_train_neg) / final_train_count) * 100 if final_train_count else 0

    final_val_count = len(val_pos) + len(sampled_val_neg)
    val_neg_pct = (len(sampled_val_neg) / final_val_count) * 100 if final_val_count else 0

    print(f"\nFinal Dataset Balance:")
    print(f"  Train set: {final_train_count} images (Positives: {len(train_pos)}, Negatives: {len(sampled_train_neg)} -> {train_neg_pct:.1f}% background)")
    print(f"  Val set:   {final_val_count} images (Positives: {len(val_pos)}, Negatives: {len(sampled_val_neg)} -> {val_neg_pct:.1f}% background)")

    print(f"\nSubsampled Negatives Breakdown by Tier:")
    for tier_name in [
        "Tier 1: D20 (alligator) + D10 (transverse)",
        "Tier 2: D20 (alligator crack)",
        "Tier 3: D10 (transverse crack)",
    ]:
        t_count = sum(1 for item in total_sampled_neg if item["tier"] == tier_name)
        print(f"  {tier_name}: {t_count} images")

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

    # Save explicit balanced manifests
    train_balanced_txt = proc_dir / "train_balanced.txt"
    val_balanced_txt = proc_dir / "val_balanced.txt"

    train_balanced_txt.write_text("\n".join(new_train_lines) + "\n", encoding="utf-8")
    val_balanced_txt.write_text("\n".join(new_val_lines) + "\n", encoding="utf-8")
    print(f"  [OK] Wrote balanced train list to: {train_balanced_txt}")
    print(f"  [OK] Wrote balanced val list to:   {val_balanced_txt}")

    # Backup originals and update train.txt / val.txt
    if args.update_primary_manifests:
        if not train_backup.exists():
            shutil.copy2(train_txt, train_backup)
            print(f"  [OK] Backed up original full train list to: {train_backup}")
        if not val_backup.exists():
            shutil.copy2(val_txt, val_backup)
            print(f"  [OK] Backed up original full val list to:   {val_backup}")

        train_txt.write_text("\n".join(new_train_lines) + "\n", encoding="utf-8")
        val_txt.write_text("\n".join(new_val_lines) + "\n", encoding="utf-8")
        print(f"  [OK] Updated primary {train_txt.name} ({len(new_train_lines)} images)")
        print(f"  [OK] Updated primary {val_txt.name} ({len(new_val_lines)} images)")

    # Write detector YAML
    write_balanced_yaml(args.yaml_config, train_txt, val_txt)

    print("\n[SUCCESS] Negatives subsampled and dataset configuration ready!")


if __name__ == "__main__":
    main()


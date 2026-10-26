"""Convert RDD2022 (Road Damage Dataset) annotations to YOLO format.

Supports the multi-country RDD2022 structure:
    RDD2022/
    ├── Country_Name/ (e.g. United_States, Japan, India, Czech, Norway)
    │   └── train/
    │       ├── images/
    │       └── annotations/xmls/
    ...

Also supports flat VOC structures where images and XMLs are in the same or paired folders.

By default, extracts potholes (RDD2022 class 'D40') as YOLO class 0 ('pothole'),
matching the local Green Bay annotation schema.

Usage:
    # Basic conversion (pothole-only, 80/20 train/val split)
    python scripts/convert_rdd2022.py --input /path/to/RDD2022 --output data/processed/rdd2022

    # Include specific countries (e.g. United_States only or US + Japan)
    python scripts/convert_rdd2022.py --input /path/to/RDD2022 --countries United_States,Japan

    # Dry run to inspect counts without writing files
    python scripts/convert_rdd2022.py --input /path/to/RDD2022 --dry-run
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

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


def parse_voc_xml(xml_path: Path, target_classes: dict[str, int]) -> tuple[int, int, list[tuple[int, float, float, float, float]]]:
    """Parse a Pascal VOC XML file and return (width, height, list of [class_id, cx, cy, w, h])."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as exc:
        print(f"  [WARN] Failed to parse {xml_path.name}: {exc}")
        return 0, 0, []

    size_elem = root.find("size")
    if size_elem is None:
        return 0, 0, []

    w_val = size_elem.find("width")
    h_val = size_elem.find("height")
    if w_val is None or h_val is None:
        return 0, 0, []

    try:
        img_w = int(w_val.text.strip())
        img_h = int(h_val.text.strip())
    except (ValueError, AttributeError):
        return 0, 0, []

    if img_w <= 0 or img_h <= 0:
        return 0, 0, []

    boxes = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue
        cls_name = name_elem.text.strip()
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
        xmin = max(0.0, min(xmin, img_w))
        xmax = max(0.0, min(xmax, img_w))
        ymin = max(0.0, min(ymin, img_h))
        ymax = max(0.0, min(ymax, img_h))

        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 1.0 or bh <= 1.0:
            continue

        cx = (xmin + bw / 2.0) / img_w
        cy = (ymin + bh / 2.0) / img_h
        nw = bw / img_w
        nh = bh / img_h

        boxes.append((cls_id, cx, cy, nw, nh))

    return img_w, img_h, boxes


def discover_pairs(input_dir: Path, countries: list[str] | None = None) -> list[tuple[Path, Path, str]]:
    """Locate all (image_path, xml_path, country) pairs in the RDD2022 dataset."""
    pairs: list[tuple[Path, Path, str]] = []

    # Check for country subdirectories
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    country_dirs = [d for d in subdirs if not countries or d.name in countries or d.name.lower() in [c.lower() for c in countries]]

    search_dirs = country_dirs if country_dirs else [input_dir]

    for cdir in search_dirs:
        country_name = cdir.name if cdir != input_dir else "general"

        # Search for XML folders
        xml_candidates = list(cdir.rglob("annotations/xmls/*.xml"))
        if not xml_candidates:
            xml_candidates = list(cdir.rglob("*.xml"))

        for xml_file in xml_candidates:
            stem = xml_file.stem
            # Find corresponding image
            parent = xml_file.parent
            # Typical locations: ../images/{stem}.jpg, or same dir, or ../../images/
            img_candidates = [
                parent.parent / "images" / f"{stem}.jpg",
                parent.parent / "images" / f"{stem}.png",
                parent / f"{stem}.jpg",
                parent / f"{stem}.png",
            ]
            img_file = next((p for p in img_candidates if p.exists()), None)
            if not img_file:
                # Broader search in cdir
                found = list(cdir.rglob(f"{stem}.jpg")) + list(cdir.rglob(f"{stem}.png"))
                if found:
                    img_file = found[0]

            if img_file and img_file.exists():
                pairs.append((img_file, xml_file, country_name))

    return pairs


def write_yolo_yaml(yaml_path: Path, output_dir: Path, train_list: Path, val_list: Path, class_names: list[str]) -> None:
    """Generate YOLO data configuration yaml."""
    content = [
        "# YOLO Dataset Configuration - Generated from RDD2022",
        f"path: {output_dir.resolve().as_posix()}",
        f"train: {train_list.resolve().as_posix()}",
        f"val: {val_list.resolve().as_posix()}",
        "",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        content.append(f"  {idx}: {name}")
    content.append("")

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(content), encoding="utf-8")
    print(f"[OK] Wrote YOLO dataset config to: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RDD2022 Pascal VOC annotations to YOLO format.")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to RDD2022 root directory")
    parser.add_argument("--output", "-o", type=Path, default=PROJECT_ROOT / "data" / "processed" / "rdd2022",
                        help="Output directory for YOLO labels and manifest")
    parser.add_argument("--countries", type=str, default="",
                        help="Comma-separated country list (e.g. 'United_States,Japan') or empty for all")
    parser.add_argument("--pothole-only", action="store_true", default=True,
                        help="Map only D40 (pothole) to class 0 (default: True)")
    parser.add_argument("--all-classes", action="store_true",
                        help="Map all damage classes: D00:0, D10:1, D20:2, D40:3")
    parser.add_argument("--val-split", type=float, default=0.20, help="Fraction of data reserved for validation (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and count annotations without saving")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images to output directory (default: write image paths without copying)")

    args = parser.parse_args()

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

    # Process and collect stats
    total_boxes = 0
    images_with_targets = 0
    processed_items = []

    for img_path, xml_path, country in pairs:
        w, h, boxes = parse_voc_xml(xml_path, target_classes)
        if boxes:
            images_with_targets += 1
            total_boxes += len(boxes)
        processed_items.append((img_path, xml_path, country, boxes))

    print("\n--- Summary of Discovered Data ---")
    print(f"Total images checked:        {len(processed_items)}")
    print(f"Images with target potholes: {images_with_targets}")
    print(f"Total target bounding boxes: {total_boxes}")

    if args.dry_run:
        print("\n[Dry-run completed. No files written.]")
        return

    # Train / Val Split
    random.seed(args.seed)
    random.shuffle(processed_items)

    val_count = int(len(processed_items) * args.val_split)
    val_items = processed_items[:val_count]
    train_items = processed_items[val_count:]

    out_dir = args.output.resolve()
    labels_train_dir = out_dir / "labels" / "train"
    labels_val_dir = out_dir / "labels" / "val"
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)

    train_img_paths: list[str] = []
    val_img_paths: list[str] = []

    def export_split(items, labels_dir, img_path_list):
        for img_path, xml_path, country, boxes in items:
            lbl_file = labels_dir / f"{img_path.stem}.txt"
            lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
            lbl_file.write_text("\n".join(lines), encoding="utf-8")
            img_path_list.append(img_path.resolve().as_posix())

    print("\nWriting YOLO label files...")
    export_split(train_items, labels_train_dir, train_img_paths)
    export_split(val_items, labels_val_dir, val_img_paths)

    train_txt = out_dir / "train.txt"
    val_txt = out_dir / "val.txt"
    train_txt.write_text("\n".join(train_img_paths), encoding="utf-8")
    val_txt.write_text("\n".join(val_img_paths), encoding="utf-8")

    # Generate configs/detector.yaml
    config_yaml = PROJECT_ROOT / "configs" / "detector.yaml"
    write_yolo_yaml(config_yaml, out_dir, train_txt, val_txt, class_names)

    print(f"\n[SUCCESS] Conversion complete:")
    print(f"  Train images: {len(train_img_paths)} (list: {train_txt})")
    print(f"  Val images:   {len(val_img_paths)} (list: {val_txt})")
    print(f"  Labels dir:   {out_dir / 'labels'}")
    print(f"  YOLO config:  {config_yaml}")
    print(f"\nTo train in the lab with GPU, run:")
    print(f"  yolo train model=yolov8n.pt data=configs/detector.yaml epochs=50 imgsz=640 batch=32 device=0")


if __name__ == "__main__":
    main()

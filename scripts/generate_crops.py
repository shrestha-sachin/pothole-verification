"""Generate verifier crops from human-reviewed annotation boxes.

Reads data/annotations/<source>/<image_id>.json files and the image
inventory.  For each reviewed (non-uncertain) image with saved boxes,
crops are cut from the original image with configurable surrounding
context and saved under data/processed/verifier_crops/.

Positive crops (pothole boxes) → label = pothole
Negative regions (shadow / repaired_patch / manhole / ordinary_road)
  → label = non_pothole, subtype stored in metadata.

Rules enforced:
  • Only reviewed, non-uncertain images are processed.
  • Crops inherit their source image's split assignment.
  • Crop resolution is preserved (no resizing — resize during model preprocessing).
  • Filenames are stable: <image_id>_<label>_<box_index>_c<context_px>.jpg
  • Coordinates are recorded as pixel values in the source image.
  • Context pixels are added around each box and clamped to image boundaries.
  • An existing crop is skipped (not overwritten) unless --overwrite is passed.
  • The crop inventory CSV is written to data/metadata/crop_inventory.csv.

NOTE: These crops are for inspection and prototype development.
Final pipeline evaluation MUST use detector-generated candidate crops,
with separate held-out ground truth to measure missed potholes AND
false positives independently.

Usage:
    python scripts/generate_crops.py [--context 20] [--overwrite]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: Pillow is required.  Run: .venv/bin/pip install Pillow")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_ROOT = PROJECT_ROOT / "data" / "annotations"
INVENTORY_PATH = PROJECT_ROOT / "data" / "metadata" / "image_inventory.csv"
CROPS_ROOT = PROJECT_ROOT / "data" / "processed" / "verifier_crops"
CROP_INVENTORY_PATH = PROJECT_ROOT / "data" / "metadata" / "crop_inventory.csv"

CROP_FIELDS = [
    "crop_id",
    "source_image_id",
    "label",          # pothole | non_pothole
    "subtype",        # shadow | repaired_patch | manhole | ordinary_road | (empty for pothole)
    "split",          # inherited from source image
    "box_x",
    "box_y",
    "box_w",
    "box_h",
    "context_px",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "crop_width",
    "crop_height",
    "crop_filepath",
    "annotation_source",  # human_reviewed
]

NEGATIVE_SUBTYPES = {"shadow", "repaired_patch", "manhole", "ordinary_road"}


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_inventory() -> dict[str, dict]:
    if not INVENTORY_PATH.exists():
        sys.exit(
            f"ERROR: Image inventory not found at {rel(INVENTORY_PATH)}\n"
            "  Run: python scripts/build_inventory.py"
        )
    with INVENTORY_PATH.open(newline="", encoding="utf-8") as fh:
        return {r["image_id"]: r for r in csv.DictReader(fh) if r.get("image_id")}


def load_annotation(image_id: str) -> dict | None:
    for ann_file in ANNOTATIONS_ROOT.rglob(f"{image_id}.json"):
        with ann_file.open(encoding="utf-8") as fh:
            return json.load(fh)
    return None


def crop_and_save(
    img: Image.Image,
    box: dict,
    context_px: int,
    out_path: Path,
    overwrite: bool,
) -> tuple[int, int, int, int, int, int] | None:
    """Cut a crop with context from img.  Returns (x1,y1,x2,y2,w,h) or None on error."""
    try:
        bx = int(box["x"])
        by = int(box["y"])
        bw = int(box["w"])
        bh = int(box["h"])
    except (KeyError, ValueError) as exc:
        print(f"    SKIP invalid box: {box} — {exc}")
        return None

    if bw <= 0 or bh <= 0:
        print(f"    SKIP zero-size box: {box}")
        return None

    img_w, img_h = img.size
    x1 = max(0, bx - context_px)
    y1 = max(0, by - context_px)
    x2 = min(img_w, bx + bw + context_px)
    y2 = min(img_h, by + bh + context_px)

    if x2 <= x1 or y2 <= y1:
        print(f"    SKIP degenerate crop region: ({x1},{y1},{x2},{y2})")
        return None

    if out_path.exists() and not overwrite:
        return (x1, y1, x2, y2, x2 - x1, y2 - y1)

    crop = img.crop((x1, y1, x2, y2))
    temp = out_path.with_suffix(".jpg.part")
    crop.save(str(temp), "JPEG", quality=92)
    temp.replace(out_path)
    return (x1, y1, x2, y2, x2 - x1, y2 - y1)


def generate_crops(context_px: int, overwrite: bool) -> None:
    inventory = load_inventory()
    CROPS_ROOT.mkdir(parents=True, exist_ok=True)

    all_crop_rows: list[dict] = []
    total_images = reviewed = skipped = crops_written = crops_skipped = 0

    for image_id, inv_row in inventory.items():
        total_images += 1
        status = inv_row.get("review_status", "")
        uncertain = inv_row.get("uncertain", "").strip().lower() in ("1", "true", "yes")

        if status != "reviewed" or uncertain:
            skipped += 1
            continue

        ann = load_annotation(image_id)
        if ann is None:
            skipped += 1
            continue

        reviewed += 1
        img_path = PROJECT_ROOT / inv_row["filepath"]
        if not img_path.exists():
            print(f"  SKIP {image_id}: image file not found at {inv_row['filepath']}")
            skipped += 1
            reviewed -= 1
            continue

        split = inv_row.get("split", "")

        try:
            img = Image.open(img_path)
        except Exception as exc:
            print(f"  SKIP {image_id}: cannot open image — {exc}")
            skipped += 1
            reviewed -= 1
            continue

        img_w, img_h = img.size

        # ---- Positive crops (pothole) ------------------------------------
        for idx, box in enumerate(ann.get("pothole_boxes", [])):
            crop_id = f"{image_id}_pothole_{idx:03d}_c{context_px}"
            out_path = CROPS_ROOT / f"{crop_id}.jpg"
            coords = crop_and_save(img, box, context_px, out_path, overwrite)
            if coords is None:
                continue
            x1, y1, x2, y2, cw, ch = coords
            if out_path.exists():
                crops_written += 1
            all_crop_rows.append({
                "crop_id": crop_id,
                "source_image_id": image_id,
                "label": "pothole",
                "subtype": "",
                "split": split,
                "box_x": box.get("x", ""),
                "box_y": box.get("y", ""),
                "box_w": box.get("w", ""),
                "box_h": box.get("h", ""),
                "context_px": context_px,
                "crop_x1": x1, "crop_y1": y1, "crop_x2": x2, "crop_y2": y2,
                "crop_width": cw, "crop_height": ch,
                "crop_filepath": rel(out_path),
                "annotation_source": "human_reviewed",
            })

        # ---- Negative regions -------------------------------------------
        for neg in ann.get("negative_regions", []):
            subtype = neg.get("subtype", "").strip()
            if subtype not in NEGATIVE_SUBTYPES:
                print(f"  WARN {image_id}: unknown negative subtype '{subtype}'; keeping as-is")
            box = {k: neg[k] for k in ("x", "y", "w", "h") if k in neg}
            idx = ann.get("negative_regions", []).index(neg)
            crop_id = f"{image_id}_nonpothole_{subtype}_{idx:03d}_c{context_px}"
            out_path = CROPS_ROOT / f"{crop_id}.jpg"
            coords = crop_and_save(img, box, context_px, out_path, overwrite)
            if coords is None:
                continue
            x1, y1, x2, y2, cw, ch = coords
            if out_path.exists():
                crops_written += 1
            all_crop_rows.append({
                "crop_id": crop_id,
                "source_image_id": image_id,
                "label": "non_pothole",
                "subtype": subtype,
                "split": split,
                "box_x": box.get("x", ""),
                "box_y": box.get("y", ""),
                "box_w": box.get("w", ""),
                "box_h": box.get("h", ""),
                "context_px": context_px,
                "crop_x1": x1, "crop_y1": y1, "crop_x2": x2, "crop_y2": y2,
                "crop_width": cw, "crop_height": ch,
                "crop_filepath": rel(out_path),
                "annotation_source": "human_reviewed",
            })

    # ---- Save crop inventory -------------------------------------------
    CROP_INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CROP_INVENTORY_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CROP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_crop_rows)

    print(f"\nCrop generation summary:")
    print(f"  Total inventory images:    {total_images}")
    print(f"  Processed (reviewed):      {reviewed}")
    print(f"  Skipped (unreviewd/uncert):{skipped}")
    print(f"  Crops generated:           {crops_written}")
    print(f"  Crop inventory:            {rel(CROP_INVENTORY_PATH)}")
    print(f"  Crops directory:           {rel(CROPS_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--context",
        type=int,
        default=20,
        metavar="PX",
        help="Pixels of surrounding road context added around each box (default: 20).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing crop files.  By default, existing crops are kept.",
    )
    args = parser.parse_args()
    if args.context < 0:
        parser.error("--context must be >= 0")
    generate_crops(context_px=args.context, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

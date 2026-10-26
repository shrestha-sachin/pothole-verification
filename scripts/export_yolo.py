"""Export YOLO detection labels from human-reviewed annotations.

Reads data/annotations/<source>/<image_id>.json and
data/metadata/image_inventory.csv.

Rules:
  • Only images with review_status == 'reviewed' are exported.
  • Uncertain images are skipped (never become negative examples).
  • Reviewed images with no boxes → empty .txt file (true negative).
  • One txt per eligible image with lines: 0 cx cy w h  (normalized [0,1]).
  • Coordinates are validated; bad boxes are reported and skipped.
  • Existing exports are overwritten atomically (*.txt.part → *.txt).

Usage:
    python scripts/export_yolo.py [--output exports/yolo_labels]
    python scripts/export_yolo.py --validate   # runs a self-test, exits 0 if ok

Output layout:
    exports/yolo_labels/
        labels/
            <image_id>.txt
        image_paths.txt   # one project-relative image path per exported label
        export_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_ROOT = PROJECT_ROOT / "data" / "annotations"
INVENTORY_PATH = PROJECT_ROOT / "data" / "metadata" / "image_inventory.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "exports" / "yolo_labels"

YOLO_CLASS_POTHOLE = 0


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def normalize_box(box: dict, img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """Convert pixel {x, y, w, h} to YOLO cx cy w h in [0,1].

    Returns None and prints a warning if the box is invalid.
    """
    try:
        px = float(box["x"])
        py = float(box["y"])
        pw = float(box["w"])
        ph = float(box["h"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"    SKIP invalid box (missing/bad field): {box} — {exc}")
        return None

    if pw <= 0 or ph <= 0:
        print(f"    SKIP zero-size box: {box}")
        return None
    if px < 0 or py < 0 or px + pw > img_w or py + ph > img_h:
        print(f"    WARN box partially outside image; clamping: {box}")
        px = max(0.0, px)
        py = max(0.0, py)
        pw = min(pw, img_w - px)
        ph = min(ph, img_h - py)
        if pw <= 0 or ph <= 0:
            print(f"    SKIP: clamped box has zero area.")
            return None

    cx = (px + pw / 2) / img_w
    cy = (py + ph / 2) / img_h
    nw = pw / img_w
    nh = ph / img_h

    for val, name in [(cx, "cx"), (cy, "cy"), (nw, "w"), (nh, "h")]:
        if not (0.0 <= val <= 1.0):
            print(f"    SKIP: normalized {name}={val:.4f} out of [0,1] after clamping")
            return None

    return cx, cy, nw, nh


# ---------------------------------------------------------------------------
# Load inventory
# ---------------------------------------------------------------------------

def load_inventory() -> dict[str, dict]:
    if not INVENTORY_PATH.exists():
        sys.exit(
            f"ERROR: Image inventory not found at {rel(INVENTORY_PATH)}\n"
            "  Run: python scripts/build_inventory.py"
        )
    with INVENTORY_PATH.open(newline="", encoding="utf-8") as fh:
        return {r["image_id"]: r for r in csv.DictReader(fh) if r.get("image_id")}


# ---------------------------------------------------------------------------
# Load annotation for a single image
# ---------------------------------------------------------------------------

def load_annotation(image_id: str) -> dict | None:
    """Search annotations directories for <image_id>.json.  Returns None if missing."""
    for ann_file in ANNOTATIONS_ROOT.rglob(f"{image_id}.json"):
        with ann_file.open(encoding="utf-8") as fh:
            return json.load(fh)
    return None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(output_dir: Path) -> None:
    inventory = load_inventory()
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    image_paths: list[str] = []

    total = reviewed = skipped_uncertain = skipped_no_ann = exported_pos = exported_neg = 0

    for image_id, inv_row in inventory.items():
        total += 1
        status = inv_row.get("review_status", "")
        uncertain = inv_row.get("uncertain", "").strip().lower() in ("1", "true", "yes")

        if status != "reviewed":
            continue
        reviewed += 1

        if uncertain:
            skipped_uncertain += 1
            print(f"  SKIP {image_id}: uncertain — excluded from export")
            continue

        ann = load_annotation(image_id)
        if ann is None:
            skipped_no_ann += 1
            print(f"  SKIP {image_id}: reviewed but no annotation file found")
            continue

        img_w = int(ann.get("image_width", inv_row.get("width", 0)))
        img_h = int(ann.get("image_height", inv_row.get("height", 0)))
        if img_w <= 0 or img_h <= 0:
            print(f"  SKIP {image_id}: invalid image dimensions ({img_w}×{img_h})")
            continue

        boxes: list[dict] = ann.get("pothole_boxes", [])
        yolo_lines: list[str] = []

        for box in boxes:
            coords = normalize_box(box, img_w, img_h)
            if coords is not None:
                cx, cy, nw, nh = coords
                yolo_lines.append(f"{YOLO_CLASS_POTHOLE} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_path = labels_dir / f"{image_id}.txt"
        temp_path = label_path.with_suffix(".txt.part")
        temp_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        temp_path.replace(label_path)

        if yolo_lines:
            exported_pos += 1
        else:
            exported_neg += 1

        image_paths.append(inv_row.get("filepath", ""))
        manifest_rows.append({
            "image_id": image_id,
            "label_file": rel(label_path),
            "image_path": inv_row.get("filepath", ""),
            "pothole_boxes": len(yolo_lines),
            "is_negative": "1" if not yolo_lines else "0",
        })

    # ---- image_paths.txt ------------------------------------------------
    paths_file = output_dir / "image_paths.txt"
    paths_file.write_text("\n".join(image_paths) + "\n", encoding="utf-8")

    # ---- export_manifest.csv -------------------------------------------
    manifest_path = output_dir / "export_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["image_id", "label_file", "image_path", "pothole_boxes", "is_negative"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nExport summary:")
    print(f"  Total inventory images:  {total}")
    print(f"  Reviewed:                {reviewed}")
    print(f"  Exported with potholes:  {exported_pos}")
    print(f"  Exported as negatives:   {exported_neg}")
    print(f"  Skipped (uncertain):     {skipped_uncertain}")
    print(f"  Skipped (no annotation): {skipped_no_ann}")
    print(f"  Output:                  {output_dir}")
    print(f"  Manifest:                {rel(manifest_path)}")


# ---------------------------------------------------------------------------
# Self-test / validation
# ---------------------------------------------------------------------------

def run_validation() -> None:
    """Smoke-test coordinate normalization without touching source data."""
    print("Running YOLO export validation (no files modified)...")

    # Test 1: normal box
    box = {"x": 100, "y": 50, "w": 200, "h": 100}
    result = normalize_box(box, img_w=800, img_h=600)
    assert result is not None, "Valid box should normalize"
    cx, cy, nw, nh = result
    assert abs(cx - 0.25) < 1e-6
    assert abs(nw - 0.25) < 1e-6

    # Test 2: zero-size box → None
    assert normalize_box({"x": 0, "y": 0, "w": 0, "h": 10}, 800, 600) is None

    # Test 3: slightly out-of-bounds → clamps to valid
    box_clamp = {"x": -5, "y": 0, "w": 100, "h": 100}
    result_c = normalize_box(box_clamp, 800, 600)
    assert result_c is not None, "Clampable box should produce result"

    # Test 4: completely outside → None
    assert normalize_box({"x": 1000, "y": 1000, "w": 50, "h": 50}, 800, 600) is None

    # Test 5: label file round-trip
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "test.txt"
        p.write_text("0 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")
        line = p.read_text().strip()
        parts = line.split()
        assert parts[0] == "0"
        assert len(parts) == 5
        for v in parts[1:]:
            assert 0.0 <= float(v) <= 1.0

    print("✓ All validation checks passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        metavar="DIR",
        help=f"Output directory (default: {rel(DEFAULT_OUTPUT)})",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run self-tests on coordinate normalization without touching source data, then exit.",
    )
    args = parser.parse_args()

    if args.validate:
        run_validation()
        return

    export(Path(args.output))


if __name__ == "__main__":
    main()

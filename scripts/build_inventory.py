"""Build or refresh data/metadata/image_inventory.csv.

Scans all raw Green Bay images (Mapillary segments and any images placed in
data/raw/green_bay_images/).  Reads each segment's manifest.csv to preserve
Mapillary IDs and sequence metadata.  Detects exact duplicates by SHA-256
and reports them without deleting originals.

Usage (from project root, with venv active):
    python scripts/build_inventory.py [--dry-run]

Options:
    --dry-run   Print what would be written but do not save the CSV.

Safe to rerun: existing review_status, labels, and notes columns are
preserved from a previously saved inventory.  Only new images are added
with default 'unreviewed' status.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("ERROR: Pillow is required.  Run: .venv/bin/pip install Pillow")

# ---------------------------------------------------------------------------
# Paths (project-relative; no machine-specific absolute paths committed)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_MAPILLARY = PROJECT_ROOT / "data" / "raw" / "mapillary"
RAW_GB_EXTRA = PROJECT_ROOT / "data" / "raw" / "green_bay_images"
VIDEO_FRAMES_ROOT = PROJECT_ROOT / "data" / "processed" / "video_frames"
INVENTORY_PATH = PROJECT_ROOT / "data" / "metadata" / "image_inventory.csv"

INVENTORY_FIELDS = [
    "image_id",
    "original_filename",
    "filepath",          # project-relative
    "source",
    "source_image_id",
    "sha256",
    "width",
    "height",
    "sequence_id",
    "recording_id",
    "split",
    "review_status",
    "pothole_count",
    "shadows_present",
    "patches_present",
    "manholes_present",
    "uncertain",
    "notes",
]

# Fields that must not be overwritten when refreshing an existing inventory
PRESERVE_FIELDS = {
    "review_status",
    "pothole_count",
    "shadows_present",
    "patches_present",
    "manholes_present",
    "uncertain",
    "notes",
    "split",
}

REVIEW_DEFAULT = "unreviewed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def image_dims(path: Path) -> tuple[int, int]:
    """Return (width, height) without fully decoding the image."""
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception as exc:
        print(f"  WARNING: cannot read dimensions for {path}: {exc}")
        return (0, 0)


def rel(path: Path) -> str:
    """Return a project-relative POSIX path string."""
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_existing_inventory() -> dict[str, dict]:
    """Return {image_id: row_dict} for a previously saved inventory."""
    if not INVENTORY_PATH.exists():
        return {}
    with INVENTORY_PATH.open(newline="", encoding="utf-8") as fh:
        return {row["image_id"]: row for row in csv.DictReader(fh) if row.get("image_id")}


def load_mapillary_manifests() -> dict[str, dict]:
    """Return {mapillary_image_id: manifest_row} from all segment manifests."""
    meta: dict[str, dict] = {}
    for manifest_path in sorted(RAW_MAPILLARY.glob("*/manifest.csv")):
        with manifest_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                img_id = row.get("image_id", "").strip()
                if img_id:
                    meta[img_id] = row
    return meta


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def discover_mapillary_images() -> list[dict]:
    """Yield one entry dict per image file found under data/raw/mapillary/."""
    entries = []
    mapillary_meta = load_mapillary_manifests()
    for img_path in sorted(RAW_MAPILLARY.rglob("*.jpg")):
        mapillary_id = img_path.stem  # filename without extension
        manifest = mapillary_meta.get(mapillary_id, {})
        entries.append({
            "_path": img_path,
            "source": "mapillary",
            "source_image_id": mapillary_id,
            "sequence_id": manifest.get("sequence_id", ""),
            "recording_id": "",   # Mapillary images have sequence_id, not video recording
        })
    return entries


def discover_extra_images() -> list[dict]:
    """Yield one entry dict per image found under data/raw/green_bay_images/."""
    entries = []
    if not RAW_GB_EXTRA.exists():
        return entries
    for img_path in sorted(RAW_GB_EXTRA.rglob("*")):
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            entries.append({
                "_path": img_path,
                "source": "local",
                "source_image_id": "",
                "sequence_id": "",
                "recording_id": "",
            })
    return entries


def discover_video_frames() -> list[dict]:
    """Yield one entry dict per extracted video frame under data/processed/video_frames/."""
    entries = []
    if not VIDEO_FRAMES_ROOT.exists():
        return entries
    seen_paths: set[Path] = set()
    for frames_csv in sorted(VIDEO_FRAMES_ROOT.rglob("frames.csv")):
        with frames_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                fp = row.get("filepath", "")
                if fp:
                    full_p = PROJECT_ROOT / fp
                    if full_p.exists() and full_p not in seen_paths:
                        seen_paths.add(full_p)
                        f_idx = int(row.get("frame_index", 0))
                        vid_id = row.get("video_id", "vid")
                        entries.append({
                            "_path": full_p,
                            "source": "video",
                            "source_image_id": f"vid_{vid_id}_{f_idx:05d}",
                            "sequence_id": "",
                            "recording_id": vid_id,
                        })
    return entries


# ---------------------------------------------------------------------------
# Inventory builder
# ---------------------------------------------------------------------------

def build_inventory(dry_run: bool = False) -> None:
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_inventory()
    print(f"Loaded {len(existing)} existing inventory rows.")

    raw_entries = discover_mapillary_images() + discover_extra_images() + discover_video_frames()
    print(f"Found {len(raw_entries)} image files on disk.")

    # ---- Compute hashes and detect duplicates ---------------------------
    hash_to_paths: dict[str, list[Path]] = {}
    file_hashes: dict[Path, str] = {}
    for entry in raw_entries:
        p = entry["_path"]
        h = sha256(p)
        file_hashes[p] = h
        hash_to_paths.setdefault(h, []).append(p)

    duplicates = {h: ps for h, ps in hash_to_paths.items() if len(ps) > 1}
    if duplicates:
        print(f"\n⚠  DUPLICATE REPORT  ({len(duplicates)} duplicate groups; originals NOT deleted):")
        for h, paths in sorted(duplicates.items()):
            print(f"  sha256:{h[:16]}…")
            for p in paths:
                print(f"    {rel(p)}")
        print()
    else:
        print("No exact-duplicate images detected.\n")

    # ---- Assign stable image_ids ----------------------------------------
    # Use Mapillary ID where available; otherwise sha256[:12] + counter.
    rows: list[dict] = []
    assigned_ids: set[str] = set()
    counter = 0

    for entry in raw_entries:
        p: Path = entry["_path"]
        h = file_hashes[p]
        src_id = entry["source_image_id"]

        if src_id:
            image_id = f"mly_{src_id}" if entry["source"] == "mapillary" else src_id
        else:
            counter += 1
            image_id = f"local_{h[:12]}_{counter:04d}"

        # Avoid emitting two rows with the same image_id (same file in two dirs)
        if image_id in assigned_ids:
            continue
        assigned_ids.add(image_id)

        w, h_px = image_dims(p)

        # Preserve any labels already saved in the existing inventory
        prev = existing.get(image_id, {})
        row: dict = {
            "image_id": image_id,
            "original_filename": p.name,
            "filepath": rel(p),
            "source": entry["source"],
            "source_image_id": src_id,
            "sha256": h,
            "width": w,
            "height": h_px,
            "sequence_id": entry["sequence_id"],
            "recording_id": entry["recording_id"],
            "split": prev.get("split", "pilot"),   # default: pilot evaluation set
            "review_status": prev.get("review_status", REVIEW_DEFAULT),
            "pothole_count": prev.get("pothole_count", ""),
            "shadows_present": prev.get("shadows_present", ""),
            "patches_present": prev.get("patches_present", ""),
            "manholes_present": prev.get("manholes_present", ""),
            "uncertain": prev.get("uncertain", ""),
            "notes": prev.get("notes", ""),
        }
        rows.append(row)

    # ---- Sort: sequence_id, then filepath for stable ordering ------------
    rows.sort(key=lambda r: (r["sequence_id"], r["filepath"]))

    print(f"Inventory will contain {len(rows)} unique images.")

    # ---- Duplicate summary in inventory ----------------------------------
    # (Duplicates are kept at their canonical path; both paths are in rows
    #  because they have different filepaths but may share source_image_id.)
    sha_counts: dict[str, int] = {}
    for r in rows:
        sha_counts[r["sha256"]] = sha_counts.get(r["sha256"], 0) + 1
    dup_count = sum(1 for v in sha_counts.values() if v > 1)
    if dup_count:
        print(f"  ↳ {dup_count} SHA-256 values appear in >1 inventory row (cross-segment duplicates).")

    if dry_run:
        print("\n[dry-run] No file written.")
        for r in rows[:3]:
            print(" ", r)
        return

    with INVENTORY_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=INVENTORY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Inventory saved → {rel(INVENTORY_PATH)}")
    print(f"  Split breakdown:")
    split_counts: dict[str, int] = {}
    for r in rows:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
    for s, n in sorted(split_counts.items()):
        print(f"    {s}: {n}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing files")
    args = parser.parse_args()
    build_inventory(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

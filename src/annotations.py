"""Shared helpers for reading and writing annotation JSON files.

Annotation format (data/annotations/<source>/<image_id>.json):
{
  "image_id": "mly_299972942579578",
  "source_filepath": "data/raw/mapillary/segment_001/images/299972942579578.jpg",
  "image_width": 2048,
  "image_height": 1536,
  "review_status": "reviewed",          // unreviewed | reviewed | uncertain
  "shadows_present": false,
  "patches_present": false,
  "manholes_present": false,
  "uncertain": false,
  "notes": "",
  "pothole_boxes": [
    {"x": 120, "y": 200, "w": 80, "h": 60}   // pixel coords, top-left origin
  ],
  "negative_regions": [
    {"x": 400, "y": 100, "w": 120, "h": 80, "subtype": "shadow"}
  ],
  "annotation_version": 1
}

Rules:
- Coordinates are integers in pixels, top-left origin.
- pothole_boxes: clearly visible missing pavement / cavities only.
- negative_regions: shadow | repaired_patch | manhole | ordinary_road
- review_status drives export eligibility; uncertain excludes an image.
- Model predictions are never stored here; keep them separate.
"""

from __future__ import annotations

import json
from pathlib import Path

ANNOTATION_VERSION = 1

VALID_STATUSES = {"unreviewed", "reviewed", "uncertain"}
VALID_NEGATIVE_SUBTYPES = {"shadow", "repaired_patch", "manhole", "ordinary_road"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_ROOT = PROJECT_ROOT / "data" / "annotations"


def _ann_path(image_id: str, source: str = "green_bay") -> Path:
    return ANNOTATIONS_ROOT / source / f"{image_id}.json"


def _source_for_image_id(image_id: str) -> str:
    """Guess annotation subdirectory from image_id prefix."""
    if image_id.startswith("vid_") or image_id.startswith("frame_"):
        return "video"
    return "green_bay"


def annotation_path(image_id: str, source: str | None = None) -> Path:
    src = source or _source_for_image_id(image_id)
    return _ann_path(image_id, src)


def load(image_id: str, source: str | None = None) -> dict:
    """Load annotation for image_id; return empty scaffold if none exists."""
    path = annotation_path(image_id, source)
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        # Back-compat: fill any keys added in newer versions
        data.setdefault("pothole_boxes", [])
        data.setdefault("negative_regions", [])
        data.setdefault("annotation_version", ANNOTATION_VERSION)
        return data
    return _empty(image_id)


def _empty(image_id: str) -> dict:
    return {
        "image_id": image_id,
        "source_filepath": "",
        "image_width": 0,
        "image_height": 0,
        "review_status": "unreviewed",
        "shadows_present": False,
        "patches_present": False,
        "manholes_present": False,
        "uncertain": False,
        "notes": "",
        "pothole_boxes": [],
        "negative_regions": [],
        "annotation_version": ANNOTATION_VERSION,
    }


def save(annotation: dict, source: str | None = None) -> Path:
    """Atomically save annotation to its JSON file.  Returns the saved path."""
    image_id = annotation["image_id"]
    path = annotation_path(image_id, source)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate before saving
    status = annotation.get("review_status", "unreviewed")
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid review_status: {status!r}")
    for box in annotation.get("pothole_boxes", []):
        _validate_box(box, context="pothole_box")
    for neg in annotation.get("negative_regions", []):
        _validate_box(neg, context="negative_region")
        subtype = neg.get("subtype", "")
        if subtype not in VALID_NEGATIVE_SUBTYPES:
            raise ValueError(f"Invalid negative subtype: {subtype!r}")

    temp = path.with_suffix(".json.part")
    with temp.open("w", encoding="utf-8") as fh:
        json.dump(annotation, fh, indent=2, ensure_ascii=False)
    temp.replace(path)
    return path


def _validate_box(box: dict, context: str) -> None:
    for field in ("x", "y", "w", "h"):
        if field not in box:
            raise ValueError(f"{context} missing field '{field}': {box}")
        v = box[field]
        if not isinstance(v, (int, float)):
            raise ValueError(f"{context} field '{field}' must be numeric: {box}")
    if box["w"] <= 0 or box["h"] <= 0:
        raise ValueError(f"{context} has non-positive w/h: {box}")


def find_all(image_id: str) -> list[Path]:
    """Return all annotation files matching this image_id across all subdirectories."""
    return list(ANNOTATIONS_ROOT.rglob(f"{image_id}.json"))

"""Validated settings shared by all dashboard sessions on this installation.

Keep durable values outside Streamlit widget state: Streamlit removes widget
state when a user navigates away from the page that owns the widget.
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path


DEFAULT_SETTINGS = {
    "detector_path": "models/best.pt",
    "verifier_path": "",
    "detector_conf": 0.35,
    "detector_iou": 0.45,
    "sample_fps": 5.0,
    "imgsz": 640,
    "min_obs": 2,
    "temporal_window": 5,
    "verifier_threshold": 0.5,
    "tracking": True,
    "generate_clips": False,
    "research_tools": False,
}


def validate_settings(settings: dict) -> dict:
    """Merge defaults, discard unknown keys, and reject invalid configured values.

    Numeric controls accept numbers only, never strings or booleans. Integer
    controls require an actual integer. No path, number, or flag is coerced.
    """
    if not isinstance(settings, dict):
        raise ValueError("Settings must be a JSON object.")
    result = {key: settings.get(key, value) for key, value in DEFAULT_SETTINGS.items()}
    for key in ("detector_path", "verifier_path"):
        if not isinstance(result[key], str):
            raise ValueError(f"{key} must be text.")
    for key in ("tracking", "generate_clips", "research_tools"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be true or false.")
    for key, lower, upper in (
        ("detector_conf", 0.1, 0.9),
        ("detector_iou", 0.1, 0.9),
        ("sample_fps", 1, 15),
        ("verifier_threshold", 0.1, 0.9),
    ):
        value = result[key]
        if type(value) not in (int, float) or not lower <= value <= upper or not math.isfinite(value):
            raise ValueError(f"{key} must be a number between {lower} and {upper}.")
    for key, lower, upper in (("min_obs", 1, 10), ("temporal_window", 2, 10)):
        value = result[key]
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{key} must be a whole number between {lower} and {upper}.")
    if type(result["imgsz"]) is not int or result["imgsz"] not in (416, 640, 960, 1280):
        raise ValueError("imgsz must be 416, 640, 960, or 1280.")
    if result["min_obs"] > result["temporal_window"]:
        raise ValueError("Required sightings cannot exceed the number of frames checked.")
    return result


def load_settings(path: Path) -> dict:
    """Load a valid settings snapshot, or safe defaults for missing/broken files."""
    try:
        return validate_settings(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return dict(DEFAULT_SETTINGS)


def save_settings(path: Path, settings: dict) -> None:
    """Publish a complete validated snapshot using an atomic sibling replacement."""
    values = validate_settings(settings)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    try:
        with pending.open("x", encoding="utf-8") as stream:
            json.dump(values, stream, indent=2, allow_nan=False)
            stream.write("\n")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)

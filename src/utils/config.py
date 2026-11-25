"""Centralized repository configuration and portable path resolution."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ANNOTATED_DIR = DATA_DIR / "annotated"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFESTS_DIR = DATA_DIR / "manifests"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
DETECTOR_MODELS_DIR = MODELS_DIR / "detector"
VERIFIER_MODELS_DIR = MODELS_DIR / "verifier"

RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
EVALUATIONS_DIR = RESULTS_DIR / "evaluations"
REPORTS_DIR = RESULTS_DIR / "reports"

APPROVED_SOURCES = ("rdd2022_usa", "local_smartphone", "local_dashcam", "mapillary")


def load_yaml_config(name: str) -> dict[str, Any]:
    path = CONFIGS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_device(device: str = "auto") -> str:
    """Resolve compute device across Windows RTX 4090, macOS MPS, and Modal/CPU."""
    if device and device != "auto":
        return device
    env_dev = os.environ.get("POTHOLE_DEVICE", "").strip()
    if env_dev:
        return env_dev
    import torch
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def checkpoint_meta_path(checkpoint_path: str | Path) -> Path:
    p = Path(checkpoint_path)
    return p.with_suffix(".meta.json")


def save_checkpoint_metadata(checkpoint_path: str | Path, metadata: dict[str, Any]) -> Path:
    """Save companion preprocessing & provenance metadata alongside a trained checkpoint."""
    import cv2
    import ultralytics
    import torch

    enriched = {
        **metadata,
        "package_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
        },
    }
    meta_path = checkpoint_meta_path(checkpoint_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    return meta_path


def load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load preprocessing metadata for a checkpoint; defaults cleanly to full_frame if absent."""
    meta_path = checkpoint_meta_path(checkpoint_path)
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"preprocessing_profile": "full_frame", "tile_wide_roi": False, "source_rois": {}, "input_imgsz": 960}

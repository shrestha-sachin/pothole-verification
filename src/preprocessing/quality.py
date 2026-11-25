"""Reproducible, calibrated image quality assessment and near-duplicate checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    """Calibrated thresholds derived from empirical dataset percentiles (non-punitive on hard scenes)."""
    min_laplacian_reject: float = 120.0      # Below 120 at <=960w = unreadable motion blur
    low_sharpness_warn: float = 220.0        # Soft warning (retains genuine soft Mapillary frames >=139)
    min_brightness_reject: float = 30.0      # Pitch-black unusable frames
    low_brightness_warn: float = 65.0        # Dark overcast frames kept, flagged for visibility
    max_brightness_reject: float = 235.0     # Severely washed-out glare
    high_brightness_warn: float = 200.0
    min_contrast_reject: float = 15.0        # Flat/blank lens cap frames
    min_bbox_px_reject: float = 6.0          # Sub-6px native boxes carry zero visual signal
    tiny_bbox_px_warn: float = 12.0          # Soft warning for tiny objects
    near_dup_hamming: int = 2                # Adjacent negative frames with <=2 bit dHash diff


@dataclass
class ImageQualityResult:
    valid_image: bool
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    laplacian_var: float = 0.0
    brightness_mean: float = 0.0
    contrast_std: float = 0.0
    dhash_int: int = 0
    dhash_hex: str = ""
    flags: list[str] = field(default_factory=list)
    rejection_reason: str = ""


def compute_dhash(gray: np.ndarray, hash_size: int = 8) -> tuple[int, str]:
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    val = 0
    for bit in diff.flatten():
        val = (val << 1) | int(bool(bit))
    return val, f"{val:016x}"


def assess_image_quality(
    image_path: Path,
    boxes_norm: Iterable[tuple[int, float, float, float, float]] = (),
    thresholds: QualityThresholds = QualityThresholds(),
) -> ImageQualityResult:
    """Assess image readability, resolution, sharpness, brightness, contrast, and box sizes."""
    arr = cv2.imread(str(image_path))
    if arr is None or arr.size == 0:
        return ImageQualityResult(valid_image=False, flags=["corrupt_image"], rejection_reason="corrupt_image")

    ih, iw = arr.shape[:2]
    if iw < 160 or ih < 160:
        return ImageQualityResult(
            valid_image=False,
            width=iw,
            height=ih,
            aspect_ratio=round(iw / max(1, ih), 3),
            flags=["too_small_resolution"],
            rejection_reason="too_small_resolution",
        )

    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    # Normalize width to <=960px so Laplacian variance is comparable across 640px, 2K, and 4K
    if iw > 960:
        scale = 960.0 / iw
        g_norm = cv2.resize(gray, (960, max(1, int(round(ih * scale)))), interpolation=cv2.INTER_AREA)
    else:
        g_norm = gray

    lap_var = float(cv2.Laplacian(g_norm, cv2.CV_64F).var())
    bright = float(np.mean(gray))
    contrast = float(np.std(gray))
    dh_int, dh_hex = compute_dhash(gray)

    flags: list[str] = []
    reject_reasons: list[str] = []

    if lap_var < thresholds.min_laplacian_reject:
        flags.append("excessive_blur")
        reject_reasons.append("excessive_blur")
    elif lap_var < thresholds.low_sharpness_warn:
        flags.append("low_sharpness_warn")

    if bright < thresholds.min_brightness_reject:
        flags.append("severely_underexposed")
        reject_reasons.append("severely_underexposed")
    elif bright < thresholds.low_brightness_warn:
        flags.append("low_brightness_warn")

    if bright > thresholds.max_brightness_reject:
        flags.append("severely_overexposed")
        reject_reasons.append("severely_overexposed")
    elif bright > thresholds.high_brightness_warn:
        flags.append("high_brightness_warn")

    if contrast < thresholds.min_contrast_reject:
        flags.append("low_contrast_flat")
        reject_reasons.append("low_contrast_flat")

    min_box_side = None
    for _, _, _, bw, bh in boxes_norm:
        px_w, px_h = bw * iw, bh * ih
        side = min(px_w, px_h)
        min_box_side = side if min_box_side is None else min(min_box_side, side)

    if min_box_side is not None:
        if min_box_side < thresholds.min_bbox_px_reject:
            flags.append("subpixel_bbox")
            reject_reasons.append("subpixel_bbox")
        elif min_box_side < thresholds.tiny_bbox_px_warn:
            flags.append("tiny_bbox_warn")

    return ImageQualityResult(
        valid_image=not reject_reasons,
        width=iw,
        height=ih,
        aspect_ratio=round(iw / ih, 3),
        laplacian_var=round(lap_var, 2),
        brightness_mean=round(bright, 2),
        contrast_std=round(contrast, 2),
        dhash_int=dh_int,
        dhash_hex=dh_hex,
        flags=flags,
        rejection_reason=";".join(reject_reasons),
    )


def compute_effective_yolo_box_px(
    orig_w: int,
    orig_h: int,
    box_w_norm: float,
    box_h_norm: float,
    imgsz: int,
) -> tuple[float, float, float]:
    """Compute effective box pixel width/height and linear scale after Ultralytics letterboxing."""
    scale = min(imgsz / max(1, orig_w), imgsz / max(1, orig_h))
    eff_w = box_w_norm * orig_w * scale
    eff_h = box_h_norm * orig_h * scale
    return eff_w, eff_h, scale

"""Standardized preprocessing and crop utilities for pothole verification.

Enforces a single shared preprocessing pipeline for training, inference, and evidence extraction:
  1. Box context expansion: 15% padding on each side, clamped to image boundaries.
  2. Aspect ratio preservation: scale longest side to 224 px.
  3. Padding: center in 224x224 canvas filled with ImageNet mean.
  4. Normalization: ImageNet mean and std.
"""

from __future__ import annotations

import logging
from typing import Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard ImageNet normalization constants (RGB order)
IMAGENET_MEAN_RGB = (0.485, 0.456, 0.406)
IMAGENET_STD_RGB = (0.229, 0.224, 0.225)

# BGR equivalents for OpenCV operations
IMAGENET_MEAN_BGR = (0.406, 0.456, 0.485)

TARGET_SIZE = 224
DEFAULT_PAD_FRAC = 0.15


def expand_and_clamp_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
    pad_frac: float = DEFAULT_PAD_FRAC,
) -> tuple[int, int, int, int]:
    """Expand bounding box by pad_frac on all four sides and clamp to image dimensions.

    Parameters
    ----------
    x1, y1, x2, y2 : int
        Pixel coordinates of the box (top-left origin).
    img_w, img_h : int
        Image width and height in pixels.
    pad_frac : float
        Fraction of box width/height to pad on each side (default 0.15 = 15%).

    Returns
    -------
    tuple[int, int, int, int]
        Clamped (crop_x1, crop_y1, crop_x2, crop_y2).
    """
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    pad_x = int(round(bw * pad_frac))
    pad_y = int(round(bh * pad_frac))

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(img_w, x2 + pad_x)
    cy2 = min(img_h, y2 + pad_y)

    return cx1, cy1, cx2, cy2


def extract_padded_crop(
    image: np.ndarray,
    box: Union[Tuple[int, int, int, int], Sequence[int], dict],
    pad_frac: float = DEFAULT_PAD_FRAC,
) -> np.ndarray:
    """Extract a cropped region from an image with context padding clamped to borders.

    Parameters
    ----------
    image : np.ndarray
        Source image array of shape (H, W, C) or (H, W).
    box : tuple or dict
        Either (x1, y1, x2, y2) or dict with keys {'x', 'y', 'w', 'h'}.
    pad_frac : float
        Padding fraction on each side (default 0.15).

    Returns
    -------
    np.ndarray
        Cropped image slice.
    """
    if image is None or image.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    img_h, img_w = image.shape[:2]

    if isinstance(box, dict):
        bx = int(box.get("x", 0))
        by = int(box.get("y", 0))
        bw = int(box.get("w", 0))
        bh = int(box.get("h", 0))
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
    else:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    cx1, cy1, cx2, cy2 = expand_and_clamp_box(x1, y1, x2, y2, img_w, img_h, pad_frac=pad_frac)

    if cx2 <= cx1 or cy2 <= cy1:
        return np.zeros((0, 0, 3), dtype=image.dtype)

    return image[cy1:cy2, cx1:cx2].copy()


def compute_laplacian_variance(crop_bgr: np.ndarray) -> float:
    """Compute Laplacian variance (focus/sharpness measure) of an image crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    if crop_bgr.ndim == 3 and crop_bgr.shape[2] == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    elif crop_bgr.ndim == 2:
        gray = crop_bgr
    else:
        return 0.0

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def preprocess_crop_for_verifier(
    crop_bgr: np.ndarray,
    target_size: int = TARGET_SIZE,
) -> np.ndarray:
    """Preprocess a BGR crop into a normalized CHW float32 tensor for MobileNetV3-Small.

    Steps:
      1. BGR -> RGB conversion and scale to [0, 1].
      2. Aspect-ratio preserving resize (longest side = target_size).
         Uses cv2.INTER_AREA for downsampling, cv2.INTER_LINEAR for upsampling.
      3. Center-pad onto target_size x target_size canvas filled with ImageNet mean.
      4. Normalize with ImageNet mean and std.
      5. Output CHW float32 array.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((3, target_size, target_size), dtype=np.float32)

    # Convert BGR to RGB and scale to [0, 1]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((3, target_size, target_size), dtype=np.float32)

    scale = target_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)

    # Canvas initialized with ImageNet mean in RGB
    canvas = np.full((target_size, target_size, 3), IMAGENET_MEAN_RGB, dtype=np.float32)
    y0 = (target_size - new_h) // 2
    x0 = (target_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    # ImageNet normalization
    mean = np.array(IMAGENET_MEAN_RGB, dtype=np.float32)
    std = np.array(IMAGENET_STD_RGB, dtype=np.float32)
    normalized = (canvas - mean) / std

    # HWC -> CHW
    return normalized.transpose(2, 0, 1).astype(np.float32)


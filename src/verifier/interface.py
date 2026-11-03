"""MobileNetV3-Small verifier interface.

This module defines the VerifierInterface that the dashboard and evaluation
pipeline use.  The actual MobileNetV3-Small model is not trained yet;
this interface gracefully degrades when no checkpoint is available.

Preprocessing contract (must match training):
  1. Expand detected bounding box by 15% on each side
  2. Preserve aspect ratio
  3. Resize longest side to fit 224×224, pad remainder with ImageNet mean
  4. Normalize with ImageNet mean and std

When the checkpoint does not exist, is_configured() returns False and
the workflow continues with detector-only results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TARGET_SIZE = 224


@dataclass
class VerifierResult:
    pothole_score: float          # probability ∈ [0, 1]; higher = more likely pothole
    is_pothole: bool              # True if score >= threshold
    threshold_used: float
    model_version: str = "not_configured"


class VerifierInterface:
    """
    Wrapper around a trained MobileNetV3-Small classifier.

    Usage::

        verifier = VerifierInterface("models/verifier_v1.pt", threshold=0.5)
        if verifier.is_configured():
            result = verifier.score_crop(crop_bgr)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        self.threshold = threshold
        self._model = None
        self._configured = False
        self._model_version = "not_configured"
        self._device = self._resolve_device(device)

        if checkpoint_path is not None:
            self._try_load(Path(checkpoint_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """True if a trained checkpoint was successfully loaded."""
        return self._configured

    def score_crop(self, crop_bgr: np.ndarray) -> VerifierResult:
        """
        Score a single BGR crop.
        Returns VerifierResult with pothole_score ∈ [0, 1].
        Raises RuntimeError if not configured.
        """
        if not self._configured:
            raise RuntimeError("Verifier not configured — no checkpoint loaded.")
        tensor = self._preprocess(crop_bgr)
        score = self._run_inference(tensor)
        return VerifierResult(
            pothole_score=score,
            is_pothole=score >= self.threshold,
            threshold_used=self.threshold,
            model_version=self._model_version,
        )

    def score_batch(self, crops_bgr: list[np.ndarray]) -> list[VerifierResult]:
        """Score a list of crops; returns results in order."""
        return [self.score_crop(c) for c in crops_bgr]

    # ------------------------------------------------------------------
    # Preprocessing (identical at training and inference)
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess a BGR crop for MobileNetV3-Small:
          1. Convert BGR → RGB
          2. Resize longest side to 224 with preserved aspect ratio
          3. Pad to 224×224 with ImageNet mean
          4. Normalize to [0,1] and apply ImageNet mean/std
          5. Return CHW float32 ndarray
        """
        import cv2

        if crop_bgr is None or crop_bgr.size == 0:
            # Return a blank tensor
            return np.zeros((3, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        h, w = rgb.shape[:2]
        scale = TARGET_SIZE / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to TARGET_SIZE × TARGET_SIZE with ImageNet mean
        canvas = np.full((TARGET_SIZE, TARGET_SIZE, 3), IMAGENET_MEAN, dtype=np.float32)
        y0 = (TARGET_SIZE - new_h) // 2
        x0 = (TARGET_SIZE - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

        # Normalize
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        canvas = (canvas - mean) / std

        # HWC → CHW
        return canvas.transpose(2, 0, 1).astype(np.float32)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _try_load(self, path: Path) -> None:
        if not path.exists():
            logger.info("Verifier checkpoint not found at %s — running without verifier.", path)
            return
        try:
            import torch
            from torchvision.models import mobilenet_v3_small

            # Build architecture matching training
            model = mobilenet_v3_small(weights=None)
            # Replace classifier head for binary classification
            in_features = model.classifier[-1].in_features
            import torch.nn as nn
            model.classifier[-1] = nn.Linear(in_features, 2)

            state = torch.load(str(path), map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
                self._model_version = state.get("version", path.stem)
            else:
                model.load_state_dict(state)
                self._model_version = path.stem

            model.eval()
            model.to(self._device)
            self._model = model
            self._configured = True
            logger.info("Verifier loaded from %s (device=%s)", path, self._device)
        except Exception as exc:
            logger.warning("Failed to load verifier from %s: %s", path, exc)
            self._configured = False

    def _preprocess(self, crop_bgr: np.ndarray):
        """Preprocess and return a torch tensor."""
        import torch
        arr = self.preprocess_crop(crop_bgr)
        return torch.from_numpy(arr).unsqueeze(0).to(self._device)

    def _run_inference(self, tensor) -> float:
        import torch
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
            # Class 1 = pothole (convention: 0=non_pothole, 1=pothole)
            return float(probs[0, 1].item())

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"


# ---------------------------------------------------------------------------
# Convenience: load from path with graceful degradation
# ---------------------------------------------------------------------------

def load_verifier(
    checkpoint_path: Optional[str | Path] = None,
    threshold: float = 0.5,
    device: str = "auto",
) -> VerifierInterface:
    """Factory function — always returns a VerifierInterface; check .is_configured()."""
    return VerifierInterface(checkpoint_path=checkpoint_path, threshold=threshold, device=device)

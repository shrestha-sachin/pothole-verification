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

from src.verifier.preprocessing import (
    IMAGENET_MEAN_RGB,
    IMAGENET_STD_RGB,
    TARGET_SIZE,
    preprocess_crop_for_verifier,
)

logger = logging.getLogger(__name__)


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
        self.class_to_idx: dict[str, int] = {"non_pothole": 0, "pothole": 1}
        self.pothole_class_idx: int = 1

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
    # Preprocessing (delegates to centralized shared preprocessing)
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
        return preprocess_crop_for_verifier(crop_bgr)

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
            import torch.nn as nn

            state = torch.load(str(path), map_location="cpu")
            class_to_idx = {"non_pothole": 0, "pothole": 1}

            if isinstance(state, dict):
                model_state = state.get("model_state_dict", state)
                self._model_version = state.get("version", path.stem)
                if "class_to_idx" in state and isinstance(state["class_to_idx"], dict):
                    class_to_idx = state["class_to_idx"]
                elif "classes" in state and isinstance(state["classes"], list):
                    class_to_idx = {name: idx for idx, name in enumerate(state["classes"])}
            else:
                model_state = state
                self._model_version = path.stem

            self.class_to_idx = class_to_idx
            self.pothole_class_idx = class_to_idx.get("pothole", 1)

            # Build architecture matching training
            num_classes = max(2, len(class_to_idx))
            model = mobilenet_v3_small(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)

            model.load_state_dict(model_state)
            model.eval()
            model.to(self._device)
            self._model = model
            self._configured = True
            logger.info(
                "Verifier loaded from %s (device=%s, pothole_idx=%d, classes=%s)",
                path, self._device, self.pothole_class_idx, self.class_to_idx,
            )
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
            return float(probs[0, self.pothole_class_idx].item())

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

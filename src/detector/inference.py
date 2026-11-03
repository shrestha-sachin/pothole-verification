"""YOLO pothole detector — load once, reuse for every frame.

Usage::

    detector = PotholeDetector("models/best.pt", conf=0.35, device="auto")
    for result in detector.run_on_frame(frame_bgr):
        print(result)  # FrameDetection(...)

Rules enforced:
  - The model is loaded once at construction; never reloaded per frame.
  - Device selection is automatic: MPS → CUDA → CPU.
  - Inference runs on BGR frames supplied by OpenCV (converted to RGB internally).
  - Only class 0 (pothole) boxes are returned.
  - Raw YOLO Results are not stored; only lightweight FrameDetection dicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameDetection:
    """A single bounding-box detection on one frame."""
    frame_index: int
    timestamp_ms: int
    # Box in pixel coords (top-left origin)
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    class_id: int = 0
    class_name: str = "pothole"

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def xywh(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    @property
    def cx_cy_w_h(self) -> tuple[float, float, float, float]:
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        return (self.x1 + w / 2, self.y1 + h / 2, float(w), float(h))


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class PotholeDetector:
    """Singleton-style YOLO detector.  Construct once, call run_on_frame many times."""

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        target_class_id: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        model_path  : path to .pt checkpoint (YOLO11n or any Ultralytics-compatible)
        conf        : detector confidence threshold
        iou         : NMS IoU threshold
        imgsz       : inference image size
        device      : "auto" | "mps" | "cuda" | "cpu"
        target_class_id : class index to keep (0 = pothole by convention)
        """
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.target_class_id = target_class_id

        self.device = self._resolve_device(device)
        logger.info("Loading YOLO model from %s on device=%s", self.model_path, self.device)
        self.model = YOLO(str(self.model_path))

        # Warm-up: run one dummy pass so first real frame is not slow
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self.model.predict(
            dummy,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        logger.info("Detector ready.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_on_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: int = 0,
    ) -> list[FrameDetection]:
        """Run inference on a single BGR frame; return list of FrameDetection."""
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(
            frame_rgb,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[FrameDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls[0].item())
                if cls != self.target_class_id:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0].item())
                detections.append(
                    FrameDetection(
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        conf=conf,
                        class_id=cls,
                        class_name="pothole",
                    )
                )
        return detections

    # ------------------------------------------------------------------
    # Batch / video processing
    # ------------------------------------------------------------------

    def process_video_frames(
        self,
        frame_iter,
        progress_callback=None,
    ) -> list[FrameDetection]:
        """
        Process an iterator of (frame_index, timestamp_ms, bgr_frame) tuples.
        progress_callback(done, total) is called after each frame if provided.
        """
        all_detections: list[FrameDetection] = []
        frames = list(frame_iter)
        total = len(frames)
        for i, (frame_index, timestamp_ms, bgr) in enumerate(frames):
            dets = self.run_on_frame(bgr, frame_index=frame_index, timestamp_ms=timestamp_ms)
            all_detections.extend(dets)
            if progress_callback:
                progress_callback(i + 1, total)
        return all_detections

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

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

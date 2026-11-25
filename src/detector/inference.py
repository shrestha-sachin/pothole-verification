"""YOLO pothole detector — load once, reuse for every frame.

Usage::

    detector = PotholeDetector("models/best.pt", conf=0.35, device="auto")
    for result in detector.run_on_frame(frame_bgr):
        print(result)  # FrameDetection(...)

Rules enforced:
  - The model is loaded once at construction; never reloaded per frame.
  - Device selection is automatic: MPS -> CUDA -> CPU.
  - Inference runs on BGR frames supplied by OpenCV (Ultralytics handles BGR->RGB).
  - Only class 0 (pothole) boxes are returned.
  - Shared ROI / horizontal-tile preprocessing maps and NMS-merges predictions in original-frame coords.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.preprocessing.roi import (
    SourceROI,
    compute_roi_and_tile_rects,
    map_predictions_to_original,
)

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
# Shared ROI + tiled inference & BoT-SORT tracking helpers
# ---------------------------------------------------------------------------

def predict_frame_detections(
    model: Any,
    frame_bgr: np.ndarray,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 960,
    device: str = "auto",
    roi: Optional[SourceROI] = None,
    tile_wide_roi: bool = False,
    merge_iou: float = 0.50,
    overlap_frac: float = 0.15,
    min_tile_trigger_width: int = 2000,
    target_class_id: int = 0,
) -> list[dict[str, Any]]:
    """Run YOLO prediction with full_frame, roi, or roi_tiled preprocessing and NMS-merge in original coords."""
    h, w = frame_bgr.shape[:2]
    rects = compute_roi_and_tile_rects(
        w,
        h,
        roi,
        tile_wide_roi=tile_wide_roi,
        overlap_frac=overlap_frac,
        min_tile_trigger_width=min_tile_trigger_width,
    )

    merged_raw: list[dict[str, Any]] = []
    for rect in rects:
        rx1, ry1, rx2, ry2 = rect
        sub_bgr = frame_bgr[ry1:ry2, rx1:rx2]
        results = model.predict(
            sub_bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        rect_preds: list[dict[str, Any]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                if cls_id != target_class_id:
                    continue
                b_conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                rect_preds.append(
                    {
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                        "conf": b_conf,
                        "class_id": cls_id,
                    }
                )
        # Map boxes back to original-frame coords without pre-NMS distortion across tiles
        merged_raw.extend(map_predictions_to_original(rect_preds, rect, nms_iou_thresh=1.01))

    if len(rects) > 1 and len(merged_raw) > 1:
        return map_predictions_to_original(merged_raw, (0, 0, 0, 0), nms_iou_thresh=merge_iou)
    merged_raw.sort(key=lambda p: float(p.get("conf", 0.0)), reverse=True)
    return merged_raw


def create_botsort_tracker(tracker_cfg: dict[str, Any], device: str = "cpu") -> Any:
    """Instantiate an Ultralytics BoT-SORT tracker directly from config dict."""
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.utils import IterableSimpleNamespace

    cfg = {
        "tracker_type": "botsort",
        "track_high_thresh": float(tracker_cfg.get("track_high_thresh", 0.3)),
        "track_low_thresh": float(tracker_cfg.get("track_low_thresh", 0.1)),
        "new_track_thresh": float(tracker_cfg.get("new_track_thresh", 0.25)),
        "track_buffer": int(tracker_cfg.get("track_buffer", 30)),
        "match_thresh": float(tracker_cfg.get("match_thresh", 0.8)),
        "fuse_score": bool(tracker_cfg.get("fuse_score", True)),
        "gmc_method": str(tracker_cfg.get("gmc_method", "none")),
        "with_reid": bool(tracker_cfg.get("with_reid", False)),
        "proximity_thresh": float(tracker_cfg.get("proximity_thresh", 0.5)),
        "appearance_thresh": float(tracker_cfg.get("appearance_thresh", 0.8)),
        "model": str(tracker_cfg.get("model", "auto")),
        "device": device,
    }
    return BOTSORT(args=IterableSimpleNamespace(**cfg))


def track_merged_detections(
    tracker: Any,
    detections: list[dict[str, Any]],
    frame_bgr: np.ndarray,
) -> list[dict[str, Any]]:
    """Run BoT-SORT update on original-frame merged detections -> list of {x1,y1,x2,y2,conf,track_id,class_id}."""
    import torch
    from ultralytics.engine.results import Boxes

    if detections:
        arr = torch.tensor(
            [
                [float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"]), float(d["conf"]), float(d.get("class_id", 0))]
                for d in detections
            ],
            dtype=torch.float32,
        )
    else:
        arr = torch.empty((0, 6), dtype=torch.float32)

    det_boxes = Boxes(arr, frame_bgr.shape[:2]).cpu().numpy()
    tracks = tracker.update(det_boxes, frame_bgr)
    tracked_out: list[dict[str, Any]] = []
    for row in tracks:
        tracked_out.append(
            {
                "x1": float(row[0]),
                "y1": float(row[1]),
                "x2": float(row[2]),
                "y2": float(row[3]),
                "track_id": int(row[4]),
                "conf": float(row[5]),
                "class_id": int(row[6]),
            }
        )
    return tracked_out


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
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.target_class_id = target_class_id

        self.device = self._resolve_device(device)
        logger.info("Loading YOLO model from %s on device=%s", self.model_path, self.device)
        self.model = YOLO(str(self.model_path))

        self._validate_checkpoint_classes()

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

    def _validate_checkpoint_classes(self) -> None:
        """Confirm target class represents pothole in the checkpoint; reject incompatible models."""
        names = getattr(self.model, "names", {})
        if not names:
            return

        cls_name = names.get(self.target_class_id, "")
        if "pothole" not in str(cls_name).lower():
            pothole_classes = [cid for cid, cname in names.items() if "pothole" in str(cname).lower()]
            sample_classes = {cid: names[cid] for cid in list(names.keys())[:5]}
            raise ValueError(
                f"Incompatible YOLO checkpoint '{self.model_path.name}': "
                f"class {self.target_class_id} is '{cls_name}', not 'pothole'. "
                f"(Loaded model classes: {sample_classes}...). "
                + (f"Pothole exists at class {pothole_classes[0]}; set target_class_id={pothole_classes[0]}."
                   if pothole_classes else
                   "This appears to be an un-fine-tuned base checkpoint (such as COCO yolo11n.pt). "
                   "Please supply a fine-tuned pothole detector checkpoint (e.g. models/best.pt).")
            )

    def run_on_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int = 0,
        timestamp_ms: int = 0,
        roi: Optional[SourceROI] = None,
        tile_wide_roi: bool = False,
        merge_iou: float = 0.50,
    ) -> list[FrameDetection]:
        """Run inference on a single BGR frame (with optional SourceROI crop / horizontal tiling & NMS merge)."""
        preds = predict_frame_detections(
            self.model,
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            roi=roi,
            tile_wide_roi=tile_wide_roi,
            merge_iou=merge_iou,
            target_class_id=self.target_class_id,
        )
        return [
            FrameDetection(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                x1=int(round(p["x1"])),
                y1=int(round(p["y1"])),
                x2=int(round(p["x2"])),
                y2=int(round(p["y2"])),
                conf=float(p["conf"]),
                class_id=int(p["class_id"]),
                class_name="pothole",
            )
            for p in preds
        ]

    def process_video_frames(
        self,
        frame_iter,
        total_frames: Optional[int] = None,
        progress_callback=None,
        roi: Optional[SourceROI] = None,
        tile_wide_roi: bool = False,
        merge_iou: float = 0.50,
    ) -> list[FrameDetection]:
        """Process an iterator of (frame_index, timestamp_ms, bgr_frame) tuples with bounded memory."""
        all_detections: list[FrameDetection] = []
        for i, (frame_index, timestamp_ms, bgr) in enumerate(frame_iter):
            dets = self.run_on_frame(
                bgr,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                roi=roi,
                tile_wide_roi=tile_wide_roi,
                merge_iou=merge_iou,
            )
            all_detections.extend(dets)
            if progress_callback:
                progress_callback(i + 1, total_frames or (i + 1))
        return all_detections

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

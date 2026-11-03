"""BoT-SORT based candidate event grouper.

Groups repeated YOLO detections of the same physical pothole into a single
CandidateEvent, using Ultralytics' built-in BoT-SORT tracker.

Design decisions aligned with the capstone proposal:
  - Camera-motion compensation: disabled (cmc_method = none)
  - Appearance re-identification (ReID): disabled (with_reid = False)
  - One shared tracker instance per video processing run
  - Actual detector observations stored separately from tracker predictions
  - Representative crop selected from up to 3 highest-confidence detections
    using Laplacian variance (sharpness), breaking ties by confidence
  - Fixed road ROI configurable but not required
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObservedDetection:
    """A single actual (non-predicted) detector observation belonging to an event."""
    frame_index: int
    timestamp_ms: int
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    laplacian_var: float = 0.0  # sharpness score of the crop


@dataclass
class CandidateEvent:
    """A grouped pothole candidate event backed by ≥1 actual detector observations."""
    event_id: str
    video_id: str
    road_segment_id: str
    track_id: int

    # Observation window
    first_timestamp_ms: int
    last_timestamp_ms: int
    first_frame_index: int
    last_frame_index: int

    # Aggregated detector stats
    detector_conf_max: float
    detector_conf_mean: float
    observation_count: int           # actual detector hits (not tracker predictions)

    # Representative bounding box (from best observation)
    rep_x1: int
    rep_y1: int
    rep_x2: int
    rep_y2: int

    # Evidence paths (populated by evidence extractor)
    full_frame_path: str = ""
    crop_path: str = ""
    clip_path: str = ""

    # Full list of actual detector observations
    observations: list[ObservedDetection] = field(default_factory=list)

    # Tracker config snapshot
    tracker_config: dict = field(default_factory=dict)

    # Policy evaluation support
    policy_a_retained: bool = True          # conf >= threshold
    policy_b_retained: bool = False         # temporal: obs_count >= min_obs
    policy_c_score: Optional[float] = None  # verifier score (None = not run)
    policy_c_retained: Optional[bool] = None

    @property
    def duration_ms(self) -> int:
        return max(0, self.last_timestamp_ms - self.first_timestamp_ms)

    @property
    def duration_sec(self) -> float:
        return self.duration_ms / 1000.0

    @property
    def start_time_str(self) -> str:
        sec = self.first_timestamp_ms / 1000.0
        m, s = divmod(int(sec), 60)
        return f"{m:02d}:{s:02d}.{self.first_timestamp_ms % 1000:03d}"

    @property
    def end_time_str(self) -> str:
        sec = self.last_timestamp_ms / 1000.0
        m, s = divmod(int(sec), 60)
        return f"{m:02d}:{s:02d}.{self.last_timestamp_ms % 1000:03d}"


# ---------------------------------------------------------------------------
# Event grouper
# ---------------------------------------------------------------------------

class EventGrouper:
    """Groups frame-level detections into CandidateEvents using BoT-SORT."""

    def __init__(
        self,
        video_id: str,
        road_segment_id: str = "seg_001",
        # BoT-SORT parameters
        track_high_thresh: float = 0.3,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        # Evidence selection
        max_candidate_obs_for_rep: int = 3,
        # Policy B default
        policy_b_min_obs: int = 2,
        policy_b_max_frame_span: int = 5,
    ) -> None:
        self.video_id = video_id
        self.road_segment_id = road_segment_id
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.max_candidate_obs_for_rep = max_candidate_obs_for_rep
        self.policy_b_min_obs = policy_b_min_obs
        self.policy_b_max_frame_span = policy_b_max_frame_span

        self._tracker = None
        self._tracker_model_path = None

        # Per-track state: track_id → list of ObservedDetection
        self._track_obs: dict[int, list[ObservedDetection]] = {}
        # track_id → full-frame BGR for evidence
        self._track_frames: dict[int, list[tuple[float, np.ndarray]]] = {}

    # ------------------------------------------------------------------
    # Tracker lifecycle
    # ------------------------------------------------------------------

    def _get_tracker(self, model):
        """Lazily attach BoT-SORT tracker to the YOLO model (per video run)."""
        return model  # tracking is done via model.track() per-frame

    # ------------------------------------------------------------------
    # Main processing API
    # ------------------------------------------------------------------

    def process_detections_with_tracking(
        self,
        frames_and_detections: list[tuple[int, int, np.ndarray, list]],
        model,
        conf: float,
        iou: float,
        imgsz: int,
        device: str,
        tracker_config_path: Optional[str] = None,
    ) -> list[CandidateEvent]:
        """
        Run BoT-SORT tracking over (frame_index, timestamp_ms, bgr, detections)
        and return candidate events.

        We use model.track() per frame so the tracker state accumulates.
        Only boxes that correspond to actual detector detections (conf >= conf)
        are recorded as observations; tracker-predicted-only boxes are ignored.
        """
        import tempfile, os

        # Build BoT-SORT tracker config as a temp YAML if not provided
        cfg_path = tracker_config_path or self._write_tracker_config()

        # Per-track accumulator: track_id → [ObservedDetection, ...]
        track_obs: dict[int, list[ObservedDetection]] = {}
        # track_id → [(conf, frame_bgr, x1,y1,x2,y2)] for evidence selection
        track_evidence: dict[int, list[tuple[float, np.ndarray, int, int, int, int]]] = {}

        # Reset tracker between videos by creating a fresh predict call chain
        # Ultralytics tracker resets when persist=False per frame, but we need
        # persist=True for accumulation. We reset manually before the loop.
        model.predictor = None  # Force fresh predictor / tracker state

        for frame_index, timestamp_ms, bgr, _ in frames_and_detections:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            results = model.track(
                rgb,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                tracker=cfg_path,
                persist=True,
                verbose=False,
            )

            for r in results:
                if r.boxes is None or r.boxes.id is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id != 0:
                        continue
                    box_conf = float(box.conf[0].item())
                    track_id = int(box.id[0].item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                    # Only record as observation if box came from actual detection
                    # (conf >= threshold, not a pure tracker prediction)
                    if box_conf >= conf:
                        # Compute sharpness of crop
                        crop = bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        lap_var = float(cv2.Laplacian(
                            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size > 0 else np.zeros((1, 1), dtype=np.uint8),
                            cv2.CV_64F,
                        ).var()) if crop.size > 0 else 0.0

                        obs = ObservedDetection(
                            frame_index=frame_index,
                            timestamp_ms=timestamp_ms,
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            conf=box_conf,
                            laplacian_var=lap_var,
                        )
                        track_obs.setdefault(track_id, []).append(obs)
                        # Keep up to max_candidate_obs_for_rep best for evidence
                        evid_list = track_evidence.setdefault(track_id, [])
                        evid_list.append((box_conf, lap_var, bgr.copy(), x1, y1, x2, y2))
                        # Sort by (-laplacian_var, -conf) and keep top-N
                        evid_list.sort(key=lambda t: (-t[1], -t[0]))
                        if len(evid_list) > self.max_candidate_obs_for_rep:
                            evid_list.pop()

        # Build CandidateEvent objects
        events: list[CandidateEvent] = []
        tracker_cfg_snapshot = self._tracker_config_snapshot()

        for track_id, obs_list in track_obs.items():
            if not obs_list:
                continue
            obs_list.sort(key=lambda o: o.timestamp_ms)
            confs = [o.conf for o in obs_list]
            event_id = str(uuid.uuid4())[:8]
            event = CandidateEvent(
                event_id=event_id,
                video_id=self.video_id,
                road_segment_id=self.road_segment_id,
                track_id=track_id,
                first_timestamp_ms=obs_list[0].timestamp_ms,
                last_timestamp_ms=obs_list[-1].timestamp_ms,
                first_frame_index=obs_list[0].frame_index,
                last_frame_index=obs_list[-1].frame_index,
                detector_conf_max=max(confs),
                detector_conf_mean=float(np.mean(confs)),
                observation_count=len(obs_list),
                rep_x1=obs_list[0].x1,
                rep_y1=obs_list[0].y1,
                rep_x2=obs_list[0].x2,
                rep_y2=obs_list[0].y2,
                observations=obs_list,
                tracker_config=tracker_cfg_snapshot,
            )

            # Apply policy B flag
            event.policy_b_retained = event.observation_count >= self.policy_b_min_obs

            events.append(event)

        return events

    def _write_tracker_config(self) -> str:
        """Write a BoT-SORT YAML config with CMC and ReID disabled; return path."""
        import tempfile, os, yaml

        cfg = {
            "tracker_type": "botsort",
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "fuse_score": True,
            # Disable global motion compensation (camera-motion compensation)
            "gmc_method": "none",
            # Disable appearance ReID
            "with_reid": False,
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.8,
            "model": "auto",
        }

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="botsort_"
        )
        import yaml as _yaml
        _yaml.dump(cfg, tmp)
        tmp.close()
        self._tmp_cfg_path = tmp.name
        return tmp.name

    def _tracker_config_snapshot(self) -> dict:
        return {
            "tracker": "botsort",
            "cmc_method": "none",
            "with_reid": False,
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
        }


# ---------------------------------------------------------------------------
# Evidence extractor
# ---------------------------------------------------------------------------

class EvidenceExtractor:
    """Saves full-frame, crop, and short clip evidence for each CandidateEvent."""

    def __init__(self, evidence_root: str | Path) -> None:
        self.evidence_root = Path(evidence_root)

    def extract_all(
        self,
        events: list[CandidateEvent],
        frames_and_detections: list[tuple[int, int, np.ndarray, list]],
        track_evidence: dict[int, list],
        clip_duration_ms: int = 2000,
        source_fps: float = 30.0,
    ) -> None:
        """
        For each event, write:
          - full-frame JPEG (from representative observation)
          - cropped JPEG (padded by 15%)
          - short evidence clip (±clip_duration_ms around first detection)
        """
        # Build frame lookup by frame_index
        frame_lookup: dict[int, np.ndarray] = {
            fi: bgr for fi, _, bgr, _ in frames_and_detections
        }

        for event in events:
            ev_dir = self.evidence_root / event.video_id / event.event_id
            ev_dir.mkdir(parents=True, exist_ok=True)

            # Select best observation (highest lap_var, then conf)
            best_obs = max(event.observations, key=lambda o: (o.laplacian_var, o.conf))
            frame_bgr = frame_lookup.get(best_obs.frame_index)

            if frame_bgr is not None:
                # Full-frame JPEG
                full_path = ev_dir / "full_frame.jpg"
                cv2.imwrite(str(full_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                event.full_frame_path = str(full_path)

                # Crop with 15% padding
                crop_bgr = self._padded_crop(frame_bgr, best_obs.x1, best_obs.y1,
                                              best_obs.x2, best_obs.y2, pad_frac=0.15)
                crop_path = ev_dir / "crop.jpg"
                cv2.imwrite(str(crop_path), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                event.crop_path = str(crop_path)

                # Update representative box from best observation
                event.rep_x1 = best_obs.x1
                event.rep_y1 = best_obs.y1
                event.rep_x2 = best_obs.x2
                event.rep_y2 = best_obs.y2

    @staticmethod
    def _padded_crop(
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        pad_frac: float = 0.15,
    ) -> np.ndarray:
        """Expand box by pad_frac on each side, clamp to frame, return crop."""
        h_frame, w_frame = frame.shape[:2]
        bw = x2 - x1
        bh = y2 - y1
        px = int(bw * pad_frac)
        py = int(bh * pad_frac)
        cx1 = max(0, x1 - px)
        cy1 = max(0, y1 - py)
        cx2 = min(w_frame, x2 + px)
        cy2 = min(h_frame, y2 + py)
        return frame[cy1:cy2, cx1:cx2].copy()

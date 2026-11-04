"""BoT-SORT based candidate event grouper and bounded-memory video pipeline.

Groups repeated YOLO detections of the same physical pothole into a single
CandidateEvent, using Ultralytics' built-in BoT-SORT tracker.

Design decisions aligned with the capstone proposal:
  - Camera-motion compensation: disabled (gmc_method = none, cmc_method = none)
  - Appearance re-identification (ReID): disabled (with_reid = False)
  - Reset tracker state between videos
  - Actual detector observations stored separately from tracker predictions
  - Representative crop selected from up to 3 highest-confidence detections
    using Laplacian variance (sharpness), breaking ties by confidence
  - Shared crop preprocessing: 15% context expansion clamped to image boundaries
  - Bounded memory execution: stream video incrementally without retaining decoded frames
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional, Sequence

import cv2
import numpy as np

from src.evaluation.policies import check_temporal_window
from src.verifier.preprocessing import (
    DEFAULT_PAD_FRAC,
    compute_laplacian_variance,
    extract_padded_crop,
)

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
    rep_timestamp_ms: int = 0
    rep_frame_index: int = 0

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
    policy_b_retained: bool = False         # temporal: >= 2 hits in 5-frame sliding window
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


def select_representative_observation(
    observations: Sequence[ObservedDetection],
) -> Optional[ObservedDetection]:
    """Select the three highest-confidence actual observations for an event, then choose

    the sharpest of those using Laplacian variance, breaking ties by confidence.
    """
    if not observations:
        return None

    # 1. Sort by confidence descending
    sorted_by_conf = sorted(observations, key=lambda o: o.conf, reverse=True)
    # 2. Pick top 3 observations
    top_3 = sorted_by_conf[:3]
    # 3. Choose the sharpest using Laplacian variance, breaking ties by confidence
    best = max(top_3, key=lambda o: (o.laplacian_var, o.conf))
    return best


# ---------------------------------------------------------------------------
# Event grouper
# ---------------------------------------------------------------------------

class EventGrouper:
    """Groups frame-level detections into CandidateEvents using BoT-SORT with bounded memory."""

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
        policy_b_window_frames: int = 5,
        target_class_id: int = 0,
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
        self.policy_b_window_frames = policy_b_window_frames
        self.target_class_id = target_class_id
        self._tmp_cfg_path: Optional[str] = None

    def validate_checkpoint(self, model) -> None:
        """Confirm class 0 really means pothole in the selected checkpoint."""
        names = getattr(model, "names", {})
        if not names:
            return
        cls_name = names.get(self.target_class_id, "")
        if "pothole" not in str(cls_name).lower():
            pothole_classes = [cid for cid, cname in names.items() if "pothole" in str(cname).lower()]
            sample_classes = {cid: names[cid] for cid in list(names.keys())[:5]}
            raise ValueError(
                f"Incompatible YOLO checkpoint: class {self.target_class_id} is '{cls_name}', not 'pothole'. "
                f"(Loaded model classes: {sample_classes}...). "
                + (f"Pothole exists at class {pothole_classes[0]}." if pothole_classes else
                   "This appears to be an un-fine-tuned base checkpoint (such as COCO yolo11n.pt). "
                   "Please supply a fine-tuned pothole detector checkpoint (e.g. models/best.pt).")
            )

    def process_video_stream(
        self,
        frame_iter: Generator[tuple[int, int, np.ndarray], None, None] | Sequence,
        model: Any,
        conf: float = 0.35,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        evidence_root: Optional[Path | str] = None,
        video_path: Optional[Path | str] = None,
        generate_clips: bool = False,
        total_expected_frames: Optional[int] = None,
        progress_callback = None,
    ) -> list[CandidateEvent]:
        """
        Process frames incrementally with bounded memory:
          1. Stream frames without retaining 4K bitmaps in RAM.
          2. Reset tracker state between videos.
          3. Pass BGR directly to model.track (Ultralytics handles flip).
          4. Maintain bounded evidence buffers per active track.
          5. Write evidence and assemble CandidateEvents.
        """
        self.validate_checkpoint(model)

        cfg_path = self._write_tracker_config()

        # Clean tracker reset
        model.predictor = None

        track_obs: dict[int, list[ObservedDetection]] = {}
        # track_id -> dict of candidate observation info:
        # {obs_index: (obs, crop_bgr, full_frame_jpg_bytes)}
        track_candidates: dict[int, list[tuple[ObservedDetection, np.ndarray, bytes]]] = {}

        for frame_idx, (fi, ts_ms, bgr) in enumerate(frame_iter):
            # Pass BGR directly to model.track (BasePredictor preprocess flips BGR to RGB)
            results = model.track(
                bgr,
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
                    if cls_id != self.target_class_id:
                        continue
                    box_conf = float(box.conf[0].item())
                    track_id = int(box.id[0].item())
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                    # Only record as actual observation if conf >= threshold
                    if box_conf >= conf:
                        crop_bgr = extract_padded_crop(bgr, (x1, y1, x2, y2), pad_frac=DEFAULT_PAD_FRAC)
                        lap_var = compute_laplacian_variance(crop_bgr)

                        obs = ObservedDetection(
                            frame_index=fi,
                            timestamp_ms=ts_ms,
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            conf=box_conf,
                            laplacian_var=lap_var,
                        )
                        track_obs.setdefault(track_id, []).append(obs)

                        # Bounded evidence: keep up to top 3 confidence candidates for this track
                        cands = track_candidates.setdefault(track_id, [])
                        if len(cands) < self.max_candidate_obs_for_rep or box_conf > min(c[0].conf for c in cands):
                            # Compress full frame as JPEG to bound memory
                            annotated = bgr.copy()
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 3)
                            lbl = f"pothole {box_conf:.2f}"
                            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                            cv2.rectangle(annotated, (x1, max(0, y1 - th - 12)), (x1 + tw + 8, y1), (0, 200, 80), -1)
                            cv2.putText(annotated, lbl, (x1 + 4, max(0, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                            _, jpg_bytes = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])

                            cands.append((obs, crop_bgr, jpg_bytes.tobytes()))
                            # Keep only top 3 by confidence
                            cands.sort(key=lambda t: t[0].conf, reverse=True)
                            if len(cands) > self.max_candidate_obs_for_rep:
                                cands.pop()

            if progress_callback:
                progress_callback(frame_idx + 1, total_expected_frames)

        # Assemble CandidateEvent objects
        events: list[CandidateEvent] = []
        tracker_cfg_snapshot = self._tracker_config_snapshot()
        ev_root = Path(evidence_root) if evidence_root else None

        for track_id, obs_list in track_obs.items():
            if not obs_list:
                continue
            obs_list.sort(key=lambda o: o.timestamp_ms)
            confs = [o.conf for o in obs_list]
            event_id = str(uuid.uuid4())[:8]

            best_obs = select_representative_observation(obs_list)
            if best_obs is None:
                best_obs = obs_list[0]

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
                rep_x1=best_obs.x1,
                rep_y1=best_obs.y1,
                rep_x2=best_obs.x2,
                rep_y2=best_obs.y2,
                rep_timestamp_ms=best_obs.timestamp_ms,
                rep_frame_index=best_obs.frame_index,
                observations=obs_list,
                tracker_config=tracker_cfg_snapshot,
            )

            # Save evidence files if root provided
            if ev_root:
                ev_dir = ev_root / self.video_id / event_id
                ev_dir.mkdir(parents=True, exist_ok=True)

                # Find candidate match
                cands = track_candidates.get(track_id, [])
                matching_cand = next((c for c in cands if c[0].frame_index == best_obs.frame_index), None)
                if matching_cand is None and cands:
                    matching_cand = cands[0]

                if matching_cand is not None:
                    _, crop_bgr, full_jpg = matching_cand

                    crop_path = ev_dir / "crop.jpg"
                    cv2.imwrite(str(crop_path), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    event.crop_path = str(crop_path)

                    full_path = ev_dir / "full_frame.jpg"
                    full_path.write_bytes(full_jpg)
                    event.full_frame_path = str(full_path)

                # Generate short evidence clip if requested and video exists
                if generate_clips and video_path and Path(video_path).exists():
                    clip_path = ev_dir / "clip.mp4"
                    success = self._extract_evidence_clip(
                        video_path=Path(video_path),
                        target_ts_ms=best_obs.timestamp_ms,
                        output_clip_path=clip_path,
                        window_sec=1.5,
                    )
                    if success:
                        event.clip_path = str(clip_path)

            # Apply temporal Policy B check
            temporal_pass = check_temporal_window(
                obs_list,
                min_obs=self.policy_b_min_obs,
                window_size_frames=self.policy_b_window_frames,
                conf_threshold=conf,
            )
            event.policy_a_retained = event.detector_conf_max >= conf
            event.policy_b_retained = event.policy_a_retained and temporal_pass

            events.append(event)

        return events

    def _extract_evidence_clip(
        self,
        video_path: Path,
        target_ts_ms: int,
        output_clip_path: Path,
        window_sec: float = 1.5,
    ) -> bool:
        """Extract a short MP4 evidence clip centered around target_ts_ms."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            start_ms = max(0, target_ts_ms - int(window_sec * 1000))
            end_ms = target_ts_ms + int(window_sec * 1000)

            cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_clip_path), fourcc, fps, (width, height))

            while cap.isOpened():
                curr_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                ok, frame = cap.read()
                if not ok or curr_ms > end_ms:
                    break
                out.write(frame)

            cap.release()
            out.release()
            return output_clip_path.exists() and output_clip_path.stat().st_size > 0
        except Exception as exc:
            logger.warning("Could not extract clip for %s: %s", output_clip_path, exc)
            return False

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
        """Backwards-compatible API accepting a list of frames."""
        frame_iter = ((fi, ts, bgr) for fi, ts, bgr, _ in frames_and_detections)
        return self.process_video_stream(
            frame_iter=frame_iter,
            model=model,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            total_expected_frames=len(frames_and_detections),
        )

    def _write_tracker_config(self) -> str:
        """Write a BoT-SORT YAML config with CMC and ReID disabled; return path."""
        import tempfile, yaml

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
        yaml.dump(cfg, tmp)
        tmp.close()
        self._tmp_cfg_path = tmp.name
        return tmp.name

    def _tracker_config_snapshot(self) -> dict:
        return {
            "tracker": "botsort",
            "gmc_method": "none",
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
        track_evidence: dict[int, list] | None = None,
        clip_duration_ms: int = 2000,
        source_fps: float = 30.0,
    ) -> None:
        """For each event, write full-frame JPEG, 15% padded crop JPEG, and update representative box."""
        frame_lookup: dict[int, np.ndarray] = {
            fi: bgr for fi, _, bgr, _ in frames_and_detections
        }

        for event in events:
            ev_dir = self.evidence_root / event.video_id / event.event_id
            ev_dir.mkdir(parents=True, exist_ok=True)

            best_obs = select_representative_observation(event.observations)
            if best_obs is None:
                continue

            frame_bgr = frame_lookup.get(best_obs.frame_index)
            if frame_bgr is not None:
                # Full-frame JPEG
                full_path = ev_dir / "full_frame.jpg"
                cv2.imwrite(str(full_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                event.full_frame_path = str(full_path)

                # Crop with standardized 15% context padding clamped to borders
                crop_bgr = extract_padded_crop(
                    frame_bgr,
                    (best_obs.x1, best_obs.y1, best_obs.x2, best_obs.y2),
                    pad_frac=DEFAULT_PAD_FRAC,
                )
                crop_path = ev_dir / "crop.jpg"
                cv2.imwrite(str(crop_path), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                event.crop_path = str(crop_path)

                event.rep_x1 = best_obs.x1
                event.rep_y1 = best_obs.y1
                event.rep_x2 = best_obs.x2
                event.rep_y2 = best_obs.y2
                event.rep_timestamp_ms = best_obs.timestamp_ms
                event.rep_frame_index = best_obs.frame_index

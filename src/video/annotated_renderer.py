"""Complete annotated video renderer for pothole detection and tracking.

Renders full-frame road videos frame-by-frame with bounding boxes,
confidence scores, and BoT-SORT tracking IDs synchronized to original footage.
Produces browser-compatible H.264 MP4 videos with bounded memory execution.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Ensure Ultralytics config stays inside temporary directory
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

logger = logging.getLogger(__name__)

# Default managed directory for annotated videos
DEFAULT_OUTPUT_DIR = Path("results/annotated_videos")

# Standard bounding-box styling (no emojis, professional green)
BOX_COLOR_BGR = (0, 200, 80)        # Distinct bright green in BGR
TEXT_COLOR_BGR = (255, 255, 255)    # High-contrast white


@dataclass
class RenderProgress:
    """Real-time progress telemetry emitted during video rendering."""
    current_frame: int
    total_frames: int
    percent: float               # 0.0 to 100.0
    elapsed_sec: float
    current_fps: float
    eta_sec: Optional[float]
    status_text: str
    detections_count: int


@dataclass
class RenderMetadata:
    """Persistent metadata record for a completed video rendering job."""
    job_id: str
    video_id: str
    input_video_path: str
    output_video_path: str
    model_path: str
    model_checkpoint: str
    detector_conf: float
    iou_threshold: float
    imgsz: int
    source_fps: float
    source_frame_count: int
    source_width: int
    source_height: int
    source_duration_sec: float
    processed_frame_count: int
    output_fps: float
    output_frame_count: int
    output_width: int
    output_height: int
    processing_duration_sec: float
    effective_inference_fps: float
    status: str                  # "completed" | "failed"
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_detections_drawn: int = 0
    unique_track_ids: list[int] = field(default_factory=list)
    ffmpeg_transcoded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, destination: Union[str, Path]) -> Path:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.to_dict(), indent=2))
        return dest


def resolve_device(device: str = "auto") -> str:
    """Resolve compute device following priority: MPS -> CUDA -> CPU."""
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


def is_ffmpeg_available() -> bool:
    """Check whether FFmpeg binary is installed and executable in PATH."""
    bin_path = shutil.which("ffmpeg")
    if not bin_path:
        return False
    try:
        result = subprocess.run([bin_path, "-version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def transcode_to_h264_browser(raw_video_path: Path, output_video_path: Path) -> bool:
    """Transcode raw MP4 to web-optimized H.264 (yuv420p, faststart) using FFmpeg.

    Returns True if transcode succeeded, False otherwise.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(raw_video_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.warning("FFmpeg transcoding failed (code %d): %s", result.returncode, result.stderr)
            return False
        return True
    except Exception as exc:
        logger.warning("FFmpeg transcoding invocation error: %s", exc)
        return False


def validate_rendered_video(video_path: Path, expected_frames: Optional[int] = None) -> dict[str, Any]:
    """Validate that rendered video exists, has non-zero size, and can be read by OpenCV."""
    if not video_path.exists():
        raise FileNotFoundError(f"Rendered video does not exist: {video_path}")
    size = video_path.stat().st_size
    if size <= 0:
        raise ValueError(f"Rendered video file is empty (0 bytes): {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open rendered video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if width <= 0 or height <= 0:
            raise ValueError(f"Rendered video has invalid dimensions: {width}x{height}")
        if frames <= 0:
            raise ValueError(f"Rendered video reports 0 frames: {video_path}")

        return {
            "fps": fps,
            "frame_count": frames,
            "width": width,
            "height": height,
            "file_size_bytes": size,
        }
    finally:
        cap.release()


def draw_detection_box(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    conf: float,
    track_id: Optional[int] = None,
    color: tuple[int, int, int] = BOX_COLOR_BGR,
) -> None:
    """Draw a clean, scaled pothole bounding box and high-contrast label directly on frame.

    Format: 'Pothole 0.94 | ID 3' (with track ID) or 'Pothole 0.94' (without track ID).
    No emojis. All coordinates are clamped within image boundaries.
    """
    h, w = frame.shape[:2]
    # Clamp coordinates to frame boundaries
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))

    if x2 <= x1 or y2 <= y1:
        return

    # Scaling relative to resolution: 1080p ~ 3px, 4k ~ 6px, 720p ~ 2px
    min_dim = min(h, w)
    thickness = max(2, int(round(min_dim / 360.0)))
    font_scale = max(0.45, min_dim / 1150.0)
    font_thickness = max(1, int(round(thickness * 0.6)))

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Format label text
    if track_id is not None:
        label = f"Pothole {conf:.2f} | ID {track_id}"
    else:
        label = f"Pothole {conf:.2f}"

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    pad_x = max(4, int(round(min_dim / 300.0)))
    pad_y = max(3, int(round(min_dim / 400.0)))

    # Position label banner above box if space permits, else inside top edge
    banner_height = th + (pad_y * 2)
    if y1 - banner_height >= 0:
        banner_y1 = y1 - banner_height
        banner_y2 = y1
        text_y = y1 - pad_y - (baseline // 2)
    else:
        banner_y1 = y1
        banner_y2 = y1 + banner_height
        text_y = banner_y2 - pad_y - (baseline // 2)

    banner_x1 = max(0, min(x1, w - 1))
    banner_x2 = min(w - 1, banner_x1 + tw + (pad_x * 2))
    if banner_x2 == w - 1:
        banner_x1 = max(0, banner_x2 - tw - (pad_x * 2))

    # Fill solid background banner
    cv2.rectangle(frame, (banner_x1, banner_y1), (banner_x2, banner_y2), color, -1)

    # Render crisp white text
    text_x = banner_x1 + pad_x
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        TEXT_COLOR_BGR,
        font_thickness,
        lineType=cv2.LINE_AA,
    )


class AnnotatedVideoRenderer:
    """Processes video footage frame-by-frame, applying YOLO detections & BoT-SORT tracking.

    Maintains bounded memory by streaming frames incrementally, writing
    annotated frames directly to disk, and transcoding to browser-ready H.264 MP4.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        track_high_thresh: float = 0.3,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # BoT-SORT tracker configuration (camera-motion compensation and ReID disabled)
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self._tmp_tracker_cfg: Optional[str] = None

    def validate_checkpoint(self, model: Any, target_class_id: int = 0) -> None:
        """Confirm class 0 is 'pothole' in checkpoint; reject incompatible base models."""
        names = getattr(model, "names", {})
        if not names:
            return
        cls_name = names.get(target_class_id, "")
        if "pothole" not in str(cls_name).lower():
            pothole_classes = [cid for cid, cname in names.items() if "pothole" in str(cname).lower()]
            sample_classes = {cid: names[cid] for cid in list(names.keys())[:5]}
            raise ValueError(
                f"Incompatible YOLO checkpoint: class {target_class_id} is '{cls_name}', not 'pothole'. "
                f"(Loaded classes: {sample_classes}...). "
                + (f"Pothole exists at class {pothole_classes[0]}." if pothole_classes else
                   "This appears to be an un-fine-tuned base checkpoint (such as COCO yolo11n.pt). "
                   "Please supply a fine-tuned pothole detector checkpoint (e.g. models/best.pt).")
            )

    def _write_tracker_config(self) -> str:
        """Write a clean BoT-SORT YAML config matching project specifications."""
        import yaml
        cfg = {
            "tracker_type": "botsort",
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "new_track_thresh": self.new_track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "fuse_score": True,
            "gmc_method": "none",
            "with_reid": False,
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.8,
            "model": "auto",
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="botsort_vis_")
        yaml.dump(cfg, tmp)
        tmp.close()
        self._tmp_tracker_cfg = tmp.name
        return tmp.name

    def render_annotated_video(
        self,
        video_path: Union[str, Path],
        model: Any,
        output_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "auto",
        target_class_id: int = 0,
        use_tracking: bool = True,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[RenderProgress], None]] = None,
    ) -> RenderMetadata:
        """Process video frame-by-frame and produce a complete annotated MP4.

        Parameters
        ----------
        video_path        : Path to source video (.mp4, .mov, etc.)
        model             : Ultralytics YOLO model instance or path to .pt file
        output_path       : Optional custom destination path for output MP4
        conf_threshold    : Minimum detector confidence to draw bounding box
        iou_threshold     : NMS IoU threshold for detection
        imgsz             : Inference image size
        device            : "auto" | "mps" | "cuda" | "cpu"
        target_class_id   : Target class index (0 = pothole)
        use_tracking      : Whether to apply BoT-SORT tracking and display track IDs
        max_frames        : Optional frame limit for testing or short previews
        progress_callback : Callback receiving RenderProgress updates
        """
        source_path = Path(video_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Input video file not found: {source_path}")

        # Open source video
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise IOError(f"Could not open input video with OpenCV: {source_path}")

        try:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            raw_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            if source_fps <= 0 or source_width <= 0 or source_height <= 0:
                raise ValueError(
                    f"Video metadata invalid: fps={source_fps}, dimensions={source_width}x{source_height}"
                )

            source_duration_sec = (raw_total_frames / source_fps) if source_fps > 0 else 0.0
            total_target_frames = min(raw_total_frames, max_frames) if max_frames else raw_total_frames
        finally:
            cap.release()

        # Resolve model instance
        if isinstance(model, (str, Path)):
            m_path = Path(model)
            if not m_path.exists():
                raise FileNotFoundError(f"Model checkpoint file not found: {m_path}")
            from ultralytics import YOLO
            actual_model = YOLO(str(m_path))
            model_checkpoint_name = m_path.name
            model_path_str = str(m_path)
        else:
            actual_model = model
            ckpt_attr = getattr(actual_model, "ckpt_path", None)
            if isinstance(ckpt_attr, (str, Path)):
                model_checkpoint_name = str(Path(ckpt_attr).name)
                model_path_str = str(ckpt_attr)
            else:
                name_attr = getattr(actual_model, "model_name", None)
                if isinstance(name_attr, str):
                    model_checkpoint_name = str(Path(name_attr).name)
                    model_path_str = name_attr
                else:
                    model_checkpoint_name = "yolo_model"
                    model_path_str = "yolo_model"

        # Validate checkpoint
        self.validate_checkpoint(actual_model, target_class_id=target_class_id)

        # Reset predictor state between videos to prevent stale tracking IDs
        if hasattr(actual_model, "predictor") and actual_model.predictor is not None:
            actual_model.predictor = None

        resolved_device = resolve_device(device)

        # Setup collision-safe unique job and output paths
        video_id = source_path.stem.replace(" ", "_")
        timestamp_int = int(time.time())
        unique_token = uuid.uuid4().hex[:8]
        job_id = f"annotated_{video_id}_{timestamp_int}_{unique_token}"

        if output_path:
            final_output_path = Path(output_path)
        else:
            final_output_path = self.output_dir / f"{video_id}_annotated_{timestamp_int}_{unique_token[:6]}.mp4"
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

        tracker_cfg_path = self._write_tracker_config() if use_tracking else None

        # Create temporary raw output video file
        temp_raw_file = tempfile.NamedTemporaryFile(suffix=".raw.mp4", delete=False, prefix="ann_raw_")
        temp_raw_path = Path(temp_raw_file.name)
        temp_raw_file.close()

        # Initialize VideoWriter
        # Prefer 'avc1' (H.264) for MP4; fallback to 'mp4v' if unavailable
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(temp_raw_path), fourcc, source_fps, (source_width, source_height))
        if not writer.isOpened():
            logger.info("Codec 'avc1' not opened; falling back to 'mp4v'")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temp_raw_path), fourcc, source_fps, (source_width, source_height))
            if not writer.isOpened():
                temp_raw_path.unlink(missing_ok=True)
                raise RuntimeError("Failed to initialize OpenCV VideoWriter for output video.")

        # Re-open capture for streaming frames
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            temp_raw_path.unlink(missing_ok=True)
            raise IOError(f"Could not re-open video for streaming: {source_path}")

        start_time = time.time()
        processed_frames = 0
        total_detections_drawn = 0
        seen_track_ids: set[int] = set()

        try:
            while True:
                if max_frames and processed_frames >= max_frames:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                # Frame-by-frame inference:
                # Passes BGR ndarray directly to model.track or model.predict
                if use_tracking:
                    results = actual_model.track(
                        frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        imgsz=imgsz,
                        device=resolved_device,
                        tracker=tracker_cfg_path,
                        persist=True,
                        verbose=False,
                    )
                else:
                    results = actual_model.predict(
                        frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        imgsz=imgsz,
                        device=resolved_device,
                        verbose=False,
                    )

                frame_detections = 0
                for r in results:
                    if r.boxes is None or len(r.boxes) == 0:
                        continue
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                        if cls_id != target_class_id:
                            continue
                        b_conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                        if b_conf < conf_threshold:
                            continue

                        # Extract track ID if present
                        track_id = None
                        if hasattr(boxes, "id") and boxes.id is not None and len(boxes.id) > 0:
                            raw_id = box.id[0].item() if hasattr(box.id[0], "item") else box.id[0]
                            if raw_id is not None:
                                track_id = int(raw_id)
                                seen_track_ids.add(track_id)

                        # Extract bounding box coordinates
                        xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                        x1, y1, x2, y2 = xyxy

                        draw_detection_box(
                            frame=frame,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            conf=b_conf,
                            track_id=track_id,
                        )
                        frame_detections += 1
                        total_detections_drawn += 1

                writer.write(frame)
                processed_frames += 1

                # Emit progress telemetry
                if progress_callback:
                    elapsed = time.time() - start_time
                    fps_val = (processed_frames / elapsed) if elapsed > 0 else 0.0
                    tot = total_target_frames or processed_frames
                    pct = min(100.0, (processed_frames / tot) * 100.0) if tot > 0 else 0.0
                    rem = max(0, tot - processed_frames)
                    eta = (rem / fps_val) if fps_val > 0 else None
                    progress_callback(
                        RenderProgress(
                            current_frame=processed_frames,
                            total_frames=tot,
                            percent=pct,
                            elapsed_sec=elapsed,
                            current_fps=fps_val,
                            eta_sec=eta,
                            status_text=f"Rendered {processed_frames}/{tot} frames ({pct:.0f}%)",
                            detections_count=total_detections_drawn,
                        )
                    )

        except Exception as exc:
            logger.exception("Exception during video rendering loop: %s", exc)
            # Cleanup on failure
            if writer:
                writer.release()
            if cap:
                cap.release()
            temp_raw_path.unlink(missing_ok=True)
            if self._tmp_tracker_cfg and Path(self._tmp_tracker_cfg).exists():
                Path(self._tmp_tracker_cfg).unlink(missing_ok=True)
            raise RuntimeError(f"Video rendering failed at frame {processed_frames}: {exc}") from exc
        finally:
            cap.release()
            writer.release()
            if self._tmp_tracker_cfg and Path(self._tmp_tracker_cfg).exists():
                Path(self._tmp_tracker_cfg).unlink(missing_ok=True)

        processing_duration = time.time() - start_time
        effective_fps = (processed_frames / processing_duration) if processing_duration > 0 else 0.0

        # Transcode with FFmpeg to H.264/yuv420p/faststart for browser playback
        ffmpeg_transcoded = False
        if is_ffmpeg_available():
            logger.info("Transcoding raw video with FFmpeg to %s", final_output_path)
            transcode_ok = transcode_to_h264_browser(temp_raw_path, final_output_path)
            if transcode_ok:
                ffmpeg_transcoded = True
                temp_raw_path.unlink(missing_ok=True)
            else:
                logger.warning("FFmpeg transcoding failed; moving raw output directly")
                shutil.move(str(temp_raw_path), str(final_output_path))
        else:
            logger.info("FFmpeg not installed; moving raw video directly to %s", final_output_path)
            shutil.move(str(temp_raw_path), str(final_output_path))

        # Validate final output video
        output_info = validate_rendered_video(final_output_path)

        metadata = RenderMetadata(
            job_id=job_id,
            video_id=video_id,
            input_video_path=str(source_path.resolve()),
            output_video_path=str(final_output_path.resolve()),
            model_path=str(model_path_str),
            model_checkpoint=model_checkpoint_name,
            detector_conf=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
            source_fps=source_fps,
            source_frame_count=raw_total_frames,
            source_width=source_width,
            source_height=source_height,
            source_duration_sec=source_duration_sec,
            processed_frame_count=processed_frames,
            output_fps=output_info["fps"],
            output_frame_count=output_info["frame_count"],
            output_width=output_info["width"],
            output_height=output_info["height"],
            processing_duration_sec=round(processing_duration, 2),
            effective_inference_fps=round(effective_fps, 2),
            status="completed",
            total_detections_drawn=total_detections_drawn,
            unique_track_ids=sorted(seen_track_ids),
            ffmpeg_transcoded=ffmpeg_transcoded,
        )

        # Write companion JSON metadata
        json_meta_path = final_output_path.with_suffix(".json")
        metadata.save_json(json_meta_path)
        logger.info(
            "Annotated video successfully rendered to %s (frames: %d, detections: %d, time: %.1fs)",
            final_output_path,
            processed_frames,
            total_detections_drawn,
            processing_duration,
        )

        return metadata

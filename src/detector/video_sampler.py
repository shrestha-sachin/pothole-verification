"""Video frame sampler — yields (frame_index, timestamp_ms, bgr_frame) tuples.

Preserves original video timestamps; does not modify the source file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def iter_frames(
    video_path: str | Path,
    target_fps: float = 5.0,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> Generator[tuple[int, int, np.ndarray], None, None]:
    """
    Yield (frame_index, timestamp_ms, bgr_frame) at approximately target_fps.

    Parameters
    ----------
    video_path  : path to MP4 (or any OpenCV-readable format)
    target_fps  : sampling rate in frames per second (default 5.0)
    start_sec   : skip frames before this timestamp
    end_sec     : stop after this timestamp (None = process entire video)

    Yields
    ------
    (frame_index, timestamp_ms, bgr_frame)
      frame_index  — sequential counter of sampled frames (0-based)
      timestamp_ms — original video position in milliseconds
      bgr_frame    — full-resolution BGR ndarray from OpenCV
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    try:
        source_fps: float = cap.get(cv2.CAP_PROP_FPS)
        total_raw: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if source_fps <= 0:
            raise ValueError(f"Video reports invalid FPS={source_fps}. Try re-encoding with ffmpeg.")

        interval_sec = 1.0 / target_fps
        next_sample_time = max(0.0, start_sec)

        raw_frame_index = 0
        sampled_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_sec = raw_frame_index / source_fps
            timestamp_ms = int(round(timestamp_sec * 1000))

            if end_sec is not None and timestamp_sec > end_sec:
                break

            # Skip until we reach start_sec
            if timestamp_sec < start_sec:
                raw_frame_index += 1
                continue

            # Sample at target_fps cadence
            if timestamp_sec + (0.5 / source_fps) >= next_sample_time:
                yield sampled_index, timestamp_ms, frame
                sampled_index += 1
                next_sample_time += interval_sec

            raw_frame_index += 1

    finally:
        cap.release()


def get_video_metadata(video_path: str | Path) -> dict:
    """Return basic metadata for a video file without extracting frames."""
    path = Path(video_path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": f"Cannot open: {path}"}

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
        duration_sec = frame_count / fps if fps > 0 else 0.0
        return {
            "path": str(path),
            "filename": path.name,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "fourcc": fourcc_str.strip("\x00"),
            "duration_sec": duration_sec,
            "duration_min": duration_sec / 60.0,
            "file_size_mb": path.stat().st_size / (1024 * 1024),
        }
    finally:
        cap.release()

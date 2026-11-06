"""Video rendering and visualization module for pothole detection."""

from src.video.annotated_renderer import (
    AnnotatedVideoRenderer,
    RenderMetadata,
    RenderProgress,
    draw_detection_box,
    is_ffmpeg_available,
    transcode_to_h264_browser,
    validate_rendered_video,
)

__all__ = [
    "AnnotatedVideoRenderer",
    "RenderMetadata",
    "RenderProgress",
    "draw_detection_box",
    "is_ffmpeg_available",
    "transcode_to_h264_browser",
    "validate_rendered_video",
]


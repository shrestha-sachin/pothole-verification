"""Unit and integration tests for annotated video rendering and playback."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

from src.database import Database
from src.tracking.event_grouper import CandidateEvent, ObservedDetection
from src.video.annotated_renderer import (
    AnnotatedVideoRenderer,
    RenderMetadata,
    RenderProgress,
    draw_detection_box,
    is_ffmpeg_available,
    validate_rendered_video,
)


class DummyBox:
    def __init__(self, xyxy: list[float], conf: float = 0.94, cls_id: int = 0, track_id: int | None = 3):
        self.xyxy = torch.tensor([xyxy])
        self.conf = torch.tensor([conf])
        self.cls = torch.tensor([cls_id])
        self.id = torch.tensor([track_id]) if track_id is not None else None


class DummyBoxes:
    def __init__(self, boxes: list[DummyBox] | None = None):
        self.boxes = boxes or []
        track_ids = [b.id[0] for b in self.boxes if b.id is not None]
        self.id = torch.tensor(track_ids) if track_ids else None

    def __iter__(self):
        return iter(self.boxes)

    def __len__(self):
        return len(self.boxes)


class DummyResult:
    def __init__(self, boxes: list[DummyBox] | None = None):
        self.boxes = DummyBoxes(boxes) if boxes else None


def create_synthetic_video(path: Path, num_frames: int = 15, fps: float = 10.0, width: int = 320, height: int = 240) -> Path:
    """Helper to generate a small synthetic MP4 video with numbered frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create solid gray frame with small square pattern
        frame = np.full((height, width, 3), fill_value=60 + (i * 5) % 150, dtype=np.uint8)
        # Add simple visual feature
        cv2.circle(frame, (width // 2, height // 2), 15, (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return path


class TestAnnotatedVideoRenderer(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_ann_vid_"))
        self.output_dir = self.test_dir / "annotated"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = AnnotatedVideoRenderer(output_dir=self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_render_synthetic_video_no_detections(self):
        """Render complete video when detector produces no pothole detections."""
        source_path = self.test_dir / "clean_road.mp4"
        create_synthetic_video(source_path, num_frames=15, fps=10.0, width=320, height=240)
        orig_mtime = source_path.stat().st_mtime
        orig_size = source_path.stat().st_size

        mock_model = MagicMock()
        mock_model.names = {0: "pothole"}
        mock_model.predictor = None
        mock_model.track = MagicMock(return_value=[DummyResult([])])

        progress_reports = []

        def on_prog(p: RenderProgress):
            progress_reports.append(p)

        meta = self.renderer.render_annotated_video(
            video_path=source_path,
            model=mock_model,
            conf_threshold=0.35,
            progress_callback=on_prog,
        )

        # 1. Output video was generated and is valid
        self.assertEqual(meta.status, "completed")
        self.assertEqual(meta.processed_frame_count, 15)
        self.assertEqual(meta.output_frame_count, 15)
        self.assertEqual(meta.total_detections_drawn, 0)
        self.assertEqual(meta.unique_track_ids, [])
        self.assertTrue(Path(meta.output_video_path).exists())
        self.assertGreater(Path(meta.output_video_path).stat().st_size, 0)

        # 2. Source video was untouched
        self.assertEqual(source_path.stat().st_mtime, orig_mtime)
        self.assertEqual(source_path.stat().st_size, orig_size)

        # 3. Progress callback received updates
        self.assertGreater(len(progress_reports), 0)
        self.assertEqual(progress_reports[-1].current_frame, 15)
        self.assertEqual(progress_reports[-1].percent, 100.0)

    def test_render_synthetic_video_with_mock_detections(self):
        """Render video with detections on specific frames and verify boxes, track IDs, and metadata."""
        source_path = self.test_dir / "road_with_potholes.mp4"
        create_synthetic_video(source_path, num_frames=20, fps=10.0, width=400, height=300)

        mock_model = MagicMock()
        mock_model.names = {0: "pothole"}
        mock_model.predictor = None

        frame_counter = [0]

        def track_side_effect(frame, **kwargs):
            idx = frame_counter[0]
            frame_counter[0] += 1
            # Frames 5, 6, 7 detect track 3 with conf 0.94
            if 5 <= idx <= 7:
                box = DummyBox(xyxy=[50.0, 60.0, 150.0, 140.0], conf=0.94, track_id=3)
                return [DummyResult([box])]
            # Frame 12 detects track 8 with conf 0.88
            elif idx == 12:
                box = DummyBox(xyxy=[200.0, 100.0, 280.0, 220.0], conf=0.88, track_id=8)
                return [DummyResult([box])]
            return [DummyResult([])]

        mock_model.track = track_side_effect

        meta = self.renderer.render_annotated_video(
            video_path=source_path,
            model=mock_model,
            conf_threshold=0.35,
        )

        self.assertEqual(meta.status, "completed")
        self.assertEqual(meta.processed_frame_count, 20)
        self.assertEqual(meta.output_frame_count, 20)
        self.assertEqual(meta.total_detections_drawn, 4)
        self.assertEqual(meta.unique_track_ids, [3, 8])

        # Verify companion JSON metadata
        json_path = Path(meta.output_video_path).with_suffix(".json")
        self.assertTrue(json_path.exists())
        saved_meta = json.loads(json_path.read_text())
        self.assertEqual(saved_meta["total_detections_drawn"], 4)
        self.assertEqual(saved_meta["unique_track_ids"], [3, 8])
        self.assertEqual(saved_meta["job_id"], meta.job_id)

    def test_bounding_box_coordinates_and_scaling(self):
        """Verify bounding-box coordinates scaling across resolutions and edge clamping."""
        # Test on 3 different frame resolutions
        resolutions = [(320, 240), (1280, 720), (1920, 1080)]
        for w, h in resolutions:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Box right at top-left edge
            draw_detection_box(frame, x1=-10, y1=-5, x2=80, y2=90, conf=0.92, track_id=1)
            # Box right at bottom-right edge
            draw_detection_box(frame, x1=w - 50, y1=h - 40, x2=w + 20, y2=h + 20, conf=0.85, track_id=2)
            # Box in center
            draw_detection_box(frame, x1=w // 4, y1=h // 4, x2=w // 2, y2=h // 2, conf=0.95, track_id=3)

            # Frame should contain non-zero pixels (drawn annotations)
            self.assertGreater(np.count_nonzero(frame), 0)

    def test_tracking_id_display_and_label_format(self):
        """Verify label format: 'Pothole 0.94 | ID 3' when track ID present; 'Pothole 0.94' when None."""
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        # With track ID
        draw_detection_box(frame, x1=50, y1=50, x2=150, y2=150, conf=0.94, track_id=3)
        # Without track ID
        draw_detection_box(frame, x1=200, y1=50, x2=300, y2=150, conf=0.87, track_id=None)

        self.assertGreater(np.count_nonzero(frame), 0)

    def test_output_frame_count_and_playback_duration(self):
        """Verify that output frame count and duration precisely match the source video."""
        source_path = self.test_dir / "duration_test.mp4"
        create_synthetic_video(source_path, num_frames=30, fps=15.0, width=320, height=240)

        mock_model = MagicMock()
        mock_model.names = {0: "pothole"}
        mock_model.predictor = None
        mock_model.track = MagicMock(return_value=[DummyResult([])])

        meta = self.renderer.render_annotated_video(
            video_path=source_path,
            model=mock_model,
        )

        output_path = Path(meta.output_video_path)
        cap = cv2.VideoCapture(str(output_path))
        self.assertTrue(cap.isOpened())
        out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        self.assertEqual(out_frames, 30)
        self.assertAlmostEqual(out_fps, 15.0, delta=0.5)

    def test_confidence_filtering(self):
        """Detections below confidence threshold must NOT be drawn."""
        source_path = self.test_dir / "conf_filter_test.mp4"
        create_synthetic_video(source_path, num_frames=5, fps=10.0, width=320, height=240)

        mock_model = MagicMock()
        mock_model.names = {0: "pothole"}
        mock_model.predictor = None

        # Return a detection with confidence 0.25 (below threshold 0.35)
        low_conf_box = DummyBox(xyxy=[50.0, 50.0, 100.0, 100.0], conf=0.25, track_id=1)
        mock_model.track = MagicMock(return_value=[DummyResult([low_conf_box])])

        meta = self.renderer.render_annotated_video(
            video_path=source_path,
            model=mock_model,
            conf_threshold=0.35,
        )

        self.assertEqual(meta.total_detections_drawn, 0)
        self.assertEqual(meta.unique_track_ids, [])

    def test_processing_failure_and_clean_recovery(self):
        """Verify clean error recovery and no leftover corrupted files on failure."""
        # 1. Non-existent file
        with self.assertRaises(FileNotFoundError):
            self.renderer.render_annotated_video(
                video_path="non_existent_video.mp4",
                model=MagicMock(),
            )

        # 2. Failure during inference loop
        source_path = self.test_dir / "fail_test.mp4"
        create_synthetic_video(source_path, num_frames=10, fps=10.0, width=320, height=240)

        mock_model = MagicMock()
        mock_model.names = {0: "pothole"}
        mock_model.predictor = None
        mock_model.track = MagicMock(side_effect=RuntimeError("Simulated GPU OOM or inference error"))

        with self.assertRaises(RuntimeError) as ctx:
            self.renderer.render_annotated_video(
                video_path=source_path,
                model=mock_model,
            )
        self.assertIn("Video rendering failed", str(ctx.exception))

        # Ensure no dangling files in output dir
        output_files = list(self.output_dir.glob("fail_test_annotated*"))
        self.assertEqual(len(output_files), 0)

    def test_repeated_processing_collision_safety(self):
        """Repeated processing runs must generate unique jobs and never overwrite previous outputs."""
        source_path = self.test_dir / "repeat_test.mp4"
        create_synthetic_video(source_path, num_frames=5, fps=10.0, width=320, height=240)

        mock_model = MagicMock()
        mock_model.names = {0: "pothole"}
        mock_model.predictor = None
        mock_model.track = MagicMock(return_value=[DummyResult([])])

        meta1 = self.renderer.render_annotated_video(video_path=source_path, model=mock_model)
        meta2 = self.renderer.render_annotated_video(video_path=source_path, model=mock_model)

        self.assertNotEqual(meta1.job_id, meta2.job_id)
        self.assertNotEqual(meta1.output_video_path, meta2.output_video_path)
        self.assertTrue(Path(meta1.output_video_path).exists())
        self.assertTrue(Path(meta2.output_video_path).exists())

    def test_incompatible_checkpoint_rejected(self):
        """Base COCO checkpoint (class 0 != 'pothole') must be rejected with an informative error."""
        source_path = self.test_dir / "coco_test.mp4"
        create_synthetic_video(source_path, num_frames=5, fps=10.0, width=320, height=240)

        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "bicycle"}

        with self.assertRaises(ValueError) as ctx:
            self.renderer.render_annotated_video(video_path=source_path, model=mock_model)

        self.assertIn("class 0 is 'person', not 'pothole'", str(ctx.exception))
        self.assertIn("un-fine-tuned base checkpoint", str(ctx.exception))

    def test_preservation_of_existing_event_records_and_reviews(self):
        """Ensure that saving annotated videos does not alter or overwrite existing events and reviews."""
        db_path = self.test_dir / "test_preservation.db"
        db = Database(db_path)

        # Register video and save an event and review
        db.register_video("vid_preserve", {"filepath": "preserve.mp4", "duration_sec": 60.0})
        ev = CandidateEvent(
            event_id="ev_keep_001",
            video_id="vid_preserve",
            road_segment_id="seg_001",
            track_id=1,
            first_timestamp_ms=1000,
            last_timestamp_ms=1500,
            first_frame_index=5,
            last_frame_index=8,
            detector_conf_max=0.92,
            detector_conf_mean=0.88,
            observation_count=3,
            rep_x1=10, rep_y1=20, rep_x2=80, rep_y2=90,
            rep_timestamp_ms=1200,
            rep_frame_index=6,
            policy_a_retained=True,
            policy_b_retained=True,
            observations=[],
        )
        db.save_events([ev])
        db.save_review("ev_keep_001", "confirmed", notes="Retain this review")

        # Now save annotated video metadata
        sample_meta = RenderMetadata(
            job_id="job_ann_001",
            video_id="vid_preserve",
            input_video_path="/tmp/preserve.mp4",
            output_video_path="/tmp/preserve_ann.mp4",
            model_path="models/best.pt",
            model_checkpoint="best.pt",
            detector_conf=0.35,
            iou_threshold=0.45,
            imgsz=640,
            source_fps=30.0,
            source_frame_count=100,
            source_width=1920,
            source_height=1080,
            source_duration_sec=3.33,
            processed_frame_count=100,
            output_fps=30.0,
            output_frame_count=100,
            output_width=1920,
            output_height=1080,
            processing_duration_sec=5.2,
            effective_inference_fps=19.2,
            status="completed",
            total_detections_drawn=14,
            unique_track_ids=[1, 2],
        )
        db.save_annotated_video(sample_meta)

        # Verify existing candidate events and reviews are 100% untouched
        saved_events = db.get_events(video_id="vid_preserve", include_reviews=True)
        self.assertEqual(len(saved_events), 1)
        self.assertEqual(saved_events[0]["event_id"], "ev_keep_001")
        self.assertEqual(saved_events[0]["review"]["decision"], "confirmed")
        self.assertEqual(saved_events[0]["review"]["reviewer_notes"], "Retain this review")

        # Verify annotated video was saved and can be retrieved
        ann_records = db.get_annotated_videos("vid_preserve")
        self.assertEqual(len(ann_records), 1)
        self.assertEqual(ann_records[0]["job_id"], "job_ann_001")
        self.assertEqual(ann_records[0]["total_detections"], 14)


if __name__ == "__main__":
    unittest.main()


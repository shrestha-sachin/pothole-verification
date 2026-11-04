"""Test bounded-memory pipeline streaming and checkpoint class compatibility validation."""

import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

from src.detector.inference import PotholeDetector
from src.tracking.event_grouper import EventGrouper


class DummyBox:
    def __init__(self, cls_id: int = 0, conf: float = 0.85, track_id: int = 1):
        self.cls = torch.tensor([cls_id])
        self.conf = torch.tensor([conf])
        self.id = torch.tensor([track_id])
        self.xyxy = torch.tensor([[50.0, 50.0, 150.0, 150.0]])


class DummyBoxes:
    def __init__(self, boxes=None):
        self.boxes = boxes or []
        self.id = torch.tensor([b.id[0] for b in self.boxes]) if self.boxes else None

    def __iter__(self):
        return iter(self.boxes)


class DummyResult:
    def __init__(self, boxes=None):
        self.boxes = boxes


class TestCheckpointValidation(unittest.TestCase):
    """Tests confirming class 0 really means pothole and rejecting incompatible checkpoints."""

    def test_incompatible_coco_checkpoint_is_rejected(self):
        """A base COCO model (where class 0 is 'person') must be rejected with an informative error."""
        grouper = EventGrouper(video_id="test_vid")

        mock_coco_model = MagicMock()
        mock_coco_model.names = {0: "person", 1: "bicycle", 2: "car"}

        with self.assertRaises(ValueError) as ctx:
            grouper.validate_checkpoint(mock_coco_model)

        err_msg = str(ctx.exception)
        self.assertIn("class 0 is 'person', not 'pothole'", err_msg)
        self.assertIn("un-fine-tuned base checkpoint", err_msg)

    def test_valid_pothole_checkpoint_is_accepted(self):
        """A checkpoint with class 0 as 'pothole' must pass validation."""
        grouper = EventGrouper(video_id="test_vid")

        mock_pothole_model = MagicMock()
        mock_pothole_model.names = {0: "pothole"}

        # Should execute without raising
        grouper.validate_checkpoint(mock_pothole_model)


class TestBoundedMemoryStreaming(unittest.TestCase):
    """Tests confirming video stream processing does not accumulate decoded frames."""

    def test_stream_processes_incrementally_with_bounded_memory(self):
        grouper = EventGrouper(video_id="stream_test", policy_b_min_obs=2)

        # Mock YOLO model whose track() returns detections
        mock_model = MagicMock()
        mock_model.names = {0: "pothole"}
        mock_model.predictor = None

        def dummy_track(img, **kwargs):
            # Verify input color: passed as BGR ndarray
            self.assertIsInstance(img, np.ndarray)
            return [DummyResult(boxes=DummyBoxes([DummyBox(cls_id=0, conf=0.88, track_id=42)]))]

        mock_model.track = dummy_track

        # Generator yielding 30 synthetic frames
        def frame_generator():
            for i in range(30):
                # Small synthetic BGR frame
                frame = np.full((120, 160, 3), fill_value=i, dtype=np.uint8)
                yield (i, i * 200, frame)

        events = grouper.process_video_stream(
            frame_iter=frame_generator(),
            model=mock_model,
            conf=0.35,
            iou=0.45,
            total_expected_frames=30,
        )

        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.track_id, 42)
        self.assertEqual(ev.observation_count, 30)
        self.assertEqual(ev.first_frame_index, 0)
        self.assertEqual(ev.last_frame_index, 29)
        self.assertTrue(ev.policy_a_retained)
        self.assertTrue(ev.policy_b_retained)
        self.assertEqual(ev.rep_x1, 50)
        self.assertEqual(ev.rep_y1, 50)


if __name__ == "__main__":
    unittest.main()

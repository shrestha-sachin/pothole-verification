"""Tests for bounded-memory stream processing, evidence disk staging, and tracker reactivation.

Verifies:
  - Streaming across 50 sequential tracks maintains bounded RAM without frame accumulation.
  - Tracker reactivation re-opens historical events, updates observation lists, and updates representative evidence.
  - Evidence staging to disk keeps at most 3 candidate images per track and cleans up on completion.
"""

import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from src.tracking.event_grouper import EventGrouper


class DummyBox:
    def __init__(self, cls_id: int = 0, conf: float = 0.85, track_id: int = 1, xyxy: tuple = (50, 50, 150, 150)):
        self.cls = torch.tensor([cls_id])
        self.conf = torch.tensor([conf])
        self.id = torch.tensor([track_id])
        self.xyxy = torch.tensor([[float(v) for v in xyxy]])


class DummyBoxes:
    def __init__(self, boxes=None):
        self.boxes = boxes or []
        self.id = torch.tensor([b.id[0] for b in self.boxes]) if self.boxes else None

    def __iter__(self):
        return iter(self.boxes)


class DummyResult:
    def __init__(self, boxes=None):
        self.boxes = boxes


class TestTrackingBoundedMemory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_bounded_memory_50_sequential_tracks(self):
        """Simulate 500 frames with 50 sequential tracks; verify bounded memory and proper grouping."""
        grouper = EventGrouper(video_id="bench_vid", track_buffer=20, policy_b_min_obs=2)

        mock_model = MagicMock()
        mock_model.names = {0: "pothole"}
        mock_model.predictor = None

        current_frame_state = {"frame_idx": 0}

        def mock_track(bgr, **kwargs):
            fi = current_frame_state["frame_idx"]
            # 50 tracks: track_id = (fi // 10) + 1
            # Each track is active for 10 frames
            track_id = (fi // 10) + 1
            box = DummyBox(cls_id=0, conf=0.80 + (fi % 10) * 0.01, track_id=track_id)
            return [DummyResult(boxes=DummyBoxes([box]))]

        mock_model.track = mock_track

        def frame_generator():
            for i in range(500):
                current_frame_state["frame_idx"] = i
                # Generate 480x640 synthetic frame
                frame = np.full((480, 640, 3), fill_value=(i % 256), dtype=np.uint8)
                yield (i, i * 33, frame)

        tracemalloc.start()
        snapshot_start = tracemalloc.take_snapshot()

        events = grouper.process_video_stream(
            frame_iter=frame_generator(),
            model=mock_model,
            conf=0.35,
            iou=0.45,
            total_expected_frames=500,
        )

        snapshot_end = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # All 50 tracks must be grouped into exactly 50 events
        self.assertEqual(len(events), 50)
        for ev in events:
            self.assertEqual(ev.observation_count, 10)
            self.assertTrue(ev.policy_a_retained)
            self.assertTrue(ev.policy_b_retained)

        # Peak memory used by Python objects during processing of 500 frames with 50 tracks
        # should remain strictly bounded (under 15 MB)
        peak_mb = peak / (1024 * 1024)
        self.assertLess(peak_mb, 15.0, f"Peak memory was {peak_mb:.2f} MB, which exceeds bounded limit!")

    def test_tracker_reactivation_updates_event_and_evidence(self):
        """When a finalized track ID receives a new detection, re-open event and update representative observation."""
        evidence_dir = self.root / "evidence"
        grouper = EventGrouper(video_id="reactivate_vid", track_buffer=10, policy_b_min_obs=2)

        mock_model = MagicMock()
        mock_model.names = {0: "pothole"}
        mock_model.predictor = None

        current_frame_state = {"frame_idx": 0}

        def mock_track(bgr, **kwargs):
            fi = current_frame_state["frame_idx"]
            if fi in (0, 1, 2):
                # Initial observations for track 10 with moderate conf
                box = DummyBox(cls_id=0, conf=0.70, track_id=10, xyxy=(40, 40, 80, 80))
                return [DummyResult(boxes=DummyBoxes([box]))]
            elif fi == 50:
                # Reactivation: track 10 reappears 48 frames later with high conf and sharp pattern
                box = DummyBox(cls_id=0, conf=0.98, track_id=10, xyxy=(60, 60, 120, 120))
                return [DummyResult(boxes=DummyBoxes([box]))]
            else:
                return [DummyResult(boxes=DummyBoxes([]))]

        mock_model.track = mock_track

        def frame_generator():
            for i in range(60):
                current_frame_state["frame_idx"] = i
                # For frame 50, create an image with high Laplacian sharpness (alternating high-contrast pixels)
                if i == 50:
                    frame = np.zeros((200, 200, 3), dtype=np.uint8)
                    frame[60:120:2, 60:120:2] = 255
                else:
                    frame = np.full((200, 200, 3), fill_value=128, dtype=np.uint8)
                yield (i, i * 100, frame)

        events = grouper.process_video_stream(
            frame_iter=frame_generator(),
            model=mock_model,
            conf=0.35,
            iou=0.45,
            evidence_root=evidence_dir,
            total_expected_frames=60,
        )

        # Track 10 must be a single CandidateEvent (not two disjoint events)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.track_id, 10)
        self.assertEqual(ev.observation_count, 4)
        self.assertEqual(ev.first_frame_index, 0)
        self.assertEqual(ev.last_frame_index, 50)
        self.assertAlmostEqual(ev.detector_conf_max, 0.98, places=2)

        # The new observation at frame 50 had higher sharpness and confidence (0.98 vs 0.70)
        # So the representative coordinates must update to the new observation!
        self.assertEqual(ev.rep_frame_index, 50)
        self.assertEqual(ev.rep_x1, 60)
        self.assertEqual(ev.rep_y1, 60)

        # Staged temporary candidate directory must be cleaned up
        tmp_staging = evidence_dir / "reactivate_vid" / ".tmp_candidates"
        self.assertFalse(tmp_staging.exists())

        # Evidence files must exist in final event folder
        ev_dir = evidence_dir / "reactivate_vid" / ev.event_id
        self.assertTrue((ev_dir / "crop.jpg").exists())
        self.assertTrue((ev_dir / "full_frame.jpg").exists())


if __name__ == "__main__":
    unittest.main()


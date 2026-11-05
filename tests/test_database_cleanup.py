import tempfile
import unittest
from pathlib import Path

from src.database import Database
from src.tracking.event_grouper import CandidateEvent, ObservedDetection


class TestDatabaseCleanup(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.db = Database(self.db_path)

        # Register video
        self.video_id = "test_vid_1"
        self.db.register_video(self.video_id, {"filepath": "/path/to/test.mp4", "duration_sec": 10.0})

        obs = ObservedDetection(frame_index=1, timestamp_ms=1000, x1=10, y1=10, x2=50, y2=50, conf=0.8)
        ev = CandidateEvent(
            event_id="ev_001",
            video_id=self.video_id,
            road_segment_id="seg_001",
            track_id=1,
            first_timestamp_ms=1000,
            last_timestamp_ms=2000,
            first_frame_index=1,
            last_frame_index=2,
            detector_conf_max=0.8,
            detector_conf_mean=0.8,
            observation_count=1,
            rep_x1=10, rep_y1=10, rep_x2=50, rep_y2=50,
            observations=[obs],
        )
        self.db.save_events([ev])
        self.db.save_review("ev_001", decision="confirmed", notes="Looks real")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_video_stats(self):
        stats = self.db.get_video_stats(self.video_id)
        self.assertEqual(stats["event_count"], 1)
        self.assertEqual(stats["detection_count"], 1)
        self.assertEqual(stats["review_count"], 1)

    def test_delete_video_data(self):
        deleted = self.db.delete_video_data(self.video_id, clear_reviews=True, reset_status=True)
        self.assertEqual(deleted, 1)
        stats = self.db.get_video_stats(self.video_id)
        self.assertEqual(stats["event_count"], 0)
        self.assertEqual(stats["detection_count"], 0)
        self.assertEqual(stats["review_count"], 0)
        self.assertEqual(stats["status"], "pending")

    def test_delete_video_completely(self):
        ok = self.db.delete_video(self.video_id, clear_reviews=True)
        self.assertTrue(ok)
        videos = self.db.get_videos()
        self.assertEqual(len(videos), 0)

    def test_purge_all_inference_data(self):
        res = self.db.purge_all_inference_data(clear_reviews=True)
        self.assertEqual(res["events_deleted"], 1)
        self.assertEqual(res["reviews_deleted"], 1)
        stats = self.db.get_video_stats(self.video_id)
        self.assertEqual(stats["event_count"], 0)


if __name__ == "__main__":
    unittest.main()

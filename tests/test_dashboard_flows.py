"""Integration tests for Streamlit dashboard workflows, review persistence, and export."""

import unittest
from pathlib import Path

from src.database import Database
from src.evaluation.matcher import (
    GroundTruthAnnotation,
    evaluate_event_detections,
    load_video_ground_truth,
)
from src.tracking.event_grouper import CandidateEvent, ObservedDetection


class TestDashboardWorkflows(unittest.TestCase):
    """Test dashboard data operations and review interactions."""

    def setUp(self):
        self.test_db_path = Path("results/test_dashboard_flows.db")
        if self.test_db_path.exists():
            self.test_db_path.unlink()
        self.db = Database(self.test_db_path)

        # Register video
        self.db.register_video(
            "vid_flow_test",
            {
                "filepath": "data/raw/local_video/test.mp4",
                "road_segment_id": "seg_001",
                "duration_sec": 120.0,
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
                "file_size_mb": 50.0,
            },
        )

        # Create candidate events
        obs1 = ObservedDetection(
            frame_index=10,
            timestamp_ms=2000,
            x1=100, y1=150, x2=250, y2=250,
            conf=0.88,
            laplacian_var=180.0,
        )
        obs2 = ObservedDetection(
            frame_index=12,
            timestamp_ms=2400,
            x1=105, y1=152, x2=255, y2=252,
            conf=0.91,
            laplacian_var=210.0,
        )
        ev1 = CandidateEvent(
            event_id="ev_001",
            video_id="vid_flow_test",
            road_segment_id="seg_001",
            track_id=1,
            first_timestamp_ms=2000,
            last_timestamp_ms=2400,
            first_frame_index=10,
            last_frame_index=12,
            detector_conf_max=0.91,
            detector_conf_mean=0.895,
            observation_count=2,
            rep_x1=105, rep_y1=152, rep_x2=255, rep_y2=252,
            rep_timestamp_ms=2400,
            rep_frame_index=12,
            policy_a_retained=True,
            policy_b_retained=True,
            policy_c_score=None,
            policy_c_retained=None,
            observations=[obs1, obs2],
        )

        ev2 = CandidateEvent(
            event_id="ev_002",
            video_id="vid_flow_test",
            road_segment_id="seg_001",
            track_id=2,
            first_timestamp_ms=5000,
            last_timestamp_ms=5000,
            first_frame_index=25,
            last_frame_index=25,
            detector_conf_max=0.45,
            detector_conf_mean=0.45,
            observation_count=1,
            rep_x1=500, rep_y1=400, rep_x2=600, rep_y2=500,
            rep_timestamp_ms=5000,
            rep_frame_index=25,
            policy_a_retained=True,
            policy_b_retained=False,
            policy_c_score=None,
            policy_c_retained=None,
            observations=[],
        )

        self.db.save_events([ev1, ev2])

    def tearDown(self):
        if self.test_db_path.exists():
            self.test_db_path.unlink()

    def test_save_and_update_human_reviews(self):
        # Initial review
        self.db.save_review("ev_001", decision="confirmed", fp_category=None, notes="Clear cavity")
        ev = self.db.get_event("ev_001")
        self.assertIsNotNone(ev["review"])
        self.assertEqual(ev["review"]["decision"], "confirmed")
        self.assertEqual(ev["review"]["reviewer_notes"], "Clear cavity")

        # Update to rejected with subtype
        self.db.save_review("ev_001", decision="rejected", fp_category="shadow", notes="Tree shadow")
        ev_updated = self.db.get_event("ev_001")
        self.assertEqual(ev_updated["review"]["decision"], "rejected")
        self.assertEqual(ev_updated["review"]["fp_category"], "shadow")

    def test_reprocessing_preserves_reviews(self):
        # Record review for ev_001
        self.db.save_review("ev_001", decision="confirmed", fp_category=None, notes="Pothole verified")

        # Reprocess: delete with preserve_reviews=True
        preserved = self.db.delete_events_for_video("vid_flow_test", clear_reviews=False)
        self.assertEqual(len(preserved), 1)

        # New re-tracked event with similar spatial/temporal coordinates
        new_ev = CandidateEvent(
            event_id="ev_001_retracked",
            video_id="vid_flow_test",
            road_segment_id="seg_001",
            track_id=10,
            first_timestamp_ms=2050,
            last_timestamp_ms=2450,
            first_frame_index=10,
            last_frame_index=12,
            detector_conf_max=0.92,
            detector_conf_mean=0.90,
            observation_count=2,
            rep_x1=106, rep_y1=153, rep_x2=256, rep_y2=253,
            rep_timestamp_ms=2450,
            rep_frame_index=12,
            policy_a_retained=True,
            policy_b_retained=True,
            observations=[],
        )
        self.db.save_events([new_ev])

        reattached = self.db.reattach_preserved_reviews("vid_flow_test", preserved)
        self.assertEqual(reattached, 1)

        check_ev = self.db.get_event("ev_001_retracked")
        self.assertIsNotNone(check_ev["review"])
        self.assertEqual(check_ev["review"]["decision"], "confirmed")

    def test_csv_export_contains_all_metadata_and_review_fields(self):
        self.db.save_review("ev_001", decision="confirmed", fp_category=None, notes="Validated")
        csv_str = self.db.export_events_csv("vid_flow_test")

        self.assertIn("ev_001", csv_str)
        self.assertIn("ev_002", csv_str)
        self.assertIn("rep_timestamp_ms", csv_str)
        self.assertIn("rep_frame_index", csv_str)
        self.assertIn("confirmed", csv_str)
        self.assertIn("Validated", csv_str)


if __name__ == "__main__":
    unittest.main()


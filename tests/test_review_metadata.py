"""Regression coverage for DB/sidecar compatibility in video review."""
import unittest

from src.video.review_metadata import normalize_review_metadata


class TestReviewMetadata(unittest.TestCase):
    def test_renderer_sidecar_supports_dashboard_fields(self):
        sidecar = {
            "effective_inference_fps": 12.5,
            "processing_duration_sec": 30,
            "total_detections_drawn": 48,
            "unique_track_ids": [1, 3, 5],
        }
        result = normalize_review_metadata(sidecar)
        self.assertEqual(result["effective_fps"], 12.5)
        self.assertEqual(result["processing_duration"], 30)
        self.assertEqual(result["total_detections"], 48)
        self.assertEqual(result["unique_tracks"], 3)
        self.assertNotIn("unique_tracks", sidecar)

    def test_database_fields_take_precedence(self):
        result = normalize_review_metadata({"total_detections": 10,
                                           "total_detections_drawn": 20,
                                           "unique_tracks": 7})
        self.assertEqual(result["total_detections"], 10)
        self.assertEqual(result["unique_tracks"], 7)

    def test_missing_optional_telemetry_is_safe(self):
        result = normalize_review_metadata({"unique_track_ids": None})
        self.assertEqual(result["unique_tracks"], 0)
        self.assertEqual(result["effective_fps"], 0)

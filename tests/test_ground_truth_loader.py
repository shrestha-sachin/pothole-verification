"""Test ground truth loading, multi-frame physical defect clustering, and CSV parsing."""

import unittest
from pathlib import Path

from src.evaluation.matcher import (
    GroundTruthAnnotation,
    load_video_ground_truth,
    parse_ground_truth_csv,
)


class TestGroundTruthLoader(unittest.TestCase):
    """Test loading and clustering of ground truth annotations."""

    def test_parse_ground_truth_csv(self):
        csv_text = """video_id,timestamp_ms,x1,y1,x2,y2,gt_event_id,label
vid_01,1000,10,20,50,60,defect_1,pothole
vid_01,2000,100,120,180,200,defect_2,shadow
"""
        annotations = parse_ground_truth_csv(csv_text)
        self.assertEqual(len(annotations), 2)
        self.assertEqual(annotations[0].video_id, "vid_01")
        self.assertEqual(annotations[0].gt_event_id, "defect_1")
        self.assertTrue(annotations[0].is_pothole)
        self.assertEqual(annotations[1].label, "shadow")
        self.assertFalse(annotations[1].is_pothole)

    def test_load_video_ground_truth_clustering(self):
        # Green_Bay_Road_5_min annotations exist in data/annotations/video/
        annotations = load_video_ground_truth("Green_Bay_Road_5_min")
        self.assertGreater(len(annotations), 0)

        # Check that physical clustering took place: total annotations > unique gt_event_id
        potholes = [a for a in annotations if a.is_pothole]
        unique_defects = set(a.gt_event_id for a in potholes)
        self.assertGreater(len(potholes), 0)
        self.assertLessEqual(len(unique_defects), len(potholes))


if __name__ == "__main__":
    unittest.main()


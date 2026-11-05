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

    def test_legacy_annotations_are_not_automatic_ground_truth(self):
        # Old files lack explicit video identity and verified physical IDs.
        self.assertEqual(load_video_ground_truth("Green_Bay_Road_5_min"), [])


if __name__ == "__main__":
    unittest.main()


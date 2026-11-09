"""Regression checks for split integrity and non-destructive defaults."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.convert_rdd2022 import RDDItem, assign_group_ids, stratified_group_split
from scripts import export_yolo


def item(image_id, sha, country="United_States"):
    return RDDItem(
        image_id=image_id,
        img_path=Path(f"/tmp/{country}/{image_id}.jpg"),
        xml_path=Path(f"/tmp/{country}/{image_id}.xml"),
        country=country,
        img_w=100, img_h=100, boxes=[], class_counts={},
        sha256=sha, has_pothole=True, has_crack=False, negative_tier="",
    )


class DatasetSafetyTests(unittest.TestCase):
    def test_duplicate_connects_separate_sequences(self):
        items = [
            item("United_States_seq1_001", "hashA"),
            item("United_States_seq1_002", "hashB"),
            item("United_States_seq2_001", "hashB"),
            item("United_States_seq2_002", "hashC"),
            item("United_States_seq3_001", "hashD"),
            item("United_States_seq4_001", "hashE"),
        ]
        assign_group_ids(items)
        self.assertEqual(len({x.group_id for x in items[:4]}), 1)
        self.assertEqual(len({x.group_id for x in items}), 3)
        train, val, test = stratified_group_split(items, 0.6, 0.2, 0.2)
        split_ids = [{x.group_id for x in split} for split in (train, val, test)]
        self.assertFalse(split_ids[0] & split_ids[1])
        self.assertFalse(split_ids[0] & split_ids[2])
        self.assertFalse(split_ids[1] & split_ids[2])

    def test_local_export_rejects_recording_split_leakage_before_writing(self):
        rows = {
            "frame1": {"image_id": "frame1", "review_status": "reviewed",
                       "split": "train", "recording_id": "driveA"},
            "frame2": {"image_id": "frame2", "review_status": "reviewed",
                       "split": "test", "recording_id": "driveA"},
        }
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "export"
            with patch.object(export_yolo, "load_inventory", return_value=rows):
                with self.assertRaisesRegex(ValueError, "Recording-level data leakage"):
                    export_yolo.export(out)
            self.assertFalse(out.exists())

    def test_local_export_rejects_missing_split(self):
        rows = {"frame1": {"image_id": "frame1", "review_status": "reviewed",
                            "split": "", "recording_id": "driveA"}}
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(export_yolo, "load_inventory", return_value=rows):
                with self.assertRaisesRegex(ValueError, "Invalid or missing split"):
                    export_yolo.export(Path(temp) / "export")

    def test_local_export_protects_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "export"
            out.mkdir()
            (out / "train.txt").write_text("previous run\n")
            with patch.object(export_yolo, "load_inventory", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "not empty"):
                    export_yolo.export(out)
            self.assertEqual((out / "train.txt").read_text(), "previous run\n")


if __name__ == "__main__":
    unittest.main()

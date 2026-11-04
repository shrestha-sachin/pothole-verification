"""Test image-to-label resolution, split integrity, and dataset configuration portability."""

import tempfile
import unittest
from pathlib import Path

from ultralytics.data.utils import img2label_paths
from scripts.convert_rdd2022 import parse_voc_xml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestImageLabelResolution(unittest.TestCase):

    def test_ultralytics_label_resolution_on_exported_train(self):
        """Verify that Ultralytics img2label_paths resolves every image in train.txt to an existing label."""
        train_txt = PROJECT_ROOT / "exports" / "yolo_labels" / "train.txt"
        self.assertTrue(train_txt.exists(), f"Missing {train_txt}")

        image_paths = [l.strip() for l in train_txt.read_text().splitlines() if l.strip()]
        self.assertGreater(len(image_paths), 0, "train.txt is empty")

        resolved_labels = img2label_paths(image_paths)
        self.assertEqual(len(resolved_labels), len(image_paths))

        missing = [l for l in resolved_labels if not Path(l).exists()]
        self.assertEqual(len(missing), 0, f"Found {len(missing)} unresolved label files: {missing[:3]}")

    def test_detector_yaml_has_no_hardcoded_absolute_paths(self):
        """Verify configs/detector.yaml does not contain machine-specific absolute Windows or local paths."""
        yaml_path = PROJECT_ROOT / "configs" / "detector.yaml"
        self.assertTrue(yaml_path.exists(), f"Missing {yaml_path}")
        content = yaml_path.read_text()

        self.assertNotIn("C:/", content, "Found hardcoded Windows path in configs/detector.yaml")
        self.assertNotIn("C:\\", content, "Found hardcoded Windows path in configs/detector.yaml")
        self.assertNotIn("/Users/", content, "Found hardcoded machine absolute path in configs/detector.yaml")
        self.assertNotIn("data/processes", content, "Found typo 'data/processes' in configs/detector.yaml")

    def test_corrupt_voc_xml_is_rejected_not_treated_as_negative(self):
        """Verify that corrupted or unparseable XML annotations return None, preventing silent false negatives."""
        target_classes = {"D40": 0}

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Truncated XML (syntax error)
            corrupt_xml = tmp_path / "corrupt.xml"
            corrupt_xml.write_text("<annotation><object><name>D40</name>", encoding="utf-8")
            self.assertIsNone(parse_voc_xml(corrupt_xml, target_classes))

            # 2. Missing <size> element
            no_size_xml = tmp_path / "no_size.xml"
            no_size_xml.write_text(
                "<annotation><object><name>D40</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object></annotation>",
                encoding="utf-8",
            )
            self.assertIsNone(parse_voc_xml(no_size_xml, target_classes))

            # 3. Invalid zero/negative dimensions
            invalid_dim_xml = tmp_path / "invalid_dim.xml"
            invalid_dim_xml.write_text(
                "<annotation><size><width>0</width><height>100</height></size></annotation>",
                encoding="utf-8",
            )
            self.assertIsNone(parse_voc_xml(invalid_dim_xml, target_classes))

            # 4. Valid XML with D40 pothole
            valid_pos_xml = tmp_path / "valid_pos.xml"
            valid_pos_xml.write_text(
                "<annotation><size><width>640</width><height>480</height></size>"
                "<object><name>D40</name><bndbox><xmin>100</xmin><ymin>100</ymin><xmax>200</xmax><ymax>200</ymax></bndbox></object></annotation>",
                encoding="utf-8",
            )
            res_pos = parse_voc_xml(valid_pos_xml, target_classes)
            self.assertIsNotNone(res_pos)
            w, h, boxes = res_pos
            self.assertEqual((w, h), (640, 480))
            self.assertEqual(len(boxes), 1)
            self.assertEqual(boxes[0][0], 0)  # class 0

            # 5. Valid negative road XML (no D40 object)
            valid_neg_xml = tmp_path / "valid_neg.xml"
            valid_neg_xml.write_text(
                "<annotation><size><width>640</width><height>480</height></size></annotation>",
                encoding="utf-8",
            )
            res_neg = parse_voc_xml(valid_neg_xml, target_classes)
            self.assertIsNotNone(res_neg)
            w_neg, h_neg, boxes_neg = res_neg
            self.assertEqual((w_neg, h_neg), (640, 480))
            self.assertEqual(len(boxes_neg), 0)  # Valid true negative


if __name__ == "__main__":
    unittest.main()

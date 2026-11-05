"""Unit and integration tests for RDD2022 to YOLO dataset conversion.

Verifies:
  - D40 maps to pothole class 0.
  - Manifests contain absolute output paths without dereferencing symlinks.
  - Ultralytics label resolution succeeds for 100% of generated images.
  - Both symlink and copy modes work properly.
  - Source data is never modified.
  - Non-empty train and val sets are enforced.
  - Corrupt XMLs are rejected rather than silently treated as negative images.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from scripts.convert_rdd2022 import (
    discover_pairs,
    main,
    parse_voc_xml,
)


def _create_mock_voc_xml(xml_path: Path, filename: str, w: int, h: int, objects: list[tuple[str, int, int, int, int]]) -> None:
    """Helper to generate a valid Pascal VOC XML string."""
    obj_lines = []
    for cls_name, xmin, ymin, xmax, ymax in objects:
        obj_lines.append(f"""  <object>
    <name>{cls_name}</name>
    <bndbox>
      <xmin>{xmin}</xmin>
      <ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax>
      <ymax>{ymax}</ymax>
    </bndbox>
  </object>""")
    xml_content = f"""<annotation>
  <filename>{filename}</filename>
  <size>
    <width>{w}</width>
    <height>{h}</height>
    <depth>3</depth>
  </size>
{"".join(obj_lines)}
</annotation>"""
    xml_path.write_text(xml_content, encoding="utf-8")


class TestRDD2022Conversion(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()
        self.rdd_dir = self.root / "RDD2022"
        self.out_dir = self.root / "processed"
        self.config_yaml = self.root / "detector.yaml"

        # Create synthetic RDD2022 directory layout
        # United_States/train/images and United_States/train/annotations/xmls
        us_img_dir = self.rdd_dir / "United_States" / "train" / "images"
        us_xml_dir = self.rdd_dir / "United_States" / "train" / "annotations" / "xmls"
        us_img_dir.mkdir(parents=True, exist_ok=True)
        us_xml_dir.mkdir(parents=True, exist_ok=True)

        # Create 6 synthetic images across 3 sequences
        self.synthetic_items = []
        for seq_idx in (1, 2, 3):
            for frame_idx in (1, 2):
                stem = f"United_States_seq{seq_idx}_{frame_idx:03d}"
                img_path = us_img_dir / f"{stem}.jpg"
                xml_path = us_xml_dir / f"{stem}.xml"

                # Create 100x100 dummy image
                img = Image.new("RGB", (100, 100), color=(128, 128, 128))
                img.save(img_path)

                # Sequence 1: D40 (pothole)
                # Sequence 2: D00 (crack, valid negative for pothole detector)
                # Sequence 3: D40 (pothole)
                if seq_idx in (1, 3):
                    objs = [("D40", 20, 20, 80, 80)]
                else:
                    objs = [("D00", 10, 10, 50, 50)]

                _create_mock_voc_xml(xml_path, img_path.name, 100, 100, objs)
                self.synthetic_items.append((img_path, xml_path))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_voc_xml_d40_mapped_to_pothole_class_zero(self):
        """RDD2022 D40 class must map to class 0 in pothole-only mode."""
        xml_path = self.synthetic_items[0][1]
        parsed = parse_voc_xml(xml_path, target_classes={"D40": 0})
        self.assertIsNotNone(parsed)
        w, h, boxes = parsed
        self.assertEqual(w, 100)
        self.assertEqual(h, 100)
        self.assertEqual(len(boxes), 1)
        cls_id, cx, cy, bw, bh = boxes[0]
        self.assertEqual(cls_id, 0)
        self.assertAlmostEqual(cx, 0.5, places=2)
        self.assertAlmostEqual(cy, 0.5, places=2)
        self.assertAlmostEqual(bw, 0.6, places=2)
        self.assertAlmostEqual(bh, 0.6, places=2)

    def test_parse_voc_xml_negative_image_returns_empty_boxes(self):
        """An image containing only non-target damage types must return an empty box list (valid negative)."""
        xml_path = self.synthetic_items[2][1]  # Sequence 2 has D00
        parsed = parse_voc_xml(xml_path, target_classes={"D40": 0})
        self.assertIsNotNone(parsed)
        _, _, boxes = parsed
        self.assertEqual(len(boxes), 0)

    def test_parse_voc_xml_corrupt_file_returns_none(self):
        """Corrupt XML must return None and not be treated as a negative sample."""
        corrupt_xml = self.root / "corrupt.xml"
        corrupt_xml.write_text("<annotation>BROKEN", encoding="utf-8")
        parsed = parse_voc_xml(corrupt_xml, target_classes={"D40": 0})
        self.assertIsNone(parsed)

    def test_conversion_symlink_mode_manifest_paths_not_dereferenced(self):
        """In symlink mode, manifests must reference the output directory, not dereference back to source."""
        argv = [
            "--input", str(self.rdd_dir),
            "--output", str(self.out_dir),
            "--config-yaml", str(self.config_yaml),
            "--val-split", "0.33",
            "--seed", "42",
        ]
        main(argv)

        train_txt = self.out_dir / "train.txt"
        val_txt = self.out_dir / "val.txt"
        self.assertTrue(train_txt.exists())
        self.assertTrue(val_txt.exists())

        train_lines = [line.strip() for line in train_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        val_lines = [line.strip() for line in val_txt.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertGreater(len(train_lines), 0)
        self.assertGreater(len(val_lines), 0)

        for p_str in train_lines + val_lines:
            p = Path(p_str)
            # CRITICAL CHECK: The path in manifest MUST be inside out_dir/images, NOT rdd_dir!
            self.assertTrue(p_str.startswith(str(self.out_dir / "images")),
                            f"Path {p_str} was dereferenced back to source instead of output directory!")

            # Verify Ultralytics img2label resolution rule:
            # Replace /images/ with /labels/ and .jpg with .txt
            label_p_str = p_str.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
            label_p = Path(label_p_str).with_suffix(".txt")
            self.assertTrue(label_p.exists(), f"Label {label_p} does not exist for manifest image {p_str}")

    def test_conversion_copy_images_mode(self):
        """When --copy-images is specified, files must be real copies, not symlinks."""
        argv = [
            "--input", str(self.rdd_dir),
            "--output", str(self.out_dir),
            "--config-yaml", str(self.config_yaml),
            "--copy-images",
            "--val-split", "0.33",
            "--seed", "42",
        ]
        main(argv)

        train_txt = self.out_dir / "train.txt"
        train_lines = [line.strip() for line in train_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
        for p_str in train_lines:
            p = Path(p_str)
            self.assertTrue(p.exists())
            self.assertFalse(p.is_symlink(), f"Image {p} is a symlink but --copy-images was requested")

    def test_source_data_is_never_modified(self):
        """Source directory files must remain completely untouched after conversion."""
        before_timestamps = {p: p.stat().st_mtime for p in self.rdd_dir.rglob("*") if p.is_file()}
        argv = [
            "--input", str(self.rdd_dir),
            "--output", str(self.out_dir),
            "--config-yaml", str(self.config_yaml),
            "--val-split", "0.33",
        ]
        main(argv)
        after_timestamps = {p: p.stat().st_mtime for p in self.rdd_dir.rglob("*") if p.is_file()}
        self.assertEqual(before_timestamps, after_timestamps)


if __name__ == "__main__":
    unittest.main()

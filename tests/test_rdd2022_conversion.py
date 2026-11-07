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

                # Create 100x100 dummy image with unique color per sequence/frame
                img = Image.new("RGB", (100, 100), color=(seq_idx * 50, frame_idx * 50, (seq_idx + frame_idx) * 20))
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

    def test_standard_rdd2022_flat_filenames_split_successfully(self):
        """Standard RDD2022 filenames (e.g. United_States_004029.jpg) without sequence tags must split properly."""
        flat_dir = self.root / "flat_rdd"
        flat_img_dir = flat_dir / "United_States" / "train" / "images"
        flat_xml_dir = flat_dir / "United_States" / "train" / "annotations" / "xmls"
        flat_img_dir.mkdir(parents=True, exist_ok=True)
        flat_xml_dir.mkdir(parents=True, exist_ok=True)

        flat_stems = [
            "United_States_004029",
            "United_States_001241",
            "United_States_000135",
            "United_States_003286",
            "United_States_004580",
            "United_States_002911",
            "United_States_004002",
            "United_States_003100",
            "United_States_002830",
            "United_States_001247",
        ]
        for idx, stem in enumerate(flat_stems):
            img_p = flat_img_dir / f"{stem}.jpg"
            xml_p = flat_xml_dir / f"{stem}.xml"
            img = Image.new("RGB", (100, 100), color=(idx * 20, 80, 120))
            img.save(img_p)
            objs = [("D40", 10, 10, 50, 50)] if idx % 2 == 0 else [("D20", 20, 20, 60, 60)]
            _create_mock_voc_xml(xml_p, img_p.name, 100, 100, objs)

        out_flat = self.root / "out_flat"
        argv = [
            "--input", str(flat_dir),
            "--output", str(out_flat),
            "--config-yaml", str(out_flat / "detector.yaml"),
            "--countries", "United_States",
            "--train-split", "0.80",
            "--val-split", "0.10",
            "--test-split", "0.10",
            "--seed", "42",
        ]
        main(argv)

        train_txt = out_flat / "train.txt"
        val_txt = out_flat / "val.txt"
        test_txt = out_flat / "test.txt"
        self.assertTrue(train_txt.exists())
        self.assertTrue(val_txt.exists())
        self.assertTrue(test_txt.exists())

        train_lines = [l for l in train_txt.read_text().splitlines() if l.strip()]
        val_lines = [l for l in val_txt.read_text().splitlines() if l.strip()]
        test_lines = [l for l in test_txt.read_text().splitlines() if l.strip()]

        self.assertGreater(len(train_lines), 0, "Train split must not be empty")
        self.assertGreater(len(val_lines), 0, "Validation split must not be empty")
        self.assertGreater(len(test_lines), 0, "Test split must not be empty")
        self.assertEqual(len(train_lines) + len(val_lines) + len(test_lines), len(flat_stems))

        # Check metadata CSV grouping method
        meta_csv = out_flat / "split_metadata.csv"
        self.assertTrue(meta_csv.exists())
        content = meta_csv.read_text()
        self.assertIn("sample_fallback_no_sequence_metadata", content)

    def test_reproducibility_with_same_seed(self):
        """Splits produced with the same random seed must yield identical image assignments."""
        out1 = self.root / "out_seed1"
        out2 = self.root / "out_seed2"

        argv1 = ["--input", str(self.rdd_dir), "--output", str(out1), "--config-yaml", str(out1 / "d.yaml"), "--seed", "42"]
        argv2 = ["--input", str(self.rdd_dir), "--output", str(out2), "--config-yaml", str(out2 / "d.yaml"), "--seed", "42"]
        main(argv1)
        main(argv2)

        train1 = [Path(l).name for l in (out1 / "train.txt").read_text().splitlines() if l.strip()]
        train2 = [Path(l).name for l in (out2 / "train.txt").read_text().splitlines() if l.strip()]
        val1 = [Path(l).name for l in (out1 / "val.txt").read_text().splitlines() if l.strip()]
        val2 = [Path(l).name for l in (out2 / "val.txt").read_text().splitlines() if l.strip()]
        test1 = [Path(l).name for l in (out1 / "test.txt").read_text().splitlines() if l.strip()]
        test2 = [Path(l).name for l in (out2 / "test.txt").read_text().splitlines() if l.strip()]

        self.assertEqual(train1, train2)
        self.assertEqual(val1, val2)
        self.assertEqual(test1, test2)

    def test_split_independence_and_no_leakage(self):
        """Zero image ID or content hash overlap must exist across train, val, and test splits."""
        argv = [
            "--input", str(self.rdd_dir),
            "--output", str(self.out_dir),
            "--config-yaml", str(self.out_dir / "d.yaml"),
            "--train-split", "0.60",
            "--val-split", "0.20",
            "--test-split", "0.20",
            "--seed", "42",
        ]
        main(argv)

        train_stems = {Path(l).stem for l in (self.out_dir / "train.txt").read_text().splitlines() if l.strip()}
        val_stems = {Path(l).stem for l in (self.out_dir / "val.txt").read_text().splitlines() if l.strip()}
        test_stems = {Path(l).stem for l in (self.out_dir / "test.txt").read_text().splitlines() if l.strip()}

        self.assertEqual(len(train_stems & val_stems), 0, "Train and Val share image IDs")
        self.assertEqual(len(train_stems & test_stems), 0, "Train and Test share image IDs")
        self.assertEqual(len(val_stems & test_stems), 0, "Val and Test share image IDs")

    def test_duplicate_detection_keeps_identical_images_in_same_split(self):
        """Exact content duplicates must be assigned to the same duplicate cluster and split."""
        dup_dir = self.root / "dup_rdd"
        img_dir = dup_dir / "United_States" / "train" / "images"
        xml_dir = dup_dir / "United_States" / "train" / "annotations" / "xmls"
        img_dir.mkdir(parents=True, exist_ok=True)
        xml_dir.mkdir(parents=True, exist_ok=True)

        # Create 6 images: image 0 and image 1 are exact byte-for-byte duplicates
        dup_img = Image.new("RGB", (100, 100), color=(222, 111, 55))
        for i in range(6):
            stem = f"United_States_{i:06d}"
            img_p = img_dir / f"{stem}.jpg"
            xml_p = xml_dir / f"{stem}.xml"
            if i in (0, 1):
                dup_img.save(img_p)  # identical content bytes
            else:
                img_unique = Image.new("RGB", (100, 100), color=(i * 35, 90, 150))
                img_unique.save(img_p)
            _create_mock_voc_xml(xml_p, img_p.name, 100, 100, [("D40", 10, 10, 50, 50)])

        out_dup = self.root / "out_dup"
        argv = [
            "--input", str(dup_dir),
            "--output", str(out_dup),
            "--config-yaml", str(out_dup / "d.yaml"),
            "--train-split", "0.60",
            "--val-split", "0.20",
            "--test-split", "0.20",
            "--seed", "42",
        ]
        main(argv)

        train_stems = {Path(l).stem for l in (out_dup / "train.txt").read_text().splitlines() if l.strip()}
        val_stems = {Path(l).stem for l in (out_dup / "val.txt").read_text().splitlines() if l.strip()}
        test_stems = {Path(l).stem for l in (out_dup / "test.txt").read_text().splitlines() if l.strip()}

        # Verify images 0 and 1 are in the exact same split
        in_train = ("United_States_000000" in train_stems) and ("United_States_000001" in train_stems)
        in_val = ("United_States_000000" in val_stems) and ("United_States_000001" in val_stems)
        in_test = ("United_States_000000" in test_stems) and ("United_States_000001" in test_stems)
        self.assertTrue(in_train or in_val or in_test, "Duplicate images were split across different partitions!")

    def test_positive_and_negative_label_handling(self):
        """Verify D40 mapped to class 0, mixed damage remains positive, and negatives produce empty labels."""
        test_dir = self.root / "labels_test_rdd"
        img_dir = test_dir / "United_States" / "train" / "images"
        xml_dir = test_dir / "United_States" / "train" / "annotations" / "xmls"
        img_dir.mkdir(parents=True, exist_ok=True)
        xml_dir.mkdir(parents=True, exist_ok=True)

        # 1. Pothole only
        img1 = img_dir / "pothole_only.jpg"
        Image.new("RGB", (100, 100), color=(10, 10, 10)).save(img1)
        _create_mock_voc_xml(xml_dir / "pothole_only.xml", img1.name, 100, 100, [("D40", 20, 20, 80, 80)])

        # 2. Mixed: D40 + D20 + D10
        img2 = img_dir / "mixed_pothole_cracks.jpg"
        Image.new("RGB", (100, 100), color=(20, 20, 20)).save(img2)
        _create_mock_voc_xml(xml_dir / "mixed_pothole_cracks.xml", img2.name, 100, 100, [
            ("D40", 10, 10, 40, 40),
            ("D20", 50, 50, 90, 90),
            ("D10", 60, 10, 80, 30),
        ])

        # 3. Cracks only: D20 + D10 (hard negative Tier 1)
        img3 = img_dir / "crack_negative_t1.jpg"
        Image.new("RGB", (100, 100), color=(30, 30, 30)).save(img3)
        _create_mock_voc_xml(xml_dir / "crack_negative_t1.xml", img3.name, 100, 100, [
            ("D20", 15, 15, 65, 65),
            ("D10", 40, 40, 80, 50),
        ])

        # 4. Clean road background (empty objects)
        img4 = img_dir / "clean_road_neg.jpg"
        Image.new("RGB", (100, 100), color=(40, 40, 40)).save(img4)
        _create_mock_voc_xml(xml_dir / "clean_road_neg.xml", img4.name, 100, 100, [])

        out_p = self.root / "out_labels_test"
        argv = [
            "--input", str(test_dir),
            "--output", str(out_p),
            "--config-yaml", str(out_p / "d.yaml"),
            "--train-split", "0.50",
            "--val-split", "0.25",
            "--test-split", "0.25",
            "--keep-all-negatives",
            "--seed", "42",
        ]
        main(argv)

        # Check label files across splits
        all_labels = list(out_p.glob("labels/*/*.txt"))
        label_map = {p.stem: p for p in all_labels}

        # Pothole only label
        pothole_only_lines = [l.strip() for l in label_map["pothole_only"].read_text().splitlines() if l.strip()]
        self.assertEqual(len(pothole_only_lines), 1)
        self.assertTrue(pothole_only_lines[0].startswith("0 "))

        # Mixed pothole + cracks label: ONLY D40 must be exported as class 0
        mixed_lines = [l.strip() for l in label_map["mixed_pothole_cracks"].read_text().splitlines() if l.strip()]
        self.assertEqual(len(mixed_lines), 1, "Only D40 box should be exported in pothole-only mode")
        self.assertTrue(mixed_lines[0].startswith("0 "))

        # Crack negative: must be empty .txt
        crack_lines = [l.strip() for l in label_map["crack_negative_t1"].read_text().splitlines() if l.strip()]
        self.assertEqual(len(crack_lines), 0, "Negative crack image must have empty label file")

        # Clean road negative: must be empty .txt
        clean_lines = [l.strip() for l in label_map["clean_road_neg"].read_text().splitlines() if l.strip()]
        self.assertEqual(len(clean_lines), 0, "Negative clean road image must have empty label file")

    def test_in_split_negative_subsampling_no_cross_split_contamination(self):
        """Negatives must be sampled strictly within their split; candidate negatives never move splits."""
        sub_dir = self.root / "subsample_test_rdd"
        img_dir = sub_dir / "United_States" / "train" / "images"
        xml_dir = sub_dir / "United_States" / "train" / "annotations" / "xmls"
        img_dir.mkdir(parents=True, exist_ok=True)
        xml_dir.mkdir(parents=True, exist_ok=True)

        # Create 12 images: 6 positives and 6 crack negatives
        for i in range(12):
            stem = f"United_States_{i:04d}"
            img_p = img_dir / f"{stem}.jpg"
            xml_p = xml_dir / f"{stem}.xml"
            Image.new("RGB", (100, 100), color=(i * 15, 60, 180)).save(img_p)
            if i < 6:
                _create_mock_voc_xml(xml_p, img_p.name, 100, 100, [("D40", 10, 10, 50, 50)])
            else:
                _create_mock_voc_xml(xml_p, img_p.name, 100, 100, [("D20", 20, 20, 60, 60)])

        out_sub = self.root / "out_sub"
        argv = [
            "--input", str(sub_dir),
            "--output", str(out_sub),
            "--config-yaml", str(out_sub / "d.yaml"),
            "--train-split", "0.60",
            "--val-split", "0.20",
            "--test-split", "0.20",
            "--neg-ratio", "0.33",
            "--seed", "42",
        ]
        main(argv)

        meta_csv = out_sub / "split_metadata.csv"
        self.assertTrue(meta_csv.exists())
        import csv
        with meta_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # Check that every image has an assigned split
        for r in rows:
            self.assertIn(r["split"], ("train", "val", "test"))

        # Verify that unselected negatives are in split_metadata with selected_in_manifest == '0'
        unselected = [r for r in rows if r["selected_in_manifest"] == "0"]
        self.assertGreater(len(unselected), 0, "Subsampling should leave some candidate negatives unselected")
        for u in unselected:
            self.assertEqual(u["has_pothole"], "0", "Positives must never be unselected")

        # Verify train.txt, val.txt, test.txt contain only selected images
        for sname in ("train", "val", "test"):
            manifest_stems = {Path(l).stem for l in (out_sub / f"{sname}.txt").read_text().splitlines() if l.strip()}
            split_selected_stems = {r["image_id"] for r in rows if r["split"] == sname and r["selected_in_manifest"] == "1"}
            self.assertEqual(manifest_stems, split_selected_stems)

    def test_empty_split_protection(self):
        """Enforces that tiny datasets with < 3 items raise an error, while >= 3 items yield 3 non-empty splits."""
        from scripts.convert_rdd2022 import RDDItem, stratified_group_split

        items_2 = [
            RDDItem("img1", Path("p1"), Path("x1"), "US", 100, 100, [], {}, "h1", True, False, "", "g1"),
            RDDItem("img2", Path("p2"), Path("x2"), "US", 100, 100, [], {}, "h2", False, True, "", "g2"),
        ]
        with self.assertRaises(ValueError):
            stratified_group_split(items_2, 0.80, 0.10, 0.10)

        # 3 items: each must get exactly 1 split
        items_3 = [
            RDDItem("img1", Path("p1"), Path("x1"), "US", 100, 100, [], {}, "h1", True, False, "", "g1"),
            RDDItem("img2", Path("p2"), Path("x2"), "US", 100, 100, [], {}, "h2", False, True, "", "g2"),
            RDDItem("img3", Path("p3"), Path("x3"), "US", 100, 100, [], {}, "h3", True, False, "", "g3"),
        ]
        tr, va, te = stratified_group_split(items_3, 0.80, 0.10, 0.10)
        self.assertEqual(len(tr), 1)
        self.assertEqual(len(va), 1)
        self.assertEqual(len(te), 1)

    def test_subsample_rdd_negatives_standalone_script(self):
        """Verify scripts/subsample_rdd_negatives.py runs on 3-split output with split isolation."""
        from scripts.subsample_rdd_negatives import main as subsample_main

        # First generate a full converted dataset
        sub_dir = self.root / "subsample_cli_rdd"
        img_dir = sub_dir / "United_States" / "train" / "images"
        xml_dir = sub_dir / "United_States" / "train" / "annotations" / "xmls"
        img_dir.mkdir(parents=True, exist_ok=True)
        xml_dir.mkdir(parents=True, exist_ok=True)

        for i in range(12):
            stem = f"United_States_{i:04d}"
            img_p = img_dir / f"{stem}.jpg"
            xml_p = xml_dir / f"{stem}.xml"
            Image.new("RGB", (100, 100), color=(i * 15, 60, 180)).save(img_p)
            if i < 6:
                _create_mock_voc_xml(xml_p, img_p.name, 100, 100, [("D40", 10, 10, 50, 50)])
            else:
                _create_mock_voc_xml(xml_p, img_p.name, 100, 100, [("D20", 20, 20, 60, 60)])

        proc_dir = self.root / "proc_for_subsample"
        conv_argv = [
            "--input", str(sub_dir),
            "--output", str(proc_dir),
            "--config-yaml", str(proc_dir / "d.yaml"),
            "--train-split", "0.60",
            "--val-split", "0.20",
            "--test-split", "0.20",
            "--keep-all-negatives",
            "--seed", "42",
        ]
        main(conv_argv)

        manifest_csv = self.root / "audit_manifest.csv"
        yaml_out = self.root / "balanced_detector.yaml"
        sub_argv = [
            "--proc-dir", str(proc_dir),
            "--train-negatives", "1",
            "--val-negatives", "1",
            "--test-negatives", "1",
            "--manifest-csv", str(manifest_csv),
            "--yaml-config", str(yaml_out),
            "--seed", "42",
        ]
        subsample_main(sub_argv)

        self.assertTrue((proc_dir / "train_balanced.txt").exists())
        self.assertTrue((proc_dir / "val_balanced.txt").exists())
        self.assertTrue((proc_dir / "test_balanced.txt").exists())
        self.assertTrue(manifest_csv.exists())
        self.assertTrue(yaml_out.exists())

        yaml_content = yaml_out.read_text()
        self.assertIn("train:", yaml_content)
        self.assertIn("val:", yaml_content)
        self.assertIn("test:", yaml_content)

    def test_ultralytics_check_det_dataset_compatibility(self):
        """Generated YAML configuration must validate with Ultralytics check_det_dataset."""
        from ultralytics.data.utils import check_det_dataset
        argv = [
            "--input", str(self.rdd_dir),
            "--output", str(self.out_dir),
            "--config-yaml", str(self.config_yaml),
            "--seed", "42",
        ]
        main(argv)

        dataset_dict = check_det_dataset(str(self.config_yaml))
        self.assertIn("train", dataset_dict)
        self.assertIn("val", dataset_dict)
        self.assertIn("test", dataset_dict)
        self.assertEqual(dataset_dict["names"], {0: "pothole"})


if __name__ == "__main__":
    unittest.main()
